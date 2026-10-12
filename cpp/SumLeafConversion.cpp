#include "SumLeafConversion.hpp"
#include "ConvertToSCF.hpp"
#include "RebuiltSum.hpp"
#include "Sum.hpp"
#include "SumOps.hpp"
#include "SumTypes.hpp"
#include <LeafConversion.hpp>
#include <StridedEnvelope.hpp>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::sum {

namespace {

/// The leaf standing for a payload leaf of `type` the sum does not carry. A
/// buffer is an allocation in the automatic scope, which is legal wherever the
/// conversion runs and which an alias analysis counts as a buffer of its own; any
/// other leaf is upstream's absent value, a poison.
///
/// The buffer has the leaf's type exactly, its layout included: storage of an
/// identity layout reinterpreted at the leaf's offset, sizes and strides, a
/// dynamic offset or size read as 0 and a dynamic stride as 1. The storage holds
/// the view's envelope (lowering::stridedEnvelope), so a dynamically sized
/// leaf's storage is empty and a rank-0 leaf, which holds one element by its
/// type, gets one; the view is never read, since no variant carries it, and a
/// copy of it copies the envelope from its base. Null for a buffer whose layout
/// is not strided.
Value absentLeaf(OpBuilder &builder, Location loc, Type type) {
  auto buffer = dyn_cast<MemRefType>(type);
  if (!buffer)
    return ub::PoisonOp::create(builder, loc, type);
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(buffer.getStridesAndOffset(strides, offset)))
    return Value();
  int64_t viewOffset = ShapedType::isDynamic(offset) ? 0 : offset;
  SmallVector<int64_t> viewSizes, viewStrides;
  for (auto [size, stride] : llvm::zip_equal(buffer.getShape(), strides)) {
    viewSizes.push_back(ShapedType::isDynamic(size) ? 0 : size);
    viewStrides.push_back(ShapedType::isDynamic(stride) ? 1 : stride);
  }
  int64_t reach = *getConstantIntValue(lowering::stridedEnvelope(
      builder, loc, builder.getIndexAttr(viewOffset),
      getAsIndexOpFoldResult(builder.getContext(), viewSizes),
      getAsIndexOpFoldResult(builder.getContext(), viewStrides)));
  auto storageType =
      MemRefType::get({reach}, buffer.getElementType(),
                      MemRefLayoutAttrInterface{}, buffer.getMemorySpace());
  Value storage = memref::AllocaOp::create(builder, loc, storageType);
  return memref::ReinterpretCastOp::create(builder, loc, buffer, storage,
                                           viewOffset, viewSizes, viewStrides);
}

bool isLeafRebuild(scf::IndexSwitchOp select, const TypeConverter &converter);

/// Whether every operation of `arm` but its terminator rebuilds from values
/// defined outside the switch: a nested rebuild (isLeafRebuild), or an operation
/// free of memory effects and speculatable, as the makes the materialization
/// mints are. An arm that allocates, reads or writes memory, or calls computes
/// its payload rather than rebuilding it.
bool rebuildsFromOutside(Region &arm, const TypeConverter &converter) {
  for (Operation &op : arm.front().without_terminator()) {
    auto nested = dyn_cast<scf::IndexSwitchOp>(op);
    if (!(nested && isLeafRebuild(nested, converter)) && !isPure(&op))
      return false;
  }
  return true;
}

/// Whether `select` is the source materialization's rebuild of a sum
/// (RebuiltSum) exactly: every arm rebuilds from values defined outside the
/// switch (rebuildsFromOutside), and a payload defined outside is a leaf under
/// `converter`. A switch whose arm computes a buffer is no rebuild, so the
/// conversion flattens it and the buffer crosses as a leaf of its results
/// instead of hiding inside the sum.
bool isLeafRebuild(scf::IndexSwitchOp select, const TypeConverter &converter) {
  if (select->getNumResults() != 1)
    return false;
  std::optional<RebuiltSum> rebuilt = RebuiltSum::match(select->getResult(0));
  if (!rebuilt)
    return false;
  for (size_t variant = 0, e = rebuilt->getNumVariants(); variant != e;
       ++variant) {
    Region &arm = rebuilt->getArm(variant);
    if (!rebuildsFromOutside(arm, converter))
      return false;
    Value payload = rebuilt->getMake(variant).getPayload();
    if (payload && !arm.isAncestor(payload.getParentRegion()) &&
        !converter.isLegal(payload.getType()))
      return false;
  }
  return true;
}

} // namespace

void populateSumLeafTypes(TypeConverter &converter) {
  converter.addConversion(
      [&converter](SumType type, SmallVectorImpl<Type> &leaves)
          -> std::optional<LogicalResult> {
        // A refusal leaves `leaves` as it found them, the converter's contract.
        SmallVector<Type> sumLeaves = {IndexType::get(type.getContext())};
        for (size_t index = 0, e = type.getVariants().size(); index != e; ++index)
          if (failed(converter.convertTypes(type.getPayloadTypes(index),
                                            sumLeaves)))
            return failure();
        llvm::append_range(leaves, sumLeaves);
        return success();
      });

  // The leaves are the tag and then each payload's leaves. The switch on the
  // tag selects the arm of the variant the tag names, and only that arm rebuilds
  // its payload, so a payload the sum does not carry, a nested sum's tag
  // included, is never read: the rebuild keeps the guard the take-apart states.
  converter.addSourceMaterialization(
      [&converter](OpBuilder &builder, SumType type, ValueRange leaves,
                   Location loc) -> Value {
        if (leaves.empty() || !isa<IndexType>(leaves.front().getType()))
          return Value();
        ArrayRef<Type> variants = type.getVariants();
        SmallVector<TypeRange> payloads;
        for (size_t index = 0; index != variants.size(); ++index)
          payloads.push_back(type.getPayloadTypes(index));
        FailureOr<SmallVector<ValueRange>> slices =
            lowering::leafSlices(converter, payloads, leaves.drop_front());
        if (failed(slices))
          return Value();

        bool rebuilt = true;
        scf::IndexSwitchOp select = createVariantSwitch(
            builder, loc, TypeRange{type}, leaves.front(), variants.size(),
            [&](size_t variant, Region &arm) {
              OpBuilder::InsertionGuard guard(builder);
              builder.createBlock(&arm);
              FailureOr<SmallVector<Value>> payload = lowering::valuesFromLeaves(
                  converter, builder, loc, payloads[variant], (*slices)[variant]);
              Value made;
              if (succeeded(payload)) {
                made = MakeOp::create(
                    builder, loc, type, builder.getIndexAttr(variant),
                    payload->empty() ? Value() : payload->front());
              } else {
                // The arm still yields, so the switch stays valid while the
                // materialization reports its failure.
                rebuilt = false;
                made = ub::PoisonOp::create(builder, loc, type);
              }
              scf::YieldOp::create(builder, loc, made);
            });
        return rebuilt ? select.getResult(0) : Value();
      });

  // A payload is read only under its own variant's guard, the precondition
  // `sum.get` states; the other path yields the payload's absent leaves.
  converter.addTargetMaterialization(
      [&converter](OpBuilder &builder, TypeRange, ValueRange inputs,
                   Location loc, Type) -> SmallVector<Value> {
        if (inputs.size() != 1)
          return {};
        Value value = inputs.front();
        auto type = dyn_cast<SumType>(value.getType());
        if (!type)
          return {};
        SmallVector<Value> leaves = {TagOp::create(builder, loc, value)};
        for (size_t index = 0, e = type.getVariants().size(); index != e;
             ++index) {
          SmallVector<Type> payloadLeaves;
          if (failed(converter.convertTypes(type.getPayloadTypes(index),
                                            payloadLeaves)))
            return {};
          if (payloadLeaves.empty())
            continue;
          Value holds = IsVariantOp::create(builder, loc, value, int64_t(index));
          auto read = scf::IfOp::create(builder, loc, payloadLeaves, holds,
                                        /*addThenBlock=*/true,
                                        /*addElseBlock=*/true);
          {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(read.thenBlock());
            Value payload = GetOp::create(builder, loc, value, int64_t(index));
            SmallVector<Value> carried;
            if (failed(lowering::appendLeaves(converter, builder, loc, payload,
                                              carried)))
              return {};
            scf::YieldOp::create(builder, loc, carried);
            builder.setInsertionPointToStart(read.elseBlock());
            SmallVector<Value> absent;
            for (Type leaf : payloadLeaves) {
              Value stand = absentLeaf(builder, loc, leaf);
              if (!stand)
                return {};
              absent.push_back(stand);
            }
            scf::YieldOp::create(builder, loc, absent);
          }
          llvm::append_range(leaves, read.getResults());
        }
        return leaves;
      });
}

void populateSumLeafLegality(ConversionTarget &target,
                             const TypeConverter &converter,
                             RewritePatternSet &) {
  // The rebuild is an scf.index_switch carrying the sum, a boundary the
  // structural conversion would flatten into another rebuild for its every
  // reader. It and the yields of its arms are legal instead, so one run of the
  // conversion leaves it standing, a read of it folds where the canonicalizer
  // can see through it (RebuiltSum), and a reader of the whole sum keeps the
  // switch. Every other switch and yield defers to the boundary's rule.
  target.addDynamicallyLegalOp<scf::IndexSwitchOp>(
      [&converter](scf::IndexSwitchOp select) -> std::optional<bool> {
        if (isLeafRebuild(select, converter))
          return true;
        return std::nullopt;
      });
  target.addDynamicallyLegalOp<scf::YieldOp>(
      [&converter](scf::YieldOp yield) -> std::optional<bool> {
        auto select = dyn_cast<scf::IndexSwitchOp>(yield->getParentOp());
        if (select && isLeafRebuild(select, converter))
          return true;
        return std::nullopt;
      });
}

} // namespace mlir::sum
