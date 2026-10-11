#include "SumLeafConversion.hpp"
#include "ConvertToSCF.hpp"
#include "Sum.hpp"
#include "SumOps.hpp"
#include "SumTypes.hpp"
#include <LeafConversion.hpp>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/BuiltinTypes.h>
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
/// as many elements as that view reaches, so a dynamically sized leaf's is empty,
/// and a copy of the view reads only storage that exists: a static shape keeps its
/// elements, and a rank-0 leaf, which holds exactly one element by its type, gets
/// one. Null for a buffer whose layout is not strided.
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
  int64_t reach = viewOffset + 1;
  for (auto [size, stride] : llvm::zip_equal(buffer.getShape(), strides)) {
    viewSizes.push_back(ShapedType::isDynamic(size) ? 0 : size);
    viewStrides.push_back(ShapedType::isDynamic(stride) ? 1 : stride);
    reach += (viewSizes.back() - 1) * viewStrides.back();
  }
  if (llvm::is_contained(viewSizes, 0))
    reach = 0;
  auto storageType =
      MemRefType::get({reach}, buffer.getElementType(),
                      MemRefLayoutAttrInterface{}, buffer.getMemorySpace());
  Value storage = memref::AllocaOp::create(builder, loc, storageType);
  return memref::ReinterpretCastOp::create(builder, loc, buffer, storage,
                                           viewOffset, viewSizes, viewStrides);
}

} // namespace

void populateSumLeafConversion(ConversionTarget &, TypeConverter &converter,
                               RewritePatternSet &) {
  converter.addConversion(
      [&converter](SumType type, SmallVectorImpl<Type> &leaves)
          -> std::optional<LogicalResult> {
        // A refusal leaves `leaves` as it found them, the converter's contract.
        SmallVector<Type> sumLeaves = {IndexType::get(type.getContext())};
        for (Type variant : type.getVariants())
          if (!isa<NoneType>(variant) &&
              failed(converter.convertType(variant, sumLeaves)))
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
        SmallVector<Type> payloadTypes;
        for (Type variant : variants)
          if (!isa<NoneType>(variant))
            payloadTypes.push_back(variant);
        FailureOr<SmallVector<ValueRange>> slices = lowering::leafSlices(
            converter, payloadTypes, leaves.drop_front());
        if (failed(slices))
          return Value();

        bool rebuilt = true;
        unsigned slice = 0;
        scf::IndexSwitchOp select = createVariantSwitch(
            builder, loc, TypeRange{type}, leaves.front(), variants.size(),
            [&](size_t variant, Region &arm) {
              OpBuilder::InsertionGuard guard(builder);
              builder.createBlock(&arm);
              Value made;
              if (isa<NoneType>(variants[variant])) {
                made = MakeOp::create(builder, loc, type, int64_t(variant));
              } else if (Value payload = lowering::valueFromLeaves(
                             converter, builder, loc, variants[variant],
                             (*slices)[slice++])) {
                made = MakeOp::create(builder, loc, type,
                                      builder.getIndexAttr(variant), payload);
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
        for (auto [index, variant] : llvm::enumerate(type.getVariants())) {
          if (isa<NoneType>(variant))
            continue;
          SmallVector<Type> payloadLeaves;
          if (failed(converter.convertType(variant, payloadLeaves)))
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

} // namespace mlir::sum
