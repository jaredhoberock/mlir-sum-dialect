#include "SumLeafConversion.hpp"
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
/// buffer is an empty allocation in the automatic scope, which is legal wherever
/// the conversion runs and which an alias analysis counts as a buffer of its own;
/// any other leaf is upstream's absent value, a poison.
Value absentLeaf(OpBuilder &builder, Location loc, Type type) {
  auto buffer = dyn_cast<MemRefType>(type);
  if (!buffer)
    return ub::PoisonOp::create(builder, loc, type);
  SmallVector<int64_t> emptyShape(buffer.getRank(), 0);
  auto emptyType =
      MemRefType::get(emptyShape, buffer.getElementType(),
                      MemRefLayoutAttrInterface{}, buffer.getMemorySpace());
  Value empty = memref::AllocaOp::create(builder, loc, emptyType);
  if (emptyType == buffer)
    return empty;
  return memref::CastOp::create(builder, loc, buffer, empty);
}

} // namespace

void populateSumLeafConversion(ConversionTarget &, TypeConverter &converter,
                               RewritePatternSet &) {
  converter.addConversion(
      [&converter](SumType type, SmallVectorImpl<Type> &leaves)
          -> std::optional<LogicalResult> {
        leaves.push_back(IndexType::get(type.getContext()));
        for (Type variant : type.getVariants())
          if (!isa<NoneType>(variant) &&
              failed(converter.convertType(variant, leaves)))
            return failure();
        return success();
      });

  // The leaves are the tag and then each payload's leaves, consumed front to
  // back; every payload is rebuilt before the switch, which yields the make of
  // the variant the tag names. Variants 0 through n - 2 are cases and the last
  // is the default, so every tag a sum of n variants holds has its arm.
  converter.addSourceMaterialization(
      [&converter](OpBuilder &builder, SumType type, ValueRange leaves,
                   Location loc) -> Value {
        if (leaves.empty() || !isa<IndexType>(leaves.front().getType()))
          return Value();
        Value tag = leaves.front();
        leaves = leaves.drop_front();
        ArrayRef<Type> variants = type.getVariants();
        SmallVector<Value> payloads(variants.size());
        for (auto [index, variant] : llvm::enumerate(variants)) {
          if (isa<NoneType>(variant))
            continue;
          SmallVector<Type> payloadLeaves;
          if (failed(converter.convertType(variant, payloadLeaves)) ||
              payloadLeaves.size() > leaves.size())
            return Value();
          payloads[index] = lowering::valueFromLeaves(
              converter, builder, loc, variant,
              leaves.take_front(payloadLeaves.size()));
          if (!payloads[index])
            return Value();
          leaves = leaves.drop_front(payloadLeaves.size());
        }
        if (!leaves.empty())
          return Value();

        SmallVector<int64_t> cases;
        for (int64_t index = 0; index + 1 < int64_t(variants.size()); ++index)
          cases.push_back(index);
        auto select = scf::IndexSwitchOp::create(builder, loc, TypeRange{type},
                                                 tag, cases, cases.size());
        for (auto [index, variant] : llvm::enumerate(variants)) {
          Region &arm = index + 1 == variants.size()
                            ? select.getDefaultRegion()
                            : select.getCaseRegions()[index];
          OpBuilder::InsertionGuard guard(builder);
          builder.createBlock(&arm);
          Value made =
              payloads[index]
                  ? MakeOp::create(builder, loc, type,
                                   builder.getIndexAttr(index), payloads[index])
                        .getResult()
                  : MakeOp::create(builder, loc, type, int64_t(index))
                        .getResult();
          scf::YieldOp::create(builder, loc, made);
        }
        return select.getResult(0);
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
            for (Type leaf : payloadLeaves)
              absent.push_back(absentLeaf(builder, loc, leaf));
            scf::YieldOp::create(builder, loc, absent);
          }
          llvm::append_range(leaves, read.getResults());
        }
        return leaves;
      });
}

} // namespace mlir::sum
