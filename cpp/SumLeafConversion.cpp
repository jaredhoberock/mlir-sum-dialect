#include "SumLeafConversion.hpp"
#include "Sum.hpp"
#include "SumOps.hpp"
#include "SumTypes.hpp"
#include <LeafBufferAlignment.hpp>
#include <LeafConversion.hpp>
#include <StridedEnvelope.hpp>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
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
/// the view's envelope (lowering::stridedEnvelope, which folds to a constant
/// over those static values), so a dynamically sized leaf's storage is empty and
/// a rank-0 leaf, which holds one element by its type, gets one; the view is never read, since no variant carries it, and a
/// copy of it copies the envelope from its base. The storage starts at the
/// boundary every leaf buffer starts at (lowering::kLeafBufferAlignment), so the
/// stand-in keeps the contract of the buffer it stands for wherever that
/// contract is stated of it. Null for a buffer whose layout is not strided.
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
  Value storage = memref::AllocaOp::create(
      builder, loc, storageType,
      builder.getI64IntegerAttr(lowering::kLeafBufferAlignment));
  return memref::ReinterpretCastOp::create(builder, loc, buffer, storage,
                                           viewOffset, viewSizes, viewStrides);
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

  // The leaves are the tag and then each payload's leaves. The sum is one
  // assemble of the tag and each variant's payload rebuilt from its slice of
  // the leaves, so every read of it is a projection that folds to the leaf it
  // reads (RebuiltSum), however deeply sums nest, and no read branches on a
  // tag.
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
        SmallVector<Value> rebuilt;
        for (size_t index = 0; index != variants.size(); ++index) {
          FailureOr<SmallVector<Value>> payload = lowering::valuesFromLeaves(
              converter, builder, loc, payloads[index], (*slices)[index]);
          if (failed(payload))
            return Value();
          llvm::append_range(rebuilt, *payload);
        }
        return AssembleOp::create(builder, loc, type, leaves.front(), rebuilt);
      });

  // Each payload leaf is the read of its payload where the sum holds the
  // payload's variant and the leaf's stand-in (absentLeaf) elsewhere, chosen by
  // a select: a read of the payload of a variant the sum does not hold is
  // poison (GetOp), which the select does not choose there, so no branch guards
  // the read. A stand-in that is poison folds the select away.
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
          Value payload = GetOp::create(builder, loc, value, int64_t(index));
          SmallVector<Value> read;
          if (failed(lowering::appendLeaves(converter, builder, loc, payload,
                                            read)))
            return {};
          for (auto [leaf, leafType] : llvm::zip_equal(read, payloadLeaves)) {
            Value stand = absentLeaf(builder, loc, leafType);
            if (!stand)
              return {};
            leaves.push_back(
                arith::SelectOp::create(builder, loc, holds, leaf, stand));
          }
        }
        return leaves;
      });
}

} // namespace mlir::sum
