#include "SumLeafConversion.hpp"
#include "Sum.hpp"
#include "SumOps.hpp"
#include "SumTypes.hpp"
#include <LeafConversion.hpp>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::sum {

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
  // make of the tag and each variant's payload rebuilt from its slice of the
  // leaves, so every read of it is a projection that folds to the leaf it
  // reads, however deeply sums nest, and no read branches on a tag.
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
        return MakeOp::create(builder, loc, type, leaves.front(), rebuilt);
      });

  // Each payload leaf is the read of its payload: a read of the payload of a
  // variant the sum does not hold is poison (GetOp), which is what a leaf of a
  // variant the sum does not hold may be, so no branch guards the read and no
  // value stands in for it. A sum made from leaves gives those leaves back,
  // each read folding to the make's operand. Each leaf type's owner lowers a
  // poison of its type to a defined value where its lowering states a fact of
  // every value of the type.
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
          Value payload = GetOp::create(builder, loc, value, int64_t(index));
          if (failed(lowering::appendLeaves(converter, builder, loc, payload,
                                            leaves)))
            return {};
        }
        return leaves;
      });
}

} // namespace mlir::sum
