#include "c_api.h"
#include "ConvertToSCF.hpp"
#include "Sum.hpp"
#include "SumOps.hpp"
#include "SumTypeInterface.hpp"
#include "SumTypes.hpp"
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/IR/Builders.h>

using namespace mlir;
using namespace mlir::sum;

extern "C" {

void sumRegisterDialect(MlirContext context) {
  unwrap(context)->loadDialect<SumDialect>();
}

MlirType sumSumTypeCreate(MlirContext ctx, const MlirType *wrappedVariants, intptr_t nVariants) {
  SmallVector<Type> variants;
  variants.reserve(nVariants);
  for (intptr_t i = 0; i < nVariants; ++i) {
    variants.push_back(unwrap(wrappedVariants[i]));
  }
  return wrap(SumType::get(unwrap(ctx), variants));
}

MlirOperation sumGetOpCreate(MlirLocation loc, MlirValue input, int64_t index) {
  OpBuilder builder(unwrap(loc)->getContext());
  auto op = builder.create<GetOp>(unwrap(loc), unwrap(input), index);
  return wrap(op.getOperation());
}

MlirOperation sumIsVariantOpCreate(MlirLocation loc, MlirValue input, int64_t index) {
  OpBuilder builder(unwrap(loc)->getContext());
  auto op = builder.create<IsVariantOp>(unwrap(loc), unwrap(input), index);
  return wrap(op.getOperation());
}

MlirOperation sumMakeOpCreate(MlirLocation loc, MlirType resultTy, int64_t index, MlirValue payload) {
  OpBuilder builder(unwrap(loc)->getContext());
  auto indexAttr = builder.getIndexAttr(index);
  Value payloadVal = payload.ptr ? unwrap(payload) : Value();
  auto op = builder.create<MakeOp>(
    unwrap(loc),
    unwrap(resultTy),
    indexAttr,
    payloadVal
  );
  return wrap(op.getOperation());
}

MlirOperation sumMatchOpCreate(MlirLocation loc, MlirValue input, const MlirType *resultTypes, intptr_t nResults) {
  OpBuilder builder(unwrap(loc).getContext());

  auto inputValue = unwrap(input);
  auto sumTy = cast<SumTypeInterface>(inputValue.getType());

  SmallVector<Type> results;
  for (intptr_t i = 0; i < nResults; ++i) {
    results.push_back(unwrap(resultTypes[i]));
  }

  OperationState state(unwrap(loc), MatchOp::getOperationName());
  state.addOperands(inputValue);
  state.addTypes(results);

  // Add empty regions, one per variant
  for (size_t i = 0; i < sumTy.getNumVariants(); ++i) {
    state.addRegion();
  }

  return wrap(builder.create(state));
}

MlirOperation sumTagOpCreate(MlirLocation loc, MlirValue input) {
  OpBuilder builder(unwrap(loc)->getContext());
  auto op = builder.create<TagOp>(unwrap(loc), unwrap(input));
  return wrap(op.getOperation());
}

MlirOperation sumYieldOpCreate(MlirLocation loc, const MlirValue *results, intptr_t nResults) {
  OpBuilder builder(unwrap(loc).getContext());
  
  SmallVector<Value> values;
  for (intptr_t i = 0; i < nResults; ++i) {
    values.push_back(unwrap(results[i]));
  }
  
  auto op = builder.create<YieldOp>(unwrap(loc), values);
  return wrap(op.getOperation());
}

MlirPass sumCreateConvertToSCFPass() {
  return wrap(createConvertSumToSCFPass().release());
}

} // end extern "C"
