#include "c_api.h"
#include "ConvertToSCF.hpp"
#include "Sum.hpp"
#include "SumOps.hpp"
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

  Value inputVal = unwrap(input);
  auto sumTy = dyn_cast<SumType>(inputVal.getType());
  if (!sumTy)
    return {};

  auto variants = sumTy.getVariants();
  if (index < 0 || static_cast<size_t>(index) >= variants.size())
    return {};

  Type resultTy = variants[static_cast<size_t>(index)];
  auto indexAttr = builder.getIndexAttr(index);

  OperationState state(unwrap(loc), GetOp::getOperationName());
  state.addOperands(inputVal);
  state.addTypes(resultTy);
  state.addAttribute("index", indexAttr);

  return wrap(builder.create(state));
}

MlirOperation sumIsVariantOpCreate(MlirLocation loc, MlirValue input, int64_t index) {
  OpBuilder builder(unwrap(loc)->getContext());
  auto op = builder.create<IsVariantOp>(unwrap(loc), unwrap(input), index);
  return wrap(op.getOperation());
}

MlirOperation sumMakeOpCreate(MlirLocation loc, MlirType resultTy, int64_t index, MlirValue payload) {
  OpBuilder builder(unwrap(loc)->getContext());
  auto indexAttr = builder.getIndexAttr(index);
  auto op = builder.create<MakeOp>(
    unwrap(loc),
    cast<SumType>(unwrap(resultTy)),
    indexAttr,
    unwrap(payload)
  );
  return wrap(op.getOperation());
}

MlirOperation sumMatchOpCreate(MlirLocation loc, MlirValue input, const MlirType *resultTypes, intptr_t nResults) {
  OpBuilder builder(unwrap(loc).getContext());
  
  auto inputValue = unwrap(input);
  auto sumType = dyn_cast<SumType>(inputValue.getType());
  if (!sumType) return {};
  auto variants = sumType.getVariants();
  
  SmallVector<Type> results;
  for (intptr_t i = 0; i < nResults; ++i) {
    results.push_back(unwrap(resultTypes[i]));
  }
  
  OperationState state(unwrap(loc), MatchOp::getOperationName());
  state.addOperands(inputValue);
  state.addTypes(results);
  
  // Add empty regions, one per variant
  for (size_t i = 0; i < variants.size(); ++i) {
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
