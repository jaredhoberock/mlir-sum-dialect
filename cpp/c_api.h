#pragma once

#include <mlir-c/IR.h>
#include <mlir-c/Pass.h>
#include <mlir-c/Support.h>

#ifdef __cplusplus
extern "C" {
#endif

void sumRegisterDialect(MlirContext ctx);

MlirType sumSumTypeCreate(MlirContext ctx, const MlirType *variants, intptr_t nVariants);

// Whether `type` is a `!sum.sum`.
bool sumTypeIsASum(MlirType type);

// How many variants a `!sum.sum` carries; `type` must be a `!sum.sum`.
intptr_t sumSumTypeGetNumVariants(MlirType type);

// The variant type at `index` of a `!sum.sum`; `type` must be a `!sum.sum` and
// `index` in range.
MlirType sumSumTypeGetVariant(MlirType type, intptr_t index);

MlirOperation sumGetOpCreate(MlirLocation loc, MlirValue input, int64_t index);

MlirOperation sumIsVariantOpCreate(MlirLocation loc, MlirValue input, int64_t index);

// payload may be {nullptr} for nullary variants
MlirOperation sumMakeOpCreate(MlirLocation loc, MlirType resultTy, int64_t index, MlirValue payload);

MlirOperation sumMatchOpCreate(MlirLocation loc, MlirValue input, const MlirType *resultTypes, intptr_t nResults);

MlirOperation sumTagOpCreate(MlirLocation, MlirValue input);

MlirOperation sumYieldOpCreate(MlirLocation loc, const MlirValue *results, intptr_t nResults);

MlirPass sumCreateConvertToSCFPass();

#ifdef __cplusplus
}
#endif
