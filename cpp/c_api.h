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

// tag is an index; payloads holds one value per variant that carries a payload,
// in variant order
MlirOperation sumMakeOpCreate(MlirLocation loc, MlirType resultTy, MlirValue tag,
                              const MlirValue *payloads, intptr_t nPayloads);

MlirOperation sumMatchOpCreate(MlirLocation loc, MlirValue input, const MlirType *resultTypes, intptr_t nResults);

MlirOperation sumTagOpCreate(MlirLocation, MlirValue input);

MlirOperation sumYieldOpCreate(MlirLocation loc, const MlirValue *results, intptr_t nResults);

MlirPass sumCreateConvertToSCFPass();

#ifdef __cplusplus
}
#endif
