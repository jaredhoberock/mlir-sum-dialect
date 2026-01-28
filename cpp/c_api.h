#pragma once

#include <mlir-c/IR.h>
#include <mlir-c/Pass.h>
#include <mlir-c/Support.h>

#ifdef __cplusplus
extern "C" {
#endif

void sumRegisterDialect(MlirContext ctx);

MlirType sumSumTypeCreate(MlirContext ctx, const MlirType *variants, intptr_t nVariants);

MlirOperation sumGetOpCreate(MlirLocation loc, MlirValue input, int64_t index);

MlirOperation sumMakeOpCreate(MlirLocation loc, MlirType resultTy, int64_t index, MlirValue payload);

MlirOperation sumMatchOpCreate(MlirLocation loc, MlirValue input, const MlirType *resultTypes, intptr_t nResults);

MlirOperation sumTagOpCreate(MlirLocation, MlirValue input);

MlirOperation sumYieldOpCreate(MlirLocation loc, const MlirValue *results, intptr_t nResults);


#ifdef __cplusplus
}
#endif
