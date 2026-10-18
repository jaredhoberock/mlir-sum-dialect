#ifndef SUM_TD
#define SUM_TD

include "mlir/IR/OpBase.td"

def SumDialect : Dialect {
  let name = "sum";
  let summary = "Sum dialect.";
  let cppNamespace = "::mlir::sum";

  let useDefaultTypePrinterParser = 1;

  // a read of a sum built where it is read folds to the index or flag it
  // reads, which materializes as an arith constant
  let hasConstantMaterializer = 1;

  // the rewrites of a branch one of whose arms cannot complete
  // (BranchArms.hpp), over upstream's scf and cf operations
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    void registerTypes();
  }];

  // a sum's data-layout model measures the LLVM struct the sum lowers to, so
  // the LLVM dialect loads with the sum dialect rather than inside a query; the
  // leaf conversion's materializations build arith, memref and ub operations
  // wherever a sum is taken apart, so those load with it too
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::ub::UBDialect"
  ];
}

class SumDialectOp<string mnemonic, list<Trait> traits = []> :
  Op<SumDialect, mnemonic, traits>;

#endif // SUM_TD
