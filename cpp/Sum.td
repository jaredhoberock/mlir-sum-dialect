#ifndef SUM_TD
#define SUM_TD

include "mlir/IR/OpBase.td"

def SumDialect : Dialect {
  let name = "sum";
  let summary = "Sum dialect.";
  let cppNamespace = "::mlir::sum";

  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerTypes();
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect"
  ];
}

class SumDialectOp<string mnemonic, list<Trait> traits = []> :
  Op<SumDialect, mnemonic, traits>;

#endif // SUM_TD
