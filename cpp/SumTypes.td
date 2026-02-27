#ifndef SUM_TYPES_TD
#define SUM_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "Sum.td"
include "SumTypeInterface.td"

def SumType : TypeDef<SumDialect, "Sum", [
  SumTypeInterface
]> {
  let mnemonic = "sum";
  let summary = "A sum type (tagged union) with indexed variants";
  let description = [{
    Represents a sum type where each variant holds a payload type.
    Variants are identified by index, not by name.

    Example:

        !sum.sum<(i64, tuple<>)> // variant 0: i64, variant 1: unit
  }];

  let parameters = (ins
    ArrayRefParameter<"Type", "variant payload types">:$variants
  );

  let assemblyFormat = "`<` `(` $variants `)` `>`";
}

#endif // SUM_TYPES_TD
