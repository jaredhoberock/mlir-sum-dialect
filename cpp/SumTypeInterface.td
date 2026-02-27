#ifndef SUM_TYPE_INTERFACE_TD
#define SUM_TYPE_INTERFACE_TD

include "mlir/IR/OpBase.td"

def SumTypeInterface : TypeInterface<"SumTypeInterface"> {
  let cppNamespace = "::mlir::sum";
  let methods = [
    InterfaceMethod<"Return the number of variants",
                    "size_t", "getNumVariants">,
    InterfaceMethod<"Return the payload type of variant at index",
                    "::mlir::Type", "getVariantType",
                    (ins "size_t":$index)>,
  ];
}

#endif // SUM_TYPE_INTERFACE_TD
