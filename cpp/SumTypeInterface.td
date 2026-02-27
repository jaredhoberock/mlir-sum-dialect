#ifndef SUM_TYPE_INTERFACE_TD
#define SUM_TYPE_INTERFACE_TD

include "mlir/IR/OpBase.td"

def SumTypeInterface : TypeInterface<"SumTypeInterface"> {
  let cppNamespace = "::mlir::sum";
  let methods = [
    InterfaceMethod<"Return whether this type implements SumTypeInterface",
                    "bool", "implementsInterface", (ins), /*methodBody=*/[{}],
                    /*defaultImplementation=*/[{ return true; }]>,
    InterfaceMethod<"Return the number of variants",
                    "size_t", "getNumVariants", (ins), /*methodBody=*/[{}],
                    /*defaultImplementation=*/[{
                      return $_type.getVariants().size();
                    }]>,
    InterfaceMethod<"Return the payload type of variant at index",
                    "::mlir::Type", "getVariantType",
                    (ins "size_t":$index), /*methodBody=*/[{}],
                    /*defaultImplementation=*/[{
                      return $_type.getVariants()[index];
                    }]>,
  ];

  let extraClassOf = [{
    return $_type.implementsInterface();
  }];
}

#endif // SUM_TYPE_INTERFACE_TD
