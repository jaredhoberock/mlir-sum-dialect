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
    InterfaceMethod<"Extract the payload of a variant from a sum value (unchecked)",
                    "::mlir::Value", "buildGet",
                    (ins "::mlir::OpBuilder &":$builder,
                         "::mlir::Location":$loc,
                         "::mlir::Value":$sumValue,
                         "int64_t":$index)>,
    InterfaceMethod<"Construct a sum value with the given variant and payload",
                    "::mlir::Value", "buildMake",
                    (ins "::mlir::OpBuilder &":$builder,
                         "::mlir::Location":$loc,
                         "int64_t":$index,
                         "::mlir::Value":$payload)>,
  ];

  let extraClassOf = [{
    return $_type.implementsInterface();
  }];
}

#endif // SUM_TYPE_INTERFACE_TD
