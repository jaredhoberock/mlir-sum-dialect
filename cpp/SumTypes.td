#ifndef SUM_TYPES_TD
#define SUM_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "Sum.td"
include "SumTypeInterface.td"

def SumType : TypeDef<SumDialect, "Sum", [
  DataLayoutTypeInterface,
  MemRefElementTypeInterface,
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

  let extraClassDeclaration = [{
    // SumTypeInterface
    ::mlir::Value buildGet(::mlir::OpBuilder &builder, ::mlir::Location loc,
                           ::mlir::Value sumValue, int64_t index) const;
    ::mlir::Value buildMake(::mlir::OpBuilder &builder, ::mlir::Location loc,
                            int64_t index, ::mlir::Value payload) const;

    /// Returns the number of bytes used for the tag (discriminant).
    /// Minimum 1 byte, rounded up to the next power of two.
    size_t getSizeOfTagInBytes() const;

    // DataLayoutTypeInterface
    ::llvm::TypeSize getTypeSizeInBits(const ::mlir::DataLayout &dataLayout,
                                       ::mlir::DataLayoutEntryListRef params) const;
    uint64_t getABIAlignment(const ::mlir::DataLayout &dataLayout,
                             ::mlir::DataLayoutEntryListRef params) const;
  }];
}

#endif // SUM_TYPES_TD
