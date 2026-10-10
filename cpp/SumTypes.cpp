#include "Sum.hpp"
#include "SumOps.hpp"
#include "SumTypes.hpp"
#include <InterfaceLLVMTypeConverter.hpp>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>

using namespace mlir;
using namespace mlir::sum;

#define GET_TYPEDEF_CLASSES
#include <SumTypes.cpp.inc>

void SumDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include <SumTypes.cpp.inc>
  >();
}

size_t SumType::getSizeOfTagInBytes() const {
  size_t numVariants = getVariants().size();
  size_t tagBits = std::max<size_t>(8, llvm::PowerOf2Ceil(
      llvm::Log2_64_Ceil(numVariants)));
  return tagBits / 8;
}

/// The size of a sum type in bits: the size of the literal LLVM struct it lowers
/// to -- the tag, then one field per variant -- measured in `dataLayout`.
/// The struct comes from the lowering's own type conversion
/// [lowering::loweredForDataLayout], so the size a sum reports is the size of the
/// exact struct the lowering emits, laid out by the same data layout.
llvm::TypeSize SumType::getTypeSizeInBits(
    const DataLayout &dataLayout, DataLayoutEntryListRef params) const {
  return dataLayout.getTypeSizeInBits(
      lowering::loweredForDataLayout(*this, dataLayout));
}

/// The ABI alignment of a sum type: that of the struct it lowers to, measured
/// in `dataLayout`.
uint64_t SumType::getABIAlignment(
    const DataLayout &dataLayout, DataLayoutEntryListRef params) const {
  return dataLayout.getTypeABIAlignment(
      lowering::loweredForDataLayout(*this, dataLayout));
}
