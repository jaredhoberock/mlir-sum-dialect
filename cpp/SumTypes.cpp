#include "Sum.hpp"
#include "SumOps.hpp"
#include "SumTypes.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>

using namespace mlir;
using namespace mlir::sum;

#define GET_TYPEDEF_CLASSES
#include "SumTypes.cpp.inc"

void SumDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "SumTypes.cpp.inc"
  >();
}

size_t SumType::getSizeOfTagInBytes() const {
  size_t numVariants = getVariants().size();
  size_t tagBits = std::max<size_t>(8, llvm::PowerOf2Ceil(
      llvm::Log2_64_Ceil(numVariants)));
  return tagBits / 8;
}

/// Returns the size of a sum type in bits: tag + max(variant sizes).
llvm::TypeSize SumType::getTypeSizeInBits(
    const DataLayout &dataLayout, DataLayoutEntryListRef params) const {
  size_t tagBits = getSizeOfTagInBytes() * 8;

  size_t maxPayloadBits = 0;
  for (Type variant : getVariants()) {
    if (isa<NoneType>(variant))
      continue;
    maxPayloadBits = std::max(maxPayloadBits,
        dataLayout.getTypeSizeInBits(variant).getFixedValue());
  }

  return llvm::TypeSize::getFixed(tagBits + maxPayloadBits);
}

/// Returns the ABI alignment of a sum type: the maximum alignment
/// required by any non-nullary variant.
uint64_t SumType::getABIAlignment(
    const DataLayout &dataLayout, DataLayoutEntryListRef params) const {
  uint64_t maxAlign = 1;
  for (Type variant : getVariants()) {
    if (isa<NoneType>(variant))
      continue;
    maxAlign = std::max(maxAlign, dataLayout.getTypeABIAlignment(variant));
  }
  return maxAlign;
}


