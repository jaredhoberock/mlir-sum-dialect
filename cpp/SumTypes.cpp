#include "Sum.hpp"
#include "SumOps.hpp"
#include "SumTypes.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
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

/// Returns the natural alignment of a type in bits: power-of-2 ceil of
/// its size. MLIR's default DataLayout reports i64 as 4-byte aligned,
/// but LLVM lowers it as 8-byte aligned, so we can't trust
/// `getTypeABIAlignment` for layout-affecting decisions.
static uint64_t naturalAlignmentBits(const DataLayout &dataLayout, Type t) {
  uint64_t sizeBits = dataLayout.getTypeSizeInBits(t).getFixedValue();
  return llvm::PowerOf2Ceil(sizeBits);
}

/// The size and alignment, in bits, a variant occupies inside the sum's payload,
/// each variant measured in the LLVM form it lowers to. A scalar, a nominal
/// struct, a tuple, or a nested sum each carries a data layout that already
/// reports the layout of the struct it lowers to, so it is measured directly. A
/// memref carries no data layout of its own -- the default one would abort on
/// it -- so it is measured through the LLVM descriptor struct it lowers to,
/// whose alignment the data layout reports correctly (the struct is pointer
/// dominated), so it needs no natural-alignment widening.
static std::pair<uint64_t, uint64_t>
variantSizeAndAlignBits(const DataLayout &dataLayout, Type variant) {
  if (isa<BaseMemRefType>(variant)) {
    LLVMTypeConverter converter(variant.getContext());
    Type descriptor = converter.convertType(variant);
    return {dataLayout.getTypeSizeInBits(descriptor).getFixedValue(),
            dataLayout.getTypeABIAlignment(descriptor) * 8};
  }
  return {dataLayout.getTypeSizeInBits(variant).getFixedValue(),
          naturalAlignmentBits(dataLayout, variant)};
}

/// Returns the size of a sum type in bits: the aligned layout of
/// `{ tag, max(variant) }` as an LLVM struct. The tag is padded out to
/// the payload's natural alignment, then the payload occupies
/// `max(variant sizes)` bits, then the whole thing is rounded up so
/// arrays of the sum type pack without inter-element misalignment.
llvm::TypeSize SumType::getTypeSizeInBits(
    const DataLayout &dataLayout, DataLayoutEntryListRef params) const {
  uint64_t tagBits = getSizeOfTagInBytes() * 8;

  uint64_t maxPayloadBits = 0;
  uint64_t maxPayloadAlignBits = 8;
  for (Type variant : getVariants()) {
    if (isa<NoneType>(variant))
      continue;
    auto [sizeBits, alignBits] = variantSizeAndAlignBits(dataLayout, variant);
    maxPayloadBits = std::max(maxPayloadBits, sizeBits);
    maxPayloadAlignBits = std::max(maxPayloadAlignBits, alignBits);
  }

  uint64_t paddedTagBits = llvm::alignTo(tagBits, maxPayloadAlignBits);
  uint64_t totalBits = paddedTagBits + maxPayloadBits;
  totalBits = llvm::alignTo(totalBits, maxPayloadAlignBits);

  return llvm::TypeSize::getFixed(totalBits);
}

/// Returns the ABI alignment of a sum type: the maximum natural
/// alignment required by any non-nullary variant. We use natural
/// alignment (power-of-2 ceil of size) rather than
/// `getTypeABIAlignment`, which MLIR's default DataLayout reports too
/// loosely for large integers.
uint64_t SumType::getABIAlignment(
    const DataLayout &dataLayout, DataLayoutEntryListRef params) const {
  uint64_t maxAlignBits = 8;
  for (Type variant : getVariants()) {
    if (isa<NoneType>(variant))
      continue;
    maxAlignBits = std::max(maxAlignBits,
        variantSizeAndAlignBits(dataLayout, variant).second);
  }
  return maxAlignBits / 8;
}


