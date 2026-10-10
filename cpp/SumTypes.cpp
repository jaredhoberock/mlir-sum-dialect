#include "Sum.hpp"
#include "SumOps.hpp"
#include "SumTypes.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Transforms/DialectConversion.h>

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

/// The LLVM descriptor struct a memref lowers to, converted as the lowering
/// converts it: the converter is populated from every loaded dialect's
/// ConvertToLLVMPatternInterface, the population upstream's convert-to-llvm
/// performs in its normal mode, so a memref whose element belongs to another
/// dialect -- a sum, say -- converts through that dialect's conversion. The
/// target and pattern set are throwaway; only the type conversions the
/// interfaces add to the converter are consulted.
///
/// A memref no loaded dialect lowers has no descriptor and so no layout. That
/// stops compilation with a fatal error naming the memref, as the data layout
/// itself stops on a type that provides no layout, instead of handing a null
/// type to the layout query.
static Type loweredDescriptor(BaseMemRefType memref) {
  MLIRContext *ctx = memref.getContext();
  LLVMTypeConverter converter(ctx);
  ConversionTarget throwawayTarget(*ctx);
  RewritePatternSet throwawayPatterns(ctx);
  for (Dialect *dialect : ctx->getLoadedDialects())
    if (auto *iface = dyn_cast<ConvertToLLVMPatternInterface>(dialect))
      iface->populateConvertToLLVMConversionPatterns(throwawayTarget, converter,
                                                     throwawayPatterns);

  Type descriptor = converter.convertType(memref);
  if (!descriptor) {
    std::string message;
    llvm::raw_string_ostream os(message);
    os << "no loaded dialect lowers " << memref
       << " to LLVM, so the sum variant has no data layout";
    llvm::report_fatal_error(llvm::Twine(message));
  }
  return descriptor;
}

/// The size and alignment, in bits, a variant occupies inside the sum's payload,
/// each variant measured in the LLVM form it lowers to. A scalar, a nominal
/// struct, a tuple, or a nested sum each carries a data layout that already
/// reports the layout of the struct it lowers to, so it is measured directly. A
/// memref carries no data layout of its own -- the default one would abort on
/// it -- so it is measured through the LLVM descriptor struct it lowers to
/// [loweredDescriptor], whose alignment the data layout reports correctly (the
/// struct is pointer dominated), so it needs no natural-alignment widening.
static std::pair<uint64_t, uint64_t>
variantSizeAndAlignBits(const DataLayout &dataLayout, Type variant) {
  if (auto memref = dyn_cast<BaseMemRefType>(variant)) {
    Type descriptor = loweredDescriptor(memref);
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


