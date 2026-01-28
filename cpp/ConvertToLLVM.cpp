#include "ConvertToLLVM.hpp"
#include "Sum.hpp"
#include "SumOps.hpp"
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::sum {

// helper to get LLVM struct type and tag type from a SumType
static FailureOr<std::pair<LLVM::LLVMStructType, IntegerType>>
getSumLLVMTypes(SumType sumTy, const TypeConverter &tc) {
  auto llvmStructTy = dyn_cast_or_null<LLVM::LLVMStructType>(tc.convertType(sumTy));
  if (!llvmStructTy)
    return failure();
  return std::make_pair(llvmStructTy, cast<IntegerType>(llvmStructTy.getBody()[0]));
}

// helper to create an alloca for a single struct instance
static Value createStructAlloca(Location loc, LLVM::LLVMStructType structTy,
                                ConversionPatternRewriter &rewriter) {
  Value one = rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
  return rewriter.create<LLVM::AllocaOp>(
      loc, LLVM::LLVMPointerType::get(rewriter.getContext()), structTy, one);
}

struct LoweredSumView {
  LLVM::LLVMStructType llvmStructTy;
  IntegerType tagTy;
  Value alloca;
  Value tagIndex;   // index-typed tag
  Value payloadPtr; // gep [0,1] into alloca
};

static FailureOr<LoweredSumView>
getLoweredSumView(Location loc,
                  Value loweredSumValue, // adaptor.getInput()
                  SumType sumTy,
                  const TypeConverter &tc,
                  ConversionPatternRewriter &rewriter) {
  auto typesOrFail = getSumLLVMTypes(sumTy, tc);
  if (failed(typesOrFail))
    return failure();
  auto [llvmStructTy, tagTy] = *typesOrFail;

  MLIRContext *ctx = rewriter.getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);

  // alloca + store the lowered struct
  Value alloca = createStructAlloca(loc, llvmStructTy, rewriter);
  rewriter.create<LLVM::StoreOp>(loc, loweredSumValue, alloca);

  // tag = load *(gep [0,0])
  Value tagPtr = rewriter.create<LLVM::GEPOp>(
      loc, ptrTy, llvmStructTy, alloca, ArrayRef<LLVM::GEPArg>{0, 0});
  Value tag = rewriter.create<LLVM::LoadOp>(loc, tagTy, tagPtr);

  // tag integer -> index
  Value tagIndex = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), tag);

  // payloadPtr = gep [0,1]
  Value payloadPtr = rewriter.create<LLVM::GEPOp>(
      loc, ptrTy, llvmStructTy, alloca, ArrayRef<LLVM::GEPArg>{0, 1});

  return LoweredSumView{llvmStructTy, tagTy, alloca, tagIndex, payloadPtr};
}

struct GetOpLowering : OpConversionPattern<GetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto sumTy = cast<SumType>(op.getInput().getType());

    auto viewOrFail = getLoweredSumView(
        loc, adaptor.getInput(), sumTy, *getTypeConverter(), rewriter);
    if (failed(viewOrFail))
      return op.emitOpError() << "cannot lower sum type to LLVM: " << sumTy;

    Type loweredResultTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!loweredResultTy)
      return op.emitOpError() << "cannot lower result type to LLVM: "
                              << op.getResult().getType();

    Value payload = rewriter.create<LLVM::LoadOp>(loc, loweredResultTy, viewOrFail->payloadPtr);
    rewriter.replaceOp(op, payload);
    return success();
  }
};

struct IsVariantOpLowering : OpConversionPattern<IsVariantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IsVariantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto sumTy = cast<SumType>(op.getInput().getType());

    auto typesOrFail = getSumLLVMTypes(sumTy, *getTypeConverter());
    if (failed(typesOrFail))
      return op.emitOpError() << "cannot lower sum type to LLVM: " << sumTy;
    auto [llvmStructTy, tagTy] = *typesOrFail;

    // Extract tag from struct value
    Value tag = rewriter.create<LLVM::ExtractValueOp>(loc, adaptor.getInput(), 0);

    // Create constant for expected index
    int64_t index = op.getIndex().getZExtValue();
    Value expected = rewriter.create<LLVM::ConstantOp>(
        loc, tagTy, rewriter.getIntegerAttr(tagTy, index));

    // Compare tag == expected
    Value result = rewriter.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::eq, tag, expected);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct MakeOpLowering : OpConversionPattern<MakeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MakeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto sumTy = cast<SumType>(op.getResult().getType());

    auto typesOrFail = getSumLLVMTypes(sumTy, *getTypeConverter());
    if (failed(typesOrFail))
      return op.emitError() << "cannot lower sum type to LLVM: " << sumTy;
    auto [llvmStructTy, tagTy] = *typesOrFail;

    auto ptrTy = LLVM::LLVMPointerType::get(getContext());

    // allocate temporary
    auto alloca = createStructAlloca(loc, llvmStructTy, rewriter);

    // store tag
    uint64_t index = op.getIndex().getZExtValue();
    auto tag = rewriter.create<LLVM::ConstantOp>(
        loc, tagTy, rewriter.getIntegerAttr(tagTy, index));
    auto tagPtr = rewriter.create<LLVM::GEPOp>(
        loc, ptrTy, llvmStructTy, alloca, ArrayRef<LLVM::GEPArg>{0, 0});
    rewriter.create<LLVM::StoreOp>(loc, tag, tagPtr);

    // store payload through typed pointer
    auto payloadPtr = rewriter.create<LLVM::GEPOp>(
        loc, ptrTy, llvmStructTy, alloca, ArrayRef<LLVM::GEPArg>{0, 1});
    rewriter.create<LLVM::StoreOp>(loc, adaptor.getPayload(), payloadPtr);

    // load result
    auto result = rewriter.create<LLVM::LoadOp>(loc, llvmStructTy, alloca);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct TagOpLowering : OpConversionPattern<TagOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TagOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto sumTy = cast<SumType>(op.getInput().getType());

    if (failed(getSumLLVMTypes(sumTy, *getTypeConverter())))
      return op.emitOpError() << "cannot lower sum type to LLVM: " << sumTy;

    // Extract tag from struct value
    Value tag = rewriter.create<LLVM::ExtractValueOp>(loc, adaptor.getInput(), 0);

    // tag integer -> index
    Value tagIndex = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), tag);

    rewriter.replaceOp(op, tagIndex);
    return success();
  }
};

void populateSumToLLVMConversionPatterns(LLVMTypeConverter& typeConverter,
                                        RewritePatternSet& patterns) {
  typeConverter.addConversion([&](SumType sumTy) -> std::optional<Type> {
    // compute the size of the largest variant, after conversion
    DataLayout layout;
    size_t maxSize = 0;
    for (Type variant : sumTy.getVariants()) {
      Type converted = typeConverter.convertType(variant);
      if (!converted)
        return std::nullopt;
      maxSize = std::max(maxSize, layout.getTypeSize(converted).getFixedValue());
    }

    // compute tag width
    size_t numVariants = sumTy.getVariants().size();
    auto tagBits = std::max<std::size_t>(8, llvm::PowerOf2Ceil(llvm::Log2_64_Ceil(numVariants)));

    auto *ctx = sumTy.getContext();
    auto i8Ty = IntegerType::get(ctx, 8);
    auto tagTy = IntegerType::get(ctx, tagBits);
    auto payloadTy = LLVM::LLVMArrayType::get(i8Ty, maxSize);
    return LLVM::LLVMStructType::getLiteral(ctx, {tagTy, payloadTy});
  });

  patterns.add<
    GetOpLowering,
    IsVariantOpLowering,
    MakeOpLowering,
    TagOpLowering
  >(typeConverter, patterns.getContext());
}

} // end mlir::sum
