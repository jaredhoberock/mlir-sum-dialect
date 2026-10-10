#include "ConvertToLLVM.hpp"
#include "Sum.hpp"
#include "SumOps.hpp"
#include "SumTypeInterface.hpp"
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::sum {

// A sum lowers to a literal LLVM struct: the tag, then one field for each
// variant that carries a payload, in variant order; a nullary variant has no
// field. Every operation reads or writes a field of that struct by position.

/// The lowered struct `sumTy` converts to under `tc`, or failure when `tc` does
/// not convert it.
static FailureOr<LLVM::LLVMStructType> getLoweredSum(Type sumTy,
                                                     const TypeConverter &tc) {
  auto lowered = dyn_cast_or_null<LLVM::LLVMStructType>(tc.convertType(sumTy));
  if (!lowered)
    return failure();
  return lowered;
}

/// The field of the lowered struct that holds the payload of variant `index`:
/// the tag is field 0, and each variant carrying a payload takes the next field
/// in variant order.
static int64_t payloadField(SumTypeInterface sumTy, size_t index) {
  int64_t field = 1;
  for (size_t variant = 0; variant != index; ++variant)
    if (!isa<NoneType>(sumTy.getVariantType(variant)))
      ++field;
  return field;
}

struct GetOpLowering : OpConversionPattern<GetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sumTy = op.getInput().getType();
    if (failed(getLoweredSum(sumTy, *getTypeConverter())))
      return op.emitOpError() << "cannot lower sum type to LLVM: " << sumTy;

    int64_t field = payloadField(cast<SumTypeInterface>(sumTy),
                                 op.getIndex().getZExtValue());
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, adaptor.getInput(),
                                                      field);
    return success();
  }
};

struct IsVariantOpLowering : OpConversionPattern<IsVariantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IsVariantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto inputTy = op.getInput().getType();

    auto lowered = getLoweredSum(inputTy, *getTypeConverter());
    if (failed(lowered))
      return op.emitOpError() << "cannot lower sum type to LLVM: " << inputTy;
    Type tagTy = lowered->getBody()[0];

    Value tag = LLVM::ExtractValueOp::create(rewriter, loc, adaptor.getInput(), 0);
    Value expected = LLVM::ConstantOp::create(
        rewriter, loc, tagTy,
        rewriter.getIntegerAttr(tagTy, op.getIndex().getZExtValue()));
    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, LLVM::ICmpPredicate::eq, tag,
                                              expected);
    return success();
  }
};

struct MakeOpLowering : OpConversionPattern<MakeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MakeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultTy = op.getResult().getType();

    auto lowered = getLoweredSum(resultTy, *getTypeConverter());
    if (failed(lowered))
      return op.emitError() << "cannot lower sum type to LLVM: " << resultTy;
    Type tagTy = lowered->getBody()[0];

    // A sum value carries no undefined field: the struct starts from zero, so
    // the payload fields of the variants the value does not hold are defined.
    Value sum = LLVM::ZeroOp::create(rewriter, loc, *lowered);
    uint64_t index = op.getIndex().getZExtValue();
    Value tag = LLVM::ConstantOp::create(rewriter, loc, tagTy,
                                         rewriter.getIntegerAttr(tagTy, index));
    sum = LLVM::InsertValueOp::create(rewriter, loc, sum, tag,
                                      ArrayRef<int64_t>{0});

    // a nullary variant has no payload and no field
    if (Value payload = adaptor.getPayload())
      sum = LLVM::InsertValueOp::create(
          rewriter, loc, sum, payload,
          payloadField(cast<SumTypeInterface>(resultTy), index));

    rewriter.replaceOp(op, sum);
    return success();
  }
};

struct TagOpLowering : OpConversionPattern<TagOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TagOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto inputTy = op.getInput().getType();

    if (failed(getLoweredSum(inputTy, *getTypeConverter())))
      return op.emitOpError() << "cannot lower sum type to LLVM: " << inputTy;

    Value tag = LLVM::ExtractValueOp::create(rewriter, loc, adaptor.getInput(), 0);
    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, rewriter.getIndexType(),
                                                    tag);
    return success();
  }
};

void populateSumToLLVMConversionPatterns(LLVMTypeConverter& typeConverter,
                                        RewritePatternSet& patterns) {
  // !sum.sum<(A, none, B, ...)> → !llvm.struct<(iTag, A', B', ...)>. The struct
  // is a literal LLVM struct of the tag and the converted payloads, so LLVM lays
  // out every byte of it -- each payload at its own alignment, the whole at the
  // largest -- by the data layout of the module it is compiled in; the
  // conversion computes no size or offset of its own. A payload the converter
  // does not convert leaves the sum unconverted, so the step that lowers the
  // payload's type runs first.
  typeConverter.addConversion([&](SumType sumTy) -> std::optional<Type> {
    MLIRContext *ctx = sumTy.getContext();
    SmallVector<Type> fields = {
        IntegerType::get(ctx, sumTy.getSizeOfTagInBytes() * 8)};
    for (Type variant : sumTy.getVariants()) {
      if (isa<NoneType>(variant))
        continue;
      Type lowered = typeConverter.convertType(variant);
      if (!lowered)
        return std::nullopt;
      fields.push_back(lowered);
    }
    return LLVM::LLVMStructType::getLiteral(ctx, fields);
  });

  patterns.add<
    GetOpLowering,
    IsVariantOpLowering,
    MakeOpLowering,
    TagOpLowering
  >(typeConverter, patterns.getContext());
}

} // end mlir::sum
