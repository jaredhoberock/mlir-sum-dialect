#include "ConvertToLLVM.hpp"
#include "Sum.hpp"
#include "SumOps.hpp"
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::sum {

// A sum lowers to a literal LLVM struct: the tag, then one field per variant in
// variant order, so the field of variant i is i + 1. A nullary variant's field
// is the empty struct, which occupies no bytes and adds no alignment. Every
// operation reads or writes a field of that struct by position.

/// The lowered struct `sumTy` converts to under `tc`, or failure when `tc` does
/// not convert it.
static FailureOr<LLVM::LLVMStructType> getLoweredSum(Type sumTy,
                                                     const TypeConverter &tc) {
  auto lowered = dyn_cast_or_null<LLVM::LLVMStructType>(tc.convertType(sumTy));
  if (!lowered)
    return failure();
  return lowered;
}

struct GetOpLowering : OpConversionPattern<GetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sumTy = op.getInput().getType();
    if (failed(getLoweredSum(sumTy, *getTypeConverter())))
      return op.emitOpError() << "cannot lower sum type to LLVM: " << sumTy;

    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
        op, adaptor.getInput(), op.getIndex().getZExtValue() + 1);
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

    // a nullary variant has no payload; its empty field stays as built
    if (Value payload = adaptor.getPayload())
      sum = LLVM::InsertValueOp::create(rewriter, loc, sum, payload, index + 1);

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

    // The tag holds a variant ordinal, which is never negative: it widens to
    // an index zero-extended, so a tag whose top bit is set (variant 128 of an
    // i8 tag) reads as its ordinal rather than as a negative number.
    Value tag = LLVM::ExtractValueOp::create(rewriter, loc, adaptor.getInput(), 0);
    rewriter.replaceOpWithNewOp<arith::IndexCastUIOp>(
        op, rewriter.getIndexType(), tag);
    return success();
  }
};

void populateSumToLLVMConversionPatterns(LLVMTypeConverter& typeConverter,
                                        RewritePatternSet& patterns) {
  // !sum.sum<(A, none, B, ...)> → !llvm.struct<(iTag, A', struct<()>, B', ...)>.
  // The struct is a literal LLVM struct of the tag and one field per variant, so
  // LLVM lays out every byte of it -- each payload at its own alignment, the
  // whole at the largest -- by the data layout of the module it is compiled in;
  // the conversion computes no size or offset of its own. A payload the
  // converter does not convert leaves the sum unconverted, so the step that
  // lowers the payload's type runs first.
  //
  // The payloads do not share storage as the variants of a C union or a Rust
  // enum do, because a sum lowers to an SSA aggregate and an SSA aggregate has
  // no union: insertvalue requires the inserted value's type to be the field's
  // type, and an aggregate cannot be bitcast, so a field typed as the largest
  // payload receives another payload only through memory, stored at one type
  // and loaded at another -- a stack round trip at every make and get, which a
  // struct of one field per variant does not take. (The type conversion also
  // cannot see the module's data layout to size a union, but that is the lesser
  // reason: a converter that could would still need memory.) If a sum with
  // several payloads ever needs shared storage, the form that needs neither
  // memory nor a size query is GHC's unboxed-sum layout (ubxSumRepType in
  // GHC.Types.RepType, which GHC.Stg.Unarise applies): each payload flattens
  // into scalar leaves, each leaf takes a slot of its kind (pointer, word,
  // float, double), and the variants share the slots, the sum holding of each
  // kind as many as the variant that needs the most.
  typeConverter.addConversion([&](SumType sumTy) -> std::optional<Type> {
    MLIRContext *ctx = sumTy.getContext();
    SmallVector<Type> fields = {
        IntegerType::get(ctx, sumTy.getSizeOfTagInBytes() * 8)};
    for (Type variant : sumTy.getVariants()) {
      Type lowered = isa<NoneType>(variant)
                         ? LLVM::LLVMStructType::getLiteral(ctx, {})
                         : typeConverter.convertType(variant);
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
