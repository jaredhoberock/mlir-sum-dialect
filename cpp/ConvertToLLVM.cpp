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

/// A make lowers to a struct built over poison: the tag narrowed or widened to
/// the tag field's width, and each payload operand in the field of its variant.
/// A sum value defines its tag and the payload of the variant it holds, and
/// nothing else: the fields of the variants the tag does not name hold values
/// no read under `sum.get`'s precondition observes, poison where the make's
/// operand is.
struct MakeOpLowering : OpConversionPattern<MakeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MakeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultTy = cast<SumTypeInterface>(op.getResult().getType());

    auto lowered = getLoweredSum(resultTy, *getTypeConverter());
    if (failed(lowered))
      return op.emitError() << "cannot lower sum type to LLVM: " << resultTy;
    Type tagTy = lowered->getBody()[0];

    // The tag holds a variant ordinal, which is never negative, so it narrows
    // or widens as an unsigned value (TagOpLowering reads it back so).
    Value tag = adaptor.getTag();
    unsigned from = tag.getType().getIntOrFloatBitWidth();
    unsigned to = tagTy.getIntOrFloatBitWidth();
    if (from > to)
      tag = LLVM::TruncOp::create(rewriter, loc, tagTy, tag);
    else if (from < to)
      tag = LLVM::ZExtOp::create(rewriter, loc, tagTy, tag);
    Value sum = LLVM::PoisonOp::create(rewriter, loc, *lowered);
    sum = LLVM::InsertValueOp::create(rewriter, loc, sum, tag,
                                      ArrayRef<int64_t>{0});
    // A nullary variant has no payload; its empty field stays as built.
    ValueRange payloads = adaptor.getPayloads();
    for (size_t variant = 0, e = resultTy.getNumVariants(); variant != e;
         ++variant) {
      if (resultTy.getPayloadTypes(variant).empty())
        continue;
      sum = LLVM::InsertValueOp::create(rewriter, loc, sum, payloads.front(),
                                        variant + 1);
      payloads = payloads.drop_front();
    }
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
    for (size_t index = 0, e = sumTy.getVariants().size(); index != e;
         ++index) {
      ArrayRef<Type> payload = sumTy.getPayloadTypes(index);
      Type lowered = payload.empty()
                         ? LLVM::LLVMStructType::getLiteral(ctx, {})
                         : typeConverter.convertType(payload.front());
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
