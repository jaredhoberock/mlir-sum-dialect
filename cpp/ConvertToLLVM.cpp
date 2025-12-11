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

struct MakeOpLowering : OpConversionPattern<MakeOp> {
  using OpConversionPattern::OpConversionPattern;

  // Lowers sum.make to alloca + store tag + store payload + load:
  //
  //   %one = llvm.mlir.constant(1 : i64) : i64
  //   %alloca = llvm.alloca %one x !llvm.struct<(i8, array<8 x i8>)>
  //   %tag = llvm.mlir.constant(0 : i8) : i8
  //   %tag_ptr = llvm.getelementptr %alloca[0, 0]
  //   llvm.store %tag, %tag_ptr
  //   %payload_ptr = llvm.getelementptr %alloca[0, 1]
  //   llvm.store %payload, %payload_ptr
  //   %result = llvm.load %alloca
  LogicalResult
  matchAndRewrite(MakeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto sumTy = cast<SumType>(op.getResult().getType());
    auto llvmStructTy = dyn_cast_or_null<LLVM::LLVMStructType>(getTypeConverter()->convertType(sumTy));
    if (!llvmStructTy) return op.emitError() << "cannot lower sum type to LLVM: " << sumTy;

    // Get tag type from the converted struct
    auto tagTy = cast<IntegerType>(llvmStructTy.getBody()[0]);

    // Allocate temporary
    auto one = rewriter.create<LLVM::ConstantOp>(
      loc,
      rewriter.getI64Type(),
      rewriter.getI64IntegerAttr(1));

    auto alloca = rewriter.create<LLVM::AllocaOp>(
      loc,
      LLVM::LLVMPointerType::get(getContext()),
      llvmStructTy,
      one);

    // Store tag
    uint64_t index = op.getIndex().getZExtValue();
    auto tag = rewriter.create<LLVM::ConstantOp>(
      loc,
      tagTy,
      rewriter.getIntegerAttr(tagTy, index)
    );

    auto tagPtr = rewriter.create<LLVM::GEPOp>(
      loc,
      LLVM::LLVMPointerType::get(getContext()),
      llvmStructTy,
      alloca,
      ArrayRef<LLVM::GEPArg>{0, 0}
    );

    rewriter.create<LLVM::StoreOp>(loc, tag, tagPtr);

    // Store payload through typed pointer
    auto payloadPtr = rewriter.create<LLVM::GEPOp>(
      loc,
      LLVM::LLVMPointerType::get(getContext()),
      llvmStructTy,
      alloca,
      ArrayRef<LLVM::GEPArg>{0, 1}
    );

    rewriter.create<LLVM::StoreOp>(loc, adaptor.getPayload(), payloadPtr);

    // Load result
    auto result = rewriter.create<LLVM::LoadOp>(
      loc,
      llvmStructTy,
      alloca
    );

    rewriter.replaceOp(op, result);

    return success();
  }
};

struct MatchOpLowering : OpConversionPattern<MatchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MatchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Lowers sum.match to:
    // 1. Store input struct to alloca
    // 2. Load tag, convert to index
    // 3. scf.index_switch on tag
    // 4. Each case: load payload, execute case body, yield result

    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();
    auto sumTy = cast<SumType>(op.getInput().getType());
    auto llvmStructTy = dyn_cast_or_null<LLVM::LLVMStructType>(getTypeConverter()->convertType(sumTy));
    if (!llvmStructTy) return op.emitError() << "cannot lower sum type to LLVM: " << sumTy;

    auto tagTy = cast<IntegerType>(llvmStructTy.getBody()[0]);
    auto variants = sumTy.getVariants();

    // Store input to alloca for payload extraction
    Value one = rewriter.create<LLVM::ConstantOp>(
      loc,
      rewriter.getI64Type(),
      rewriter.getI64IntegerAttr(1)
    );

    Value alloca = rewriter.create<LLVM::AllocaOp>(
      loc,
      LLVM::LLVMPointerType::get(ctx),
      llvmStructTy,
      one
    );
    rewriter.create<LLVM::StoreOp>(loc, adaptor.getInput(), alloca);

    // Extract tag
    Value tagPtr = rewriter.create<LLVM::GEPOp>(
      loc,
      LLVM::LLVMPointerType::get(ctx),
      llvmStructTy,
      alloca,
      ArrayRef<LLVM::GEPArg>{0, 0}
    );
    Value tag = rewriter.create<LLVM::LoadOp>(loc, tagTy, tagPtr);

    // Convert tag to index
    Value tagIndex = rewriter.create<arith::IndexCastOp>(
      loc, 
      rewriter.getIndexType(),
      tag
    );

    // Payload pointer (type-punned via opaque ptr)
    Value payloadPtr = rewriter.create<LLVM::GEPOp>(
      loc,
      LLVM::LLVMPointerType::get(ctx),
      llvmStructTy,
      alloca,
      ArrayRef<LLVM::GEPArg>{0, 1}
    );

    // Build case values: 0, 1, ..., N-2 (last variant goes in default)
    SmallVector<int64_t> caseValues;
    for (size_t i = 0; i + 1 < variants.size(); ++i)
      caseValues.push_back(i);

    // Result types
    SmallVector<Type> resultTypes;
    for (Type resultType : op.getResultTypes())
      resultTypes.push_back(getTypeConverter()->convertType(resultType));

    // Create scf.index_switch
    auto switchOp = rewriter.create<scf::IndexSwitchOp>(
      loc,
      resultTypes,
      tagIndex,
      caseValues,
      caseValues.size()
    );

    // Fill each case region
    for (size_t i = 0; i < variants.size(); ++i) {
      PatternRewriter::InsertionGuard guard(rewriter);

      Region &tgtRegion = (i < variants.size() - 1)
        ? switchOp.getCaseRegions()[i]
        : switchOp.getDefaultRegion();
      Region &srcRegion = op.getCases()[i];

      // Create entry block in target region
      Block *entryBlock = rewriter.createBlock(&tgtRegion);
      rewriter.setInsertionPointToStart(entryBlock);

      // Load payload at start of target region
      Type variantLLVMTy = getTypeConverter()->convertType(variants[i]);
      Value payload = rewriter.create<LLVM::LoadOp>(
        loc,
        variantLLVMTy,
        payloadPtr
      );

      // Inline the source region after our entry block
      rewriter.inlineRegionBefore(srcRegion, tgtRegion, tgtRegion.end());

      // Merge the inlined block into entry block, replacing block arg with payload
      auto secondBlock = std::next(tgtRegion.begin());
      rewriter.mergeBlocks(&*secondBlock, entryBlock, {payload});

      // Convert region types
      if (failed(rewriter.convertRegionTypes(&tgtRegion, *getTypeConverter())))
        return rewriter.notifyMatchFailure(op, "region type conversion failed");

      // Replace sum.yield to scf.yield
      auto oldYield = dyn_cast<YieldOp>(tgtRegion.back().getTerminator());
      if (not oldYield)
        return rewriter.notifyMatchFailure(op, "failed to find sum.yield");
      rewriter.replaceOpWithNewOp<scf::YieldOp>(oldYield, oldYield.getResults());
    }

    rewriter.replaceOp(op, switchOp.getResults());
    return success();
  }
};

void populateSumToLLVMConversionPatterns(LLVMTypeConverter& typeConverter, RewritePatternSet& patterns) {
  typeConverter.addConversion([&](SumType sumTy) -> std::optional<Type> {
    // Compute the size of the largest variant, after conversion
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
    MakeOpLowering,
    MatchOpLowering
  >(typeConverter, patterns.getContext());

  // include conversion patterns for scf.index_switch
  populateSCFToControlFlowConversionPatterns(patterns);
}

} // end mlir::sum
