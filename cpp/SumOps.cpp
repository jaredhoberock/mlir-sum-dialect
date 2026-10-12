#include "RebuiltSum.hpp"
#include "Sum.hpp"
#include "SumOps.hpp"
#include "SumTypeInterface.hpp"
#include "SumTypes.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>

#define GET_OP_CLASSES
#include <SumOps.cpp.inc>

namespace mlir::sum {

//===----------------------------------------------------------------------===//
// GetOp
//===----------------------------------------------------------------------===//

LogicalResult GetOp::verify() {
  auto sumTy = dyn_cast<SumTypeInterface>(getInput().getType());
  if (!sumTy)
    return emitOpError("expected operand to implement SumTypeInterface");

  size_t numVariants = sumTy.getNumVariants();
  if (numVariants == 0)
    return emitOpError("cannot get payload from sum type with zero variants");

  uint64_t idx = getIndex().getZExtValue();
  if (idx >= numVariants)
    return emitOpError("index ")
           << idx << " out of range for sum type with "
           << numVariants << " variants";

  ArrayRef<Type> payload = sumTy.getPayloadTypes(idx);
  if (payload.empty())
    return emitOpError("cannot extract payload from nullary variant ")
           << idx;

  Type actualTy = getPayload().getType();
  if (actualTy != payload.front())
    return emitOpError("result type ")
           << actualTy << " does not match variant " << idx
           << " payload type " << payload.front();

  return success();
}

/// The payload of variant k where the sum is built in view: a make of variant k
/// gives its payload, and a rebuilt sum gives the payload its arm k makes when
/// that payload is defined outside the arm, so it dominates the read. A make of
/// another variant declines, since the read's precondition does not hold there
/// and no value is the right answer.
OpFoldResult GetOp::fold(FoldAdaptor) {
  uint64_t index = getIndex().getZExtValue();
  if (auto make = getInput().getDefiningOp<MakeOp>()) {
    if (make.getIndex().getZExtValue() == index)
      return make.getPayload();
    return {};
  }
  if (std::optional<RebuiltSum> rebuilt = RebuiltSum::match(getInput())) {
    Value payload = rebuilt->getMake(index).getPayload();
    if (!rebuilt->getArm(index).isAncestor(payload.getParentRegion()))
      return payload;
  }
  return {};
}

//===----------------------------------------------------------------------===//
// IsVariantOp
//===----------------------------------------------------------------------===//

LogicalResult IsVariantOp::verify() {
  auto sumType = llvm::cast<SumTypeInterface>(getInput().getType());
  int64_t index = getIndex().getZExtValue();
  int64_t numVariants = sumType.getNumVariants();

  if (index < 0 || index >= numVariants) {
    return emitOpError("variant index ")
           << index << " is out of bounds for sum type with "
           << numVariants << " variants";
  }

  return success();
}

/// Whether a make holds variant k is whether it makes variant k.
OpFoldResult IsVariantOp::fold(FoldAdaptor) {
  auto make = getInput().getDefiningOp<MakeOp>();
  if (!make)
    return {};
  return BoolAttr::get(getContext(), make.getIndex() == getIndex());
}

namespace {

/// Whether a rebuilt sum holds variant k, read off its tag: the tag equals k
/// for a variant its switch has a case for, and the last variant's default arm
/// takes every tag from n - 1 up.
struct IsVariantOfRebuiltSum : OpRewritePattern<IsVariantOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IsVariantOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<RebuiltSum> rebuilt = RebuiltSum::match(op.getInput());
    if (!rebuilt)
      return failure();
    uint64_t index = op.getIndex().getZExtValue();
    bool last = index + 1 == rebuilt->getNumVariants();
    Value variant = arith::ConstantIndexOp::create(rewriter, op.getLoc(), index);
    rewriter.replaceOpWithNewOp<arith::CmpIOp>(
        op, last ? arith::CmpIPredicate::uge : arith::CmpIPredicate::eq,
        rebuilt->getTag(), variant);
    return success();
  }
};

} // namespace

void IsVariantOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {
  patterns.add<IsVariantOfRebuiltSum>(context);
}

//===----------------------------------------------------------------------===//
// MakeOp
//===----------------------------------------------------------------------===//

LogicalResult MakeOp::verify() {
  auto sumType = cast<SumTypeInterface>(getResult().getType());
  size_t numVariants = sumType.getNumVariants();
  uint64_t index = getIndex().getZExtValue();

  if (index >= numVariants)
    return emitOpError("variant index ") << index << " is out of bounds for sum type with " << numVariants << " variants";

  ArrayRef<Type> payload = sumType.getPayloadTypes(index);
  if (payload.empty()) {
    if (getPayload())
      return emitOpError("nullary variant ") << index << " must not have a payload";
  } else {
    if (!getPayload())
      return emitOpError("variant ") << index << " requires a payload";
    if (getPayload().getType() != payload.front())
      return emitOpError("payload type ") << getPayload().getType() << " does not match variant type " << payload.front();
  }

  return success();
}

ParseResult MakeOp::parse(OpAsmParser &parser, OperationState &result) {
  // examples:
  //   sum.make 0 %val : !sum.sum<(i64, none)>
  //   sum.make 1 : !sum.sum<(i64, none)>

  Type parsedType;

  // Parse index as an integer, then convert to index type
  int64_t index;
  if (parser.parseInteger(index))
    return failure();

  // Try to parse an optional operand (payload)
  OpAsmParser::UnresolvedOperand payload;
  auto optResult = parser.parseOptionalOperand(payload);
  bool hasPayload = optResult.has_value() && succeeded(*optResult);

  if (parser.parseColon() ||
      parser.parseType(parsedType))
    return failure();

  auto resultType = dyn_cast<SumTypeInterface>(parsedType);
  if (!resultType)
    return parser.emitError(parser.getNameLoc(), "expected type implementing SumTypeInterface");

  // Create index attribute with index type
  auto indexType = parser.getBuilder().getIndexType();
  result.addAttribute("index", parser.getBuilder().getIntegerAttr(indexType, index));

  size_t numVariants = resultType.getNumVariants();
  if (index >= (int64_t)numVariants)
    return parser.emitError(parser.getNameLoc(), "variant index out of bounds");

  ArrayRef<Type> payloadTypes = resultType.getPayloadTypes(index);
  if (payloadTypes.empty()) {
    if (hasPayload)
      return parser.emitError(parser.getCurrentLocation(),
                               "nullary variant must not have a payload");
  } else {
    if (!hasPayload)
      return parser.emitError(parser.getCurrentLocation(),
                               "expected payload operand for non-nullary variant");
    if (parser.resolveOperand(payload, payloadTypes.front(), result.operands))
      return failure();
  }

  result.addTypes(parsedType);
  return success();
}

void MakeOp::print(OpAsmPrinter &p) {
  p << ' ' << getIndex();
  if (getPayload())
    p << ' ' << getPayload();
  p << " : " << getResult().getType();
}


//===----------------------------------------------------------------------===//
// MatchOp
//===----------------------------------------------------------------------===//

ParseResult MatchOp::parse(OpAsmParser &parser, OperationState &result) {
  // Example:
  //
  // sum.match %x : !sum.sum<(i64, none)> -> i64
  // case 0 (%inner: i64) {
  //   sum.yield %inner : i64
  // }
  // case 1 {
  //   %c = arith.constant 0 : i64
  //   sum.yield %c : i64
  // }

  OpAsmParser::UnresolvedOperand input;
  Type parsedType;
  Type resultType;

  // Parse: %x : <type>
  if (parser.parseOperand(input) ||
      parser.parseColon() ||
      parser.parseType(parsedType))
    return failure();

  auto inputType = dyn_cast<SumTypeInterface>(parsedType);
  if (!inputType)
    return parser.emitError(parser.getNameLoc(), "expected type implementing SumTypeInterface");

  if (parser.resolveOperand(input, parsedType, result.operands))
    return failure();

  // Parse optional: -> resultType
  if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseType(resultType))
      return failure();
    result.addTypes(resultType);
  }

  // Parse cases
  size_t numVariants = inputType.getNumVariants();
  for (size_t i = 0; i < numVariants; ++i) {
    // Parse: case N
    if (parser.parseKeyword("case"))
      return failure();

    int64_t caseIndex;
    if (parser.parseInteger(caseIndex))
      return failure();

    if (caseIndex != (int64_t)i)
      return parser.emitError(parser.getCurrentLocation(),
                               "expected case ") << i << ", got " << caseIndex;

    ArrayRef<Type> payloadTypes = inputType.getPayloadTypes(i);

    // A case takes its variant's payload as its block argument, and a nullary
    // case takes none: (%arg: type)
    SmallVector<OpAsmParser::Argument> args;
    if (!payloadTypes.empty()) {
      if (parser.parseArgumentList(args, OpAsmParser::Delimiter::Paren, true))
        return failure();

      if (args.size() != 1)
        return parser.emitError(parser.getCurrentLocation(),
                                 "expected exactly one argument for case ") << i;
      if (args[0].type != payloadTypes.front())
        return parser.emitError(parser.getCurrentLocation(),
                                 "argument type mismatch for case ") << i;
    }

    // Parse region body
    Region *caseRegion = result.addRegion();
    if (parser.parseRegion(*caseRegion, args))
      return failure();

    // Ensure terminator
    MatchOp::ensureTerminator(*caseRegion, parser.getBuilder(),
                               result.location);
  }

  return success();
}

void MatchOp::print(OpAsmPrinter &p) {
  p << ' ' << getInput() << " : " << getInput().getType();

  if (getNumResults())
    p << " -> " << getResultTypes();

  auto sumType = cast<SumTypeInterface>(getInput().getType());
  for (auto [i, caseRegion] : llvm::enumerate(getCases())) {
    p.printNewline();
    if (sumType.getPayloadTypes(i).empty()) {
      p << "case " << i << " ";
    } else {
      p << "case " << i << " (";
      p.printRegionArgument(caseRegion.getArgument(0));
      p << ") ";
    }
    p.printRegion(caseRegion, false, true);
  }
}

LogicalResult MatchOp::verify() {
  auto sumType = cast<SumTypeInterface>(getInput().getType());
  size_t numVariants = sumType.getNumVariants();

  // Check number of cases matches number of variants
  if (getCases().size() != numVariants)
    return emitOpError("expected ") << numVariants << " cases, got " << getCases().size();

  // Check each case
  for (auto [i, caseRegion] : llvm::enumerate(getCases())) {
    Block &block = caseRegion.front();
    ArrayRef<Type> payloadTypes = sumType.getPayloadTypes(i);
    if (payloadTypes.empty()) {
      if (block.getNumArguments() != 0)
        return emitOpError("case ") << i << " (nullary) expected 0 arguments, got " << block.getNumArguments();
    } else {
      if (block.getNumArguments() != 1)
        return emitOpError("case ") << i << " expected 1 argument, got " << block.getNumArguments();
      if (block.getArgument(0).getType() != payloadTypes.front())
        return emitOpError("case ") << i << " argument type " << block.getArgument(0).getType()
                                    << " does not match variant type " << payloadTypes.front();
    }

    auto yield = cast<YieldOp>(block.getTerminator());

    // Check yield result count matches
    if (yield.getResults().size() != getNumResults())
      return emitOpError("case ") << i << " yields " << yield.getResults().size()
                                  << " values but expected " << getNumResults();

    // Check each yield type matches
    for (auto [j, pair] : llvm::enumerate(llvm::zip(yield.getResults(), getResults()))) {
      auto [yieldedVal, matchResult] = pair;
      if (yieldedVal.getType() != matchResult.getType())
        return emitOpError("case ") << i << " yield #" << j << " has type "
                                    << yieldedVal.getType() << " but expected "
                                    << matchResult.getType();
    }
  }

  return success();
}

void MatchOp::getSuccessorRegions(RegionBranchPoint point,
                                  SmallVectorImpl<RegionSuccessor> &successors) {
  // Every case returns to the op.
  if (!point.isParent()) {
    successors.emplace_back(getOperation(), getResults());
    return;
  }
  // Any case may be entered; its payload argument is read from the sum value,
  // not forwarded from an operand.
  for (Region &caseRegion : getCases())
    successors.emplace_back(&caseRegion);
}

void MatchOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands, SmallVectorImpl<InvocationBounds> &bounds) {
  bounds.append(getNumRegions(), InvocationBounds(/*lb=*/0, /*ub=*/1));
}

//===----------------------------------------------------------------------===//
// TagOp
//===----------------------------------------------------------------------===//

LogicalResult TagOp::verify() {
  auto sumTy = dyn_cast<SumTypeInterface>(getInput().getType());
  if (!sumTy)
    return emitOpError("expected operand to implement SumTypeInterface");

  if (sumTy.getNumVariants() == 0)
    return emitOpError("cannot take tag of zero-variant sum type");

  return success();
}

/// The tag of a make is the variant it makes.
OpFoldResult TagOp::fold(FoldAdaptor) {
  if (auto make = getInput().getDefiningOp<MakeOp>())
    return make.getIndexAttr();
  return {};
}

namespace {

/// The tag of a rebuilt sum is its switch's tag clamped to the last variant,
/// min(tag, n - 1), since the default arm takes every tag past the cases.
struct TagOfRebuiltSum : OpRewritePattern<TagOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TagOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<RebuiltSum> rebuilt = RebuiltSum::match(op.getInput());
    if (!rebuilt)
      return failure();
    Value last = arith::ConstantIndexOp::create(rewriter, op.getLoc(),
                                                rebuilt->getNumVariants() - 1);
    rewriter.replaceOpWithNewOp<arith::MinUIOp>(op, rebuilt->getTag(), last);
    return success();
  }
};

} // namespace

void TagOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
  patterns.add<TagOfRebuiltSum>(context);
}

} // end mlir::sum
