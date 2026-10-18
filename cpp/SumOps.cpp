#include "RebuiltSum.hpp"
#include "Sum.hpp"
#include "SumOps.hpp"
#include "SumTypeInterface.hpp"
#include "SumTypes.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Matchers.h>
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
/// gives its payload, and a rebuilt sum (an assemble, or a switch or an if
/// selecting among makes) gives the payload of variant k that dominates the
/// read (RebuiltSum::getDominatingPayload). The read is meaningful only where
/// the sum holds variant k, and there the sum carries that payload; elsewhere
/// the read is poison, which any value refines. A make of another variant
/// declines, since it carries no payload of variant k to answer with.
OpFoldResult GetOp::fold(FoldAdaptor) {
  uint64_t index = getIndex().getZExtValue();
  if (auto make = getInput().getDefiningOp<MakeOp>()) {
    if (make.getIndex().getZExtValue() == index)
      return make.getPayload();
    return {};
  }
  if (std::optional<RebuiltSum> rebuilt = RebuiltSum::match(getInput()))
    if (Value payload = rebuilt->getDominatingPayload(index))
      return payload;
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

/// Whether a rebuilt sum holds variant k, read off what names its variant: an
/// assemble's or a switch's tag compared with k, or an if's condition where its
/// then arm makes k and the condition's negation where its else arm does
/// (RebuiltSum).
struct IsVariantOfRebuiltSum : OpRewritePattern<IsVariantOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IsVariantOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<RebuiltSum> rebuilt = RebuiltSum::match(op.getInput());
    if (!rebuilt)
      return failure();
    rewriter.replaceOp(op, rebuilt->buildHolds(rewriter, op.getLoc(),
                                               op.getIndex().getZExtValue()));
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

namespace {

/// A match of a sum a branch selects among makes (RebuiltSum) moves into the
/// branch's arms, each running the case of the variant it makes on its own
/// make's payload, so the sum is no longer built to be taken apart.
struct SinkMatchIntoRebuiltSum : OpRewritePattern<MatchOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatchOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<RebuiltSum> rebuilt = RebuiltSum::match(op.getInput());
    if (!rebuilt)
      return failure();
    return rebuilt->sink(op, rewriter);
  }
};

} // namespace

void MatchOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                          MLIRContext *context) {
  patterns.add<SinkMatchIntoRebuiltSum>(context);
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

/// The tag of a rebuilt sum is the variant it holds: an assemble's tag, a
/// switch's tag clamped to the last variant, or an if's choice between the
/// variants its two arms make (RebuiltSum).
struct TagOfRebuiltSum : OpRewritePattern<TagOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TagOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<RebuiltSum> rebuilt = RebuiltSum::match(op.getInput());
    if (!rebuilt)
      return failure();
    rewriter.replaceOp(op, rebuilt->buildTag(rewriter, op.getLoc()));
    return success();
  }
};

/// The switch on a rebuilt sum's tag that a match lowers to moves into the
/// branch's arms as the match does (SinkMatchIntoRebuiltSum). It is tried
/// before the tag folds to a choice between variants, which would cut the
/// switch's tie to the sum.
struct SinkSwitchOnRebuiltTag : OpRewritePattern<TagOp> {
  SinkSwitchOnRebuiltTag(MLIRContext *context)
      : OpRewritePattern(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(TagOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<RebuiltSum> rebuilt = RebuiltSum::match(op.getInput());
    if (!rebuilt)
      return failure();
    for (Operation *user : op->getUsers())
      if (auto select = dyn_cast<scf::IndexSwitchOp>(user))
        if (succeeded(rebuilt->sink(select, rewriter)))
          return success();
    return failure();
  }
};

/// A comparison of a sum's tag for equality with a constant k is the question
/// whether the sum holds variant k, which `sum.is_variant` asks of the sum
/// itself: each `arith.cmpi eq` of the tag with k becomes `is_variant k`, and
/// each `ne` its negation, so the question folds through whatever builds the
/// sum (IsVariantOfRebuiltSum) and an assertion of it states the condition of
/// the branch that selects the sum. It is tried before the tag of a rebuilt sum
/// becomes a choice between variants (TagOfRebuiltSum), which no fold ties back
/// to that condition. A constant that names no variant is left to arith.
struct CompareOfTag : OpRewritePattern<TagOp> {
  CompareOfTag(MLIRContext *context)
      : OpRewritePattern(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(TagOp op,
                                PatternRewriter &rewriter) const override {
    Value tag = op.getResult();
    size_t variants =
        cast<SumTypeInterface>(op.getInput().getType()).getNumVariants();
    bool rewrote = false;
    for (Operation *user : llvm::make_early_inc_range(op->getUsers())) {
      auto compare = dyn_cast<arith::CmpIOp>(user);
      if (!compare || (compare.getPredicate() != arith::CmpIPredicate::eq &&
                       compare.getPredicate() != arith::CmpIPredicate::ne))
        continue;
      Value other =
          compare.getLhs() == tag ? compare.getRhs() : compare.getLhs();
      APInt variant;
      if (!matchPattern(other, m_ConstantInt(&variant)) ||
          variant.uge(variants))
        continue;
      rewriter.setInsertionPoint(compare);
      Location loc = compare.getLoc();
      Value holds = IsVariantOp::create(rewriter, loc, op.getInput(),
                                        variant.getZExtValue());
      if (compare.getPredicate() == arith::CmpIPredicate::ne) {
        Value isTrue = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
        holds = arith::XOrIOp::create(rewriter, loc, holds, isTrue);
      }
      rewriter.replaceOp(compare, holds);
      rewrote = true;
    }
    return success(rewrote);
  }
};

} // namespace

void TagOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
  patterns.add<SinkSwitchOnRebuiltTag, CompareOfTag, TagOfRebuiltSum>(context);
}

//===----------------------------------------------------------------------===//
// AssembleOp
//===----------------------------------------------------------------------===//

Value AssembleOp::getPayloadOf(size_t variant) {
  auto type = cast<SumTypeInterface>(getResult().getType());
  if (type.getPayloadTypes(variant).empty())
    return {};
  size_t position = 0;
  for (size_t earlier = 0; earlier != variant; ++earlier)
    if (!type.getPayloadTypes(earlier).empty())
      ++position;
  return getPayloads()[position];
}

LogicalResult AssembleOp::verify() {
  auto type = cast<SumTypeInterface>(getResult().getType());
  if (type.getNumVariants() == 0)
    return emitOpError("cannot assemble a sum type with zero variants");
  SmallVector<Type> expected;
  for (size_t variant = 0, e = type.getNumVariants(); variant != e; ++variant)
    llvm::append_range(expected, type.getPayloadTypes(variant));
  if (TypeRange(getPayloads()) != TypeRange(expected))
    return emitOpError("expects one payload per variant that carries one, of "
                       "that variant's payload type, in variant order");
  return success();
}

/// An assemble of a sum's own tag and its own payloads is that sum: under the
/// variant the sum holds, each payload operand is the sum's own payload -- its
/// `sum.get`, or the select of that read where the sum holds the variant, the
/// form the leaf conversion takes a payload apart in -- and the other
/// variants' payloads are not part of the value.
OpFoldResult AssembleOp::fold(FoldAdaptor) {
  auto tag = getTag().getDefiningOp<TagOp>();
  if (!tag || tag.getInput().getType() != getResult().getType())
    return {};
  Value sum = tag.getInput();
  auto type = cast<SumTypeInterface>(sum.getType());
  for (size_t variant = 0, e = type.getNumVariants(); variant != e; ++variant) {
    Value payload = getPayloadOf(variant);
    if (!payload)
      continue;
    if (auto select = payload.getDefiningOp<arith::SelectOp>()) {
      auto holds = select.getCondition().getDefiningOp<IsVariantOp>();
      if (!holds || holds.getInput() != sum ||
          holds.getIndex().getZExtValue() != variant)
        return {};
      payload = select.getTrueValue();
    }
    auto read = payload.getDefiningOp<GetOp>();
    if (!read || read.getInput() != sum ||
        read.getIndex().getZExtValue() != variant)
      return {};
  }
  return sum;
}

namespace {

/// An assemble whose tag is the constant k is the make of variant k from its
/// payload operand of k.
struct AssembleOfConstantTag : OpRewritePattern<AssembleOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AssembleOp op,
                                PatternRewriter &rewriter) const override {
    APInt variant;
    auto type = cast<SumTypeInterface>(op.getResult().getType());
    if (!matchPattern(op.getTag(), m_ConstantInt(&variant)) ||
        variant.uge(type.getNumVariants()))
      return rewriter.notifyMatchFailure(op, "the tag names no known variant");
    size_t index = variant.getZExtValue();
    rewriter.replaceOpWithNewOp<MakeOp>(op, type, rewriter.getIndexAttr(index),
                                        op.getPayloadOf(index));
    return success();
  }
};

} // namespace

void AssembleOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                             MLIRContext *context) {
  patterns.add<AssembleOfConstantTag>(context);
}

} // end mlir::sum
