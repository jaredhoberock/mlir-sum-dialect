#include "RebuiltSum.hpp"
#include "Sum.hpp"
#include "SumOps.hpp"
#include "SumTypeInterface.hpp"
#include "SumTypes.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
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

/// The payload of variant k where the sum is built in view: a make gives its
/// payload operand of variant k, whatever its tag, and a sum a branch selects
/// among makes gives the payload of variant k that dominates the read
/// (RebuiltSum::getDominatingPayload). The read is meaningful only where the
/// sum holds variant k, and there the sum carries that payload; elsewhere the
/// read is poison, which any value refines. A read of a poison sum is poison:
/// a poison sum holds no payload that is not poison.
OpFoldResult GetOp::fold(FoldAdaptor adaptor) {
  if (isa_and_nonnull<ub::PoisonAttrInterface>(adaptor.getInput()))
    return adaptor.getInput();
  uint64_t index = getIndex().getZExtValue();
  if (auto make = getInput().getDefiningOp<MakeOp>())
    return make.getPayloadOf(index);
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

/// Whether a make whose tag is a constant holds variant k is whether its tag is
/// k; whether a poison sum holds it is poison.
OpFoldResult IsVariantOp::fold(FoldAdaptor adaptor) {
  if (isa_and_nonnull<ub::PoisonAttrInterface>(adaptor.getInput()))
    return adaptor.getInput();
  auto make = getInput().getDefiningOp<MakeOp>();
  if (!make)
    return {};
  std::optional<size_t> variant = make.getConstantVariant();
  if (!variant)
    return {};
  return BoolAttr::get(getContext(), *variant == getIndex().getZExtValue());
}

namespace {

/// Whether a built sum holds variant k, read off what names its variant: a
/// make's tag compared with k (a constant tag folds instead), a switch's tag
/// compared with k, or an if's condition where its then arm makes k and the
/// condition's negation where its else arm does (RebuiltSum).
struct IsVariantOfBuiltSum : OpRewritePattern<IsVariantOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IsVariantOp op,
                                PatternRewriter &rewriter) const override {
    if (auto make = op.getInput().getDefiningOp<MakeOp>()) {
      Value expected =
          arith::ConstantIndexOp::create(rewriter, op.getLoc(),
                                         op.getIndex().getZExtValue());
      rewriter.replaceOpWithNewOp<arith::CmpIOp>(
          op, arith::CmpIPredicate::eq, make.getTag(), expected);
      return success();
    }
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
  patterns.add<IsVariantOfBuiltSum>(context);
}

//===----------------------------------------------------------------------===//
// MakeOp
//===----------------------------------------------------------------------===//

Value MakeOp::getPayloadOf(size_t variant) {
  auto type = cast<SumTypeInterface>(getResult().getType());
  if (type.getPayloadTypes(variant).empty())
    return {};
  size_t position = 0;
  for (size_t earlier = 0; earlier != variant; ++earlier)
    if (!type.getPayloadTypes(earlier).empty())
      ++position;
  return getPayloads()[position];
}

std::optional<size_t> MakeOp::getConstantVariant() {
  APInt tag;
  auto type = cast<SumTypeInterface>(getResult().getType());
  if (!matchPattern(getTag(), m_ConstantInt(&tag)) ||
      tag.uge(type.getNumVariants()))
    return std::nullopt;
  return tag.getZExtValue();
}

MakeOp MakeOp::createOfVariant(OpBuilder &builder, Location loc, Type type,
                               size_t variant, Value payload) {
  auto sumType = cast<SumTypeInterface>(type);
  Value tag = arith::ConstantIndexOp::create(builder, loc, variant);
  SmallVector<Value> payloads;
  for (size_t each = 0, e = sumType.getNumVariants(); each != e; ++each) {
    ArrayRef<Type> carried = sumType.getPayloadTypes(each);
    if (carried.empty())
      continue;
    payloads.push_back(each == variant ? payload
                                       : ub::PoisonOp::create(builder, loc,
                                                              carried.front()));
  }
  return MakeOp::create(builder, loc, type, tag, payloads);
}

LogicalResult MakeOp::verify() {
  auto type = cast<SumTypeInterface>(getResult().getType());
  if (type.getNumVariants() == 0)
    return emitOpError("cannot make a sum type with zero variants");
  SmallVector<Type> expected;
  for (size_t variant = 0, e = type.getNumVariants(); variant != e; ++variant)
    llvm::append_range(expected, type.getPayloadTypes(variant));
  if (TypeRange(getPayloads()) != TypeRange(expected))
    return emitOpError("expects one payload per variant that carries one, of "
                       "that variant's payload type, in variant order");
  return success();
}

/// A make of a sum's own tag and its own payloads, each its `sum.get` (the
/// form the leaf conversion takes a payload apart in), is that sum: under the
/// variant the sum holds that payload operand is the sum's own payload, and the
/// other variants' payloads are not part of the value.
OpFoldResult MakeOp::fold(FoldAdaptor) {
  auto tag = getTag().getDefiningOp<TagOp>();
  if (!tag || tag.getInput().getType() != getResult().getType())
    return {};
  Value sum = tag.getInput();
  auto type = cast<SumTypeInterface>(sum.getType());
  for (size_t variant = 0, e = type.getNumVariants(); variant != e; ++variant) {
    Value payload = getPayloadOf(variant);
    if (!payload)
      continue;
    auto read = payload.getDefiningOp<GetOp>();
    if (!read || read.getInput() != sum ||
        read.getIndex().getZExtValue() != variant)
      return {};
  }
  return sum;
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

/// The tag of a make is its tag operand; the tag of a poison sum is poison.
OpFoldResult TagOp::fold(FoldAdaptor adaptor) {
  if (isa_and_nonnull<ub::PoisonAttrInterface>(adaptor.getInput()))
    return adaptor.getInput();
  if (auto make = getInput().getDefiningOp<MakeOp>())
    return make.getTag();
  return {};
}

namespace {

/// The tag of a sum a branch selects among makes is the variant it holds: a
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
/// sum (IsVariantOfBuiltSum) and an assertion of it states the condition of
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

} // end mlir::sum
