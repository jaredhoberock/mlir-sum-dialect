#include "Sum.hpp"
#include "SumOps.hpp"
#include "SumTypeInterface.hpp"
#include "SumTypes.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

#define GET_OP_CLASSES
#include "SumOps.cpp.inc"

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

  Type expectedTy = sumTy.getVariantType(idx);

  // Reject nullary variants
  if (isa<NoneType>(expectedTy))
    return emitOpError("cannot extract payload from nullary variant ")
           << idx;

  Type actualTy = getPayload().getType();
  if (actualTy != expectedTy)
    return emitOpError("result type ")
           << actualTy << " does not match variant " << idx
           << " payload type " << expectedTy;

  return success();
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

OpFoldResult IsVariantOp::fold(FoldAdaptor adaptor) {
  auto makeOp = getInput().getDefiningOp<MakeOp>();
  if (!makeOp)
    return {};

  bool match = makeOp.getIndex() == getIndex();
  OpBuilder builder(getContext());
  builder.setInsertionPoint(getOperation());
  return builder.create<arith::ConstantOp>(getLoc(), builder.getBoolAttr(match)).getResult();
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

  Type variantTy = sumType.getVariantType(index);
  bool isNullary = isa<NoneType>(variantTy);

  if (isNullary) {
    if (getPayload())
      return emitOpError("nullary variant ") << index << " must not have a payload";
  } else {
    if (!getPayload())
      return emitOpError("variant ") << index << " requires a payload";
    if (getPayload().getType() != variantTy)
      return emitOpError("payload type ") << getPayload().getType() << " does not match variant type " << variantTy;
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

  Type variantTy = resultType.getVariantType(index);
  bool isNullary = isa<NoneType>(variantTy);

  if (isNullary) {
    if (hasPayload)
      return parser.emitError(parser.getCurrentLocation(),
                               "nullary variant must not have a payload");
  } else {
    if (!hasPayload)
      return parser.emitError(parser.getCurrentLocation(),
                               "expected payload operand for non-nullary variant");
    if (parser.resolveOperand(payload, variantTy, result.operands))
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

    Type variantTy = inputType.getVariantType(i);
    bool isNullary = isa<NoneType>(variantTy);

    SmallVector<OpAsmParser::Argument> args;
    if (isNullary) {
      // Nullary variant: no block arguments
    } else {
      // Parse: (%arg: type)
      if (parser.parseArgumentList(args, OpAsmParser::Delimiter::Paren, true))
        return failure();

      if (args.size() != 1)
        return parser.emitError(parser.getCurrentLocation(),
                                 "expected exactly one argument for case ") << i;
      if (args[0].type != variantTy)
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
    Type variantTy = sumType.getVariantType(i);
    bool isNullary = isa<NoneType>(variantTy);

    if (isNullary) {
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
    Type variantTy = sumType.getVariantType(i);
    bool isNullary = isa<NoneType>(variantTy);

    if (isNullary) {
      if (block.getNumArguments() != 0)
        return emitOpError("case ") << i << " (nullary) expected 0 arguments, got " << block.getNumArguments();
    } else {
      if (block.getNumArguments() != 1)
        return emitOpError("case ") << i << " expected 1 argument, got " << block.getNumArguments();
      if (block.getArgument(0).getType() != variantTy)
        return emitOpError("case ") << i << " argument type " << block.getArgument(0).getType()
                                    << " does not match variant type " << variantTy;
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

} // end mlir::sum
