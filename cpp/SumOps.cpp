#include "Sum.hpp"
#include "SumOps.hpp"
#include "SumTypeInterface.hpp"
#include "SumTypes.hpp"
#include <mlir/IR/Builders.h>

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

//===----------------------------------------------------------------------===//
// MakeOp
//===----------------------------------------------------------------------===//

LogicalResult MakeOp::verify() {
  auto sumType = cast<SumTypeInterface>(getResult().getType());
  size_t numVariants = sumType.getNumVariants();
  uint64_t index = getIndex().getZExtValue();

  if (index >= numVariants)
    return emitOpError("variant index ") << index << " is out of bounds for sum type with " << numVariants << " variants";

  if (getPayload().getType() != sumType.getVariantType(index))
    return emitOpError("payload type ") << getPayload().getType() << " does not match variant type " << sumType.getVariantType(index);

  return success();
}

ParseResult MakeOp::parse(OpAsmParser &parser, OperationState &result) {
  // example:
  //   sum.make 0 %val : !sum.sum<(i64, tuple<>)>

  OpAsmParser::UnresolvedOperand payload;
  Type parsedType;

  // Parse index as an integer, then convert to index type
  int64_t index;
  if (parser.parseInteger(index) ||
      parser.parseOperand(payload) ||
      parser.parseColon() ||
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

  if (parser.resolveOperand(payload, resultType.getVariantType(index), result.operands))
    return failure();

  result.addTypes(parsedType);
  return success();
}

void MakeOp::print(OpAsmPrinter &p) {
  p << ' ' << getIndex() << ' ' << getPayload() << " : " << getResult().getType();
}


//===----------------------------------------------------------------------===//
// MatchOp
//===----------------------------------------------------------------------===//

ParseResult MatchOp::parse(OpAsmParser &parser, OperationState &result) {
  // Example:
  //
  // sum.match %x : !sum.sum<(i64, tuple<>)> -> i64
  // case 0 (%inner: i64) {
  //   sum.yield %inner : i64
  // }
  // case 1 (%unit: tuple<>) {
  //   sum.yield %default : i64
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

    // Parse: (%arg: type)
    SmallVector<OpAsmParser::Argument> args;
    if (parser.parseArgumentList(args, OpAsmParser::Delimiter::Paren, true))
      return failure();

    // Verify argument type matches variant
    if (args.size() != 1)
      return parser.emitError(parser.getCurrentLocation(),
                               "expected exactly one argument for case ") << i;
    if (args[0].type != inputType.getVariantType(i))
      return parser.emitError(parser.getCurrentLocation(),
                               "argument type mismatch for case ") << i;

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

  for (auto [i, caseRegion] : llvm::enumerate(getCases())) {
    p.printNewline();
    p << "case " << i << " (";
    p.printRegionArgument(caseRegion.getArgument(0));
    p << ") ";
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

    // Check block has exactly one argument
    if (block.getNumArguments() != 1)
      return emitOpError("case ") << i << " expected 1 argument, got " << block.getNumArguments();

    // Check argument type matches variant
    if (block.getArgument(0).getType() != sumType.getVariantType(i))
      return emitOpError("case ") << i << " argument type " << block.getArgument(0).getType()
                                  << " does not match variant type " << sumType.getVariantType(i);
    
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
