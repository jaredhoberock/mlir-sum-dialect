#include "Payload.hpp"
#include "Sum.hpp"
#include "SumOps.hpp"
#include <llvm/ADT/SetVector.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/Matchers.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>

namespace mlir::sum {

/// The sum `value` reads a payload of and the variant it reads, or none when
/// `value` is not a payload read.
static std::optional<std::pair<Value, unsigned>> payloadRead(Value value) {
  if (auto get = value.getDefiningOp<GetOp>())
    return std::make_pair(get.getInput(),
                          unsigned(get.getIndex().getZExtValue()));
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument)
    return std::nullopt;
  auto match = dyn_cast<MatchOp>(argument.getOwner()->getParentOp());
  if (!match)
    return std::nullopt;
  // A case region is one block whose one argument is its variant's payload,
  // and the case's position among the regions is the variant.
  return std::make_pair(match.getInput(),
                        argument.getParentRegion()->getRegionNumber());
}

/// Whether the block `terminator` ends can never reach it: an assertion of a
/// constant false stands before it.
static bool cannotReturn(Operation *terminator) {
  return llvm::any_of(*terminator->getBlock(), [](Operation &op) {
    auto assertion = dyn_cast<cf::AssertOp>(op);
    return assertion && matchPattern(assertion.getArg(), m_Zero());
  });
}

/// Adds to `sources` the values variant `index` of `sum` can carry; fails when
/// a producer of `sum` is not one the rules read.
static LogicalResult collectPayloads(Value sum, unsigned index,
                                     llvm::SetVector<Value> &sources) {
  if (auto make = sum.getDefiningOp<MakeOp>()) {
    if (make.getIndex().getZExtValue() == index && make.getPayload())
      sources.insert(make.getPayload());
    return success();
  }
  if (sum.getDefiningOp<ub::PoisonOp>())
    return success();
  if (std::optional<SmallVector<Value>> outer = payloadSources(sum)) {
    for (Value carried : *outer)
      if (failed(collectPayloads(carried, index, sources)))
        return failure();
    return success();
  }
  auto result = dyn_cast<OpResult>(sum);
  auto branch =
      result ? dyn_cast<RegionBranchOpInterface>(result.getOwner()) : nullptr;
  if (!branch)
    return failure();
  RegionBranchInverseSuccessorMapping forwarders;
  branch.getSuccessorInputOperandMapping(forwarders);
  auto forwarded = forwarders.find(sum);
  if (forwarded == forwarders.end())
    return failure();
  for (OpOperand *operand : forwarded->second) {
    // An operand the op forwards on entry is not a terminator's; a terminator
    // that cannot be reached forwards nothing.
    Operation *owner = operand->getOwner();
    if (owner != branch.getOperation() && cannotReturn(owner))
      continue;
    if (failed(collectPayloads(operand->get(), index, sources)))
      return failure();
  }
  return success();
}

std::optional<SmallVector<Value>> payloadSources(Value value) {
  std::optional<std::pair<Value, unsigned>> read = payloadRead(value);
  if (!read)
    return std::nullopt;
  llvm::SetVector<Value> sources;
  if (failed(collectPayloads(read->first, read->second, sources)))
    return std::nullopt;
  return sources.takeVector();
}

} // namespace mlir::sum
