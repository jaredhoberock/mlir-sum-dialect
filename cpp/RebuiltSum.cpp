#include "RebuiltSum.hpp"
#include "SumTypeInterface.hpp"
#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Transforms/RegionUtils.h>

namespace mlir::sum {

/// The arm of variant `variant` of a switch rebuilding a sum of `variants`
/// variants: its case, or the default for the last variant.
static Region &switchArm(scf::IndexSwitchOp select, size_t variant,
                         size_t variants) {
  return variant + 1 < variants ? select.getCaseRegions()[variant]
                                : select.getDefaultRegion();
}

/// The make `arm` yields at `position` when the arm builds it, or null.
static MakeOp makeYieldedAt(Region &arm, unsigned position) {
  if (!arm.hasOneBlock())
    return {};
  auto make =
      arm.front().getTerminator()->getOperand(position).getDefiningOp<MakeOp>();
  if (!make || make->getParentRegion() != &arm)
    return {};
  return make;
}

std::optional<RebuiltSum> RebuiltSum::match(Value value) {
  auto result = dyn_cast<OpResult>(value);
  auto type = dyn_cast<SumTypeInterface>(value.getType());
  if (!result || !type)
    return std::nullopt;
  unsigned position = result.getResultNumber();

  if (auto choice = dyn_cast<scf::IfOp>(result.getOwner())) {
    MakeOp then = makeYieldedAt(choice.getThenRegion(), position);
    MakeOp otherwise = makeYieldedAt(choice.getElseRegion(), position);
    if (!then || !otherwise || then.getIndex() == otherwise.getIndex())
      return std::nullopt;
    return RebuiltSum(result, {then, otherwise});
  }

  auto select = dyn_cast<scf::IndexSwitchOp>(result.getOwner());
  if (!select)
    return std::nullopt;
  size_t variants = type.getNumVariants();
  if (variants == 0 || select.getNumCases() + 1 != variants)
    return std::nullopt;
  for (auto [variant, caseValue] : llvm::enumerate(select.getCases()))
    if (caseValue != static_cast<int64_t>(variant))
      return std::nullopt;
  llvm::SmallVector<MakeOp> makes;
  for (size_t variant = 0; variant != variants; ++variant) {
    MakeOp make = makeYieldedAt(switchArm(select, variant, variants), position);
    if (!make || make.getIndex().getZExtValue() != variant)
      return std::nullopt;
    makes.push_back(make);
  }
  return RebuiltSum(result, std::move(makes));
}

Region &RebuiltSum::getArm(size_t arm) {
  if (auto choice = dyn_cast<scf::IfOp>(branch))
    return arm == 0 ? choice.getThenRegion() : choice.getElseRegion();
  return switchArm(cast<scf::IndexSwitchOp>(branch), arm, getNumArms());
}

Value RebuiltSum::getPayloadOutsideItsArm(size_t variant) {
  for (size_t arm = 0, e = getNumArms(); arm != e; ++arm) {
    if (makes[arm].getIndex().getZExtValue() != variant)
      continue;
    Value payload = makes[arm].getPayload();
    if (!payload || getArm(arm).isAncestor(payload.getParentRegion()))
      return {};
    return payload;
  }
  return {};
}

Value RebuiltSum::buildHolds(OpBuilder &builder, Location loc,
                             size_t variant) {
  if (auto choice = dyn_cast<scf::IfOp>(branch)) {
    Value condition = choice.getCondition();
    if (makes[0].getIndex().getZExtValue() == variant)
      return condition;
    if (makes[1].getIndex().getZExtValue() == variant) {
      Value always = arith::ConstantIntOp::create(builder, loc, 1, 1);
      return arith::XOrIOp::create(builder, loc, condition, always);
    }
    return arith::ConstantIntOp::create(builder, loc, 0, 1);
  }
  // The default arm takes every tag from n - 1 up.
  auto select = cast<scf::IndexSwitchOp>(branch);
  bool last = variant + 1 == getNumArms();
  Value expected = arith::ConstantIndexOp::create(builder, loc, variant);
  return arith::CmpIOp::create(
      builder, loc, last ? arith::CmpIPredicate::uge : arith::CmpIPredicate::eq,
      select.getArg(), expected);
}

Value RebuiltSum::buildTag(OpBuilder &builder, Location loc) {
  if (auto choice = dyn_cast<scf::IfOp>(branch)) {
    Value then = arith::ConstantIndexOp::create(
        builder, loc, makes[0].getIndex().getZExtValue());
    Value otherwise = arith::ConstantIndexOp::create(
        builder, loc, makes[1].getIndex().getZExtValue());
    return arith::SelectOp::create(builder, loc, choice.getCondition(), then,
                                   otherwise);
  }
  // The default arm takes every tag past the cases, so the tag clamps to the
  // last variant.
  auto select = cast<scf::IndexSwitchOp>(branch);
  Value last = arith::ConstantIndexOp::create(builder, loc, getNumArms() - 1);
  return arith::MinUIOp::create(builder, loc, select.getArg(), last);
}

/// The arm of `reader` that runs where the sum holds `variant`: a match's case,
/// or the switch's case of that tag, else its default.
static Region &readerArm(Operation *reader, uint64_t variant) {
  if (auto match = dyn_cast<MatchOp>(reader))
    return match.getCases()[variant];
  auto select = cast<scf::IndexSwitchOp>(reader);
  for (auto [value, arm] : llvm::zip(select.getCases(), select.getCaseRegions()))
    if (value == static_cast<int64_t>(variant))
      return arm;
  return select.getDefaultRegion();
}

LogicalResult RebuiltSum::sink(Operation *reader, RewriterBase &rewriter) {
  if (reader->getBlock() != branch->getBlock() ||
      !branch->isBeforeInBlock(reader))
    return failure();
  // The reader's effects move ahead of every operation it crosses, so each
  // must be pure: free of memory effects, and speculatable -- it terminates
  // and has no undefined behavior -- so running the reader first changes
  // nothing observable. A loop that never ends is memory-effect free but not
  // speculatable.
  for (Operation *op = branch->getNextNode(); op != reader;
       op = op->getNextNode())
    if (!isPure(op))
      return failure();
  // A value the reader reads from its own block, the sum aside, must exist
  // where the branch stands; one from an enclosing block already does.
  llvm::SetVector<Value> read;
  getUsedValuesDefinedAbove(reader->getRegions(), read);
  for (Value value : read)
    if (Operation *definition = value.getDefiningOp())
      if (value != sum && definition->getBlock() == branch->getBlock() &&
          !definition->isBeforeInBlock(branch))
        return failure();

  // Each reader arm is copied into at most one arm. A switch's default can
  // stand for several variants; copying it into each of their arms would split
  // the lanes that run it together into divergent copies, so a default two arms
  // reach keeps the reader where it is.
  llvm::SmallPtrSet<Region *, 4> reached;
  for (size_t arm = 0, e = getNumArms(); arm != e; ++arm)
    if (!reached.insert(&readerArm(reader, makes[arm].getIndex().getZExtValue()))
             .second)
      return failure();

  // Each arm runs the reader's arm of its variant, and its yield carries that
  // arm's results after its own.
  ValueRange added =
      grow(rewriter, reader->getResultTypes(), [&](size_t arm) {
        Region &from = readerArm(reader, makes[arm].getIndex().getZExtValue());
        IRMapping mapping;
        mapping.map(Value(sum), makes[arm].getResult());
        if (from.getNumArguments() != 0)
          mapping.map(from.getArgument(0), makes[arm].getPayload());
        for (Operation &op : from.front().without_terminator())
          rewriter.clone(op, mapping);
        return llvm::map_to_vector(
            from.front().getTerminator()->getOperands(),
            [&](Value result) { return mapping.lookupOrDefault(result); });
      });
  rewriter.replaceOp(reader, added);
  return success();
}

Value RebuiltSum::yieldPayload(size_t variant, RewriterBase &rewriter) {
  auto made = llvm::find_if(makes, [&](MakeOp make) {
    return make.getIndex().getZExtValue() == variant;
  });
  if (made == makes.end() || !made->getPayload())
    return {};
  size_t carrier = made - makes.begin();
  Value payload = made->getPayload();

  // A result the carrier arm yields the payload at and every other arm poison
  // already is the payload.
  auto yieldedAt = [&](size_t arm, unsigned position) {
    return getArm(arm).front().getTerminator()->getOperand(position);
  };
  for (unsigned position = 0, e = branch->getNumResults(); position != e;
       ++position) {
    bool carries = yieldedAt(carrier, position) == payload;
    for (size_t arm = 0, n = getNumArms(); carries && arm != n; ++arm)
      carries = arm == carrier ||
                yieldedAt(arm, position).getDefiningOp<ub::PoisonOp>();
    if (carries)
      return branch->getResult(position);
  }

  Type type = payload.getType();
  return grow(rewriter, type, [&](size_t arm) -> SmallVector<Value> {
    if (arm == carrier)
      return {payload};
    return {ub::PoisonOp::create(rewriter, payload.getLoc(), type)};
  }).front();
}

ValueRange
RebuiltSum::grow(RewriterBase &rewriter, TypeRange added,
                 llvm::function_ref<SmallVector<Value>(size_t arm)> yieldedBy) {
  for (size_t arm = 0, e = getNumArms(); arm != e; ++arm) {
    Operation *yield = getArm(arm).front().getTerminator();
    rewriter.setInsertionPoint(yield);
    SmallVector<Value> yielded(yield->getOperands());
    llvm::append_range(yielded, yieldedBy(arm));
    rewriter.modifyOpInPlace(yield, [&] { yield->setOperands(yielded); });
  }
  SmallVector<Type> types(branch->getResultTypes());
  llvm::append_range(types, added);
  rewriter.setInsertionPoint(branch);
  Operation *grown = nullptr;
  if (auto choice = dyn_cast<scf::IfOp>(branch))
    grown = scf::IfOp::create(rewriter, branch->getLoc(), types,
                              choice.getCondition(), false, false);
  else if (auto select = dyn_cast<scf::IndexSwitchOp>(branch))
    grown = scf::IndexSwitchOp::create(rewriter, branch->getLoc(), types,
                                       select.getArg(), select.getCases(),
                                       select.getNumCases());
  for (auto [from, to] : llvm::zip(branch->getRegions(), grown->getRegions()))
    rewriter.inlineRegionBefore(from, to, to.end());
  unsigned kept = branch->getNumResults();
  rewriter.replaceOp(branch, grown->getResults().take_front(kept));
  return grown->getResults().drop_front(kept);
}

} // namespace mlir::sum
