#include "RebuiltSum.hpp"
#include "SumTypeInterface.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>

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
    return RebuiltSum(choice, {then, otherwise});
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
  return RebuiltSum(select, std::move(makes));
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

} // namespace mlir::sum
