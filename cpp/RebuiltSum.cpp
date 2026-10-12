#include "RebuiltSum.hpp"
#include "SumTypeInterface.hpp"

namespace mlir::sum {

std::optional<RebuiltSum> RebuiltSum::match(Value value) {
  auto select = value.getDefiningOp<scf::IndexSwitchOp>();
  auto type = dyn_cast<SumTypeInterface>(value.getType());
  if (!select || !type || select->getNumResults() != 1)
    return std::nullopt;
  size_t variants = type.getNumVariants();
  if (variants == 0 || select.getNumCases() + 1 != variants)
    return std::nullopt;
  for (auto [variant, caseValue] : llvm::enumerate(select.getCases()))
    if (caseValue != static_cast<int64_t>(variant))
      return std::nullopt;

  llvm::SmallVector<MakeOp> makes;
  for (size_t variant = 0; variant != variants; ++variant) {
    Region &arm = variant + 1 < variants ? select.getCaseRegions()[variant]
                                         : select.getDefaultRegion();
    if (!arm.hasOneBlock())
      return std::nullopt;
    Operation *yield = arm.front().getTerminator();
    auto make = yield->getOperand(0).getDefiningOp<MakeOp>();
    if (!make || make->getParentRegion() != &arm ||
        make.getIndex().getZExtValue() != variant)
      return std::nullopt;
    makes.push_back(make);
  }
  return RebuiltSum(select, std::move(makes));
}

Region &RebuiltSum::getArm(size_t variant) {
  return variant + 1 < getNumVariants() ? select.getCaseRegions()[variant]
                                        : select.getDefaultRegion();
}

} // namespace mlir::sum
