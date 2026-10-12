#pragma once

#include "SumOps.hpp"
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <llvm/ADT/SmallVector.h>
#include <optional>

namespace mlir::sum {

/// A sum an `scf.index_switch` rebuilds from its tag: the switch's one result is
/// a sum of n variants, its cases are 0 through n - 2 in order with the default
/// standing for n - 1, and the arm of variant j yields the `sum.make j` it
/// builds -- the shape the leaf conversion's source materialization builds
/// (createVariantSwitch). Such a sum holds variant min(tag, n - 1), since the
/// default takes every tag past the cases, and under variant j its payload is
/// the payload of arm j's make. A RebuiltSum exists only for a value of that
/// shape, checked once by `match`.
class RebuiltSum {
public:
  /// The rebuilt sum `value` is, or none when `value` is not the result of such
  /// a switch.
  static std::optional<RebuiltSum> match(Value value);

  /// The switch that rebuilds the sum.
  scf::IndexSwitchOp getSwitch() { return select; }
  /// The tag the switch selects on.
  Value getTag() { return select.getArg(); }
  /// The number of variants of the sum, at least one.
  size_t getNumVariants() const { return makes.size(); }
  /// The arm of variant `variant`: its case, or the default for the last.
  Region &getArm(size_t variant);
  /// The make the arm of variant `variant` yields.
  MakeOp getMake(size_t variant) { return makes[variant]; }

private:
  RebuiltSum(scf::IndexSwitchOp select, llvm::SmallVector<MakeOp> makes)
      : select(select), makes(std::move(makes)) {}

  scf::IndexSwitchOp select;
  llvm::SmallVector<MakeOp> makes;
};

} // namespace mlir::sum
