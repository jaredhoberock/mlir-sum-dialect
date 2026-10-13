#pragma once

#include "SumOps.hpp"
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <llvm/ADT/SmallVector.h>
#include <optional>

namespace mlir::sum {

/// A sum a region-branching operation selects among the makes its arms build:
/// result p of one of two shapes.
///
/// An `scf.index_switch` on a tag rebuilds a sum of n variants when its cases
/// are 0 through n - 2 in order with the default standing for n - 1, and the arm
/// of variant j yields at p the `sum.make j` it builds -- the shape the leaf
/// conversion's source materialization builds (createVariantSwitch). Such a sum
/// holds variant min(tag, n - 1), since the default takes every tag past the
/// cases.
///
/// An `scf.if` selects a sum when each of its two arms yields at p the
/// `sum.make` it builds and the two make different variants. Such a sum holds
/// the then arm's variant where the condition holds and the else arm's where it
/// does not.
///
/// Under the variant an arm makes, the sum's payload is that make's payload. A
/// RebuiltSum exists only for a value of one of these shapes, checked once by
/// `match`, so no two of its arms make one variant.
class RebuiltSum {
public:
  /// The rebuilt sum `value` is, or none when `value` is not a result of either
  /// shape.
  static std::optional<RebuiltSum> match(Value value);

  /// The number of arms: the sum's variant count for a switch, two for an if.
  size_t getNumArms() const { return makes.size(); }
  /// Arm `arm`: of a switch, the arm of variant `arm` (its case, or the default
  /// for the last); of an if, the then arm (0) or the else arm (1).
  Region &getArm(size_t arm);
  /// The make arm `arm` yields.
  MakeOp getMake(size_t arm) { return makes[arm]; }

  /// The payload the sum carries under variant `variant` when a value outside
  /// the selecting operation answers for it: the payload of the make of
  /// `variant`, defined outside that make's arm, so it dominates every reader of
  /// the sum. Null when no arm makes `variant` or its make's payload is built
  /// inside the arm.
  Value getPayloadOutsideItsArm(size_t variant);

  /// An i1 that holds exactly where the sum holds variant `variant`: the if's
  /// condition itself, or one built at `builder`'s insertion point.
  Value buildHolds(OpBuilder &builder, Location loc, size_t variant);

  /// The sum's tag, an index built at `builder`'s insertion point.
  Value buildTag(OpBuilder &builder, Location loc);

  /// Moves `reader` -- a `sum.match` of the sum, or the `scf.index_switch` on
  /// its `sum.tag` that a match lowers to -- into the operation that selects
  /// the sum: each arm, after its own operations, runs the reader's arm of the
  /// variant it makes, with the sum read as its make and a match arm's argument
  /// as the make's payload, and yields that arm's results beside its own; the
  /// reader's results become the operation's added results. An arm of the
  /// reader no arm makes the variant of is dropped, and none is copied twice:
  /// no two arms make one variant, and a switch reader whose default stands for
  /// the variants of two arms is refused. The reader runs at the operation's
  /// place instead of its own, so it must follow the operation in its block
  /// past only pure operations -- free of memory effects and speculatable, so
  /// each terminates and has no undefined behavior -- and read nothing defined
  /// between them; fails, changing nothing, otherwise.
  LogicalResult sink(Operation *reader, RewriterBase &rewriter);

private:
  RebuiltSum(OpResult sum, llvm::SmallVector<MakeOp> makes)
      : sum(sum), branch(sum.getOwner()), makes(std::move(makes)) {}

  /// The sum, a result of `branch`.
  OpResult sum;
  /// The scf.index_switch or scf.if the sum is a result of.
  Operation *branch;
  /// The make each arm yields, by arm.
  llvm::SmallVector<MakeOp> makes;
};

} // namespace mlir::sum
