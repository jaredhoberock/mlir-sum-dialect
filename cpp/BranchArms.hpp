#pragma once

namespace mlir {

class RewritePatternSet;

namespace sum {

/// Adds the rewrites of a branch one of whose arms cannot complete to
/// `patterns`: an `scf.if` or `scf.index_switch` with an arm that aborts, or
/// that the assertion following the branch refutes, becomes the assertion that
/// the branch takes its live arm, followed by that arm. No rewrite runs an
/// operation of an arm on a path its guard excluded.
void populateBranchArmPatterns(RewritePatternSet &patterns);

} // namespace sum
} // namespace mlir
