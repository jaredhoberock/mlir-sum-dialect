#pragma once

namespace mlir {

class RewritePatternSet;

namespace sum {

/// Adds the rewrites of a branch one of whose arms leaves nothing for its
/// readers to `patterns`: an `scf.if` or `scf.index_switch` with an arm that
/// aborts becomes the assertion that the branch takes its live arm, followed by
/// that arm; and an `scf.if` with an arm that yields only poison becomes its
/// other arm, when that arm is pure. Each reads only upstream operations.
void populateBranchArmPatterns(RewritePatternSet &patterns);

} // namespace sum
} // namespace mlir
