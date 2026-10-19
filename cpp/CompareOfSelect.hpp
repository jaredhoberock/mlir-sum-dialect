#pragma once

namespace mlir {

class RewritePatternSet;

namespace sum {

/// Adds the fold of a comparison of a select between two constants against a
/// constant to `patterns`: the comparison is the select's condition, its
/// negation, or a constant truth.
void populateCompareOfSelectPatterns(RewritePatternSet &patterns);

} // namespace sum
} // namespace mlir
