#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace mlir::sum {

/// Exposes the leaf conversion's two materializations to lit: every value a
/// `func.return` returns is taken apart into its leaves and rebuilt from them,
/// through the conversion the loaded dialects contribute, so a row runs a sum
/// that crossed as its leaves. Registered by the dialect's plugin only.
struct TestLeafRoundtripPass
    : public PassWrapper<TestLeafRoundtripPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLeafRoundtripPass);

  inline StringRef getArgument() const override {
    return "test-sum-leaf-roundtrip";
  }
  inline StringRef getDescription() const override {
    return "Take every returned value apart into its leaves and rebuild it";
  }

  void runOnOperation() override;
};

} // namespace mlir::sum
