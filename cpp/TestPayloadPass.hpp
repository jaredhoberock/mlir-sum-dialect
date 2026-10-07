#pragma once

#include <mlir/Pass/Pass.h>

namespace mlir::sum {

/// Exposes `payloadSources` to lit: each unregistered op `test.payload_sources`
/// with one operand gets a remark counting the values its operand can be, with
/// a note at each, or a remark that the query gave none. Registered by the
/// dialect's plugin only.
struct TestPayloadSourcesPass
    : public PassWrapper<TestPayloadSourcesPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPayloadSourcesPass);

  inline StringRef getArgument() const override {
    return "test-sum-payload-sources";
  }
  inline StringRef getDescription() const override {
    return "Report the payload query's answer at each test.payload_sources op";
  }

  void runOnOperation() override;
};

} // namespace mlir::sum
