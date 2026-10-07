#include "TestPayloadPass.hpp"
#include "Payload.hpp"
#include <mlir/IR/Diagnostics.h>

namespace mlir::sum {

void TestPayloadSourcesPass::runOnOperation() {
  getOperation()->walk([](Operation *probe) {
    if (probe->getName().getStringRef() != "test.payload_sources" ||
        probe->getNumOperands() != 1)
      return;
    std::optional<SmallVector<Value>> sources =
        payloadSources(probe->getOperand(0));
    if (!sources) {
      probe->emitRemark("no payload sources");
      return;
    }
    InFlightDiagnostic remark = probe->emitRemark()
                                << sources->size() << " payload sources";
    for (Value source : *sources)
      remark.attachNote(source.getLoc()) << "payload source";
  });
}

} // namespace mlir::sum
