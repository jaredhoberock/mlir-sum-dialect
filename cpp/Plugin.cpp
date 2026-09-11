#include "ConvertToSCF.hpp"
#include "Sum.hpp"
#include "LoweringContribution.hpp"
#include <mlir/Tools/Plugins/DialectPlugin.h>

LOWERING_DEFINE_PLUGIN_CONTRACT_VERSION()

static void registerPlugin(mlir::DialectRegistry* registry) {
  registry->insert<mlir::sum::SumDialect>();
  ::mlir::PassRegistration<::mlir::sum::ConvertSumToSCFPass>();
}

extern "C" ::mlir::DialectPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
mlirGetDialectPluginInfo() {
  return {
    MLIR_PLUGIN_API_VERSION,
    "SumDialectPlugin", 
    "v0.1", 
    registerPlugin
  };
}
