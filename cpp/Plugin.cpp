#include "ConvertToSCF.hpp"
#include "Sum.hpp"
#include "LoweringContribution.hpp"
#include <mlir/Tools/Plugins/DialectPlugin.h>

// The driver reads this plugin's lowering-contract version through the plugin's
// own library handle before any of its C++ runs, refusing a version other than
// the driver's. It is defined here, in the translation unit that carries the
// dialect plugin entry point, so the value is this plugin's own.
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
