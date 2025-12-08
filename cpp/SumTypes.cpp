#include "Sum.hpp"
#include "SumTypes.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

using namespace mlir;
using namespace mlir::sum;

#define GET_TYPEDEF_CLASSES
#include "SumTypes.cpp.inc"

void SumDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "SumTypes.cpp.inc"
  >();
}
