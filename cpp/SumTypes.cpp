#include "Sum.hpp"
#include "SumOps.hpp"
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

Value SumType::buildGet(OpBuilder &builder, Location loc,
                        Value sumValue, int64_t index) const {
  return GetOp::create(builder, loc, sumValue, index);
}

Value SumType::buildMake(OpBuilder &builder, Location loc,
                         int64_t index, Value payload) const {
  return MakeOp::create(builder, loc, TypeRange{Type(*this)},
      builder.getIndexAttr(index), payload);
}
