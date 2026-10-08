#pragma once

#include <mlir/IR/Value.h>
#include <llvm/ADT/SmallVector.h>
#include <optional>

namespace mlir::sum {

/// The sum dialect's rule for one step of a backward walk through a payload
/// read: the values `value` can be when it reads the payload of variant k of a
/// sum, as the argument of a `sum.match`'s case k or as `sum.get %s, k` (the
/// spelling the sum lowering leaves inside an `scf.index_switch`). Each producer
/// of the sum is read by its own meaning: `sum.make k %v` gives `%v`; a make of
/// another variant and `ub.poison` give nothing; a result of a region-branching
/// op gives what each operand its RegionBranchOpInterface forwards into that
/// result gives, so an arm that cannot return gives nothing by yielding
/// `ub.poison`; a sum that is itself a payload read gives
/// variant k of each value it can be. The values are distinct, in the order
/// the producers are read. Returns none when `value` reads no payload, or when
/// some producer of the sum is none of these.
std::optional<llvm::SmallVector<Value>> payloadSources(Value value);

} // namespace mlir::sum
