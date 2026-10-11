#pragma once

namespace mlir {

class ConversionTarget;
class RewritePatternSet;
class TypeConverter;

namespace sum {

/// Adds the sum type to the leaf conversion `converter`: a sum converts to an
/// `index` holding its tag followed by the leaves of every variant that carries
/// a payload, in variant order. Its source materialization rebuilds each payload
/// from its leaves and selects, by `scf.index_switch` on the tag, the `sum.make`
/// of the variant the tag names. Its target materialization reads the tag with
/// `sum.tag` and each payload with `sum.get` only where `sum.is_variant` holds,
/// so a payload is read only where the value carries it; where it does not, a
/// payload's leaves are absent: an empty automatic allocation for a buffer and
/// `ub.poison` for anything else. Payloads convert and materialize through
/// `converter`, whatever dialect owns them.
void populateSumLeafConversion(ConversionTarget &target, TypeConverter &converter,
                               RewritePatternSet &patterns);

} // namespace sum
} // namespace mlir
