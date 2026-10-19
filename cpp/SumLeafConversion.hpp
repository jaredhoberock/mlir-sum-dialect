#pragma once

namespace mlir {

class TypeConverter;

namespace sum {

/// Adds the sum type to the leaf conversion `converter`: a sum converts to an
/// `index` holding its tag followed by the leaves of every variant that carries
/// a payload, in variant order. Its source materialization rebuilds the sum as
/// one `sum.make` of the tag and every variant's payload, each rebuilt from
/// its leaves, so every read of the rebuilt sum folds to a leaf. Its target
/// materialization reads the tag with `sum.tag` and each payload's leaves from
/// the payload `sum.get` reads, which is poison where the sum holds another
/// variant, so no leaf stands in for an absent one and neither materialization
/// branches. Payloads convert and materialize through `converter`, whatever
/// dialect owns them.
void populateSumLeafTypes(TypeConverter &converter);

} // namespace sum
} // namespace mlir
