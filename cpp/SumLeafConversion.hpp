#pragma once

namespace mlir {

class TypeConverter;

namespace sum {

/// Adds the sum type to the leaf conversion `converter`: a sum converts to an
/// `index` holding its tag followed by the leaves of every variant that carries
/// a payload, in variant order. Its source materialization rebuilds the sum as
/// one `sum.assemble` of the tag and every variant's payload, each rebuilt from
/// its leaves, so every read of the rebuilt sum folds to a leaf. Its target
/// materialization reads the tag with `sum.tag`, and each payload leaf as the
/// select, on `sum.is_variant`, of the leaf of the payload `sum.get` reads and
/// a stand-in for an absent leaf: an automatic allocation with the leaf's type
/// for a buffer, holding no element unless the type fixes some, and `ub.poison`
/// for anything else. Neither branches. Payloads convert and materialize
/// through `converter`, whatever dialect owns them.
void populateSumLeafTypes(TypeConverter &converter);

} // namespace sum
} // namespace mlir
