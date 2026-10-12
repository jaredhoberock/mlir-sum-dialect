#pragma once

namespace mlir {

class ConversionTarget;
class RewritePatternSet;
class TypeConverter;

namespace sum {

/// Adds the sum type to the leaf conversion `converter`: a sum converts to an
/// `index` holding its tag followed by the leaves of every variant that carries
/// a payload, in variant order. Its source materialization switches on the tag
/// (`scf.index_switch`), and the arm of the variant the tag names rebuilds that
/// variant's payload from its leaves and makes the sum of it. Its target
/// materialization reads the tag with `sum.tag` and each payload with `sum.get`
/// only where `sum.is_variant` holds; where it does not, a payload's leaves are
/// absent: an automatic allocation with the leaf's type for a buffer, holding no
/// element unless the type fixes some, and `ub.poison` for anything else. In both
/// directions a payload is read only where the value carries it. Payloads convert
/// and materialize through `converter`, whatever dialect owns them. The switch
/// the source materialization builds is legal under `target`, so the leaf
/// conversion does not flatten its own rebuild again.
void populateSumLeafConversion(ConversionTarget &target, TypeConverter &converter,
                               RewritePatternSet &patterns);

} // namespace sum
} // namespace mlir
