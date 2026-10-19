#ifndef SUM_OPS_TD
#define SUM_OPS_TD

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "Sum.td"
include "SumTypes.td"
include "SumTypeInterface.td"

def IsSumTypeInterface : Type<
  CPred<"::llvm::isa<::mlir::sum::SumTypeInterface>($_self)">,
  "type that implements SumTypeInterface">;

def GetOp : SumDialectOp<"get", [Pure]> {
  let summary = "Extract the payload of a sum value (unchecked)";
  let description = [{
    Extracts the payload stored in a sum value.

    This op does not check that the sum's active tag matches the requested
    variant index. It is intended to be used only when control flow already
    guarantees the variant (e.g. inside a matching arm). Where the sum holds
    another variant the result is poison: a sum defines only the payload of
    the variant it holds, so a read elsewhere has no value to observe, and a
    fold may answer it with any value.

    Example:

        %payload = sum.get %x, 0 : !sum.sum<(i64, i1)> -> i64
  }];

  let arguments = (ins
    IsSumTypeInterface:$input,
    IndexAttr:$index
  );

  let results = (outs AnyType:$payload);

  let hasVerifier = 1;
  let hasFolder = 1;
  let assemblyFormat = "$input `,` $index attr-dict `:` qualified(type($input)) `->` type($payload)";

  let builders = [
    OpBuilder<(ins "Value":$input, "int64_t":$index), [{
      auto sumTy = cast<SumTypeInterface>(input.getType());
      Type resultType = sumTy.getVariantType(index);
      build($_builder, $_state, resultType, input, $_builder.getIndexAttr(index));
    }]>
  ];
}

def IsVariantOp : SumDialectOp<"is_variant", [Pure]> {
  let summary = "Check if a sum value is a specific variant";
  let description = [{
    Returns true if the sum value's active variant matches the given index.

    Example:
        %is_first = sum.is_variant %x, 0 : !sum.sum<(i64, i32)>
  }];
  
  let arguments = (ins
    IsSumTypeInterface:$input,
    IndexAttr:$index
  );
  let results = (outs I1:$result);
  
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let assemblyFormat = "$input `,` $index attr-dict `:` qualified(type($input))";

  let builders = [
    OpBuilder<(ins "Value":$input, "int64_t":$index), [{
      build($_builder, $_state, $_builder.getI1Type(), input, $_builder.getIndexAttr(index));
    }]>
  ];
}

def MakeOp : SumDialectOp<"make", [Pure]> {
  let summary = "Construct a sum from its tag and the payload of every variant";
  let description = [{
    Creates a value of a sum type from its parts: `tag`, the index of the
    variant the sum holds, and one payload operand per variant that carries a
    payload, in variant order. The sum holds the variant `tag` names, with that
    variant's payload operand as its payload; the payload operands of the other
    variants are not part of the value, so any value of their type, poison
    included, may stand there. A tag that names no variant makes the sum
    poison. A make whose tag is a constant k builds variant k.

    Every read of a made sum is a projection by position: `sum.tag` is `tag`,
    `sum.is_variant` of variant k compares `tag` with k, and `sum.get` of
    variant k is the payload operand of variant k.

    Example:

        %c0 = arith.constant 0 : index
        %absent = ub.poison : f32
        %x = sum.make %c0 (%val, %absent : i64, f32) : !sum.sum<(i64, none, f32)>
        %n = sum.make %tag : !sum.sum<(none, none)>
  }];

  let arguments = (ins Index:$tag, Variadic<AnyType>:$payloads);
  let results = (outs IsSumTypeInterface:$result);

  let hasVerifier = 1;
  let hasFolder = 1;
  let assemblyFormat = "$tag (` ` `(` $payloads^ `:` type($payloads) `)`)? attr-dict `:` qualified(type($result))";

  let extraClassDeclaration = [{
    /// The payload operand of variant `variant`, or null for a variant that
    /// carries no payload.
    Value getPayloadOf(size_t variant);

    /// The variant the make builds when its tag is a constant that names one,
    /// or none.
    std::optional<size_t> getConstantVariant();

    /// The make of variant `variant` of `type` holding `payload` (null for a
    /// variant that carries none), built at `builder`'s insertion point: its
    /// tag the index constant `variant`, and the payload operand of every
    /// other variant that carries one a `ub.poison` of that payload's type.
    static MakeOp createOfVariant(OpBuilder &builder, Location loc, Type type,
                                  size_t variant, Value payload);
  }];
}

def MatchOp : SumDialectOp<"match", [
  OpAsmOpInterface,
  RecursiveMemoryEffects,
  SingleBlockImplicitTerminator<"YieldOp">,
  DeclareOpInterfaceMethods<RegionBranchOpInterface,
                            ["getRegionInvocationBounds"]>
]> {
  let summary = "Switch on a sum type variant";
  let description = [{
    Dispatches to one of several regions based on the sum type's variant.
    Each region receives the variant's payload as block arguments.
    Nullary variants (variant type is NoneType) have no block arguments.

    The region control flow is `scf.index_switch`'s, which the sum lowering
    rewrites a match into: from the op any one case is entered, at most once,
    with no operand forwarded (the payload argument is read from the sum
    value), and every case returns to the op, its yield's operands becoming
    the op's results.

    Example:

        %res = sum.match %x : !sum.sum<(i64, none)> -> i64
        case 0 (%inner: i64) {
          yield %inner : i64
        }
        case 1 {
          %c = arith.constant 0 : i64
          yield %c : i64
        }
  }];

  let arguments = (ins IsSumTypeInterface:$input);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region VariadicRegion<SizedRegion<1>>:$cases);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "sum"; }
  }];
}

def TagOp : SumDialectOp<"tag", [Pure]> {
  let summary = "Get the active variant index (discriminant) of a sum value";
  let description = [{
    Returns the index of the active variant of a sum value.

    Example:

        %tag = sum.tag %x : !sum.sum<(i64, i32)>
  }];

  let arguments = (ins IsSumTypeInterface:$input);
  let results = (outs Index:$tag);

  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let assemblyFormat = "$input attr-dict `:` qualified(type($input))";

  let builders = [
    OpBuilder<(ins "Value":$input), [{
      build($_builder, $_state, $_builder.getIndexType(), input);
    }]>
  ];
}

def YieldOp : SumDialectOp<"yield", [
  HasParent<"MatchOp">,
  Pure,
  ReturnLike,
  Terminator
]> {
  let summary = "Yield a value from a match arm.";
  let description = [{
    Terminates a match arm region and yields a value.

    Example:

        sum.yield %value : i64
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{  attr-dict ($results^ `:` type($results))? }];
}

#endif // SUM_OPS_TD
