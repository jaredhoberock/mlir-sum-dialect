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
    guarantees the variant (e.g. inside a matching arm).

    Example:

        %payload = sum.get %x, 0 : !sum.sum<(i64, i1)> -> i64
  }];

  let arguments = (ins
    IsSumTypeInterface:$input,
    IndexAttr:$index
  );

  let results = (outs AnyType:$payload);

  let hasVerifier = 1;
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
  let assemblyFormat = "$input `,` $index attr-dict `:` qualified(type($input))";

  let builders = [
    OpBuilder<(ins "Value":$input, "int64_t":$index), [{
      build($_builder, $_state, $_builder.getI1Type(), input, $_builder.getIndexAttr(index));
    }]>
  ];
}

def MakeOp : SumDialectOp<"make", [Pure]> {
  let summary = "Construct a sum type variant";
  let description = [{
    Creates a value of a sum type by specifying which variant to construct
    and providing the payload value (if any). For nullary variants (variant
    type is NoneType), no payload is provided.

    Example:

        %x = sum.make 0 %val : !sum.sum<(i64, none)>
        %y = sum.make 1 : !sum.sum<(i64, none)>
  }];

  let arguments = (ins
    IndexAttr:$index,
    Optional<AnyType>:$payload
  );

  let results = (outs IsSumTypeInterface:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Type":$resultType, "int64_t":$index), [{
      build($_builder, $_state, resultType,
            $_builder.getIndexAttr(index), /*payload=*/Value());
    }]>
  ];
}

def MatchOp : SumDialectOp<"match", [
  OpAsmOpInterface,
  RecursiveMemoryEffects,
  SingleBlockImplicitTerminator<"YieldOp">
]> {
  let summary = "Switch on a sum type variant";
  let description = [{
    Dispatches to one of several regions based on the sum type's variant.
    Each region receives the variant's payload as block arguments.
    Nullary variants (variant type is NoneType) have no block arguments.

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
