# mlir-sum-dialect

An MLIR dialect for sum types (tagged unions).

## Types

### `!sum.sum<(T1, T2, ...)>`

A sum type with variants. Each variant has a payload type.

```mlir
!sum.sum<(i64, tuple<>)>  // variant 0: i64, variant 1: unit

## Operations

### `sum.make`

Construct a sum type variant.

```mlir
%x = sum.make 0 %val : !sum.sum<(i64, tuple<>)>
%y = sum.make 1 %unit : !sum.sum<(i64, tuple<>)>
```

### `sum.match`

Dispatch on a sum type's variant. Each case region receives the payload as a block argument.

```mlir
%res = sum.match %x : !sum.sum<(i64, tuple<>)> -> i64
case 0 (%inner: i64) {
  sum.yield %inner : i64
}
case 1 (%unit: tuple<>) {
  %default = arith.constant 7 : i64
  sum.yield %default : i64
}
```

### `sum.yield`

Terminate a match case and yield a value.

```mlir
sum.yield %value : i64
```

## Building

Requires LLVM/MLIR. See `Makefile` for build instructions.

## Rust Bindings

The `lib.rs` file provides Rust bindings via the C API:

- `sum::sum_type(context, &[variants])` - create a sum type
- `sum::make(loc, result_ty, index, payload)` - create `sum.make` op
- `sum::match_(loc, input, &[result_types])` - create `sum.match` op
- `sum::yield_(loc, &[results])` - create `sum.yield` op
