// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A sum.assemble takes one payload per variant that carries one, of that
// variant's payload type, in variant order.

func.func @payload_missing(%t: index, %x: i64) -> !sum.sum<(i64, none, f32)> {
  // expected-error @+1 {{expects one payload per variant that carries one}}
  %s = sum.assemble %t (%x : i64) : !sum.sum<(i64, none, f32)>
  return %s : !sum.sum<(i64, none, f32)>
}

// -----

func.func @payloads_out_of_order(%t: index, %x: i64, %y: f32) -> !sum.sum<(i64, none, f32)> {
  // expected-error @+1 {{expects one payload per variant that carries one}}
  %s = sum.assemble %t (%y, %x : f32, i64) : !sum.sum<(i64, none, f32)>
  return %s : !sum.sum<(i64, none, f32)>
}
