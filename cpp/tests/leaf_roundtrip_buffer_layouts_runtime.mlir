// RUN: mlir-opt --pass-pipeline="builtin.module(leaf-conversion,llvm-target-to-data-layout,convert-sum-to-scf,convert-scf-to-cf,convert-to-llvm,reconcile-unrealized-casts)" %s \
// RUN: | mlir-runner --entry-point-result=void --shared-libs=%mlir_c_runner_utils \
// RUN: | FileCheck %s

// A sum whose payload is a buffer of a static shape, of a static non-unit stride
// with a static offset, of a dynamic offset, of rank zero, or of a stride
// stepping backward crosses every function boundary and every scf.if result as
// its leaves and is rebuilt where it is read whole, on the path that holds the
// buffer and on the path that does not. Where the sum holds no buffer its leaf
// is a stand-in of the buffer's own type, so each function verifies and runs: a
// held buffer reads back its element, and an absent one takes the empty arm.

// CHECK: 5
// CHECK-NEXT: 42
// CHECK-NEXT: 7
// CHECK-NEXT: 42
// CHECK-NEXT: 6
// CHECK-NEXT: 42
// CHECK-NEXT: 3
// CHECK-NEXT: 42
// CHECK-NEXT: 2
// CHECK-NEXT: 42
!S0 = !sum.sum<(memref<4xi64>, none)>
!S1 = !sum.sum<(memref<?xi64, strided<[2], offset: 1>>, none)>
!S2 = !sum.sum<(memref<?xi64, strided<[2], offset: ?>>, none)>
!S3 = !sum.sum<(memref<i64>, none)>
!S4 = !sum.sum<(memref<4xi64, strided<[-1], offset: 3>>, none)>
module attributes {llvm.target = #llvm.target<triple = "x86_64-unknown-linux-gnu", chip = "">} {
  func.func private @printI64(i64)
  func.func private @printNewline()

  func.func private @f0(%held: i1, %m: memref<4xi64>) -> !S0 {
    %r = scf.if %held -> !S0 {
      %s = sum.make 0 %m : !S0
      scf.yield %s : !S0
    } else {
      %s = sum.make 1 : !S0
      scf.yield %s : !S0
    }
    return %r : !S0
  }
  func.func private @f1(%held: i1, %m: memref<?xi64, strided<[2], offset: 1>>) -> !S1 {
    %r = scf.if %held -> !S1 {
      %s = sum.make 0 %m : !S1
      scf.yield %s : !S1
    } else {
      %s = sum.make 1 : !S1
      scf.yield %s : !S1
    }
    return %r : !S1
  }
  func.func private @f2(%held: i1, %m: memref<?xi64, strided<[2], offset: ?>>) -> !S2 {
    %r = scf.if %held -> !S2 {
      %s = sum.make 0 %m : !S2
      scf.yield %s : !S2
    } else {
      %s = sum.make 1 : !S2
      scf.yield %s : !S2
    }
    return %r : !S2
  }
  func.func private @f3(%held: i1, %m: memref<i64>) -> !S3 {
    %r = scf.if %held -> !S3 {
      %s = sum.make 0 %m : !S3
      scf.yield %s : !S3
    } else {
      %s = sum.make 1 : !S3
      scf.yield %s : !S3
    }
    return %r : !S3
  }
  func.func private @f4(%held: i1, %m: memref<4xi64, strided<[-1], offset: 3>>) -> !S4 {
    %r = scf.if %held -> !S4 {
      %s = sum.make 0 %m : !S4
      scf.yield %s : !S4
    } else {
      %s = sum.make 1 : !S4
      scf.yield %s : !S4
    }
    return %r : !S4
  }

  // Prints element `%i` of the buffer `%v` holds, or 42 when it holds none.
  func.func private @show0(%v: !S0, %i: index) {
    %e = sum.match %v : !S0 -> i64
    case 0 (%m: memref<4xi64>) {
      %x = memref.load %m[%i] : memref<4xi64>
      yield %x : i64
    }
    case 1 {
      %c = arith.constant 42 : i64
      yield %c : i64
    }
    func.call @printI64(%e) : (i64) -> ()
    func.call @printNewline() : () -> ()
    return
  }
  func.func private @show1(%v: !S1, %i: index) {
    %e = sum.match %v : !S1 -> i64
    case 0 (%m: memref<?xi64, strided<[2], offset: 1>>) {
      %x = memref.load %m[%i] : memref<?xi64, strided<[2], offset: 1>>
      yield %x : i64
    }
    case 1 {
      %c = arith.constant 42 : i64
      yield %c : i64
    }
    func.call @printI64(%e) : (i64) -> ()
    func.call @printNewline() : () -> ()
    return
  }
  func.func private @show2(%v: !S2, %i: index) {
    %e = sum.match %v : !S2 -> i64
    case 0 (%m: memref<?xi64, strided<[2], offset: ?>>) {
      %x = memref.load %m[%i] : memref<?xi64, strided<[2], offset: ?>>
      yield %x : i64
    }
    case 1 {
      %c = arith.constant 42 : i64
      yield %c : i64
    }
    func.call @printI64(%e) : (i64) -> ()
    func.call @printNewline() : () -> ()
    return
  }
  func.func private @show3(%v: !S3) {
    %e = sum.match %v : !S3 -> i64
    case 0 (%m: memref<i64>) {
      %x = memref.load %m[] : memref<i64>
      yield %x : i64
    }
    case 1 {
      %c = arith.constant 42 : i64
      yield %c : i64
    }
    func.call @printI64(%e) : (i64) -> ()
    func.call @printNewline() : () -> ()
    return
  }

  func.func private @show4(%v: !S4, %i: index) {
    %e = sum.match %v : !S4 -> i64
    case 0 (%m: memref<4xi64, strided<[-1], offset: 3>>) {
      %x = memref.load %m[%i] : memref<4xi64, strided<[-1], offset: 3>>
      yield %x : i64
    }
    case 1 {
      %c = arith.constant 42 : i64
      yield %c : i64
    }
    func.call @printI64(%e) : (i64) -> ()
    func.call @printNewline() : () -> ()
    return
  }

  func.func @main() {
    %true = arith.constant true
    %false = arith.constant false
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c9 = arith.constant 9 : index

    // Element i of the backing store holds i.
    %store = memref.alloca() : memref<9xi64>
    scf.for %i = %c0 to %c9 step %c1 {
      %v = arith.index_cast %i : index to i64
      memref.store %v, %store[%i] : memref<9xi64>
    }

    %m0 = memref.alloca() : memref<4xi64>
    %five = arith.constant 5 : i64
    memref.store %five, %m0[%c3] : memref<4xi64>
    %a0 = func.call @f0(%true, %m0) : (i1, memref<4xi64>) -> !S0
    func.call @show0(%a0, %c3) : (!S0, index) -> ()
    %b0 = func.call @f0(%false, %m0) : (i1, memref<4xi64>) -> !S0
    func.call @show0(%b0, %c3) : (!S0, index) -> ()

    // Element 3 of the view at offset 1, stride 2 is store[7].
    %m1 = memref.reinterpret_cast %store to offset: [1], sizes: [%c4], strides: [2]
        : memref<9xi64> to memref<?xi64, strided<[2], offset: 1>>
    %a1 = func.call @f1(%true, %m1) : (i1, memref<?xi64, strided<[2], offset: 1>>) -> !S1
    func.call @show1(%a1, %c3) : (!S1, index) -> ()
    %b1 = func.call @f1(%false, %m1) : (i1, memref<?xi64, strided<[2], offset: 1>>) -> !S1
    func.call @show1(%b1, %c3) : (!S1, index) -> ()

    // Element 2 of the view at offset 2, stride 2 is store[6].
    %m2 = memref.reinterpret_cast %store to offset: [%c2], sizes: [%c3], strides: [2]
        : memref<9xi64> to memref<?xi64, strided<[2], offset: ?>>
    %a2 = func.call @f2(%true, %m2) : (i1, memref<?xi64, strided<[2], offset: ?>>) -> !S2
    func.call @show2(%a2, %c2) : (!S2, index) -> ()
    %b2 = func.call @f2(%false, %m2) : (i1, memref<?xi64, strided<[2], offset: ?>>) -> !S2
    func.call @show2(%b2, %c2) : (!S2, index) -> ()

    %m3 = memref.reinterpret_cast %store to offset: [3], sizes: [], strides: []
        : memref<9xi64> to memref<i64, strided<[], offset: 3>>
    %m3i = memref.alloca() : memref<i64>
    %three = memref.load %m3[] : memref<i64, strided<[], offset: 3>>
    memref.store %three, %m3i[] : memref<i64>
    %a3 = func.call @f3(%true, %m3i) : (i1, memref<i64>) -> !S3
    func.call @show3(%a3) : (!S3) -> ()
    %b3 = func.call @f3(%false, %m3i) : (i1, memref<i64>) -> !S3
    func.call @show3(%b3) : (!S3) -> ()

    // Element 1 of the view at offset 3, stride -1 is store[2].
    %m4 = memref.reinterpret_cast %store to offset: [3], sizes: [4], strides: [-1]
        : memref<9xi64> to memref<4xi64, strided<[-1], offset: 3>>
    %a4 = func.call @f4(%true, %m4) : (i1, memref<4xi64, strided<[-1], offset: 3>>) -> !S4
    func.call @show4(%a4, %c1) : (!S4, index) -> ()
    %b4 = func.call @f4(%false, %m4) : (i1, memref<4xi64, strided<[-1], offset: 3>>) -> !S4
    func.call @show4(%b4, %c1) : (!S4, index) -> ()
    return
  }
}
