use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let mlir_prefix =
        env::var("MLIR_SYS_230_PREFIX").expect("MLIR_SYS_230_PREFIX must be set");
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cpp_dir = crate_dir.join("cpp");
    let build_dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("cpp");
    let lowering_driver_source_dir = env::var("DEP_LOWERING_DRIVER_SOURCE_DIR")
        .expect("DEP_LOWERING_DRIVER_SOURCE_DIR must be set by mlir-lowering-driver");

    let status = Command::new("make")
        .arg("-j")
        .arg(format!("BUILD_DIR={}", build_dir.display()))
        .arg(format!("LOWERING_DRIVER_SOURCE_DIR={lowering_driver_source_dir}"))
        .current_dir(&cpp_dir)
        .status()
        .expect("Failed to run make in cpp/");

    if !status.success() {
        panic!("C++ build failed");
    }

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=sum_dialect");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{mlir_prefix}/lib");

    println!("cargo:source_dir={}", cpp_dir.display());
    println!("cargo:include_dir={}", build_dir.display());
    println!("cargo:lib_dir={}", build_dir.display());

    // The dialect plugins this crate's lit rows load, each after the plugins it
    // references, published for the lit rows of the crates that depend on this one.
    let plugins = build_support::publish_dialect_plugins(build_dir.join("libsum_dialect.so"));
    let lowering_driver_plugin = env::var("DEP_LOWERING_DRIVER_PLUGIN")
        .expect("DEP_LOWERING_DRIVER_PLUGIN must be set by mlir-lowering-driver");
    build_support::write_lit_site_config(
        &mlir_prefix,
        "LIT_SUITE",
        "cpp/tests",
        &[
            ("dialect_plugins", build_support::python_paths(&plugins)),
            ("lowering_driver_plugin", build_support::python_path(Path::new(&lowering_driver_plugin))),
        ],
    );

    println!("cargo:rerun-if-env-changed=MLIR_SYS_230_PREFIX");
    println!("cargo:rerun-if-changed=cpp");
    // This dialect compiles against the driver's contribution headers rather than
    // linking them, so a change to any of them must recompile its C++. Watch the
    // driver's source directory, which cargo scans recursively, so no header the
    // dialect includes (PermissiveInlinerInterface.hpp among them) is omitted.
    println!("cargo:rerun-if-changed={lowering_driver_source_dir}");
}

