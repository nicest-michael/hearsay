//! Link-time fix for the Swift shims pulled in by `screencapturekit` / `apple-cf`.
//!
//! Those crates compile Swift, whose objects auto-link the `swiftCompatibility*`
//! static libraries. The upstream build scripts add the **full-Xcode** Swift lib
//! path (`…/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/macosx`). On a machine
//! with only the Command Line Tools that path doesn't exist, so the linker can't find
//! `libswiftCompatibility56.a` et al. and fails with undefined `__swift_FORCE_LOAD_*`
//! symbols. We add the Swift macosx lib dir that actually exists for the active
//! developer dir so both the test binaries and the final app link cleanly.

fn main() {
    if !cfg!(target_os = "macos") {
        return;
    }

    let mut candidates: Vec<String> = Vec::new();
    if let Ok(out) = std::process::Command::new("xcode-select")
        .arg("-p")
        .output()
    {
        if out.status.success() {
            let dev = String::from_utf8_lossy(&out.stdout).trim().to_string();
            // Command Line Tools layout.
            candidates.push(format!("{dev}/usr/lib/swift/macosx"));
            // Full-Xcode layout.
            candidates.push(format!(
                "{dev}/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/macosx"
            ));
        }
    }
    // Well-known fallback if `xcode-select` is unavailable.
    candidates.push("/Library/Developer/CommandLineTools/usr/lib/swift/macosx".to_string());

    for dir in candidates {
        if std::path::Path::new(&dir).exists() {
            println!("cargo:rustc-link-search=native={dir}");
        }
    }

    // ggml-metal guards its residency-set path behind `@available(macOS 15, *)`.
    // With our deployment target below 15, clang emits a runtime check that pulls in
    // the compiler-rt builtin `___isPlatformVersionAtLeast`. Rust links with
    // `-nodefaultlibs`, so clang's runtime isn't auto-linked and the release build
    // fails with that symbol undefined (debug happens to elide it). Link clang's
    // compiler-rt explicitly; the linker only extracts the one needed archive member,
    // so this doesn't clash with Rust's own `compiler_builtins`.
    if let Ok(out) = std::process::Command::new("clang")
        .arg("-print-runtime-dir")
        .output()
    {
        if out.status.success() {
            let dir = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if std::path::Path::new(&dir).exists() {
                println!("cargo:rustc-link-search=native={dir}");
                println!("cargo:rustc-link-lib=static=clang_rt.osx");
            }
        }
    }

    println!("cargo:rerun-if-changed=build.rs");
}
