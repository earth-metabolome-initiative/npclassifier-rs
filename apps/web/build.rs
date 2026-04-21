//! Generates the dedicated wasm worker assets used by the Dioxus web app.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

const WORKER_PACKAGE: &str = "npclassifier-web-worker";
const WORKER_STEM: &str = "npclassifier_web_worker";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../Cargo.lock");
    println!("cargo:rerun-if-changed=../../crates/npclassifier-core/Cargo.toml");
    println!("cargo:rerun-if-changed=../../crates/npclassifier-core/src");
    println!("cargo:rerun-if-changed=../web-worker/Cargo.toml");
    println!("cargo:rerun-if-changed=../web-worker/src");

    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR")
            .ok_or_else(|| "cargo did not provide CARGO_MANIFEST_DIR".to_owned())?,
    );
    let workspace_root = manifest_dir
        .join("../..")
        .canonicalize()
        .map_err(|error| format!("failed to resolve workspace root: {error}"))?;
    let generated_dir = manifest_dir.join("public/generated");

    emit_git_commit(&workspace_root);
    build_worker_assets(&workspace_root, &generated_dir)
}

fn emit_git_commit(workspace_root: &Path) {
    if let Some(git_dir) = git_output(workspace_root, ["rev-parse", "--git-dir"]) {
        let git_dir_path = if Path::new(&git_dir).is_absolute() {
            PathBuf::from(&git_dir)
        } else {
            workspace_root.join(git_dir)
        };
        println!(
            "cargo:rerun-if-changed={}",
            git_dir_path.join("HEAD").display()
        );
        if let Some(head_ref) = git_output(workspace_root, ["symbolic-ref", "-q", "HEAD"]) {
            let ref_path = git_dir_path.join(head_ref);
            println!("cargo:rerun-if-changed={}", ref_path.display());
        }
    }

    let commit = git_output(workspace_root, ["rev-parse", "--short=12", "HEAD"])
        .unwrap_or_else(|| String::from("unknown"));
    println!("cargo:rustc-env=NPCLASSIFIER_GIT_COMMIT={commit}");
}

fn git_output<const N: usize>(workspace_root: &Path, args: [&str; N]) -> Option<String> {
    Command::new("git")
        .current_dir(workspace_root)
        .args(args)
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok()
            } else {
                None
            }
        })
        .map(|stdout| stdout.trim().to_owned())
        .filter(|stdout| !stdout.is_empty())
}

fn build_worker_assets(workspace_root: &Path, generated_dir: &Path) -> Result<(), String> {
    fs::create_dir_all(generated_dir)
        .map_err(|error| format!("failed to create generated worker directory: {error}"))?;

    let out_dir = PathBuf::from(
        env::var_os("OUT_DIR").ok_or_else(|| "cargo did not provide OUT_DIR".to_owned())?,
    );
    let bindgen_dir = out_dir.join("npclassifier-worker-bindgen");
    let target_dir = out_dir.join("npclassifier-worker-target");
    let _ignored = fs::remove_dir_all(&bindgen_dir);
    fs::create_dir_all(&bindgen_dir)
        .map_err(|error| format!("failed to create worker bindgen directory: {error}"))?;

    let cargo = env::var("CARGO").unwrap_or_else(|_| String::from("cargo"));
    let profile = env::var("PROFILE").unwrap_or_else(|_| String::from("debug"));

    let mut build = Command::new(cargo);
    build
        .current_dir(workspace_root)
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .args([
            "build",
            "--package",
            WORKER_PACKAGE,
            "--lib",
            "--target",
            "wasm32-unknown-unknown",
            "--target-dir",
        ])
        .arg(&target_dir);

    match profile.as_str() {
        "debug" => {}
        "release" => {
            build.arg("--release");
        }
        other => {
            build.args(["--profile", other]);
        }
    }

    let status = build
        .status()
        .map_err(|error| format!("failed to launch worker cargo build: {error}"))?;
    if !status.success() {
        return Err(format!("worker cargo build failed with status {status}"));
    }

    let worker_wasm = target_dir
        .join("wasm32-unknown-unknown")
        .join(&profile)
        .join(format!("{WORKER_STEM}.wasm"));
    if !worker_wasm.exists() {
        return Err(format!("expected worker wasm at {}", worker_wasm.display()));
    }

    let mut bindgen = wasm_bindgen_cli_support::Bindgen::new();
    bindgen
        .input_path(&worker_wasm)
        .out_name(WORKER_STEM)
        .typescript(false)
        .web(true)
        .map_err(|error| format!("failed to configure worker bindgen for web output: {error}"))?
        .generate(&bindgen_dir)
        .map_err(|error| format!("worker bindgen generation failed: {error}"))?;

    copy_file(
        &bindgen_dir.join(format!("{WORKER_STEM}.js")),
        &generated_dir.join(format!("{WORKER_STEM}.js")),
    )?;
    copy_file(
        &bindgen_dir.join(format!("{WORKER_STEM}_bg.wasm")),
        &generated_dir.join(format!("{WORKER_STEM}_bg.wasm")),
    )?;
    fs::write(
        generated_dir.join("classifier-worker.js"),
        worker_loader_script(),
    )
    .map_err(|error| format!("failed to write worker bootstrap script: {error}"))?;
    Ok(())
}

fn copy_file(source: &Path, destination: &Path) -> Result<(), String> {
    fs::copy(source, destination).map_err(|error| {
        format!(
            "failed to copy generated worker asset from {} to {}: {error}",
            source.display(),
            destination.display()
        )
    })?;
    Ok(())
}

fn worker_loader_script() -> &'static str {
    r#"import init from "./npclassifier_web_worker.js";

await init(new URL("./npclassifier_web_worker_bg.wasm", import.meta.url));
"#
}
