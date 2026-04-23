# npclassifier-rs

[![CI](https://github.com/earth-metabolome-initiative/npclassifier-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/earth-metabolome-initiative/npclassifier-rs/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/earth-metabolome-initiative/npclassifier-rs/graph/badge.svg)](https://codecov.io/gh/earth-metabolome-initiative/npclassifier-rs)
[![Pages](https://github.com/earth-metabolome-initiative/npclassifier-rs/actions/workflows/pages.yml/badge.svg)](https://github.com/earth-metabolome-initiative/npclassifier-rs/actions/workflows/pages.yml)
[![Web App](https://img.shields.io/badge/web-npc.earthmetabolome.org-0f766e)](https://npc.earthmetabolome.org/)
[![Models](https://img.shields.io/badge/models-Hugging%20Face-f59e0b?logo=huggingface&logoColor=white)](https://huggingface.co/EarthMetabolomeInitiative/npclassifier-rs-models)
[![License: MIT](https://img.shields.io/badge/license-MIT-2563eb.svg)](LICENSE)

Rust implementation of `NPClassifier`, plus a browser frontend built with Dioxus.

The repository stays focused on a faithful `NPClassifier` line. The main practical deviation is low-bit packed inference for browser delivery.

## CLI

The user-facing CLI is the `cargo npc` subcommand. It reads one SMILES per line, classifies them in parallel in streamed batches, and writes results as JSON Lines by default.

From this repository during development:

```bash
cargo run --release -p cargo-npc -- \
  --model mini \
  --input smiles.txt
```

Once the package is published and installed:

```bash
cargo install cargo-npc
```

Then the same command becomes:

```bash
cargo npc --model mini --input smiles.txt
```

From stdin:

```bash
printf 'CCO\nC1=CC=CC=C1\n' | cargo run --release -p cargo-npc -- \
  --model mini
```

It also supports:

- `--model mini` or `--model faithful`
- `--output` to write results to a file instead of stdout
- `--format json`, `--format jsonl`, or `--format parquet`
- `--batch-size` to control streamed classification chunks
- `--threads` to cap rayon parallelism
- `--cache-dir` to control where downloaded bundles are cached
- `--progress` to emit progress updates on stderr
- explicit per-head threshold overrides

Parquet output writes a compact columnar schema with:

- `smiles`
- `error`
- `pathways`
- `superclasses`
- `classes`
- `is_glycoside`

Example:

```bash
cargo npc \
  --model mini \
  --input smiles.txt \
  --output results.parquet \
  --format parquet \
  --threads 8 \
  --progress
```

## Library

The core crate also exposes a reusable builder-backed runner for the hosted `Mini` and `Faithful` bundles.

```rust
# #[cfg(feature = "runner")]
# {
use npclassifier_core::{HostedModel, PackedClassifierBuilder, PackedModelVariant};

let cache_dir = std::env::temp_dir().join(format!(
    "npclassifier-rs-doctest-{}",
    std::process::id()
));

let classifier = PackedClassifierBuilder::new()
    .with_model(HostedModel::Mini)
    .with_cache_dir(&cache_dir)
    .with_variant(PackedModelVariant::Q4Kernel)
    .with_parallelism(4)
    .build()?;

let results = classifier.classify_lines_parallel("CCO\nC1=CC=CC=C1\n");

assert_eq!(results.len(), 2);
assert_eq!(results[0].smiles, "CCO");
assert_eq!(results[1].smiles, "C1=CC=CC=C1");
for entry in &results {
    if let Some(error) = &entry.error {
        eprintln!("{} could not be classified: {error}", entry.smiles);
    }
}
# }
# Ok::<(), Box<dyn std::error::Error>>(())
```

Maintainer notes are in [CONTRIBUTING.md](CONTRIBUTING.md).
