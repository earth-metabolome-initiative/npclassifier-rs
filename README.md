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

The user-facing CLI is the `cargo npc` subcommand. It reads one SMILES per line, classifies them in parallel, and prints JSON to stdout.

From this repository during development:

```bash
cargo run --release -p cargo-npc -- \
  --models /path/to/models/mini-shared \
  --input smiles.txt
```

Once the package is published and installed:

```bash
cargo install cargo-npc
```

Then the same command becomes:

```bash
cargo npc --models /path/to/models/mini-shared --input smiles.txt
```

From stdin:

```bash
printf 'CCO\nC1=CC=CC=C1\n' | cargo run --release -p cargo-npc -- \
  --models /path/to/models/mini-shared
```

It also supports:

- `--base-url` for a remote packed bundle
- `--threads` to cap rayon parallelism
- explicit per-head threshold overrides

## Library

The core crate also exposes a reusable builder-backed runner for local or remote packed bundles.

```rust
use npclassifier_core::{PackedClassifierBuilder, PackedModelVariant};

let builder = PackedClassifierBuilder::new()
    .with_local_dir("/path/to/models/mini-shared")
    .with_variant(PackedModelVariant::Q4Kernel)
    .with_parallelism(8);

let _ = builder;
```

Maintainer notes are in [CONTRIBUTING.md](CONTRIBUTING.md).
