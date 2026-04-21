# npclassifier-rs

Rust implementation of `NPClassifier`, plus a browser frontend built with Dioxus.

The repository stays focused on a faithful `NPClassifier` line. The main practical deviation is low-bit packed inference for browser delivery.

## CLI

The user-facing CLI is `npclassifier-classify`. It reads one SMILES per line, classifies them in parallel, and prints JSON to stdout.

From a local packed model bundle:

```bash
cargo run --release -p npclassifier-core --features classify-cli --bin npclassifier-classify -- \
  --models /path/to/models/mini-shared \
  --input smiles.txt
```

From stdin:

```bash
printf 'CCO\nC1=CC=CC=C1\n' | cargo run --release -p npclassifier-core --features classify-cli --bin npclassifier-classify -- \
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
