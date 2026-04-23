# Contributing

## Train

Training downloads the finalized Zenodo dataset automatically into `--data-dir`
when the files are missing.

Mini:

```bash
cargo run --release -p npclassifier-train -- \
  --backend cuda \
  --architecture mini-shared \
  --artifact-dir artifacts/mini-shared \
  --web-output-dir models/mini-shared \
  --num-epochs 200
```

Faithful:

```bash
cargo run --release -p npclassifier-train -- \
  --backend cuda \
  --architecture baseline \
  --artifact-dir artifacts/full \
  --web-output-dir models/full \
  --num-epochs 200
```

Use `--backend ndarray` for CPU-only smoke runs.

## Start Dioxus

```bash
dx serve --package npclassifier-web --platform web --port 8787 --release
```
