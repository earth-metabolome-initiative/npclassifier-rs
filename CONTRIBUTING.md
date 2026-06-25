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

The local web worker loads the same hosted model bundles used by the Pages
deployment by default. To test against local model exports instead, compile the
worker with explicit model base URLs:

```bash
NPCLASSIFIER_MINI_MODEL_BASE_URL=http://localhost:8787/models/mini-shared \
NPCLASSIFIER_FULL_MODEL_BASE_URL=http://localhost:8787/models/full \
dx serve --package npclassifier-web --platform web --port 8787 --release
```
