# Contributing

Keep changes aligned with the scope of this repository:

- this repo is for a faithful `NPClassifier` implementation
- quantization and browser packaging are in scope
- experimental architecture work should live elsewhere

Before sending a change, run the relevant checks for the code you touched. In practice that usually means:

```bash
cargo fmt --all
cargo check --workspace
cargo test -p npclassifier-web
```

Keep documentation small and practical:

- `README.md` should stay user-facing
- maintainer detail should stay in code, small docs, or CLI help

If you need to run the browser app locally:

```bash
dx serve --package npclassifier-web --platform web --port 8787 --release
```

To regenerate the distilled dataset from the current completed source parts:

```bash
cargo run --release -p npclassifier-core --features distillation-dataset --bin npclassifier-curate -- curate \
  --input-dir /home/luca/github/npc-labeler/work/completed \
  --output-dir data/distillation/teacher-splits
```

The source path is still local for now. When the completed source parts move to Internet Archive or another remote host, update the input path accordingly.

When touching code, remove stale paths, dead dependencies, and obsolete comments instead of layering new ones on top.
