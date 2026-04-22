//! Cargo subcommand for packed `NPClassifier` classification.

use std::{
    ffi::OsString,
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
};

use clap::{Parser, ValueEnum};
use npclassifier_core::{
    ClassificationThresholds, ModelBundleSource, PackedClassifierBuilder, PackedModelVariant,
};

#[derive(Debug, Parser)]
#[command(name = "cargo-npc")]
#[command(about = "Classify SMILES lines with a packed NPClassifier bundle.")]
struct Cli {
    /// Local packed model directory.
    #[arg(long, conflicts_with = "base_url")]
    models: Option<PathBuf>,
    /// Remote base URL for a packed model bundle.
    #[arg(long, conflicts_with = "models")]
    base_url: Option<String>,
    /// Optional input file. Defaults to stdin.
    #[arg(long)]
    input: Option<PathBuf>,
    /// Packed model variant to load.
    #[arg(long, value_enum, default_value_t = CliVariant::Q4Kernel)]
    variant: CliVariant,
    /// Override the pathway decision threshold.
    #[arg(long)]
    pathway_threshold: Option<f32>,
    /// Override the superclass decision threshold.
    #[arg(long)]
    superclass_threshold: Option<f32>,
    /// Override the class decision threshold.
    #[arg(long)]
    class_threshold: Option<f32>,
    /// Limit rayon worker threads for preprocessing and classification.
    #[arg(long)]
    threads: Option<usize>,
    /// Optional cache directory for remote bundles.
    #[arg(long)]
    cache_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliVariant {
    F32,
    Q8Kernel,
    Q4Kernel,
}

impl From<CliVariant> for PackedModelVariant {
    fn from(value: CliVariant) -> Self {
        match value {
            CliVariant::F32 => Self::F32,
            CliVariant::Q8Kernel => Self::Q8Kernel,
            CliVariant::Q4Kernel => Self::Q4Kernel,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse_from(normalize_subcommand_args(std::env::args_os()));
    let input = read_input(cli.input.as_deref())?;
    let smiles = parse_smiles_lines(&input);

    let mut builder = PackedClassifierBuilder::new().with_variant(cli.variant.into());
    builder = match select_source(&cli)? {
        ModelBundleSource::Local(path) => builder.with_local_dir(path),
        ModelBundleSource::Remote(url) => builder.with_remote_base_url(url),
    };
    if let Some(thresholds) = threshold_override(&cli) {
        builder = builder.with_thresholds(thresholds);
    }
    if let Some(cache_dir) = cli.cache_dir {
        builder = builder.with_cache_dir(cache_dir);
    }
    if let Some(threads) = cli.threads {
        builder = builder.with_parallelism(threads);
    }

    let classifier = builder.build()?;
    let output = classifier.classify_batch_parallel(&smiles);
    serde_json::to_writer_pretty(io::stdout(), &output)?;
    println!();

    Ok(())
}

fn normalize_subcommand_args(args: impl IntoIterator<Item = OsString>) -> Vec<OsString> {
    let mut args = args.into_iter().collect::<Vec<_>>();
    if args.get(1).is_some_and(|arg| arg == "npc") {
        args.remove(1);
    }
    args
}

fn select_source(cli: &Cli) -> Result<ModelBundleSource, Box<dyn std::error::Error>> {
    match (&cli.models, &cli.base_url) {
        (Some(path), None) => Ok(ModelBundleSource::Local(path.clone())),
        (None, Some(url)) => Ok(ModelBundleSource::Remote(url.clone())),
        (None, None) => Err("either --models or --base-url is required".into()),
        (Some(_), Some(_)) => Err("use only one of --models or --base-url".into()),
    }
}

fn read_input(path: Option<&Path>) -> Result<String, Box<dyn std::error::Error>> {
    if let Some(path) = path {
        Ok(fs::read_to_string(path)?)
    } else {
        let mut input = String::new();
        io::stdin().read_to_string(&mut input)?;
        Ok(input)
    }
}

fn parse_smiles_lines(input: &str) -> Vec<String> {
    input
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect()
}

fn threshold_override(cli: &Cli) -> Option<ClassificationThresholds> {
    let [pathway, superclass, class] = [
        cli.pathway_threshold,
        cli.superclass_threshold,
        cli.class_threshold,
    ];
    match (pathway, superclass, class) {
        (None, None, None) => None,
        _ => Some(ClassificationThresholds::new(
            pathway.unwrap_or_else(|| ClassificationThresholds::default().pathway),
            superclass.unwrap_or_else(|| ClassificationThresholds::default().superclass),
            class.unwrap_or_else(|| ClassificationThresholds::default().class),
        )),
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use super::{normalize_subcommand_args, parse_smiles_lines};

    #[test]
    fn parse_smiles_lines_trims_and_drops_empty_lines() {
        let parsed = parse_smiles_lines("CCO\n\n C1=CC=CC=C1 \n");
        assert_eq!(parsed, vec!["CCO", "C1=CC=CC=C1"]);
    }

    #[test]
    fn normalize_subcommand_args_strips_cargo_subcommand_slot() {
        let args = normalize_subcommand_args([
            OsString::from("cargo-npc"),
            OsString::from("npc"),
            OsString::from("--help"),
        ]);
        assert_eq!(
            args,
            vec![OsString::from("cargo-npc"), OsString::from("--help")]
        );
    }

    #[test]
    fn normalize_subcommand_args_keeps_direct_binary_invocation() {
        let args =
            normalize_subcommand_args([OsString::from("cargo-npc"), OsString::from("--help")]);
        assert_eq!(
            args,
            vec![OsString::from("cargo-npc"), OsString::from("--help")]
        );
    }
}
