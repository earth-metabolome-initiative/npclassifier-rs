//! CLI for curating high-confidence distillation splits from `npc-labeler`.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use npclassifier_core::{
    ConfidenceMargins, CurationConfig, DEFAULT_COMPLETED_DIR, DEFAULT_OUTPUT_DIR,
    DEFAULT_VOCABULARY_PATH, NpClassifierError, SplitFractions, curate_completed,
    summarize_completed,
};

#[derive(Debug, Parser)]
#[command(name = "npclassifier-curate")]
#[command(about = "Curate high-confidence NPClassifier teacher splits")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Scan the teacher dataset and print the filtered split summary.
    Summary(CommonArgs),
    /// Materialize train / validation / test splits on disk.
    Curate(CurateArgs),
}

#[derive(Debug, Clone, Args)]
struct CommonArgs {
    #[arg(long = "input-dir", default_value = DEFAULT_COMPLETED_DIR)]
    input_dirs: Vec<PathBuf>,
    #[arg(long, default_value = DEFAULT_VOCABULARY_PATH)]
    vocabulary: PathBuf,
    #[arg(long, default_value_t = 50_000)]
    batch_rows: usize,
    #[arg(long)]
    max_rows: Option<usize>,
    #[arg(long, default_value_t = 0.8)]
    train_fraction: f64,
    #[arg(long, default_value_t = 0.1)]
    validation_fraction: f64,
    #[arg(long, default_value_t = 0.1)]
    test_fraction: f64,
    #[arg(long, default_value_t = 10)]
    min_signature_count: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 0.25)]
    pathway_positive_margin: f32,
    #[arg(long, default_value_t = 0.20)]
    pathway_negative_margin: f32,
    #[arg(long, default_value_t = 0.20)]
    superclass_positive_margin: f32,
    #[arg(long, default_value_t = 0.10)]
    superclass_negative_margin: f32,
    #[arg(long, default_value_t = 0.15)]
    class_positive_margin: f32,
    #[arg(long, default_value_t = 0.05)]
    class_negative_margin: f32,
}

#[derive(Debug, Clone, Args)]
struct CurateArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(long, default_value = DEFAULT_OUTPUT_DIR)]
    output_dir: PathBuf,
}

impl CommonArgs {
    fn into_config(self, output_dir: PathBuf) -> CurationConfig {
        CurationConfig {
            input_dirs: self.input_dirs,
            vocabulary_path: self.vocabulary,
            output_dir,
            batch_rows: self.batch_rows,
            max_rows: self.max_rows,
            margins: ConfidenceMargins {
                pathway_positive: self.pathway_positive_margin,
                pathway_negative: self.pathway_negative_margin,
                superclass_positive: self.superclass_positive_margin,
                superclass_negative: self.superclass_negative_margin,
                class_positive: self.class_positive_margin,
                class_negative: self.class_negative_margin,
            },
            split_fractions: SplitFractions {
                train: self.train_fraction,
                validation: self.validation_fraction,
                test: self.test_fraction,
            },
            min_signature_count: self.min_signature_count,
            seed: self.seed,
        }
    }
}

fn main() -> Result<(), NpClassifierError> {
    let cli = Cli::parse();
    match cli.command {
        Command::Summary(args) => {
            let report = summarize_completed(&args.into_config(PathBuf::from(DEFAULT_OUTPUT_DIR)))?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Curate(args) => {
            let report = curate_completed(&args.common.into_config(args.output_dir))?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    }
    Ok(())
}
