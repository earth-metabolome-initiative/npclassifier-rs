//! Cargo subcommand for packed `NPClassifier` classification.

use std::{
    ffi::OsString,
    fs,
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use clap::{Parser, ValueEnum};
use npclassifier_core::{
    ClassificationThresholds, ModelBundleSource, PackedClassifier, PackedClassifierBuilder,
    PackedModelVariant, WebBatchEntry,
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
    /// Optional output file. Defaults to stdout.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Packed model variant to load.
    #[arg(long, value_enum, default_value_t = CliVariant::Q4Kernel)]
    variant: CliVariant,
    /// Output format for classification results.
    #[arg(long, value_enum, default_value_t = OutputFormat::Jsonl)]
    format: OutputFormat,
    /// Number of SMILES to classify per parallel chunk.
    #[arg(long, default_value_t = 2048)]
    batch_size: usize,
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
    /// Emit progress updates to stderr while classifying.
    #[arg(long)]
    progress: bool,
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

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum OutputFormat {
    Json,
    Jsonl,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse_from(normalize_subcommand_args(std::env::args_os()));
    let classifier = build_classifier(&cli)?;
    let input = open_input(cli.input.as_deref())?;
    let output = open_output(cli.output.as_deref())?;
    classify_stream(
        &classifier,
        input,
        output,
        cli.format,
        cli.batch_size,
        cli.progress,
    )?;

    Ok(())
}

fn build_classifier(cli: &Cli) -> Result<PackedClassifier, Box<dyn std::error::Error>> {
    let mut builder = PackedClassifierBuilder::new().with_variant(cli.variant.into());
    builder = match select_source(cli)? {
        ModelBundleSource::Local(path) => builder.with_local_dir(path),
        ModelBundleSource::Remote(url) => builder.with_remote_base_url(url),
    };
    if let Some(thresholds) = threshold_override(cli) {
        builder = builder.with_thresholds(thresholds);
    }
    if let Some(cache_dir) = &cli.cache_dir {
        builder = builder.with_cache_dir(cache_dir.clone());
    }
    if let Some(threads) = cli.threads {
        builder = builder.with_parallelism(threads);
    }
    Ok(builder.build()?)
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

fn open_input(path: Option<&Path>) -> Result<Box<dyn BufRead>, Box<dyn std::error::Error>> {
    if let Some(path) = path.filter(|path| *path != Path::new("-")) {
        Ok(Box::new(BufReader::new(fs::File::open(path)?)))
    } else {
        Ok(Box::new(BufReader::new(io::stdin())))
    }
}

fn open_output(path: Option<&Path>) -> Result<Box<dyn Write>, Box<dyn std::error::Error>> {
    if let Some(path) = path.filter(|path| *path != Path::new("-")) {
        Ok(Box::new(BufWriter::new(fs::File::create(path)?)))
    } else {
        Ok(Box::new(BufWriter::new(io::stdout())))
    }
}

fn classify_stream(
    classifier: &PackedClassifier,
    mut input: Box<dyn BufRead>,
    output: Box<dyn Write>,
    format: OutputFormat,
    batch_size: usize,
    progress: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let started_at = Instant::now();
    let mut batch = Vec::with_capacity(batch_size.max(1));
    let mut line = String::new();
    let mut processed = 0usize;
    let mut writer = OutputWriter::new(output, format)?;

    loop {
        line.clear();
        if input.read_line(&mut line)? == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        batch.push(trimmed.to_owned());
        if batch.len() >= batch_size.max(1) {
            processed += flush_batch(classifier, &mut batch, &mut writer)?;
            if progress {
                eprintln!("processed {processed} SMILES");
            }
        }
    }

    if !batch.is_empty() {
        processed += flush_batch(classifier, &mut batch, &mut writer)?;
    }
    writer.finish()?;

    if progress {
        eprintln!(
            "processed {processed} SMILES in {:.2?}",
            started_at.elapsed()
        );
    }

    Ok(())
}

fn flush_batch(
    classifier: &PackedClassifier,
    batch: &mut Vec<String>,
    writer: &mut OutputWriter<Box<dyn Write>>,
) -> Result<usize, Box<dyn std::error::Error>> {
    let entries = classifier.classify_batch_parallel(batch);
    let count = entries.len();
    writer.write_entries(&entries)?;
    batch.clear();
    Ok(count)
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

struct OutputWriter<W: Write> {
    writer: W,
    format: OutputFormat,
    wrote_any: bool,
}

impl<W: Write> OutputWriter<W> {
    fn new(mut writer: W, format: OutputFormat) -> io::Result<Self> {
        if matches!(format, OutputFormat::Json) {
            writer.write_all(b"[")?;
        }
        Ok(Self {
            writer,
            format,
            wrote_any: false,
        })
    }

    fn write_entries(
        &mut self,
        entries: &[WebBatchEntry],
    ) -> Result<(), Box<dyn std::error::Error>> {
        for entry in entries {
            self.write_entry(entry)?;
        }
        Ok(())
    }

    fn write_entry(&mut self, entry: &WebBatchEntry) -> Result<(), Box<dyn std::error::Error>> {
        match self.format {
            OutputFormat::Json => {
                if self.wrote_any {
                    self.writer.write_all(b",\n")?;
                }
                serde_json::to_writer(&mut self.writer, entry)?;
            }
            OutputFormat::Jsonl => {
                serde_json::to_writer(&mut self.writer, entry)?;
                self.writer.write_all(b"\n")?;
            }
        }
        self.wrote_any = true;
        Ok(())
    }

    fn finish(mut self) -> io::Result<W> {
        if matches!(self.format, OutputFormat::Json) {
            self.writer.write_all(b"]\n")?;
        }
        self.writer.flush()?;
        Ok(self.writer)
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use npclassifier_core::{PredictionLabels, WebBatchEntry, WebScoredLabel};

    use super::{OutputFormat, OutputWriter, normalize_subcommand_args};

    #[test]
    fn output_writer_emits_jsonl_lines() {
        let mut writer =
            OutputWriter::new(Vec::new(), OutputFormat::Jsonl).expect("writer should initialize");
        writer
            .write_entries(&[sample_entry("CCO"), sample_entry("CCN")])
            .expect("jsonl should serialize");
        let bytes = writer.finish().expect("writer should flush");
        let text = String::from_utf8(bytes).expect("jsonl should be valid utf-8");

        assert_eq!(text.lines().count(), 2);
        assert!(text.contains("\"smiles\":\"CCO\""));
        assert!(text.contains("\"smiles\":\"CCN\""));
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

    #[test]
    fn output_writer_emits_json_array() {
        let mut writer =
            OutputWriter::new(Vec::new(), OutputFormat::Json).expect("writer should initialize");
        writer
            .write_entries(&[sample_entry("CCO"), sample_entry("CCN")])
            .expect("json array should serialize");
        let bytes = writer.finish().expect("writer should flush");
        let text = String::from_utf8(bytes).expect("json array should be valid utf-8");

        assert!(text.starts_with('['));
        assert!(text.ends_with("]\n"));
        assert!(text.contains("\"smiles\":\"CCO\""));
        assert!(text.contains("\"smiles\":\"CCN\""));
    }

    fn sample_entry(smiles: &str) -> WebBatchEntry {
        WebBatchEntry {
            smiles: smiles.to_owned(),
            error: None,
            labels: PredictionLabels::new(Vec::new(), Vec::new(), Vec::new(), None),
            pathway_scores: vec![WebScoredLabel {
                index: 0,
                name: String::from("pathway"),
                score: 0.5,
            }],
            superclass_scores: Vec::new(),
            class_scores: Vec::new(),
        }
    }
}
