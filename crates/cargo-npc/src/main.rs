//! Cargo subcommand for packed `NPClassifier` classification.

use std::{
    ffi::OsString,
    fs,
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use arrow_array::{
    ArrayRef, RecordBatch,
    builder::{BooleanBuilder, ListBuilder, StringBuilder},
};
use arrow_schema::{DataType, Field, Schema};
use clap::{Parser, ValueEnum};
use npclassifier_core::{
    ClassificationThresholds, HostedModel, PackedClassifier, PackedClassifierBuilder,
    PackedModelVariant, WebBatchEntry,
};
use parquet::arrow::ArrowWriter;

#[derive(Debug, Parser)]
#[command(name = "cargo-npc")]
#[command(about = "Classify SMILES lines with a packed NPClassifier bundle.")]
struct Cli {
    /// Hosted model bundle to download automatically.
    #[arg(long, value_enum, default_value_t = CliModel::Mini)]
    model: CliModel,
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

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliModel {
    Mini,
    Faithful,
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

impl From<CliModel> for HostedModel {
    fn from(value: CliModel) -> Self {
        match value {
            CliModel::Mini => Self::Mini,
            CliModel::Faithful => Self::Faithful,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum OutputFormat {
    Json,
    Jsonl,
    Parquet,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse_from(normalize_subcommand_args(std::env::args_os()));
    let classifier = build_classifier(&cli)?;
    let input = open_input(cli.input.as_deref())?;
    let output = open_output_writer(&cli)?;
    classify_stream(&classifier, input, output, cli.batch_size, cli.progress)?;

    Ok(())
}

fn build_classifier(cli: &Cli) -> Result<PackedClassifier, Box<dyn std::error::Error>> {
    let mut builder = PackedClassifierBuilder::new()
        .with_model(cli.model.into())
        .with_variant(cli.variant.into());
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

fn open_input(path: Option<&Path>) -> Result<Box<dyn BufRead>, Box<dyn std::error::Error>> {
    if let Some(path) = path.filter(|path| *path != Path::new("-")) {
        Ok(Box::new(BufReader::new(fs::File::open(path)?)))
    } else {
        Ok(Box::new(BufReader::new(io::stdin())))
    }
}

fn open_output_writer(cli: &Cli) -> Result<ResultsWriter, Box<dyn std::error::Error>> {
    match cli.format {
        OutputFormat::Json | OutputFormat::Jsonl => {
            let path = cli.output.as_deref();
            let writer: Box<dyn Write> =
                if let Some(path) = path.filter(|path| *path != Path::new("-")) {
                    Box::new(BufWriter::new(fs::File::create(path)?))
                } else {
                    Box::new(BufWriter::new(io::stdout()))
                };
            Ok(ResultsWriter::Json(JsonOutputWriter::new(
                writer, cli.format,
            )?))
        }
        OutputFormat::Parquet => {
            let path = cli
                .output
                .as_deref()
                .filter(|path| *path != Path::new("-"))
                .ok_or("parquet output requires --output <PATH>")?;
            Ok(ResultsWriter::Parquet(Box::new(ParquetOutputWriter::new(
                path,
            )?)))
        }
    }
}

fn classify_stream(
    classifier: &PackedClassifier,
    mut input: Box<dyn BufRead>,
    mut output: ResultsWriter,
    batch_size: usize,
    progress: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let started_at = Instant::now();
    let mut batch = Vec::with_capacity(batch_size.max(1));
    let mut line = String::new();
    let mut processed = 0usize;

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
            processed += flush_batch(classifier, &mut batch, &mut output)?;
            if progress {
                eprintln!("processed {processed} SMILES");
            }
        }
    }

    if !batch.is_empty() {
        processed += flush_batch(classifier, &mut batch, &mut output)?;
    }
    output.finish()?;

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
    writer: &mut ResultsWriter,
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

enum ResultsWriter {
    Json(JsonOutputWriter<Box<dyn Write>>),
    Parquet(Box<ParquetOutputWriter>),
}

impl ResultsWriter {
    fn write_entries(
        &mut self,
        entries: &[WebBatchEntry],
    ) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Self::Json(writer) => writer.write_entries(entries),
            Self::Parquet(writer) => writer.write_entries(entries),
        }
    }

    fn finish(self) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Self::Json(writer) => {
                let _ignored = writer.finish()?;
                Ok(())
            }
            Self::Parquet(writer) => writer.finish(),
        }
    }
}

struct JsonOutputWriter<W: Write> {
    writer: W,
    format: OutputFormat,
    wrote_any: bool,
}

impl<W: Write> JsonOutputWriter<W> {
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
            OutputFormat::Parquet => unreachable!("parquet output uses ParquetOutputWriter"),
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

struct ParquetOutputWriter {
    writer: ArrowWriter<File>,
    schema: Arc<Schema>,
}

impl ParquetOutputWriter {
    fn new(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let schema = parquet_output_schema();
        let writer = ArrowWriter::try_new(File::create(path)?, schema.clone(), None)?;
        Ok(Self { writer, schema })
    }

    fn write_entries(
        &mut self,
        entries: &[WebBatchEntry],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if entries.is_empty() {
            return Ok(());
        }
        let batch = build_parquet_batch(entries, self.schema.clone())?;
        self.writer.write(&batch)?;
        Ok(())
    }

    fn finish(self) -> Result<(), Box<dyn std::error::Error>> {
        let _metadata = self.writer.close()?;
        Ok(())
    }
}

fn parquet_output_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("smiles", DataType::Utf8, false),
        Field::new("error", DataType::Utf8, true),
        string_list_field("pathways"),
        string_list_field("superclasses"),
        string_list_field("classes"),
        Field::new("is_glycoside", DataType::Boolean, true),
    ]))
}

fn string_list_field(name: &'static str) -> Field {
    Field::new(
        name,
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        false,
    )
}

fn build_parquet_batch(
    entries: &[WebBatchEntry],
    schema: Arc<Schema>,
) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let mut smiles = StringBuilder::new();
    let mut errors = StringBuilder::new();
    let mut pathways = ListBuilder::new(StringBuilder::new());
    let mut superclasses = ListBuilder::new(StringBuilder::new());
    let mut classes = ListBuilder::new(StringBuilder::new());
    let mut is_glycoside = BooleanBuilder::new();

    for entry in entries {
        smiles.append_value(&entry.smiles);
        if let Some(error) = &entry.error {
            errors.append_value(error);
        } else {
            errors.append_null();
        }
        append_string_list(&mut pathways, &entry.labels.pathways);
        append_string_list(&mut superclasses, &entry.labels.superclasses);
        append_string_list(&mut classes, &entry.labels.classes);
        if let Some(value) = entry.labels.is_glycoside {
            is_glycoside.append_value(value);
        } else {
            is_glycoside.append_null();
        }
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(smiles.finish()) as ArrayRef,
            Arc::new(errors.finish()) as ArrayRef,
            Arc::new(pathways.finish()) as ArrayRef,
            Arc::new(superclasses.finish()) as ArrayRef,
            Arc::new(classes.finish()) as ArrayRef,
            Arc::new(is_glycoside.finish()) as ArrayRef,
        ],
    )
    .map_err(Into::into)
}

fn append_string_list(builder: &mut ListBuilder<StringBuilder>, values: &[String]) {
    for value in values {
        builder.values().append_value(value);
    }
    builder.append(true);
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use npclassifier_core::{PredictionLabels, WebBatchEntry, WebScoredLabel};

    use super::{
        JsonOutputWriter, OutputFormat, build_parquet_batch, normalize_subcommand_args,
        parquet_output_schema,
    };

    #[test]
    fn output_writer_emits_jsonl_lines() {
        let mut writer = JsonOutputWriter::new(Vec::new(), OutputFormat::Jsonl)
            .expect("writer should initialize");
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
        let mut writer = JsonOutputWriter::new(Vec::new(), OutputFormat::Json)
            .expect("writer should initialize");
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

    #[test]
    fn parquet_batch_uses_compact_label_schema() {
        let schema = parquet_output_schema();
        let batch =
            build_parquet_batch(&[sample_entry("CCO"), sample_entry("CCN")], schema.clone())
                .expect("parquet batch should build");

        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.schema(), schema);
        assert_eq!(batch.num_columns(), 6);
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
