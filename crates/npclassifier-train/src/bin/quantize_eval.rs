//! Evaluate Burn PTQ variants against the held-out `NPClassifier` test split.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use burn::backend::{Cuda, NdArray};
use burn::module::Module;
use burn::prelude::Backend;
use burn::record::{CompactRecorder, FileRecorder};
use burn::tensor::quantization::{
    BlockSize, QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
};
use burn::train::InferenceStep;
use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};

use npclassifier_core::ModelHead;
use npclassifier_train::data::{
    NpClassifierBatch, TrainingManifest, build_dataloader, load_manifest, load_split_storage,
};
use npclassifier_train::error::TrainingError;
use npclassifier_train::metric::{ConfusionCounts, counts_from_tensors, matthews_correlation};
use npclassifier_train::model::{StudentModel, StudentModelConfig};

const DEFAULT_DATA_DIR: &str = "data/distillation/teacher-splits";

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BackendKind {
    Cuda,
    Ndarray,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum QuantVariantKind {
    Q4Block32,
}

#[derive(Debug, Parser)]
#[command(name = "npclassifier-quantize-eval")]
#[command(about = "Evaluate Burn PTQ variants on the NPClassifier test split")]
struct Cli {
    #[arg(long, default_value = DEFAULT_DATA_DIR)]
    data_dir: PathBuf,
    #[arg(long)]
    artifact_dir: PathBuf,
    #[arg(long, value_enum, default_value_t = BackendKind::Cuda)]
    backend: BackendKind,
    #[arg(long, default_value_t = 0)]
    cuda_device: usize,
    #[arg(long)]
    batch_size: Option<usize>,
    #[arg(long)]
    num_workers: Option<usize>,
    #[arg(long)]
    test_rows: Option<usize>,
    #[arg(long, value_enum)]
    variant: Vec<QuantVariantKind>,
}

#[derive(Debug, Deserialize)]
struct SavedTrainingConfig {
    model: StudentModelConfig,
    batch_size: usize,
    num_workers: usize,
    hard_label_weight: f32,
    teacher_weight: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct SplitEvaluationReport {
    rows: usize,
    pathway_mcc: f64,
    superclass_mcc: f64,
    class_mcc: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct QuantizationReportRow {
    variant: String,
    size_bytes: u64,
    pathway_mcc: f64,
    pathway_delta: f64,
    superclass_mcc: f64,
    superclass_delta: f64,
    class_mcc: f64,
    class_delta: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct QuantizationReport {
    rows: usize,
    baseline_variant: String,
    baseline_size_bytes: u64,
    baseline: SplitEvaluationReport,
    variants: Vec<QuantizationReportRow>,
}

fn main() -> Result<(), TrainingError> {
    let cli = Cli::parse();
    match cli.backend {
        BackendKind::Cuda => {
            type BackendImpl = Cuda<f32, i32>;
            let device = burn::backend::cuda::CudaDevice::new(cli.cuda_device);
            run::<BackendImpl>(&cli, device)
        }
        BackendKind::Ndarray => {
            type BackendImpl = NdArray;
            let device = burn::backend::ndarray::NdArrayDevice::Cpu;
            run::<BackendImpl>(&cli, device)
        }
    }
}

fn run<B: Backend>(cli: &Cli, device: B::Device) -> Result<(), TrainingError>
where
    B::Device: Clone,
{
    validate_manifest(&load_manifest(&cli.data_dir)?)?;
    let saved_config = load_saved_training_config(&cli.artifact_dir)?;
    let batch_size = cli.batch_size.unwrap_or(saved_config.batch_size);
    let num_workers = cli.num_workers.unwrap_or(saved_config.num_workers);
    let test_storage = load_split_storage(&cli.data_dir, "test", cli.test_rows, false)?;
    let test_loader = build_dataloader::<B>(&test_storage, batch_size, num_workers, None, false);

    let baseline_model = load_saved_model::<B>(&saved_config, &cli.artifact_dir, &device)?;
    let baseline = evaluate_split("f32", &baseline_model, &test_loader)?;
    let baseline_size_bytes = recorder_file_size::<B>(&cli.artifact_dir.join("model"))?;

    let variant_kinds = if cli.variant.is_empty() {
        vec![QuantVariantKind::Q4Block32]
    } else {
        cli.variant.clone()
    };
    let mut variants = Vec::new();

    for variant in variant_kinds {
        let model = load_saved_model::<B>(&saved_config, &cli.artifact_dir, &device)?;
        let quantized = quantize_model(model, variant);
        let metrics = evaluate_split(variant.label(), &quantized, &test_loader)?;
        let size_bytes = save_quantized_model::<B>(&quantized, &cli.artifact_dir, variant)?;
        variants.push(QuantizationReportRow {
            variant: variant.label().to_owned(),
            size_bytes,
            pathway_mcc: metrics.pathway_mcc,
            pathway_delta: metrics.pathway_mcc - baseline.pathway_mcc,
            superclass_mcc: metrics.superclass_mcc,
            superclass_delta: metrics.superclass_mcc - baseline.superclass_mcc,
            class_mcc: metrics.class_mcc,
            class_delta: metrics.class_mcc - baseline.class_mcc,
        });
    }

    let report = QuantizationReport {
        rows: baseline.rows,
        baseline_variant: "f32".to_owned(),
        baseline_size_bytes,
        baseline,
        variants,
    };
    fs::write(
        cli.artifact_dir.join("quantization-report.json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    print_quantization_table(&report);
    println!("{}", serde_json::to_string_pretty(&report)?);

    Ok(())
}

fn load_saved_training_config(artifact_dir: &Path) -> Result<SavedTrainingConfig, TrainingError> {
    Ok(serde_json::from_str(&fs::read_to_string(
        artifact_dir.join("training-config.json"),
    )?)?)
}

fn load_saved_model<B: Backend>(
    config: &SavedTrainingConfig,
    artifact_dir: &Path,
    device: &B::Device,
) -> Result<StudentModel<B>, TrainingError> {
    config
        .model
        .init::<B>(device, config.hard_label_weight, config.teacher_weight)
        .load_file(artifact_dir.join("model"), &CompactRecorder::new(), device)
        .map_err(|error| TrainingError::Burn(error.to_string()))
}

fn quantize_model<B: Backend>(
    model: StudentModel<B>,
    _variant: QuantVariantKind,
) -> StudentModel<B> {
    model.quantize_compatible_linear_weights(&quant_scheme())
}

fn quant_scheme() -> QuantScheme {
    QuantScheme {
        level: QuantLevel::Block(BlockSize::new([32])),
        mode: QuantMode::Symmetric,
        value: QuantValue::Q4S,
        store: QuantStore::PackedU32(0),
        param: QuantParam::F32,
    }
}

fn save_quantized_model<B: Backend>(
    model: &StudentModel<B>,
    artifact_dir: &Path,
    variant: QuantVariantKind,
) -> Result<u64, TrainingError> {
    let base = artifact_dir
        .join("quantized")
        .join(variant.label())
        .join("model");
    if let Some(parent) = base.parent() {
        fs::create_dir_all(parent)?;
    }
    model
        .clone()
        .save_file(base.clone(), &CompactRecorder::new())
        .map_err(|error| TrainingError::Burn(error.to_string()))?;

    recorder_file_size::<B>(&base)
}

fn recorder_file_size<B: Backend>(base_path: &Path) -> Result<u64, TrainingError> {
    let mut path = base_path.to_path_buf();
    path.set_extension(<CompactRecorder as FileRecorder<B>>::file_extension());
    Ok(fs::metadata(path)?.len())
}

fn validate_manifest(manifest: &TrainingManifest) -> Result<(), TrainingError> {
    if manifest.vector_widths.pathway != ModelHead::Pathway.output_width()
        || manifest.vector_widths.superclass != ModelHead::Superclass.output_width()
        || manifest.vector_widths.class_ != ModelHead::Class.output_width()
    {
        return Err(TrainingError::Dataset(
            "curated manifest vector widths do not match the expected NPClassifier heads"
                .to_owned(),
        ));
    }
    Ok(())
}

fn evaluate_split<B: Backend>(
    label: &str,
    model: &StudentModel<B>,
    loader: &Arc<dyn burn::data::dataloader::DataLoader<B, NpClassifierBatch>>,
) -> Result<SplitEvaluationReport, TrainingError> {
    let mut pathway = ConfusionCounts::default();
    let mut superclass = ConfusionCounts::default();
    let mut class = ConfusionCounts::default();
    let mut rows = 0;
    let progress = ProgressBar::new_spinner();
    progress.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] {spinner:.cyan} {msg}")
            .map_err(|error| TrainingError::Dataset(format!("invalid progress style: {error}")))?,
    );
    progress.set_message(format!("evaluate {label}"));
    progress.enable_steady_tick(std::time::Duration::from_millis(100));

    for batch in loader.iter() {
        rows += batch.len();
        let output = model.step(batch);
        pathway = merge_counts(
            pathway,
            counts_from_tensors(
                output.pathway_probabilities,
                output.pathway_targets,
                ModelHead::Pathway.threshold(),
            )?,
        );
        superclass = merge_counts(
            superclass,
            counts_from_tensors(
                output.superclass_probabilities,
                output.superclass_targets,
                ModelHead::Superclass.threshold(),
            )?,
        );
        class = merge_counts(
            class,
            counts_from_tensors(
                output.class_probabilities,
                output.class_targets,
                ModelHead::Class.threshold(),
            )?,
        );
        progress.set_message(format!("evaluate {label}: {rows} rows"));
    }
    progress.finish_with_message(format!("evaluated {label}: {rows} rows"));

    Ok(SplitEvaluationReport {
        rows,
        pathway_mcc: matthews_correlation(pathway),
        superclass_mcc: matthews_correlation(superclass),
        class_mcc: matthews_correlation(class),
    })
}

fn merge_counts(left: ConfusionCounts, right: ConfusionCounts) -> ConfusionCounts {
    ConfusionCounts {
        tp: left.tp + right.tp,
        tn: left.tn + right.tn,
        fp: left.fp + right.fp,
        fn_: left.fn_ + right.fn_,
    }
}

fn print_quantization_table(report: &QuantizationReport) {
    println!();
    println!("Quantization Comparison");
    println!(
        "{:<12} {:>12} {:>14} {:>10} {:>18} {:>10} {:>12} {:>10}",
        "Variant",
        "Size MiB",
        "Pathway MCC",
        "Delta",
        "Superclass MCC",
        "Delta",
        "Class MCC",
        "Delta"
    );
    println!(
        "{:-<12} {:-<12} {:-<14} {:-<10} {:-<18} {:-<10} {:-<12} {:-<10}",
        "", "", "", "", "", "", "", ""
    );

    print_quantization_row(
        &report.baseline_variant,
        report.baseline_size_bytes,
        &report.baseline,
        0.0,
        0.0,
        0.0,
    );
    for row in &report.variants {
        println!(
            "{:<12} {:>12} {:>14.6} {:>10.6} {:>18.6} {:>10.6} {:>12.6} {:>10.6}",
            row.variant,
            format_mib(row.size_bytes),
            row.pathway_mcc,
            row.pathway_delta,
            row.superclass_mcc,
            row.superclass_delta,
            row.class_mcc,
            row.class_delta
        );
    }
    println!();
}

fn print_quantization_row(
    label: &str,
    size_bytes: u64,
    report: &SplitEvaluationReport,
    pathway_delta: f64,
    superclass_delta: f64,
    class_delta: f64,
) {
    println!(
        "{:<12} {:>12} {:>14.6} {:>10.6} {:>18.6} {:>10.6} {:>12.6} {:>10.6}",
        label,
        format_mib(size_bytes),
        report.pathway_mcc,
        pathway_delta,
        report.superclass_mcc,
        superclass_delta,
        report.class_mcc,
        class_delta
    );
}

fn format_mib(bytes: u64) -> String {
    let tenths = bytes.saturating_mul(10) / (1024 * 1024);
    format!("{}.{}", tenths / 10, tenths % 10)
}

impl QuantVariantKind {
    fn label(self) -> &'static str {
        match self {
            Self::Q4Block32 => "q4-block32",
        }
    }
}
