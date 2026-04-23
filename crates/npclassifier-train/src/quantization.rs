//! Post-training quantization and comparison reporting.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use burn::prelude::Backend;
use burn::record::{CompactRecorder, FileRecorder};
use serde::Serialize;

use npclassifier_core::{ClassificationThresholds, PackedModelSet, PackedModelVariant};

use crate::data::NpClassifierBatch;
use crate::error::TrainingError;
use crate::evaluation::{SplitEvaluationReport, evaluate_packed_split};
use crate::model::StudentModel;
use crate::web_export::export_web_model_q4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuantVariantKind {
    Q4Block32,
}

/// Metrics and size deltas for one quantized model variant.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QuantizationReportRow {
    /// Variant label.
    pub variant: String,
    /// Serialized model size in bytes.
    pub size_bytes: u64,
    /// Pathway Matthews correlation coefficient.
    pub pathway_mcc: f64,
    /// Pathway MCC delta against f32.
    pub pathway_delta: f64,
    /// Superclass Matthews correlation coefficient.
    pub superclass_mcc: f64,
    /// Superclass MCC delta against f32.
    pub superclass_delta: f64,
    /// Class Matthews correlation coefficient.
    pub class_mcc: f64,
    /// Class MCC delta against f32.
    pub class_delta: f64,
}

/// Quantization comparison report written after training.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QuantizationReport {
    /// Number of test rows evaluated.
    pub rows: usize,
    /// Baseline f32 variant label.
    pub baseline_variant: String,
    /// Serialized f32 model size in bytes.
    pub baseline_size_bytes: u64,
    /// Baseline f32 metrics.
    pub baseline: SplitEvaluationReport,
    /// Quantized variant metrics.
    pub variants: Vec<QuantizationReportRow>,
}

/// Exports the selected model as q4 packed browser archives, evaluates that
/// packed runtime, and builds the comparison report.
///
/// # Errors
///
/// Returns an error if evaluation, serialization, or model-size inspection
/// fails.
pub fn quantize_and_evaluate<B: Backend>(
    model: &StudentModel<B>,
    artifact_dir: &Path,
    test_loader: &Arc<dyn burn::data::dataloader::DataLoader<B, NpClassifierBatch>>,
    baseline: SplitEvaluationReport,
    web_output_dir: &Path,
    thresholds: ClassificationThresholds,
) -> Result<QuantizationReport, TrainingError> {
    let baseline_size_bytes = recorder_file_size::<B>(&artifact_dir.join("model"))?;
    let variant = QuantVariantKind::Q4Block32;
    export_web_model_q4(web_output_dir, model, thresholds)?;
    let size_bytes = directory_size_bytes(web_output_dir)?;
    let packed = PackedModelSet::from_dir(web_output_dir, PackedModelVariant::Q4Kernel)?;
    let metrics = evaluate_packed_split(&packed, test_loader, ClassificationThresholds::default())?;
    let report = QuantizationReport {
        rows: baseline.rows,
        baseline_variant: "f32".to_owned(),
        baseline_size_bytes,
        variants: vec![QuantizationReportRow {
            variant: variant.label().to_owned(),
            size_bytes,
            pathway_mcc: metrics.pathway_mcc,
            pathway_delta: metrics.pathway_mcc - baseline.pathway_mcc,
            superclass_mcc: metrics.superclass_mcc,
            superclass_delta: metrics.superclass_mcc - baseline.superclass_mcc,
            class_mcc: metrics.class_mcc,
            class_delta: metrics.class_mcc - baseline.class_mcc,
        }],
        baseline,
    };

    Ok(report)
}

/// Writes the quantization comparison report.
///
/// # Errors
///
/// Returns an error if the report cannot be serialized or written.
pub fn write_quantization_report(
    artifact_dir: &Path,
    report: &QuantizationReport,
) -> Result<(), TrainingError> {
    fs::write(
        artifact_dir.join("quantization-report.json"),
        serde_json::to_string_pretty(report)?,
    )?;

    Ok(())
}

/// Prints a compact quantization comparison table.
pub fn print_quantization_table(report: &QuantizationReport) {
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

fn recorder_file_size<B: Backend>(base_path: &Path) -> Result<u64, TrainingError> {
    let mut path = base_path.to_path_buf();
    path.set_extension(<CompactRecorder as FileRecorder<B>>::file_extension());
    Ok(fs::metadata(path)?.len())
}

fn directory_size_bytes(path: &Path) -> Result<u64, TrainingError> {
    let mut size = 0_u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            size += directory_size_bytes(&entry.path())?;
        } else {
            size += metadata.len();
        }
    }
    Ok(size)
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
