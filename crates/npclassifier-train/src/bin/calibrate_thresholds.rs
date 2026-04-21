//! Calibrate multilabel decision thresholds on the validation split.

use std::fs;
use std::path::{Path, PathBuf};

use burn::backend::{Cuda, NdArray};
use burn::prelude::{Backend, Module};
use burn::record::CompactRecorder;
use burn::tensor::Transaction;
use burn::train::InferenceStep;
use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};

use npclassifier_core::{ClassificationThresholds, ModelHead};

use npclassifier_train::data::{
    NpClassifierBatch, TeacherSplitStorage, build_dataloader, load_split_storage,
};
use npclassifier_train::error::TrainingError;
use npclassifier_train::metric::{ConfusionCounts, matthews_correlation};
use npclassifier_train::model::{NpClassifierOutput, StudentModel, StudentModelConfig};

const DEFAULT_DATA_DIR: &str = "data/distillation/teacher-splits";
const DEFAULT_BATCH_SIZE: usize = 2048;
const DEFAULT_NUM_WORKERS: usize = 8;
const DEFAULT_THRESHOLD_BINS: u32 = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BackendKind {
    Cuda,
    Ndarray,
}

#[derive(Debug, Parser)]
#[command(name = "npclassifier-calibrate-thresholds")]
#[command(about = "Calibrate per-head decision thresholds on the validation split")]
struct Cli {
    #[arg(long, default_value = DEFAULT_DATA_DIR)]
    data_dir: PathBuf,
    #[arg(long)]
    artifact_dir: PathBuf,
    #[arg(long, value_enum, default_value_t = BackendKind::Cuda)]
    backend: BackendKind,
    #[arg(long, default_value_t = 0)]
    cuda_device: usize,
    #[arg(long, default_value_t = DEFAULT_BATCH_SIZE)]
    batch_size: usize,
    #[arg(long, default_value_t = DEFAULT_NUM_WORKERS)]
    num_workers: usize,
    #[arg(long)]
    validation_rows: Option<usize>,
    #[arg(long, default_value_t = DEFAULT_THRESHOLD_BINS)]
    bins: u32,
}

#[derive(Debug, Deserialize)]
struct SavedTrainingConfig {
    model: StudentModelConfig,
    hard_label_weight: f32,
    teacher_weight: f32,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct HeadCalibration {
    legacy_threshold: f32,
    legacy_mcc: f64,
    calibrated_threshold: f32,
    calibrated_mcc: f64,
}

#[derive(Debug, Serialize)]
struct ThresholdCalibrationReport {
    rows: usize,
    bins: u32,
    thresholds: ClassificationThresholds,
    pathway: HeadCalibration,
    superclass: HeadCalibration,
    class: HeadCalibration,
}

#[derive(Debug, Clone, Copy, Default)]
struct HistogramBucket {
    positives: u64,
    negatives: u64,
}

#[derive(Debug)]
struct ThresholdHistogram {
    bins: u32,
    buckets: Vec<HistogramBucket>,
    total_positives: u64,
    total_negatives: u64,
}

impl ThresholdHistogram {
    fn new(bins: u32) -> Result<Self, TrainingError> {
        let bucket_count =
            usize::try_from(bins.checked_add(1).ok_or_else(|| {
                TrainingError::Dataset("threshold bin count overflowed".to_owned())
            })?)
            .map_err(|error| TrainingError::Dataset(error.to_string()))?;

        Ok(Self {
            bins,
            buckets: vec![HistogramBucket::default(); bucket_count],
            total_positives: 0,
            total_negatives: 0,
        })
    }

    fn observe(&mut self, predictions: &[f32], targets: &[bool]) -> Result<(), TrainingError> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::Dataset(format!(
                "prediction/target length mismatch: {} vs {}",
                predictions.len(),
                targets.len()
            )));
        }

        for (prediction, target) in predictions.iter().zip(targets) {
            if !prediction.is_finite() {
                return Err(TrainingError::Dataset(
                    "encountered non-finite validation prediction during threshold calibration"
                        .to_owned(),
                ));
            }
            let bucket_index = score_to_bucket(*prediction, self.bins)?;
            let bucket = self
                .buckets
                .get_mut(bucket_index)
                .expect("bucket index should stay within the configured range");
            if *target {
                bucket.positives += 1;
                self.total_positives += 1;
            } else {
                bucket.negatives += 1;
                self.total_negatives += 1;
            }
        }

        Ok(())
    }

    fn calibrate(&self, legacy_threshold: f32) -> HeadCalibration {
        let legacy_mcc = matthews_correlation(self.counts_at_threshold(legacy_threshold));
        let mut best = HeadCalibration {
            legacy_threshold,
            legacy_mcc,
            calibrated_threshold: 1.0,
            calibrated_mcc: matthews_correlation(ConfusionCounts {
                tp: 0,
                tn: u64_to_u32(self.total_negatives),
                fp: 0,
                fn_: u64_to_u32(self.total_positives),
            }),
        };
        let mut cumulative_positive = 0_u64;
        let mut cumulative_negative = 0_u64;

        for index in (0..self.buckets.len()).rev() {
            let bucket = self.buckets[index];
            cumulative_positive += bucket.positives;
            cumulative_negative += bucket.negatives;

            let counts = ConfusionCounts {
                tp: u64_to_u32(cumulative_positive),
                fp: u64_to_u32(cumulative_negative),
                fn_: u64_to_u32(self.total_positives - cumulative_positive),
                tn: u64_to_u32(self.total_negatives - cumulative_negative),
            };
            let threshold = bucket_to_threshold(
                u32::try_from(index).expect("bucket index should fit into u32"),
                self.bins,
            );
            let mcc = matthews_correlation(counts);
            if mcc > best.calibrated_mcc {
                best.calibrated_threshold = threshold;
                best.calibrated_mcc = mcc;
            }
        }

        best
    }

    fn counts_at_threshold(&self, threshold: f32) -> ConfusionCounts {
        let threshold_bucket = score_to_bucket(threshold.clamp(0.0, 1.0), self.bins)
            .expect("threshold should quantize");
        let mut positive = 0_u64;
        let mut negative = 0_u64;

        for bucket in &self.buckets[threshold_bucket..] {
            positive += bucket.positives;
            negative += bucket.negatives;
        }

        ConfusionCounts {
            tp: u64_to_u32(positive),
            fp: u64_to_u32(negative),
            fn_: u64_to_u32(self.total_positives - positive),
            tn: u64_to_u32(self.total_negatives - negative),
        }
    }
}

fn main() -> Result<(), TrainingError> {
    let cli = Cli::parse();
    match cli.backend {
        BackendKind::Cuda => run_cuda(&cli),
        BackendKind::Ndarray => run_ndarray(&cli),
    }
}

fn run_cuda(cli: &Cli) -> Result<(), TrainingError> {
    type BackendImpl = Cuda<f32, i32>;
    let device = burn::backend::cuda::CudaDevice::new(cli.cuda_device);
    run_calibration::<BackendImpl>(cli, device)
}

fn run_ndarray(cli: &Cli) -> Result<(), TrainingError> {
    type BackendImpl = NdArray;
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    run_calibration::<BackendImpl>(cli, device)
}

fn run_calibration<B: Backend>(cli: &Cli, device: B::Device) -> Result<(), TrainingError> {
    let config = load_saved_training_config(&cli.artifact_dir)?;
    let validation_storage =
        load_split_storage(&cli.data_dir, "validation", cli.validation_rows, false)?;
    let loader = build_dataloader::<B>(
        &validation_storage,
        cli.batch_size,
        cli.num_workers,
        None,
        false,
    );
    let model = config
        .model
        .init::<B>(&device, config.hard_label_weight, config.teacher_weight)
        .load_file(
            cli.artifact_dir.join("model"),
            &CompactRecorder::new(),
            &device,
        )
        .map_err(|error| TrainingError::Burn(error.to_string()))?;
    let report = calibrate_validation_thresholds(&model, &validation_storage, &loader, cli.bins)?;

    fs::write(
        cli.artifact_dir.join("thresholds.json"),
        serde_json::to_string_pretty(&report.thresholds)?,
    )?;
    fs::write(
        cli.artifact_dir.join("threshold-calibration.json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);

    Ok(())
}

fn load_saved_training_config(artifact_dir: &Path) -> Result<SavedTrainingConfig, TrainingError> {
    Ok(serde_json::from_str(&fs::read_to_string(
        artifact_dir.join("training-config.json"),
    )?)?)
}

fn calibrate_validation_thresholds<B: Backend>(
    model: &StudentModel<B>,
    storage: &TeacherSplitStorage,
    loader: &std::sync::Arc<dyn burn::data::dataloader::DataLoader<B, NpClassifierBatch>>,
    bins: u32,
) -> Result<ThresholdCalibrationReport, TrainingError> {
    let mut pathway_histogram = ThresholdHistogram::new(bins)?;
    let mut superclass_histogram = ThresholdHistogram::new(bins)?;
    let mut class_histogram = ThresholdHistogram::new(bins)?;
    let progress = rows_progress("calibrate validation thresholds", storage.len())?;

    for batch in loader.iter() {
        let batch_len = batch.len();
        let output = model.step(batch);
        observe_batch(
            &mut pathway_histogram,
            &mut superclass_histogram,
            &mut class_histogram,
            output,
        )?;
        progress.inc(
            u64::try_from(batch_len).map_err(|error| TrainingError::Dataset(error.to_string()))?,
        );
    }

    progress.finish_with_message(format!(
        "calibrated thresholds on {} validation rows",
        storage.len()
    ));

    let pathway = pathway_histogram.calibrate(ModelHead::Pathway.threshold());
    let superclass = superclass_histogram.calibrate(ModelHead::Superclass.threshold());
    let class = class_histogram.calibrate(ModelHead::Class.threshold());

    Ok(ThresholdCalibrationReport {
        rows: storage.len(),
        bins,
        thresholds: ClassificationThresholds::new(
            pathway.calibrated_threshold,
            superclass.calibrated_threshold,
            class.calibrated_threshold,
        ),
        pathway,
        superclass,
        class,
    })
}

fn observe_batch<B: Backend>(
    pathway_histogram: &mut ThresholdHistogram,
    superclass_histogram: &mut ThresholdHistogram,
    class_histogram: &mut ThresholdHistogram,
    output: NpClassifierOutput<B>,
) -> Result<(), TrainingError> {
    let [
        pathway_predictions,
        superclass_predictions,
        class_predictions,
        pathway_targets,
        superclass_targets,
        class_targets,
    ] = Transaction::default()
        .register(output.pathway_probabilities)
        .register(output.superclass_probabilities)
        .register(output.class_probabilities)
        .register(output.pathway_targets)
        .register(output.superclass_targets)
        .register(output.class_targets)
        .execute()
        .try_into()
        .expect("correct number of synchronized calibration tensors");

    let pathway_predictions = pathway_predictions
        .to_vec::<f32>()
        .map_err(|error| TrainingError::Burn(error.to_string()))?;
    let superclass_predictions = superclass_predictions
        .to_vec::<f32>()
        .map_err(|error| TrainingError::Burn(error.to_string()))?;
    let class_predictions = class_predictions
        .to_vec::<f32>()
        .map_err(|error| TrainingError::Burn(error.to_string()))?;
    let pathway_targets = decode_targets(pathway_targets)?;
    let superclass_targets = decode_targets(superclass_targets)?;
    let class_targets = decode_targets(class_targets)?;

    pathway_histogram.observe(&pathway_predictions, &pathway_targets)?;
    superclass_histogram.observe(&superclass_predictions, &superclass_targets)?;
    class_histogram.observe(&class_predictions, &class_targets)?;

    Ok(())
}

fn decode_targets(targets: burn::tensor::TensorData) -> Result<Vec<bool>, TrainingError> {
    Ok(targets
        .to_vec::<i32>()
        .map_err(|error| TrainingError::Burn(error.to_string()))?
        .into_iter()
        .map(|value| value != 0)
        .collect())
}

fn rows_progress(label: &str, total_rows: usize) -> Result<ProgressBar, TrainingError> {
    let progress = ProgressBar::new(
        u64::try_from(total_rows).map_err(|error| TrainingError::Dataset(error.to_string()))?,
    );
    progress.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.magenta/blue} {pos:>10}/{len:<10} {msg}",
        )
        .map_err(|error| TrainingError::Dataset(format!("invalid progress style: {error}")))?,
    );
    progress.set_message(label.to_owned());
    Ok(progress)
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn score_to_bucket(score: f32, bins: u32) -> Result<usize, TrainingError> {
    let clamped = f64::from(score.clamp(0.0, 1.0));
    let scaled = (clamped * f64::from(bins)).floor();
    let bucket = if scaled <= 0.0 {
        0_u32
    } else if scaled >= f64::from(bins) {
        bins
    } else {
        scaled as u32
    };

    usize::try_from(bucket).map_err(|error| TrainingError::Dataset(error.to_string()))
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn bucket_to_threshold(bucket: u32, bins: u32) -> f32 {
    (f64::from(bucket) / f64::from(bins)) as f32
}

fn u64_to_u32(value: u64) -> u32 {
    u32::try_from(value).expect("validation calibration counts should fit into u32")
}
