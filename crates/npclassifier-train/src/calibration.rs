//! Validation-set threshold calibration.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use burn::prelude::Backend;
use burn::tensor::Transaction;
use burn::train::InferenceStep;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;

use npclassifier_core::{ClassificationThresholds, ModelHead};

use crate::data::{NpClassifierBatch, TeacherSplitStorage};
use crate::error::TrainingError;
use crate::metric::{ConfusionCounts, matthews_correlation};
use crate::model::{NpClassifierOutput, StudentModel};

/// Default number of score buckets used for threshold search.
pub const DEFAULT_THRESHOLD_BINS: u32 = 10_000;

/// Calibration summary for one prediction head.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct HeadCalibration {
    /// Legacy fixed threshold used before validation calibration.
    pub legacy_threshold: f32,
    /// MCC at the legacy fixed threshold.
    pub legacy_mcc: f64,
    /// Validation-selected threshold.
    pub calibrated_threshold: f32,
    /// MCC at the validation-selected threshold.
    pub calibrated_mcc: f64,
}

/// Threshold calibration report written after training.
#[derive(Debug, Serialize)]
pub struct ThresholdCalibrationReport {
    /// Number of validation rows used.
    pub rows: usize,
    /// Number of histogram buckets used for threshold search.
    pub bins: u32,
    /// Calibrated threshold triple.
    pub thresholds: ClassificationThresholds,
    /// Pathway-head calibration details.
    pub pathway: HeadCalibration,
    /// Superclass-head calibration details.
    pub superclass: HeadCalibration,
    /// Class-head calibration details.
    pub class: HeadCalibration,
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
                tn: self.total_negatives,
                fp: 0,
                fn_: self.total_positives,
            }),
        };
        let mut cumulative_positive = 0_u64;
        let mut cumulative_negative = 0_u64;

        for index in (0..self.buckets.len()).rev() {
            let bucket = self.buckets[index];
            cumulative_positive += bucket.positives;
            cumulative_negative += bucket.negatives;

            let counts = ConfusionCounts {
                tp: cumulative_positive,
                fp: cumulative_negative,
                fn_: self.total_positives - cumulative_positive,
                tn: self.total_negatives - cumulative_negative,
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
            tp: positive,
            fp: negative,
            fn_: self.total_positives - positive,
            tn: self.total_negatives - negative,
        }
    }
}

/// Calibrates thresholds for all three heads on the validation split.
///
/// # Errors
///
/// Returns an error if predictions cannot be synchronized or decoded, or if the
/// calibration histogram cannot represent the configured number of bins.
pub fn calibrate_validation_thresholds<B: Backend>(
    model: &StudentModel<B>,
    storage: &TeacherSplitStorage,
    loader: &Arc<dyn burn::data::dataloader::DataLoader<B, NpClassifierBatch>>,
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

/// Writes calibrated thresholds and the full calibration report.
///
/// # Errors
///
/// Returns an error if either JSON file cannot be serialized or written.
pub fn write_threshold_report(
    artifact_dir: &Path,
    report: &ThresholdCalibrationReport,
) -> Result<(), TrainingError> {
    fs::write(
        artifact_dir.join("thresholds.json"),
        serde_json::to_string_pretty(&report.thresholds)?,
    )?;
    fs::write(
        artifact_dir.join("threshold-calibration.json"),
        serde_json::to_string_pretty(report)?,
    )?;

    Ok(())
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
    let pathway_targets = decode_targets(&pathway_targets)?;
    let superclass_targets = decode_targets(&superclass_targets)?;
    let class_targets = decode_targets(&class_targets)?;

    pathway_histogram.observe(&pathway_predictions, &pathway_targets)?;
    superclass_histogram.observe(&superclass_predictions, &superclass_targets)?;
    class_histogram.observe(&class_predictions, &class_targets)?;

    Ok(())
}

fn decode_targets(targets: &burn::tensor::TensorData) -> Result<Vec<bool>, TrainingError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_histogram_selects_mcc_optimal_cutoff() {
        let mut histogram = ThresholdHistogram::new(10).expect("histogram");
        histogram
            .observe(&[0.05, 0.15, 0.85, 0.95], &[false, false, true, true])
            .expect("synthetic observations");

        let calibration = histogram.calibrate(0.5);

        assert_eq!(
            histogram.counts_at_threshold(0.5),
            ConfusionCounts {
                tp: 2,
                tn: 2,
                fp: 0,
                fn_: 0,
            }
        );
        assert!((calibration.legacy_mcc - 1.0).abs() < f64::EPSILON);
        assert!((calibration.calibrated_mcc - 1.0).abs() < f64::EPSILON);
        assert!((0.2..=0.9).contains(&calibration.calibrated_threshold));
    }

    #[test]
    fn threshold_histogram_rejects_non_finite_predictions() {
        let mut histogram = ThresholdHistogram::new(4).expect("histogram");

        let error = histogram
            .observe(&[f32::NAN], &[true])
            .expect_err("non-finite predictions should be rejected");

        assert!(error.to_string().contains("non-finite"));
    }
}
