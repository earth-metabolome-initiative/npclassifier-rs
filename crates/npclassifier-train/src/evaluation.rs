//! Shared split evaluation helpers.

use std::sync::Arc;

use burn::prelude::Backend;
use burn::train::InferenceStep;
use serde::Serialize;

use npclassifier_core::{
    ClassificationThresholds, FINGERPRINT_FORMULA_BITS, FINGERPRINT_INPUT_WIDTH, FingerprintInput,
    InferenceEngine, ModelHead, PackedModelSet,
};

use crate::data::NpClassifierBatch;
use crate::error::TrainingError;
use crate::metric::{ConfusionCounts, counts_from_tensors, matthews_correlation};
use crate::model::StudentModel;

/// Evaluation metrics for one dataset split.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SplitEvaluationReport {
    /// Number of evaluated rows.
    pub rows: usize,
    /// Pathway Matthews correlation coefficient.
    pub pathway_mcc: f64,
    /// Superclass Matthews correlation coefficient.
    pub superclass_mcc: f64,
    /// Class Matthews correlation coefficient.
    pub class_mcc: f64,
}

/// Evaluates a model on a prepared dataloader.
///
/// # Errors
///
/// Returns an error if Burn tensor synchronization or host-side decoding fails.
pub fn evaluate_split<B: Backend>(
    model: &StudentModel<B>,
    loader: &Arc<dyn burn::data::dataloader::DataLoader<B, NpClassifierBatch>>,
) -> Result<SplitEvaluationReport, TrainingError> {
    let mut pathway = ConfusionCounts::default();
    let mut superclass = ConfusionCounts::default();
    let mut class = ConfusionCounts::default();
    let mut rows = 0;

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
    }

    Ok(SplitEvaluationReport {
        rows,
        pathway_mcc: matthews_correlation(pathway),
        superclass_mcc: matthews_correlation(superclass),
        class_mcc: matthews_correlation(class),
    })
}

/// Evaluates the packed inference runtime on a prepared dataloader.
///
/// This intentionally measures the same packed q4 path used by the browser
/// runtime instead of trying to execute Burn's quantized tensor through a
/// generic dense layer.
///
/// # Errors
///
/// Returns an error if packed inference fails or if a batch fingerprint cannot
/// be reconstructed into the classifier input contract.
pub fn evaluate_packed_split<B: Backend>(
    model: &PackedModelSet,
    loader: &Arc<dyn burn::data::dataloader::DataLoader<B, NpClassifierBatch>>,
    thresholds: ClassificationThresholds,
) -> Result<SplitEvaluationReport, TrainingError> {
    let mut pathway = ConfusionCounts::default();
    let mut superclass = ConfusionCounts::default();
    let mut class = ConfusionCounts::default();
    let mut rows = 0;

    for batch in loader.iter() {
        let batch_len = batch.len();
        for row_index in 0..batch_len {
            let fingerprint = batch_fingerprint(&batch.inputs, row_index)?;
            let predictions = model.predict(&fingerprint)?;
            pathway = merge_counts(
                pathway,
                row_counts(
                    &predictions.pathway,
                    &batch.pathway_targets,
                    row_index,
                    ModelHead::Pathway.output_width(),
                    thresholds.pathway,
                )?,
            );
            superclass = merge_counts(
                superclass,
                row_counts(
                    &predictions.superclass,
                    &batch.superclass_targets,
                    row_index,
                    ModelHead::Superclass.output_width(),
                    thresholds.superclass,
                )?,
            );
            class = merge_counts(
                class,
                row_counts(
                    &predictions.class,
                    &batch.class_targets,
                    row_index,
                    ModelHead::Class.output_width(),
                    thresholds.class,
                )?,
            );
        }
        rows += batch_len;
    }

    Ok(SplitEvaluationReport {
        rows,
        pathway_mcc: matthews_correlation(pathway),
        superclass_mcc: matthews_correlation(superclass),
        class_mcc: matthews_correlation(class),
    })
}

/// Merges two confusion-count accumulators.
#[must_use]
pub const fn merge_counts(left: ConfusionCounts, right: ConfusionCounts) -> ConfusionCounts {
    ConfusionCounts {
        tp: left.tp + right.tp,
        tn: left.tn + right.tn,
        fp: left.fp + right.fp,
        fn_: left.fn_ + right.fn_,
    }
}

fn batch_fingerprint(inputs: &[f32], row_index: usize) -> Result<FingerprintInput, TrainingError> {
    let start = row_index * FINGERPRINT_INPUT_WIDTH;
    let end = start + FINGERPRINT_INPUT_WIDTH;
    let row = inputs
        .get(start..end)
        .ok_or_else(|| TrainingError::Dataset("batch fingerprint row is truncated".to_owned()))?;
    let (formula, radius) = row.split_at(FINGERPRINT_FORMULA_BITS);

    Ok(FingerprintInput::new(formula.to_vec(), radius.to_vec())?)
}

fn row_counts(
    predictions: &[f32],
    targets: &[i32],
    row_index: usize,
    width: usize,
    threshold: f32,
) -> Result<ConfusionCounts, TrainingError> {
    let start = row_index * width;
    let end = start + width;
    let targets = targets
        .get(start..end)
        .ok_or_else(|| TrainingError::Dataset("batch target row is truncated".to_owned()))?
        .iter()
        .map(|value| *value != 0)
        .collect::<Vec<_>>();

    Ok(crate::metric::accumulate_counts(
        predictions,
        &targets,
        threshold,
    ))
}
