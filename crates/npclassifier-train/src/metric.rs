//! Custom Burn metrics for multilabel `NPClassifier` training.

use std::marker::PhantomData;
use std::sync::Arc;

use burn::prelude::{Backend, Int, Tensor};
use burn::tensor::Transaction;
use burn::train::metric::state::FormatOptions;
use burn::train::metric::{
    Adaptor, ConfusionStatsInput, Metric, MetricAttributes, MetricMetadata, MetricName, Numeric,
    NumericAttributes, NumericEntry, SerializedEntry, format_float,
};

use npclassifier_core::ModelHead;

use crate::error::TrainingError;
use crate::model::NpClassifierOutput;

/// Head-specific confusion input for the pathway labels.
#[derive(Debug, Clone)]
pub struct PathwayConfusionInput<B: Backend> {
    inner: ConfusionStatsInput<B>,
}

/// Head-specific confusion input for the superclass labels.
#[derive(Debug, Clone)]
pub struct SuperclassConfusionInput<B: Backend> {
    inner: ConfusionStatsInput<B>,
}

/// Head-specific confusion input for the class labels.
#[derive(Debug, Clone)]
pub struct ClassConfusionInput<B: Backend> {
    inner: ConfusionStatsInput<B>,
}

impl<B: Backend> PathwayConfusionInput<B> {
    fn new(inner: ConfusionStatsInput<B>) -> Self {
        Self { inner }
    }
}

impl<B: Backend> SuperclassConfusionInput<B> {
    fn new(inner: ConfusionStatsInput<B>) -> Self {
        Self { inner }
    }
}

impl<B: Backend> ClassConfusionInput<B> {
    fn new(inner: ConfusionStatsInput<B>) -> Self {
        Self { inner }
    }
}

/// Shared Matthews correlation coefficient metric implementation.
#[derive(Clone)]
pub struct MatthewsCorrelationMetric<B: Backend, I> {
    name: MetricName,
    threshold: f32,
    state: MatthewsCorrelationState,
    _backend: PhantomData<B>,
    _input: PhantomData<I>,
}

impl<B: Backend, I> MatthewsCorrelationMetric<B, I> {
    /// Creates a named MCC metric for a multilabel head.
    #[must_use]
    pub fn new(name: impl Into<String>, threshold: f32) -> Self {
        Self {
            name: Arc::new(name.into()),
            threshold,
            state: MatthewsCorrelationState::default(),
            _backend: PhantomData,
            _input: PhantomData,
        }
    }
}

impl<B: Backend> Metric for MatthewsCorrelationMetric<B, PathwayConfusionInput<B>> {
    type Input = PathwayConfusionInput<B>;

    fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> SerializedEntry {
        self.update_inner(&item.inner, metadata)
    }

    fn clear(&mut self) {
        self.state.reset();
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: true,
        }
        .into()
    }
}

impl<B: Backend> Metric for MatthewsCorrelationMetric<B, SuperclassConfusionInput<B>> {
    type Input = SuperclassConfusionInput<B>;

    fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> SerializedEntry {
        self.update_inner(&item.inner, metadata)
    }

    fn clear(&mut self) {
        self.state.reset();
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: true,
        }
        .into()
    }
}

impl<B: Backend> Metric for MatthewsCorrelationMetric<B, ClassConfusionInput<B>> {
    type Input = ClassConfusionInput<B>;

    fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> SerializedEntry {
        self.update_inner(&item.inner, metadata)
    }

    fn clear(&mut self) {
        self.state.reset();
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: true,
        }
        .into()
    }
}

impl<B: Backend, I> MatthewsCorrelationMetric<B, I> {
    fn update_inner(
        &mut self,
        input: &ConfusionStatsInput<B>,
        metadata: &MetricMetadata,
    ) -> SerializedEntry {
        let [predictions, targets] = Transaction::default()
            .register(input.predictions.clone())
            .register(input.targets.clone())
            .execute()
            .try_into()
            .expect("correct number of synchronized tensors");
        let predictions = predictions
            .to_vec::<f32>()
            .expect("predictions should be accessible as f32");
        let targets = targets
            .to_vec::<bool>()
            .expect("targets should be accessible as bool");
        let counts = accumulate_counts(&predictions, &targets, self.threshold);

        self.state.update(
            counts,
            metadata,
            &FormatOptions::new(self.name.clone()).precision(4),
        )
    }
}

impl<B: Backend, I> Numeric for MatthewsCorrelationMetric<B, I> {
    fn value(&self) -> NumericEntry {
        self.state.value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.value()
    }
}

#[derive(Clone, Default)]
struct MatthewsCorrelationState {
    counts: ConfusionCounts,
    current: f64,
}

impl MatthewsCorrelationState {
    fn reset(&mut self) {
        self.counts = ConfusionCounts::default();
        self.current = f64::NAN;
    }

    fn update(
        &mut self,
        batch_counts: ConfusionCounts,
        metadata: &MetricMetadata,
        format: &FormatOptions,
    ) -> SerializedEntry {
        self.counts = merge_counts(self.counts, batch_counts);
        let epoch_value = matthews_correlation(self.counts);
        let batch_value = matthews_correlation(batch_counts);
        self.current = epoch_value;
        let is_last_batch = metadata.progress.items_processed >= metadata.progress.items_total;
        let serialized = NumericEntry::Aggregated {
            aggregated_value: epoch_value,
            count: usize::from(is_last_batch),
        }
        .serialize();
        let precision = format.precision_value().unwrap_or(4);
        let formatted = format!(
            "epoch {epoch} - batch {batch}",
            epoch = format_float(epoch_value, precision),
            batch = format_float(batch_value, precision),
        );

        SerializedEntry::new(formatted, serialized)
    }

    fn value(&self) -> NumericEntry {
        NumericEntry::Value(self.current)
    }
}

impl<B: Backend> Adaptor<PathwayConfusionInput<B>> for NpClassifierOutput<B> {
    fn adapt(&self) -> PathwayConfusionInput<B> {
        PathwayConfusionInput::new(ConfusionStatsInput::new(
            self.pathway_probabilities.clone(),
            self.pathway_targets.clone().bool(),
        ))
    }
}

impl<B: Backend> Adaptor<SuperclassConfusionInput<B>> for NpClassifierOutput<B> {
    fn adapt(&self) -> SuperclassConfusionInput<B> {
        SuperclassConfusionInput::new(ConfusionStatsInput::new(
            self.superclass_probabilities.clone(),
            self.superclass_targets.clone().bool(),
        ))
    }
}

impl<B: Backend> Adaptor<ClassConfusionInput<B>> for NpClassifierOutput<B> {
    fn adapt(&self) -> ClassConfusionInput<B> {
        ClassConfusionInput::new(ConfusionStatsInput::new(
            self.class_probabilities.clone(),
            self.class_targets.clone().bool(),
        ))
    }
}

/// Creates the pathway MCC metric.
#[must_use]
pub fn pathway_mcc_metric<B: Backend>() -> MatthewsCorrelationMetric<B, PathwayConfusionInput<B>> {
    MatthewsCorrelationMetric::new("Pathway MCC", ModelHead::Pathway.threshold())
}

/// Creates the superclass MCC metric.
#[must_use]
pub fn superclass_mcc_metric<B: Backend>()
-> MatthewsCorrelationMetric<B, SuperclassConfusionInput<B>> {
    MatthewsCorrelationMetric::new("Superclass MCC", ModelHead::Superclass.threshold())
}

/// Creates the class MCC metric.
#[must_use]
pub fn class_mcc_metric<B: Backend>() -> MatthewsCorrelationMetric<B, ClassConfusionInput<B>> {
    MatthewsCorrelationMetric::new("Class MCC", ModelHead::Class.threshold())
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConfusionCounts {
    /// True positives.
    pub tp: u64,
    /// True negatives.
    pub tn: u64,
    /// False positives.
    pub fp: u64,
    /// False negatives.
    pub fn_: u64,
}

fn merge_counts(left: ConfusionCounts, right: ConfusionCounts) -> ConfusionCounts {
    ConfusionCounts {
        tp: left.tp + right.tp,
        tn: left.tn + right.tn,
        fp: left.fp + right.fp,
        fn_: left.fn_ + right.fn_,
    }
}

/// Accumulates multilabel confusion counts using a fixed probability threshold.
#[must_use]
pub fn accumulate_counts(predictions: &[f32], targets: &[bool], threshold: f32) -> ConfusionCounts {
    let mut counts = ConfusionCounts::default();
    for (prediction, target) in predictions.iter().zip(targets.iter()) {
        let predicted = *prediction >= threshold;
        match (predicted, *target) {
            (true, true) => counts.tp += 1,
            (true, false) => counts.fp += 1,
            (false, true) => counts.fn_ += 1,
            (false, false) => counts.tn += 1,
        }
    }
    counts
}

/// Computes Matthews correlation coefficient from already aggregated confusion counts.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn matthews_correlation(counts: ConfusionCounts) -> f64 {
    let tp = counts.tp as f64;
    let tn = counts.tn as f64;
    let fp = counts.fp as f64;
    let fn_ = counts.fn_ as f64;
    let numerator = tp * tn - fp * fn_;
    let denominator = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
    if denominator <= f64::EPSILON {
        0.0
    } else {
        numerator / denominator
    }
}

/// Computes confusion counts from Burn tensors after synchronizing them.
///
/// # Errors
///
/// Returns an error if tensor synchronization or host-side tensor decoding
/// fails.
pub fn counts_from_tensors<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2, Int>,
    threshold: f32,
) -> Result<ConfusionCounts, TrainingError> {
    let mut tensors = Transaction::default()
        .register(predictions)
        .register(targets)
        .execute();
    let targets = tensors
        .pop()
        .ok_or_else(|| TrainingError::Burn("missing synchronized target tensor".to_owned()))?;
    let predictions = tensors
        .pop()
        .ok_or_else(|| TrainingError::Burn("missing synchronized prediction tensor".to_owned()))?;
    let predictions = predictions
        .to_vec::<f32>()
        .map_err(|error| TrainingError::Burn(error.to_string()))?;
    let targets = targets
        .to_vec::<i32>()
        .map_err(|error| TrainingError::Burn(error.to_string()))?;
    let targets = targets
        .into_iter()
        .map(|value| value != 0)
        .collect::<Vec<_>>();

    Ok(accumulate_counts(&predictions, &targets, threshold))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use burn::data::dataloader::Progress;
    use burn::train::metric::MetricMetadata;

    use super::{ConfusionCounts, FormatOptions, MatthewsCorrelationState, matthews_correlation};

    fn metadata(items_processed: usize, items_total: usize) -> MetricMetadata {
        MetricMetadata {
            progress: Progress {
                items_processed,
                items_total,
            },
            epoch: 1,
            epoch_total: 1,
            iteration: items_processed,
            lr: None,
        }
    }

    #[test]
    fn matthews_correlation_is_one_for_perfect_predictions() {
        let counts = ConfusionCounts {
            tp: 4,
            tn: 5,
            fp: 0,
            fn_: 0,
        };
        assert!((matthews_correlation(counts) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn matthews_correlation_is_zero_when_denominator_degenerates() {
        let counts = ConfusionCounts {
            tp: 0,
            tn: 8,
            fp: 0,
            fn_: 0,
        };
        assert!(matthews_correlation(counts).abs() < 1e-12);
    }

    #[test]
    fn matthews_correlation_handles_counts_above_u32() {
        let counts = ConfusionCounts {
            tp: 4_500_000_000,
            tn: 4_500_000_000,
            fp: 0,
            fn_: 0,
        };

        assert!((matthews_correlation(counts) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn state_logs_only_the_final_epoch_global_value() {
        let mut state = MatthewsCorrelationState::default();
        let name = Arc::new("Class MCC".to_owned());
        let first = state.update(
            ConfusionCounts {
                tp: 2,
                tn: 0,
                fp: 0,
                fn_: 0,
            },
            &metadata(32, 64),
            &FormatOptions::new(name.clone()).precision(4),
        );
        assert_eq!(first.serialized, "0,0");

        let second = state.update(
            ConfusionCounts {
                tp: 0,
                tn: 1,
                fp: 0,
                fn_: 1,
            },
            &metadata(64, 64),
            &FormatOptions::new(name).precision(4),
        );
        assert_eq!(second.serialized, "0.5773502691896258,1");
    }
}
