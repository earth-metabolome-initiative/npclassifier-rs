//! Synchronous best-model saving for the Burn training CLI.
//!
//! # Why this exists
//!
//! Burn's stock `with_file_checkpointer(...)` path wraps file checkpoints in an
//! `AsyncCheckpointer`. On the CUDA + fusion stack we use for the faithful
//! `NPClassifier` trainer, that background checkpoint thread currently triggers
//! upstream Burn handle panics around epoch boundaries.
//!
//! We still need one important behavior from checkpointing: exporting the best
//! validation model instead of blindly keeping the last epoch. This module
//! provides that behavior without touching the model architecture or the loss
//! functions.
//!
//! # What this strategy does
//!
//! The custom strategy below mirrors Burn's single-device training loop:
//!
//! - run one training epoch
//! - run one validation epoch
//! - compute validation `Class MCC`
//! - if it improved, synchronously save the current *inference* model on the
//!   main training thread
//!
//! Saving the inference model rather than the full learner state is deliberate:
//!
//! - it is all we need for best-epoch export and later evaluation
//! - it avoids Burn's async checkpointer thread entirely
//! - it does **not** attempt to save optimizer or scheduler state, so this mode
//!   is not resumable
//!
//! # Why the validation MCC is recomputed here
//!
//! Burn's metric pipeline is itself asynchronous. Querying the event store
//! immediately after `epoch_valid.run(...)` would race the metric worker. To
//! keep best-epoch selection deterministic, this strategy computes `Class MCC`
//! directly from the validation outputs while still forwarding the same outputs
//! to Burn's metric processor so the TUI and log files remain intact.

use std::fs;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;

use burn::module::{AutodiffModule, Module};
use burn::optim::GradientsAccumulator;
use burn::prelude::Backend;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{
    EventProcessorTraining, InferenceStep, Interrupter, Learner, LearnerEvent, LearnerItem,
    LearningComponentsTypes, SupervisedLearningStrategy, SupervisedTrainingEventProcessor,
    TrainLoader, TrainingBackend, TrainingComponents, TrainingModel, ValidLoader,
};
use serde::{Deserialize, Serialize};

use npclassifier_core::ModelHead;

use crate::error::TrainingError;
use crate::evaluation::merge_counts;
use crate::metric::{ConfusionCounts, counts_from_tensors, matthews_correlation};
use crate::model::StudentModel;

/// Metadata written next to the synchronously saved best model.
///
/// The `model.mpk` payload already contains the weights. This sidecar records
/// why that payload was selected so the final export step can report the chosen
/// epoch and metric value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyncBestModelMetadata {
    /// Epoch whose validation result produced the saved `best-model.mpk`.
    pub epoch: usize,
    /// Validation `Class MCC` used for selection.
    pub validation_class_mcc: f64,
    /// Human-readable marker to make the sidecar self-describing.
    pub mode: String,
    /// The metric used for selection.
    pub selection_metric: String,
}

/// Custom single-device training strategy that keeps only the best validation
/// inference model and saves it synchronously on the main thread.
///
/// This strategy is intentionally specific to this trainer's `StudentModel`.
/// Burn's built-in strategy helpers for single-device training are private, so
/// we mirror the public behavior here rather than reaching into private Burn
/// modules.
pub struct SyncBestModelStrategy<LC>
where
    LC: LearningComponentsTypes<
            TrainingModel = StudentModel<TrainingBackend<LC>>,
            InferenceModel = StudentModel<<TrainingBackend<LC> as AutodiffBackend>::InnerBackend>,
        >,
{
    device: <TrainingBackend<LC> as Backend>::Device,
    artifact_dir: PathBuf,
    _marker: PhantomData<LC>,
}

impl<LC> SyncBestModelStrategy<LC>
where
    LC: LearningComponentsTypes<
            TrainingModel = StudentModel<TrainingBackend<LC>>,
            InferenceModel = StudentModel<<TrainingBackend<LC> as AutodiffBackend>::InnerBackend>,
        >,
{
    /// Construct the strategy for one training device and one artifact
    /// directory.
    pub fn new(device: <TrainingBackend<LC> as Backend>::Device, artifact_dir: PathBuf) -> Self {
        Self {
            device,
            artifact_dir,
            _marker: PhantomData,
        }
    }

    fn best_model_stem(&self) -> PathBuf {
        self.artifact_dir.join(crate::SYNC_BEST_MODEL_FILE_STEM)
    }

    fn metadata_path(&self) -> PathBuf {
        self.artifact_dir.join(crate::SYNC_BEST_METADATA_FILE_NAME)
    }

    fn save_best_model(
        &self,
        learner: &Learner<LC>,
        epoch: usize,
        class_mcc: f64,
    ) -> Result<(), TrainingError> {
        let inference_model = learner.model().valid();
        let save_result = catch_unwind(AssertUnwindSafe(|| {
            inference_model.save_file(self.best_model_stem(), &CompactRecorder::new())
        }));

        match save_result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                return Err(TrainingError::Burn(format!(
                    "failed to save sync-best model for epoch {epoch}: {error}"
                )));
            }
            Err(payload) => {
                return Err(TrainingError::Burn(format!(
                    "sync-best model save panicked for epoch {epoch}: {}",
                    panic_payload_to_string(&payload)
                )));
            }
        }

        let metadata = SyncBestModelMetadata {
            epoch,
            validation_class_mcc: class_mcc,
            mode: "sync-best".to_owned(),
            selection_metric: crate::CLASS_MCC_LOG_NAME.to_owned(),
        };
        fs::write(
            self.metadata_path(),
            serde_json::to_string_pretty(&metadata)?,
        )?;

        Ok(())
    }

    fn maybe_update_best_model(
        &self,
        learner: &Learner<LC>,
        epoch: usize,
        class_mcc: f64,
        best_so_far: &mut Option<f64>,
    ) -> Result<(), TrainingError> {
        if best_so_far.is_some_and(|current| class_mcc <= current) {
            return Ok(());
        }

        self.save_best_model(learner, epoch, class_mcc)?;
        *best_so_far = Some(class_mcc);

        Ok(())
    }

    fn run_training_epoch(
        learner: &mut Learner<LC>,
        dataloader: &TrainLoader<LC>,
        epoch_total: usize,
        grad_accumulation: Option<usize>,
        epoch: usize,
        processor: &mut SupervisedTrainingEventProcessor<LC>,
        interrupter: &Interrupter,
    ) {
        tracing::info!("Executing training step for epoch {}", epoch);

        let mut iterator = dataloader.iter();
        let mut iteration = 0;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0;

        while let Some(item) = iterator.next() {
            iteration += 1;
            learner.lr_step();
            tracing::info!("Iteration {}", iteration);

            let progress = iterator.progress();
            let item = learner.train_step(item);

            match grad_accumulation {
                Some(accumulation) => {
                    accumulator.accumulate(&learner.model(), item.grads);
                    accumulation_current += 1;

                    if accumulation <= accumulation_current {
                        learner.optimizer_step(accumulator.grads());
                        accumulation_current = 0;
                    }
                }
                None => learner.optimizer_step(item.grads),
            }

            let item = LearnerItem::new(
                item.item,
                progress,
                epoch,
                epoch_total,
                iteration,
                Some(learner.lr_current()),
            );
            processor.process_train(LearnerEvent::ProcessedItem(item));

            if interrupter.should_stop() {
                break;
            }
        }

        processor.process_train(LearnerEvent::EndEpoch(epoch));
    }

    fn run_validation_epoch(
        learner: &Learner<LC>,
        dataloader: &ValidLoader<LC>,
        epoch_total: usize,
        epoch: usize,
        processor: &mut SupervisedTrainingEventProcessor<LC>,
        interrupter: &Interrupter,
    ) -> Result<f64, TrainingError> {
        tracing::info!("Executing validation step for epoch {}", epoch);

        let model = learner.model().valid();
        let mut iterator = dataloader.iter();
        let mut iteration = 0;
        let mut class_counts = ConfusionCounts::default();

        while let Some(item) = iterator.next() {
            let progress = iterator.progress();
            iteration += 1;

            let output = model.step(item);
            class_counts = merge_counts(
                class_counts,
                counts_from_tensors(
                    output.class_probabilities.clone(),
                    output.class_targets.clone(),
                    ModelHead::Class.threshold(),
                )?,
            );

            let item = LearnerItem::new(output, progress, epoch, epoch_total, iteration, None);
            processor.process_valid(LearnerEvent::ProcessedItem(item));

            if interrupter.should_stop() {
                break;
            }
        }

        processor.process_valid(LearnerEvent::EndEpoch(epoch));

        Ok(matthews_correlation(class_counts))
    }
}

impl<LC> SupervisedLearningStrategy<LC> for SyncBestModelStrategy<LC>
where
    LC: LearningComponentsTypes<
            TrainingModel = StudentModel<TrainingBackend<LC>>,
            InferenceModel = StudentModel<<TrainingBackend<LC> as AutodiffBackend>::InnerBackend>,
        >,
{
    fn fit(
        &self,
        training_components: TrainingComponents<LC>,
        mut learner: Learner<LC>,
        dataloader_train: TrainLoader<LC>,
        dataloader_valid: ValidLoader<LC>,
        starting_epoch: usize,
    ) -> (TrainingModel<LC>, SupervisedTrainingEventProcessor<LC>) {
        let TrainingComponents {
            num_epochs,
            grad_accumulation,
            interrupter,
            early_stopping,
            event_processor,
            event_store,
            ..
        } = training_components;
        let dataloader_train = dataloader_train.to_device(&self.device);
        let dataloader_valid = dataloader_valid.to_device(&self.device);
        learner.fork(&self.device);

        let mut event_processor = event_processor;
        let mut early_stopping = early_stopping;
        let mut best_class_mcc = None;

        for epoch in starting_epoch..=num_epochs {
            Self::run_training_epoch(
                &mut learner,
                &dataloader_train,
                num_epochs,
                grad_accumulation,
                epoch,
                &mut event_processor,
                &interrupter,
            );

            if interrupter.should_stop() {
                let reason = interrupter
                    .get_message()
                    .unwrap_or(String::from("Reason unknown"));
                tracing::info!("Training interrupted: {}", reason);
                break;
            }

            let class_mcc = match Self::run_validation_epoch(
                &learner,
                &dataloader_valid,
                num_epochs,
                epoch,
                &mut event_processor,
                &interrupter,
            ) {
                Ok(class_mcc) => class_mcc,
                Err(error) => {
                    let message =
                        format!("validation bookkeeping failed at epoch {epoch}: {error}");
                    tracing::error!("{}", message);
                    interrupter.stop(Some(&message));
                    break;
                }
            };

            if let Err(error) =
                self.maybe_update_best_model(&learner, epoch, class_mcc, &mut best_class_mcc)
            {
                let message = format!("sync-best save failed at epoch {epoch}: {error}");
                tracing::error!("{}", message);
                interrupter.stop(Some(&message));
                break;
            }

            if interrupter.should_stop() {
                let reason = interrupter
                    .get_message()
                    .unwrap_or(String::from("Reason unknown"));
                tracing::info!("Training interrupted: {}", reason);
                break;
            }

            if let Some(early_stopping) = &mut early_stopping
                && early_stopping.should_stop(epoch, &event_store)
            {
                break;
            }
        }

        (learner.model(), event_processor)
    }
}

fn panic_payload_to_string(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_owned();
    }

    "unknown panic payload".to_owned()
}
