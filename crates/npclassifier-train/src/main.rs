//! Burn-based student training CLI for `NPClassifier`.

use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use burn::backend::{Autodiff, Cuda, NdArray};
use burn::config::Config;
use burn::module::AutodiffModule;
use burn::prelude::{Backend, Module};
use burn::record::CompactRecorder;
use burn::tensor::Transaction;
use burn::tensor::backend::AutodiffBackend;
use burn::train::Interrupter;
use burn::train::checkpoint::{
    ComposedCheckpointingStrategy, KeepLastNCheckpoints, MetricCheckpointingStrategy,
};
use burn::train::metric::store::{Aggregate, Direction, Split};
use burn::train::metric::{CudaMetric, LearningRateMetric, LossMetric, NumericEntry};
use burn::train::renderer::tui::TuiMetricsRenderer;
use burn::train::{InferenceStep, Learner, SupervisedTraining, TrainingStrategy};
use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;

use npclassifier_core::{DEFAULT_DISTILLATION_DATA_DIR, ModelHead};
use npclassifier_train::calibration::{
    DEFAULT_THRESHOLD_BINS, calibrate_validation_thresholds, write_threshold_report,
};
use npclassifier_train::data::{
    NpClassifierBatch, TeacherSplitStorage, TrainingManifest, build_dataloader, load_manifest,
    load_split_storage,
};
use npclassifier_train::error::TrainingError;
use npclassifier_train::evaluation::{SplitEvaluationReport, evaluate_split};
use npclassifier_train::metric::{class_mcc_metric, pathway_mcc_metric, superclass_mcc_metric};
use npclassifier_train::model::{StudentModel, StudentModelConfig};
use npclassifier_train::quantization::{
    print_quantization_table, quantize_and_evaluate, write_quantization_report,
};
use npclassifier_train::sync_best::{SyncBestModelMetadata, SyncBestModelStrategy};
use npclassifier_train::{
    CLASS_MCC_LOG_NAME, SYNC_BEST_METADATA_FILE_NAME, SYNC_BEST_MODEL_FILE_STEM,
};

/// Default artifact directory for student training runs.
const DEFAULT_ARTIFACT_DIR: &str = "artifacts/baseline";
const CHECKPOINT_DIR_NAME: &str = "checkpoint";

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BackendKind {
    Cuda,
    Ndarray,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CheckpointingMode {
    /// Burn's stock async file checkpointing. This saves model, optimizer, and
    /// scheduler state, but it currently triggers upstream Burn handle panics on
    /// some fusion backends.
    Async,
    /// Save only the best validation model synchronously on the main training
    /// thread after each validation epoch.
    ///
    /// This mode exists specifically to avoid Burn's async checkpoint thread. It
    /// preserves best-epoch model selection, but it does not save optimizer or
    /// scheduler state and therefore does not support `--checkpoint` resume.
    SyncBest,
    /// Disable epoch-boundary checkpointing entirely and export the final epoch
    /// model.
    Off,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ArchitecturePreset {
    Baseline,
    MiniShared,
}

#[derive(Debug, Parser)]
#[command(name = "npclassifier-train")]
#[command(about = "Train an NPClassifier-compatible baseline with Burn")]
struct Cli {
    #[arg(long, default_value = DEFAULT_DISTILLATION_DATA_DIR)]
    data_dir: PathBuf,
    #[arg(long, default_value = DEFAULT_ARTIFACT_DIR)]
    artifact_dir: PathBuf,
    #[arg(long, value_enum, default_value_t = BackendKind::Cuda)]
    backend: BackendKind,
    #[arg(long, default_value_t = 0)]
    cuda_device: usize,
    #[arg(
        long,
        long_help = "Resume from an async Burn checkpoint epoch.\n\
\n\
This is only valid together with --checkpointing async because sync-best stores\n\
only the best inference model, not optimizer or scheduler state."
    )]
    checkpoint: Option<usize>,
    #[arg(
        long,
        value_enum,
        default_value_t = CheckpointingMode::SyncBest,
        long_help = "Checkpointing mode.\n\
\n\
async: Burn's stock file checkpointing. Saves model, optimizer, and scheduler\n\
state and supports --checkpoint resume, but currently hits upstream Burn\n\
handle panics on the CUDA+fusion path used by this trainer.\n\
\n\
sync-best: custom main-thread saver. After each validation epoch, recomputes\n\
validation Class MCC directly, and if it improved, writes best-model.mpk plus\n\
best-model.json synchronously. This avoids Burn's async checkpoint thread, but\n\
it does not save optimizer or scheduler state and therefore cannot resume.\n\
\n\
off: disable epoch-boundary checkpointing and export the final in-memory model."
    )]
    checkpointing: CheckpointingMode,
    #[arg(long, default_value_t = 20)]
    num_epochs: usize,
    #[arg(long, default_value_t = 512)]
    batch_size: usize,
    #[arg(long, default_value_t = 8)]
    num_workers: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 3e-4)]
    learning_rate: f64,
    #[arg(long, default_value_t = 1.0)]
    hard_label_weight: f32,
    #[arg(long, default_value_t = 0.0)]
    teacher_weight: f32,
    #[arg(
        long,
        value_enum,
        default_value_t = ArchitecturePreset::Baseline,
        long_help = "Model-width preset.\n\
\n\
baseline: faithful three-tower NPClassifier architecture with hidden widths\n\
6144 -> 3072 -> 2304 -> 1152 in each head tower.\n\
\n\
mini-shared: reduced browser-oriented variant with hidden widths\n\
1536 -> 768 -> 576 -> 288 and a shared first dense +\n\
batch-norm block across pathway, superclass, and class heads.\n\
\n\
The optional --hidden-1/2/3/4 flags override the selected preset."
    )]
    architecture: ArchitecturePreset,
    #[arg(long)]
    hidden_1: Option<usize>,
    #[arg(long)]
    hidden_2: Option<usize>,
    #[arg(long)]
    hidden_3: Option<usize>,
    #[arg(long)]
    hidden_4: Option<usize>,
    #[arg(long, default_value_t = 0.1)]
    dropout: f64,
    #[arg(long)]
    train_rows: Option<usize>,
    #[arg(long)]
    valid_rows: Option<usize>,
    #[arg(long)]
    test_rows: Option<usize>,
    #[arg(
        long,
        default_value_t = DEFAULT_THRESHOLD_BINS,
        long_help = "Number of validation-score histogram buckets used by automatic post-training threshold calibration."
    )]
    threshold_bins: u32,
    #[arg(
        long,
        long_help = "Directory for automatic packed q4 browser artifacts. Defaults to <artifact-dir>/web."
    )]
    web_output_dir: Option<PathBuf>,
}

#[derive(Config, Debug)]
struct TrainingConfig {
    model: StudentModelConfig,
    optimizer: burn::optim::AdamWConfig,
    #[config(default = 20)]
    num_epochs: usize,
    #[config(default = 512)]
    batch_size: usize,
    #[config(default = 8)]
    num_workers: usize,
    #[config(default = 42)]
    seed: u64,
    #[config(default = 3e-4)]
    learning_rate: f64,
    #[config(default = 1.0)]
    hard_label_weight: f32,
    #[config(default = 0.0)]
    teacher_weight: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct RunEvaluationReport {
    best_epoch: Option<usize>,
    best_validation_class_mcc: Option<f64>,
    train: SplitEvaluationReport,
    validation: SplitEvaluationReport,
    test: SplitEvaluationReport,
}

type BestCheckpointSelection<B> = (StudentModel<B>, Option<usize>, Option<f64>);
type SplitStorages = (
    Arc<TeacherSplitStorage>,
    Arc<TeacherSplitStorage>,
    Arc<TeacherSplitStorage>,
);
type SplitLoaders<B> = (
    Arc<dyn burn::data::dataloader::DataLoader<B, NpClassifierBatch>>,
    Arc<
        dyn burn::data::dataloader::DataLoader<
                <B as AutodiffBackend>::InnerBackend,
                NpClassifierBatch,
            >,
    >,
    Arc<
        dyn burn::data::dataloader::DataLoader<
                <B as AutodiffBackend>::InnerBackend,
                NpClassifierBatch,
            >,
    >,
    Arc<
        dyn burn::data::dataloader::DataLoader<
                <B as AutodiffBackend>::InnerBackend,
                NpClassifierBatch,
            >,
    >,
);

fn main() -> Result<(), TrainingError> {
    let cli = Cli::parse();
    match cli.backend {
        BackendKind::Cuda => run_cuda(&cli),
        BackendKind::Ndarray => run_ndarray(&cli),
    }
}

fn run_cuda(cli: &Cli) -> Result<(), TrainingError> {
    type BackendImpl = Autodiff<Cuda<f32, i32>>;
    let device = burn::backend::cuda::CudaDevice::new(cli.cuda_device);
    run_training::<BackendImpl>(cli, device, true)
}

fn run_ndarray(cli: &Cli) -> Result<(), TrainingError> {
    type BackendImpl = Autodiff<NdArray<f32, i32>>;
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    run_training::<BackendImpl>(cli, device, false)
}

fn run_training<B>(
    cli: &Cli,
    device: B::Device,
    include_cuda_metric: bool,
) -> Result<(), TrainingError>
where
    B: AutodiffBackend,
    B::Device: Clone,
{
    validate_training_invocation(cli)?;
    prepare_training_artifacts(cli)?;
    let config = build_training_config(cli);
    config.save(cli.artifact_dir.join("training-config.json"))?;
    B::seed(&device, config.seed);
    let include_teacher = cli.teacher_weight > 0.0;
    let (train_storage, valid_storage, test_storage) =
        load_training_storages(cli, include_teacher)?;
    println!(
        "loaded dataset rows: train={} validation={} test={}",
        train_storage.len(),
        valid_storage.len(),
        test_storage.len()
    );
    let (train_loader, valid_loader, train_eval_loader, test_loader) = build_split_loaders::<B>(
        &train_storage,
        &valid_storage,
        &test_storage,
        &config,
        include_teacher,
    );
    let mut training =
        SupervisedTraining::new(&cli.artifact_dir, train_loader, Arc::clone(&valid_loader))
            .metric_train_numeric(LossMetric::new())
            .metric_train_numeric(pathway_mcc_metric())
            .metric_train_numeric(superclass_mcc_metric())
            .metric_train_numeric(class_mcc_metric())
            .metric_valid_numeric(LossMetric::new())
            .metric_valid_numeric(pathway_mcc_metric())
            .metric_valid_numeric(superclass_mcc_metric())
            .metric_valid_numeric(class_mcc_metric())
            .metric_train_numeric(LearningRateMetric::new())
            .num_epochs(config.num_epochs)
            .summary();
    if std::io::stdout().is_terminal() {
        training = training.renderer(TuiMetricsRenderer::new(
            Interrupter::default(),
            cli.checkpoint,
        ));
    }
    if cli.checkpointing == CheckpointingMode::Async {
        training = training
            .with_file_checkpointer(CompactRecorder::new())
            .with_checkpointing_strategy(
                ComposedCheckpointingStrategy::builder()
                    .add(KeepLastNCheckpoints::new(3))
                    .add(MetricCheckpointingStrategy::new(
                        &class_mcc_metric::<NdArray>(),
                        Aggregate::Mean,
                        Direction::Highest,
                        Split::Valid,
                    ))
                    .build(),
            );
    } else if cli.checkpointing == CheckpointingMode::SyncBest {
        training = training.with_training_strategy(TrainingStrategy::Custom(Arc::new(
            SyncBestModelStrategy::new(device.clone(), cli.artifact_dir.clone()),
        )));
    }
    if include_cuda_metric {
        training = training.metric_train(CudaMetric::new());
    }
    if let Some(checkpoint) = cli.checkpoint {
        training = training.checkpoint(checkpoint);
    }

    let model = config
        .model
        .init::<B>(&device, config.hard_label_weight, config.teacher_weight);
    warm_up_tui_launch(&model, &test_loader)?;
    let learner = Learner::new(model, config.optimizer.init(), config.learning_rate);
    let result = training.launch(learner);
    let trained_model = result.model;
    let (model_to_export, best_epoch, best_validation_class_mcc) =
        load_selected_export_model(&trained_model, &cli.artifact_dir, cli.checkpointing)?;
    let report = RunEvaluationReport {
        best_epoch,
        best_validation_class_mcc,
        train: evaluate_split(&model_to_export, &train_eval_loader)?,
        validation: evaluate_split(&model_to_export, &valid_loader)?,
        test: evaluate_split(&model_to_export, &test_loader)?,
    };

    save_and_finalize_model(
        cli,
        &model_to_export,
        report,
        &valid_storage,
        &valid_loader,
        &test_loader,
    )
}

fn save_and_finalize_model<B: Backend>(
    cli: &Cli,
    model: &StudentModel<B>,
    report: RunEvaluationReport,
    valid_storage: &TeacherSplitStorage,
    valid_loader: &Arc<dyn burn::data::dataloader::DataLoader<B, NpClassifierBatch>>,
    test_loader: &Arc<dyn burn::data::dataloader::DataLoader<B, NpClassifierBatch>>,
) -> Result<(), TrainingError> {
    model
        .clone()
        .save_file(cli.artifact_dir.join("model"), &CompactRecorder::new())
        .map_err(|error| TrainingError::Burn(error.to_string()))?;
    write_run_reports(&cli.artifact_dir, &report)?;
    print_split_metrics_table(&report);
    println!("{}", serde_json::to_string_pretty(&report)?);

    let threshold_report =
        calibrate_validation_thresholds(model, valid_storage, valid_loader, cli.threshold_bins)?;
    write_threshold_report(&cli.artifact_dir, &threshold_report)?;
    println!("{}", serde_json::to_string_pretty(&threshold_report)?);

    let web_output_dir = cli
        .web_output_dir
        .clone()
        .unwrap_or_else(|| cli.artifact_dir.join("web"));
    let quantization_report = quantize_and_evaluate(
        model,
        &cli.artifact_dir,
        test_loader,
        report.test,
        &web_output_dir,
        threshold_report.thresholds,
    )?;
    write_quantization_report(&cli.artifact_dir, &quantization_report)?;
    print_quantization_table(&quantization_report);
    println!("{}", serde_json::to_string_pretty(&quantization_report)?);

    println!("exported packed web model to {}", web_output_dir.display());

    Ok(())
}

fn validate_training_invocation(cli: &Cli) -> Result<(), TrainingError> {
    if cli.checkpoint.is_some() && cli.checkpointing != CheckpointingMode::Async {
        return Err(TrainingError::Dataset(
            "--checkpoint requires --checkpointing async because sync-best only saves the best inference model".to_owned(),
        ));
    }

    Ok(())
}

fn prepare_training_artifacts(cli: &Cli) -> Result<(), TrainingError> {
    let manifest = load_manifest(&cli.data_dir)?;
    validate_manifest(&manifest)?;
    fs::create_dir_all(&cli.artifact_dir)?;
    fs::copy(
        cli.data_dir.join("manifest.json"),
        cli.artifact_dir.join("dataset-manifest.json"),
    )?;
    if cli.checkpointing == CheckpointingMode::SyncBest {
        clear_sync_best_artifacts(&cli.artifact_dir)?;
    }

    Ok(())
}

fn build_training_config(cli: &Cli) -> TrainingConfig {
    let model = base_model_config(cli.architecture);
    let model = if let Some(width) = cli.hidden_1 {
        model.with_hidden_1(width)
    } else {
        model
    };
    let model = if let Some(width) = cli.hidden_2 {
        model.with_hidden_2(width)
    } else {
        model
    };
    let model = if let Some(width) = cli.hidden_3 {
        model.with_hidden_3(width)
    } else {
        model
    };
    let model = if let Some(width) = cli.hidden_4 {
        model.with_hidden_4(width)
    } else {
        model
    };
    let model = model.with_dropout(cli.dropout);
    let optimizer = burn::optim::AdamWConfig::new();

    TrainingConfig::new(model, optimizer)
        .with_num_epochs(cli.num_epochs)
        .with_batch_size(cli.batch_size)
        .with_num_workers(cli.num_workers)
        .with_seed(cli.seed)
        .with_learning_rate(cli.learning_rate)
        .with_hard_label_weight(cli.hard_label_weight)
        .with_teacher_weight(cli.teacher_weight)
}

fn base_model_config(preset: ArchitecturePreset) -> StudentModelConfig {
    match preset {
        ArchitecturePreset::Baseline => StudentModelConfig::baseline(),
        ArchitecturePreset::MiniShared => StudentModelConfig::mini_shared(),
    }
}

fn build_split_loaders<B>(
    train_storage: &Arc<TeacherSplitStorage>,
    valid_storage: &Arc<TeacherSplitStorage>,
    test_storage: &Arc<TeacherSplitStorage>,
    config: &TrainingConfig,
    include_teacher: bool,
) -> SplitLoaders<B>
where
    B: AutodiffBackend,
{
    let train_loader = build_dataloader::<B>(
        train_storage,
        config.batch_size,
        config.num_workers,
        Some(config.seed),
        include_teacher,
    );
    let valid_loader = build_dataloader::<B::InnerBackend>(
        valid_storage,
        config.batch_size,
        config.num_workers,
        None,
        include_teacher,
    );
    let train_eval_loader = build_dataloader::<B::InnerBackend>(
        train_storage,
        config.batch_size,
        config.num_workers,
        None,
        false,
    );
    let test_loader = build_dataloader::<B::InnerBackend>(
        test_storage,
        config.batch_size,
        config.num_workers,
        None,
        false,
    );

    (train_loader, valid_loader, train_eval_loader, test_loader)
}

fn write_run_reports(
    artifact_dir: &Path,
    report: &RunEvaluationReport,
) -> Result<(), TrainingError> {
    fs::write(
        artifact_dir.join("metrics.json"),
        serde_json::to_string_pretty(report)?,
    )?;
    fs::write(
        artifact_dir.join("test-metrics.json"),
        serde_json::to_string_pretty(&report.test)?,
    )?;

    Ok(())
}

fn validate_manifest(manifest: &TrainingManifest) -> Result<(), TrainingError> {
    if manifest.vector_widths.pathway != ModelHead::Pathway.output_width()
        || manifest.vector_widths.superclass != ModelHead::Superclass.output_width()
        || manifest.vector_widths.class_ != ModelHead::Class.output_width()
    {
        return Err(TrainingError::Dataset(
            "dataset manifest vector widths do not match the expected NPClassifier heads"
                .to_owned(),
        ));
    }
    Ok(())
}

fn load_best_validation_model<B: Backend>(
    trained_model: &StudentModel<B>,
    artifact_dir: &Path,
) -> Result<BestCheckpointSelection<B>, TrainingError> {
    let Some((best_epoch, best_validation_class_mcc)) =
        find_best_metric_epoch(artifact_dir, CLASS_MCC_LOG_NAME)?
    else {
        return Ok((trained_model.clone(), None, None));
    };

    let Some(device) = trained_model.devices().into_iter().next() else {
        return Err(TrainingError::Burn(
            "trained model did not expose any device".to_owned(),
        ));
    };
    let best_model = trained_model
        .clone()
        .load_file(
            artifact_dir
                .join(CHECKPOINT_DIR_NAME)
                .join(format!("model-{best_epoch}")),
            &CompactRecorder::new(),
            &device,
        )
        .map_err(|error| {
            TrainingError::Burn(format!(
                "failed to restore best checkpoint for epoch {best_epoch}: {error}"
            ))
        })?;

    Ok((
        best_model,
        Some(best_epoch),
        Some(best_validation_class_mcc),
    ))
}

fn load_selected_export_model<B: Backend>(
    trained_model: &StudentModel<B>,
    artifact_dir: &Path,
    checkpointing: CheckpointingMode,
) -> Result<BestCheckpointSelection<B>, TrainingError> {
    match checkpointing {
        CheckpointingMode::Async => load_best_validation_model(trained_model, artifact_dir),
        CheckpointingMode::SyncBest => load_sync_best_validation_model(trained_model, artifact_dir),
        CheckpointingMode::Off => Ok((trained_model.clone(), None, None)),
    }
}

/// Remove the sync-best metadata from any previous run in the same artifact
/// directory so best-model selection cannot accidentally pick up stale data.
///
/// The saved model file itself is overwritten on every improvement, so the
/// metadata file is the authoritative marker for whether a valid sync-best
/// export exists for the current run.
fn clear_sync_best_artifacts(artifact_dir: &Path) -> Result<(), TrainingError> {
    let metadata_path = artifact_dir.join(SYNC_BEST_METADATA_FILE_NAME);
    if metadata_path.exists() {
        fs::remove_file(metadata_path)?;
    }

    // `CompactRecorder` writes MessagePack payloads with the `.mpk` suffix.
    let model_file = format!(
        "{}.mpk",
        artifact_dir.join(SYNC_BEST_MODEL_FILE_STEM).display()
    );
    let model_file = PathBuf::from(model_file);
    if model_file.exists() {
        fs::remove_file(model_file)?;
    }

    Ok(())
}

/// Restore the best model selected by the synchronous main-thread saver.
///
/// Unlike Burn's built-in file checkpointing, this path only stores the best
/// inference model and a tiny metadata sidecar. Optimizer and scheduler state
/// are intentionally omitted, because the goal is stable best-epoch export, not
/// resumable training.
fn load_sync_best_validation_model<B: Backend>(
    trained_model: &StudentModel<B>,
    artifact_dir: &Path,
) -> Result<BestCheckpointSelection<B>, TrainingError> {
    let metadata_path = artifact_dir.join(SYNC_BEST_METADATA_FILE_NAME);
    if !metadata_path.exists() {
        return Ok((trained_model.clone(), None, None));
    }

    let metadata =
        serde_json::from_str::<SyncBestModelMetadata>(&fs::read_to_string(metadata_path)?)?;
    let Some(device) = trained_model.devices().into_iter().next() else {
        return Err(TrainingError::Burn(
            "trained model did not expose any device".to_owned(),
        ));
    };
    let best_model = trained_model
        .clone()
        .load_file(
            artifact_dir.join(SYNC_BEST_MODEL_FILE_STEM),
            &CompactRecorder::new(),
            &device,
        )
        .map_err(|error| {
            TrainingError::Burn(format!(
                "failed to restore sync-best model for epoch {}: {error}",
                metadata.epoch
            ))
        })?;

    Ok((
        best_model,
        Some(metadata.epoch),
        Some(metadata.validation_class_mcc),
    ))
}

fn find_best_metric_epoch(
    artifact_dir: &Path,
    metric_name: &str,
) -> Result<Option<(usize, f64)>, TrainingError> {
    let valid_dir = artifact_dir.join(Split::Valid.to_string());
    if !valid_dir.exists() {
        return Ok(None);
    }

    let metric_file_name = format!("{}.log", metric_name.replace(' ', "_"));
    let mut best: Option<(usize, f64)> = None;

    for entry in fs::read_dir(&valid_dir)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if !file_type.is_dir() {
            continue;
        }

        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some(epoch_text) = name.strip_prefix("epoch-") else {
            continue;
        };
        let Ok(epoch) = epoch_text.parse::<usize>() else {
            continue;
        };

        let metric_path = entry.path().join(&metric_file_name);
        let Some(metric_value) = aggregate_logged_metric(&metric_path)? else {
            continue;
        };

        match best {
            Some((_, current_best)) if metric_value <= current_best => {}
            _ => best = Some((epoch, metric_value)),
        }
    }

    Ok(best)
}

fn aggregate_logged_metric(metric_path: &Path) -> Result<Option<f64>, TrainingError> {
    if !metric_path.exists() {
        return Ok(None);
    }

    let mut sum = 0.0;
    let mut count = 0_usize;
    let contents = fs::read_to_string(metric_path)?;
    for line in contents.lines().filter(|line| !line.is_empty()) {
        let entry = NumericEntry::deserialize(line).map_err(TrainingError::Dataset)?;
        match entry {
            NumericEntry::Value(value) => {
                sum += value;
                count += 1;
            }
            NumericEntry::Aggregated {
                aggregated_value,
                count: entry_count,
            } => {
                sum += aggregated_value * usize_to_f64(entry_count)?;
                count += entry_count;
            }
        }
    }

    if count == 0 {
        Ok(None)
    } else {
        Ok(Some(sum / usize_to_f64(count)?))
    }
}

fn load_training_storages(
    cli: &Cli,
    include_teacher: bool,
) -> Result<SplitStorages, TrainingError> {
    let train_storage =
        load_split_storage(&cli.data_dir, "train", cli.train_rows, include_teacher)?;
    let valid_storage =
        load_split_storage(&cli.data_dir, "validation", cli.valid_rows, include_teacher)?;
    let test_storage = load_split_storage(&cli.data_dir, "test", cli.test_rows, false)?;

    Ok((train_storage, valid_storage, test_storage))
}

fn warm_up_tui_launch<B: AutodiffBackend>(
    model: &StudentModel<B>,
    loader: &Arc<dyn burn::data::dataloader::DataLoader<B::InnerBackend, NpClassifierBatch>>,
) -> Result<(), TrainingError> {
    if !std::io::stdout().is_terminal() {
        return Ok(());
    }

    let mut batches = loader.iter();
    let Some(batch) = batches.next() else {
        return Ok(());
    };

    let progress = ProgressBar::new_spinner();
    progress.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] {spinner:.cyan} {msg}").map_err(
            |error| TrainingError::Dataset(format!("invalid warmup progress style: {error}")),
        )?,
    );
    progress.set_message("warming up CUDA and preparing the first TUI frame");
    progress.enable_steady_tick(std::time::Duration::from_millis(100));

    let start = Instant::now();
    let warmup_model = model.valid();
    let output = warmup_model.step(batch);
    let [
        _loss,
        _pathway_probabilities,
        _superclass_probabilities,
        _class_probabilities,
        _pathway_targets,
        _superclass_targets,
        _class_targets,
    ] = Transaction::default()
        .register(output.loss)
        .register(output.pathway_probabilities)
        .register(output.superclass_probabilities)
        .register(output.class_probabilities)
        .register(output.pathway_targets)
        .register(output.superclass_targets)
        .register(output.class_targets)
        .execute()
        .try_into()
        .expect("correct number of synchronized warmup tensors");

    progress.finish_with_message(format!(
        "warmup complete in {:.1}s; starting TUI",
        start.elapsed().as_secs_f32()
    ));

    Ok(())
}

fn usize_to_f64(value: usize) -> Result<f64, TrainingError> {
    let value = u32::try_from(value).map_err(|_| {
        TrainingError::Dataset(format!("numeric value {value} exceeded supported range"))
    })?;

    Ok(f64::from(value))
}

fn print_split_metrics_table(report: &RunEvaluationReport) {
    println!();
    println!("Final Metrics");
    println!(
        "{:<12} {:>10} {:>14} {:>18} {:>12}",
        "Split", "Rows", "Pathway MCC", "Superclass MCC", "Class MCC"
    );
    println!(
        "{:-<12} {:-<10} {:-<14} {:-<18} {:-<12}",
        "", "", "", "", ""
    );
    print_split_metric_row("Train", &report.train);
    print_split_metric_row("Validation", &report.validation);
    print_split_metric_row("Test", &report.test);
    println!();
}

fn print_split_metric_row(label: &str, report: &SplitEvaluationReport) {
    println!(
        "{:<12} {:>10} {:>14.6} {:>18.6} {:>12.6}",
        label, report.rows, report.pathway_mcc, report.superclass_mcc, report.class_mcc
    );
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::fs::{self, File};

    use arrow_array::builder::{ListBuilder, UInt16Builder};
    use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{Field, Schema};
    use npclassifier_core::DISTILLATION_DATASET_FILES;
    use parquet::arrow::ArrowWriter;
    use serde_json::json;
    use tempfile::TempDir;

    use super::*;

    const SYNTHETIC_SMILES: [&str; 4] = ["C", "CC", "CO", "CN"];
    const SYNTHETIC_CIDS: [i64; 4] = [1, 2, 3, 4];
    const SYNTHETIC_LABELS: [u16; 4] = [0, 1, 0, 1];

    #[test]
    fn full_ndarray_train_run_calibrates_quantizes_and_exports_web_artifacts()
    -> Result<(), Box<dyn Error>> {
        let temp_dir = TempDir::new()?;
        let data_dir = temp_dir.path().join("data");
        let artifact_dir = temp_dir.path().join("artifacts");
        let web_output_dir = temp_dir.path().join("web");
        write_synthetic_distillation_dataset(&data_dir)?;

        run_ndarray(&tiny_training_cli(
            data_dir,
            artifact_dir.clone(),
            web_output_dir.clone(),
        ))?;

        assert_complete_training_artifacts(&artifact_dir, &web_output_dir)?;
        Ok(())
    }

    fn tiny_training_cli(data_dir: PathBuf, artifact_dir: PathBuf, web_output_dir: PathBuf) -> Cli {
        Cli {
            data_dir,
            artifact_dir,
            backend: BackendKind::Ndarray,
            cuda_device: 0,
            checkpoint: None,
            checkpointing: CheckpointingMode::Off,
            num_epochs: 1,
            batch_size: 2,
            num_workers: 1,
            seed: 7,
            learning_rate: 1e-3,
            hard_label_weight: 1.0,
            teacher_weight: 0.0,
            architecture: ArchitecturePreset::MiniShared,
            hidden_1: Some(32),
            hidden_2: Some(32),
            hidden_3: Some(32),
            hidden_4: Some(32),
            dropout: 0.0,
            train_rows: None,
            valid_rows: None,
            test_rows: None,
            threshold_bins: 16,
            web_output_dir: Some(web_output_dir),
        }
    }

    fn write_synthetic_distillation_dataset(data_dir: &Path) -> Result<(), Box<dyn Error>> {
        fs::create_dir_all(data_dir)?;
        for key in DISTILLATION_DATASET_FILES {
            match *key {
                "manifest.json" => write_synthetic_manifest(data_dir)?,
                "train.parquet" | "validation.parquet" | "test.parquet" => {
                    write_synthetic_split(&data_dir.join(key))?;
                }
                "vocabulary.json" => fs::write(data_dir.join(key), "{}\n")?,
                _ => fs::write(data_dir.join(key), "synthetic fixture\n")?,
            }
        }

        Ok(())
    }

    fn write_synthetic_manifest(data_dir: &Path) -> Result<(), Box<dyn Error>> {
        fs::write(
            data_dir.join("manifest.json"),
            serde_json::to_vec_pretty(&json!({
                "vector_widths": {
                    "pathway": ModelHead::Pathway.output_width(),
                    "superclass": ModelHead::Superclass.output_width(),
                    "class_": ModelHead::Class.output_width(),
                }
            }))?,
        )?;
        Ok(())
    }

    fn write_synthetic_split(path: &Path) -> Result<(), Box<dyn Error>> {
        let smiles: ArrayRef = Arc::new(StringArray::from(SYNTHETIC_SMILES.to_vec()));
        let cids: ArrayRef = Arc::new(Int64Array::from(SYNTHETIC_CIDS.to_vec()));
        let pathway_ids = label_array(&SYNTHETIC_LABELS);
        let superclass_ids = label_array(&SYNTHETIC_LABELS);
        let class_ids = label_array(&SYNTHETIC_LABELS);
        let fields = vec![
            field_for_array("smiles", &smiles),
            field_for_array("cid", &cids),
            field_for_array("pathway_ids", &pathway_ids),
            field_for_array("superclass_ids", &superclass_ids),
            field_for_array("class_ids", &class_ids),
        ];
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![smiles, cids, pathway_ids, superclass_ids, class_ids],
        )?;
        let mut writer = ArrowWriter::try_new(File::create(path)?, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }

    fn label_array(labels: &[u16]) -> ArrayRef {
        let mut builder = ListBuilder::new(UInt16Builder::new());
        for label in labels {
            builder.values().append_value(*label);
            builder.append(true);
        }
        Arc::new(builder.finish())
    }

    fn field_for_array(name: &str, array: &ArrayRef) -> Field {
        Field::new(name, array.data_type().clone(), false)
    }

    fn assert_complete_training_artifacts(
        artifact_dir: &Path,
        web_output_dir: &Path,
    ) -> Result<(), Box<dyn Error>> {
        for path in [
            artifact_dir.join("training-config.json"),
            artifact_dir.join("dataset-manifest.json"),
            artifact_dir.join("model.mpk"),
            artifact_dir.join("metrics.json"),
            artifact_dir.join("test-metrics.json"),
            artifact_dir.join("thresholds.json"),
            artifact_dir.join("threshold-calibration.json"),
            artifact_dir.join("quantization-report.json"),
            web_output_dir.join("thresholds.json"),
            web_output_dir.join("shared/shared.q4-kernel.npz"),
            web_output_dir.join("pathway/pathway.q4-kernel.npz"),
            web_output_dir.join("superclass/superclass.q4-kernel.npz"),
            web_output_dir.join("class/class.q4-kernel.npz"),
        ] {
            assert!(path.exists(), "expected artifact {}", path.display());
        }

        let quantization_report: serde_json::Value = serde_json::from_str(&fs::read_to_string(
            artifact_dir.join("quantization-report.json"),
        )?)?;
        assert_eq!(quantization_report["variants"][0]["variant"], "q4-block32");
        assert_eq!(quantization_report["rows"], SYNTHETIC_SMILES.len());

        let calibration_report: serde_json::Value = serde_json::from_str(&fs::read_to_string(
            artifact_dir.join("threshold-calibration.json"),
        )?)?;
        assert_eq!(calibration_report["rows"], SYNTHETIC_SMILES.len());
        assert_eq!(calibration_report["bins"], 16);
        Ok(())
    }
}
