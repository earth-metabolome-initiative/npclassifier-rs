//! CUDA throughput benchmark for the Mini `NPClassifier.rs` Burn artifact.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{SyncSender, sync_channel};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use burn::backend::Cuda;
use burn::backend::cuda::CudaDevice;
use burn::module::Module;
use burn::record::CompactRecorder;
use clap::{Parser, ValueEnum};
use npclassifier_core::FINGERPRINT_INPUT_WIDTH;
use npclassifier_train::data::DenseFingerprintEncoder;
use npclassifier_train::model::{StudentModel, StudentModelConfig};
use rayon::prelude::*;
use smiles_parser::{DatasetFetchOptions, GzipMode, PUBCHEM_SMILES, SmilesDatasetSource};

type CudaBackend = Cuda<f32, i32>;

const DEFAULT_ARTIFACT_DIR: &str = "artifacts/mini-shared";
const DEFAULT_BATCH_SIZE: usize = 4_096;
const DEFAULT_PREPROCESSING_THREADS: usize = 32;
const DEFAULT_PROGRESS_EVERY: usize = 5_000_000;
const ENCODE_QUEUE_CAPACITY: usize = 2;

#[derive(Debug, Parser)]
#[command(name = "pubchem-cuda-throughput")]
#[command(
    about = "Benchmark the full-precision Mini NPClassifier.rs model on PubChem SMILES with CUDA."
)]
struct Cli {
    /// Artifact directory containing the Mini model.mpk.
    #[arg(long, default_value = DEFAULT_ARTIFACT_DIR)]
    artifact_dir: PathBuf,
    /// Maximum number of `PubChem` SMILES to classify. Omit for the full file.
    #[arg(long)]
    limit: Option<usize>,
    /// Number of SMILES to classify per fixed CUDA batch.
    #[arg(long, default_value_t = DEFAULT_BATCH_SIZE)]
    batch_size: usize,
    /// CUDA device index.
    #[arg(long, default_value_t = 0)]
    cuda_device: usize,
    /// Limit Rayon worker threads for fingerprint preprocessing.
    #[arg(long, default_value_t = DEFAULT_PREPROCESSING_THREADS)]
    threads: usize,
    /// Optional cache directory for the `PubChem` CID-SMILES dataset.
    #[arg(long)]
    dataset_cache_dir: Option<PathBuf>,
    /// Print one progress line every N classified SMILES. Use 0 to disable.
    #[arg(long, default_value_t = DEFAULT_PROGRESS_EVERY)]
    progress_every: usize,
    /// Host synchronization mode for completed CUDA batches.
    #[arg(long, value_enum, default_value_t = SyncMode::Reduced)]
    sync_mode: SyncMode,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SyncMode {
    /// Synchronize only reduced output checksums.
    Reduced,
    /// Synchronize all probability matrices.
    Full,
}

#[derive(Debug, Clone, Copy, Default)]
struct Timings {
    preprocessing: Duration,
    inference: Duration,
}

#[derive(Debug, Clone, Copy)]
struct ThroughputReport {
    limit: Option<usize>,
    processed: usize,
    failed: usize,
    elapsed: Duration,
    timings: Timings,
}

impl ThroughputReport {
    fn classified(self) -> usize {
        self.processed.saturating_sub(self.failed)
    }

    fn rows_per_second(self) -> u128 {
        rows_per_second(self.processed, self.elapsed)
    }

    fn inference_rows_per_second(self) -> u128 {
        rows_per_second(self.processed, self.timings.inference)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    configure_threads(cli.threads)?;

    let setup_started = Instant::now();
    let device = CudaDevice::new(cli.cuda_device);
    let model = load_mini_model(&cli.artifact_dir, &device)?;
    let warmup_elapsed = warm_up_model(&model, cli.batch_size.max(1), cli.sync_mode)?;
    eprintln!(
        "setup completed in {:.2?}; warmup={:.2?}; artifact_dir={}; sync={:?}",
        setup_started.elapsed(),
        warmup_elapsed,
        cli.artifact_dir.display(),
        cli.sync_mode
    );

    let report = classify_pubchem(&cli, &model)?;
    println!(
        "model=Mini backend=cuda limit={} processed={} classified={} failed={} elapsed={:.2?} throughput={} smiles/s preprocessing={:.2?} inference={:.2?} inference_throughput={} smiles/s sync={:?}",
        format_limit(report.limit),
        report.processed,
        report.classified(),
        report.failed,
        report.elapsed,
        report.rows_per_second(),
        report.timings.preprocessing,
        report.timings.inference,
        report.inference_rows_per_second(),
        cli.sync_mode
    );

    Ok(())
}

fn configure_threads(threads: usize) -> Result<(), Box<dyn Error>> {
    if threads == 0 {
        return Err("preprocessing thread count must be greater than zero".into());
    }

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    Ok(())
}

fn load_mini_model(
    artifact_dir: &Path,
    device: &CudaDevice,
) -> Result<StudentModel<CudaBackend>, Box<dyn Error>> {
    let model = StudentModelConfig::mini_shared()
        .init::<CudaBackend>(device, 1.0, 0.0)
        .load_file(artifact_dir.join("model"), &CompactRecorder::new(), device)?;
    Ok(model)
}

fn warm_up_model(
    model: &StudentModel<CudaBackend>,
    batch_size: usize,
    sync_mode: SyncMode,
) -> Result<Duration, Box<dyn Error>> {
    let started = Instant::now();
    let inputs = vec![0.0_f32; batch_size * FINGERPRINT_INPUT_WIDTH];
    synchronize_probabilities(
        model
            .predict_probabilities(inputs, batch_size)
            .map_err(|error| format!("model warm-up failed: {error}"))?,
        sync_mode,
    );
    Ok(started.elapsed())
}

fn classify_pubchem(
    cli: &Cli,
    model: &StudentModel<CudaBackend>,
) -> Result<ThroughputReport, Box<dyn Error>> {
    let batch_size = cli.batch_size.max(1);
    let (sender, receiver) = sync_channel(ENCODE_QUEUE_CAPACITY);
    let encoder = spawn_encoder(EncoderConfig::from_cli(cli, batch_size), sender);
    let started = Instant::now();
    let mut timings = Timings::default();
    let mut processed = 0usize;
    let mut failed = 0usize;
    let mut next_progress = ProgressCounter::new(cli.progress_every);

    for batch in receiver {
        failed += batch.failed;
        timings.preprocessing += batch.preprocessing;

        let inference_started = Instant::now();
        synchronize_probabilities(
            model
                .predict_probabilities(batch.inputs, batch_size)
                .map_err(|error| format!("model inference failed: {error}"))?,
            cli.sync_mode,
        );
        timings.inference += inference_started.elapsed();

        processed += batch.rows;
        maybe_report_progress(
            processed,
            failed,
            started.elapsed(),
            timings,
            &mut next_progress,
        );
    }

    join_encoder(encoder)?;

    Ok(ThroughputReport {
        limit: cli.limit,
        processed,
        failed,
        elapsed: started.elapsed(),
        timings,
    })
}

fn synchronize_probabilities(
    probabilities: npclassifier_train::model::NpClassifierProbabilities<CudaBackend>,
    sync_mode: SyncMode,
) {
    match sync_mode {
        SyncMode::Reduced => probabilities.sync_reduced(),
        SyncMode::Full => probabilities.sync(),
    }
}

#[derive(Debug, Clone)]
struct EncoderConfig {
    limit: Option<usize>,
    batch_size: usize,
    dataset_cache_dir: Option<PathBuf>,
}

impl EncoderConfig {
    fn from_cli(cli: &Cli, batch_size: usize) -> Self {
        Self {
            limit: cli.limit,
            batch_size,
            dataset_cache_dir: cli.dataset_cache_dir.clone(),
        }
    }
}

fn spawn_encoder(
    config: EncoderConfig,
    sender: SyncSender<EncodedBatch>,
) -> JoinHandle<Result<(), String>> {
    thread::spawn(move || encode_pubchem(&config, &sender))
}

fn encode_pubchem(config: &EncoderConfig, sender: &SyncSender<EncodedBatch>) -> Result<(), String> {
    let smiles = PUBCHEM_SMILES
        .iter_smiles_with_options(&DatasetFetchOptions {
            cache_dir: config.dataset_cache_dir.clone(),
            gzip_mode: GzipMode::Decompress,
            ..DatasetFetchOptions::default()
        })
        .map_err(|error| error.to_string())?;

    let mut batch = Vec::with_capacity(config.batch_size);
    let mut first_row_index = 0usize;
    let mut seen = 0usize;

    for smiles in smiles {
        if config.limit.is_some_and(|limit| seen >= limit) {
            break;
        }

        batch.push(smiles.map_err(|error| error.to_string())?);
        seen = seen.saturating_add(1);

        if batch.len() >= config.batch_size {
            send_encoded_batch(sender, &batch, first_row_index, config.batch_size)?;
            first_row_index = seen;
            batch.clear();
        }
    }

    if !batch.is_empty() {
        send_encoded_batch(sender, &batch, first_row_index, config.batch_size)?;
    }

    Ok(())
}

fn send_encoded_batch(
    sender: &SyncSender<EncodedBatch>,
    smiles: &[String],
    first_row_index: usize,
    model_batch_size: usize,
) -> Result<(), String> {
    let batch = encode_batch(smiles, first_row_index, model_batch_size)?;
    sender.send(batch).map_err(|error| error.to_string())
}

fn join_encoder(encoder: JoinHandle<Result<(), String>>) -> Result<(), Box<dyn Error>> {
    match encoder.join() {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(error.into()),
        Err(payload) => Err(panic_payload_to_string(payload).into()),
    }
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(message) => *message,
        Err(payload) => match payload.downcast::<&str>() {
            Ok(message) => (*message).to_owned(),
            Err(_) => "encoder thread panicked".to_owned(),
        },
    }
}

#[derive(Debug)]
struct EncodedBatch {
    inputs: Vec<f32>,
    rows: usize,
    failed: usize,
    preprocessing: Duration,
}

fn encode_batch(
    smiles: &[String],
    first_row_index: usize,
    model_batch_size: usize,
) -> Result<EncodedBatch, String> {
    if smiles.len() > model_batch_size {
        return Err(format!(
            "SMILES batch length {} exceeds model batch size {model_batch_size}",
            smiles.len()
        ));
    }

    let started = Instant::now();
    let failed = AtomicUsize::new(0);
    let mut inputs = vec![0.0_f32; model_batch_size * FINGERPRINT_INPUT_WIDTH];
    inputs
        .par_chunks_mut(FINGERPRINT_INPUT_WIDTH)
        .zip(smiles.par_iter().enumerate())
        .try_for_each_init(
            || {
                (
                    DenseFingerprintEncoder::default(),
                    vec![0_u16; FINGERPRINT_INPUT_WIDTH],
                )
            },
            |(encoder, counts), (row, (index, smiles))| -> Result<(), String> {
                let row_number = first_row_index
                    .checked_add(index)
                    .ok_or_else(|| "PubChem row index overflowed".to_owned())?;
                let cid = i64::try_from(row_number)
                    .map_err(|error| format!("PubChem row index overflowed i64: {error}"))?;
                counts.fill(0);

                if encoder.encode_into(smiles, cid, counts).is_err() {
                    failed.fetch_add(1, Ordering::Relaxed);
                    row.fill(0.0);
                    return Ok(());
                }

                for (value, count) in row.iter_mut().zip(counts.iter()) {
                    *value = f32::from(*count);
                }
                Ok(())
            },
        )?;

    Ok(EncodedBatch {
        inputs,
        rows: smiles.len(),
        failed: failed.load(Ordering::Relaxed),
        preprocessing: started.elapsed(),
    })
}

fn maybe_report_progress(
    processed: usize,
    failed: usize,
    elapsed: Duration,
    timings: Timings,
    progress: &mut ProgressCounter,
) {
    if !progress.should_report(processed) {
        return;
    }

    eprintln!(
        "processed={} classified={} failed={} elapsed={:.2?} throughput={} smiles/s preprocessing={:.2?} inference={:.2?} inference_throughput={} smiles/s",
        processed,
        processed.saturating_sub(failed),
        failed,
        elapsed,
        rows_per_second(processed, elapsed),
        timings.preprocessing,
        timings.inference,
        rows_per_second(processed, timings.inference)
    );
    progress.advance_to(processed);
}

#[derive(Debug, Clone, Copy)]
struct ProgressCounter {
    step: usize,
    next: usize,
}

impl ProgressCounter {
    const fn new(step: usize) -> Self {
        Self { step, next: step }
    }

    const fn should_report(self, processed: usize) -> bool {
        self.step > 0 && processed >= self.next
    }

    fn advance_to(&mut self, processed: usize) {
        while self.step > 0 && self.next <= processed {
            self.next = self.next.saturating_add(self.step);
        }
    }
}

fn rows_per_second(rows: usize, elapsed: Duration) -> u128 {
    let elapsed_ms = elapsed.as_millis().max(1);
    u128::from(u64::try_from(rows).unwrap_or(u64::MAX)).saturating_mul(1_000) / elapsed_ms
}

fn format_limit(limit: Option<usize>) -> String {
    limit.map_or_else(|| "all".to_owned(), |limit| limit.to_string())
}
