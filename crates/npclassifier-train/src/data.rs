//! Published distillation split loading and Burn batch preparation.

use std::fmt::Debug;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use arrow_array::cast::{as_list_array, as_primitive_array, as_string_array};
use arrow_array::types::Int64Type;
use arrow_array::{Array, RecordBatch, UInt16Array};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::data::dataset::Dataset;
use burn::prelude::Backend;
use finge_rs::{Fingerprint, LayeredCountEcfpFingerprint, SmilesRdkitScratch};
use half::f16;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use serde::Deserialize;
use smiles_parser::smiles::Smiles;

use npclassifier_core::{
    DISTILLATION_DATASET_DOI, FINGERPRINT_FORMULA_BITS, FINGERPRINT_INPUT_WIDTH, ModelHead,
    ensure_distillation_dataset, missing_distillation_dataset_files,
};

use crate::error::TrainingError;

const PATHWAY_WIDTH: usize = ModelHead::Pathway.output_width();
const SUPERCLASS_WIDTH: usize = ModelHead::Superclass.output_width();
const CLASS_WIDTH: usize = ModelHead::Class.output_width();
const PROGRESS_TICK_INTERVAL: Duration = Duration::from_millis(100);
const FINGERPRINT_PROGRESS_CHUNK_ROWS: usize = 256;

/// Fully loaded metadata and optional teacher vectors for one dataset split.
#[derive(Debug)]
pub struct TeacherSplitStorage {
    smiles: Vec<String>,
    cids: Vec<i64>,
    pathway_ids: Vec<Vec<u16>>,
    superclass_ids: Vec<Vec<u16>>,
    class_ids: Vec<Vec<u16>>,
    fingerprints: FingerprintCache,
    pathway_teacher: Option<Vec<f16>>,
    superclass_teacher: Option<Vec<f16>>,
    class_teacher: Option<Vec<f16>>,
}

impl TeacherSplitStorage {
    /// Returns the number of rows in the split.
    #[must_use]
    pub fn len(&self) -> usize {
        self.smiles.len()
    }

    /// Returns whether the split has no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.smiles.is_empty()
    }
}

/// Lightweight dataset over row indices for Burn's dataloader.
#[derive(Debug, Clone)]
pub struct SplitDataset {
    len: usize,
}

impl SplitDataset {
    /// Creates a new index dataset for the specified split size.
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self { len }
    }
}

impl Dataset<usize> for SplitDataset {
    fn get(&self, index: usize) -> Option<usize> {
        (index < self.len).then_some(index)
    }

    fn len(&self) -> usize {
        self.len
    }
}

/// One host-side training or validation batch.
#[derive(Clone, Debug)]
pub struct NpClassifierBatch {
    /// Dense `6144` counted Morgan inputs.
    pub(crate) inputs: Vec<f32>,
    /// Pathway multilabel targets.
    pub(crate) pathway_targets: Vec<i32>,
    /// Superclass multilabel targets.
    pub(crate) superclass_targets: Vec<i32>,
    /// Class multilabel targets.
    pub(crate) class_targets: Vec<i32>,
    /// Optional pathway teacher probabilities.
    pub(crate) pathway_teacher: Option<Vec<f32>>,
    /// Optional superclass teacher probabilities.
    pub(crate) superclass_teacher: Option<Vec<f32>>,
    /// Optional class teacher probabilities.
    pub(crate) class_teacher: Option<Vec<f32>>,
}

impl NpClassifierBatch {
    /// Returns the number of rows in the batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inputs.len() / FINGERPRINT_INPUT_WIDTH
    }

    /// Returns whether the batch has no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }
}

#[derive(Debug)]
struct FingerprintCache {
    indices: Vec<u16>,
}

impl FingerprintCache {
    fn new(indices: Vec<u16>) -> Self {
        Self { indices }
    }

    fn row(&self, index: usize) -> &[u16] {
        let start = index * FINGERPRINT_INPUT_WIDTH;
        let end = start + FINGERPRINT_INPUT_WIDTH;

        &self.indices[start..end]
    }
}

#[derive(Debug)]
struct DenseFingerprintEncoder {
    scratch: SmilesRdkitScratch,
    layered: LayeredCountEcfpFingerprint,
}

impl Default for DenseFingerprintEncoder {
    fn default() -> Self {
        Self {
            scratch: SmilesRdkitScratch::default(),
            layered: LayeredCountEcfpFingerprint::new(2, FINGERPRINT_FORMULA_BITS),
        }
    }
}

impl DenseFingerprintEncoder {
    fn encode_into(
        &mut self,
        smiles: &str,
        cid: i64,
        row: &mut [u16],
    ) -> Result<(), TrainingError> {
        let smiles = smiles.parse::<Smiles>().map_err(|error| {
            TrainingError::Dataset(format!(
                "failed to parse CID {cid} SMILES {smiles}: {error}"
            ))
        })?;
        let explicit_hydrogens = smiles.with_explicit_hydrogens();
        let graph = self
            .scratch
            .try_prepare(&explicit_hydrogens)
            .map_err(|error| {
                TrainingError::Dataset(format!(
                    "failed to normalize CID {cid} SMILES {explicit_hydrogens}: {error}"
                ))
            })?;
        let layered = self.layered.compute(&graph);
        write_active_counts_to_row(row, 0, layered.formula())?;
        write_active_counts_to_row(
            row,
            FINGERPRINT_FORMULA_BITS,
            layered.layer(1).ok_or_else(|| {
                TrainingError::Dataset("missing exact Morgan radius-1 layer".to_owned())
            })?,
        )?;
        write_active_counts_to_row(
            row,
            FINGERPRINT_FORMULA_BITS * 2,
            layered.layer(2).ok_or_else(|| {
                TrainingError::Dataset("missing exact Morgan radius-2 layer".to_owned())
            })?,
        )?;

        Ok(())
    }
}

/// Batcher that densifies precomputed counted Morgan fingerprints.
#[derive(Clone, Debug)]
pub struct NpClassifierBatcher {
    storage: Arc<TeacherSplitStorage>,
    include_teacher: bool,
}

impl NpClassifierBatcher {
    /// Creates a new batcher over the specified split storage.
    #[must_use]
    pub fn new(storage: Arc<TeacherSplitStorage>, include_teacher: bool) -> Self {
        Self {
            storage,
            include_teacher,
        }
    }

    fn append_targets(targets: &mut [i32], row_in_batch: usize, width: usize, labels: &[u16]) {
        for label in labels {
            targets[row_in_batch * width + usize::from(*label)] = 1;
        }
    }

    fn append_teacher_values(
        values: &mut Vec<f32>,
        teacher: Option<&Vec<f16>>,
        index: usize,
        width: usize,
        cid: i64,
        smiles: &str,
        head_name: &str,
    ) {
        let start = index * width;
        let end = start + width;
        let teacher = teacher.unwrap_or_else(|| {
            panic!("{head_name} teacher sidecar missing for CID {cid} ({smiles})")
        });
        values.extend(teacher[start..end].iter().map(|value| value.to_f32()));
    }

    fn append_fingerprint(values: &mut [f32], row_in_batch: usize, counts: &[u16]) {
        let row_offset = row_in_batch * FINGERPRINT_INPUT_WIDTH;
        for (slot, count) in values[row_offset..row_offset + FINGERPRINT_INPUT_WIDTH]
            .iter_mut()
            .zip(counts.iter())
        {
            *slot = f32::from(*count);
        }
    }
}

impl<B> Batcher<B, usize, NpClassifierBatch> for NpClassifierBatcher
where
    B: Backend,
{
    fn batch(&self, items: Vec<usize>, _device: &B::Device) -> NpClassifierBatch {
        let batch_len = items.len();
        let mut inputs = vec![0.0_f32; batch_len * FINGERPRINT_INPUT_WIDTH];
        let mut pathway_targets = vec![0_i32; batch_len * PATHWAY_WIDTH];
        let mut superclass_targets = vec![0_i32; batch_len * SUPERCLASS_WIDTH];
        let mut class_targets = vec![0_i32; batch_len * CLASS_WIDTH];

        let mut pathway_teacher = self
            .include_teacher
            .then(|| Vec::with_capacity(batch_len * PATHWAY_WIDTH));
        let mut superclass_teacher = self
            .include_teacher
            .then(|| Vec::with_capacity(batch_len * SUPERCLASS_WIDTH));
        let mut class_teacher = self
            .include_teacher
            .then(|| Vec::with_capacity(batch_len * CLASS_WIDTH));

        for (row_in_batch, index) in items.into_iter().enumerate() {
            let smiles = &self.storage.smiles[index];
            let cid = self.storage.cids[index];
            let fingerprint_counts = self.storage.fingerprints.row(index);
            Self::append_fingerprint(&mut inputs, row_in_batch, fingerprint_counts);

            Self::append_targets(
                &mut pathway_targets,
                row_in_batch,
                PATHWAY_WIDTH,
                &self.storage.pathway_ids[index],
            );
            Self::append_targets(
                &mut superclass_targets,
                row_in_batch,
                SUPERCLASS_WIDTH,
                &self.storage.superclass_ids[index],
            );
            Self::append_targets(
                &mut class_targets,
                row_in_batch,
                CLASS_WIDTH,
                &self.storage.class_ids[index],
            );

            if let Some(values) = pathway_teacher.as_mut() {
                Self::append_teacher_values(
                    values,
                    self.storage.pathway_teacher.as_ref(),
                    index,
                    PATHWAY_WIDTH,
                    cid,
                    smiles,
                    "pathway",
                );
            }
            if let Some(values) = superclass_teacher.as_mut() {
                Self::append_teacher_values(
                    values,
                    self.storage.superclass_teacher.as_ref(),
                    index,
                    SUPERCLASS_WIDTH,
                    cid,
                    smiles,
                    "superclass",
                );
            }
            if let Some(values) = class_teacher.as_mut() {
                Self::append_teacher_values(
                    values,
                    self.storage.class_teacher.as_ref(),
                    index,
                    CLASS_WIDTH,
                    cid,
                    smiles,
                    "class",
                );
            }
        }

        NpClassifierBatch {
            inputs,
            pathway_targets,
            superclass_targets,
            class_targets,
            pathway_teacher,
            superclass_teacher,
            class_teacher,
        }
    }
}

/// Loads the finalized split manifest from a dataset directory.
///
/// # Errors
///
/// Returns an error if the dataset cannot be downloaded, the manifest is
/// missing, or the manifest cannot be read or decoded.
pub fn load_manifest(data_dir: &Path) -> Result<TrainingManifest, TrainingError> {
    ensure_training_dataset(data_dir)?;
    let manifest_path = data_dir.join("manifest.json");
    if !manifest_path.exists() {
        return Err(TrainingError::MissingFile(manifest_path));
    }
    let contents = std::fs::read_to_string(&manifest_path)?;
    Ok(serde_json::from_str(&contents)?)
}

/// Loads one finalized split into memory.
///
/// # Errors
///
/// Returns an error if the dataset cannot be downloaded, the split files are
/// missing, or the Parquet rows, teacher vectors, or fingerprints cannot be
/// loaded.
pub fn load_split_storage(
    data_dir: &Path,
    split: &str,
    limit: Option<usize>,
    include_teacher: bool,
) -> Result<Arc<TeacherSplitStorage>, TrainingError> {
    ensure_training_dataset(data_dir)?;
    let rows_path = data_dir.join(format!("{split}.parquet"));
    ensure_exists(&rows_path)?;

    let file = File::open(&rows_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let total_rows = usize::try_from(builder.metadata().file_metadata().num_rows())
        .map_err(|error| TrainingError::Dataset(format!("split row count overflowed: {error}")))?;
    let total_rows = limit.map_or(total_rows, |row_limit| total_rows.min(row_limit));
    let reader = builder.build()?;
    let load_progress = rows_progress(format!("load {split} rows"), total_rows)?;

    let mut smiles = Vec::new();
    let mut cids = Vec::new();
    let mut pathway_ids = Vec::new();
    let mut superclass_ids = Vec::new();
    let mut class_ids = Vec::new();

    for batch in reader {
        let batch = batch.map_err(|error| TrainingError::Dataset(error.to_string()))?;
        let smiles_array = as_string_array(column(&batch, "smiles")?);
        let cid_array = as_primitive_array::<Int64Type>(column(&batch, "cid")?);
        let pathway_view = label_column(&batch, "pathway_ids")?;
        let superclass_view = label_column(&batch, "superclass_ids")?;
        let class_view = label_column(&batch, "class_ids")?;
        let mut rows_loaded = 0_usize;

        for row_index in 0..batch.num_rows() {
            cids.push(cid_array.value(row_index));
            smiles.push(smiles_array.value(row_index).to_owned());
            pathway_ids.push(pathway_view.values(row_index));
            superclass_ids.push(superclass_view.values(row_index));
            class_ids.push(class_view.values(row_index));
            rows_loaded += 1;

            if limit.is_some_and(|max_rows| smiles.len() >= max_rows) {
                break;
            }
        }
        load_progress.inc(usize_to_u64(rows_loaded, "loaded row count")?);

        if limit.is_some_and(|max_rows| smiles.len() >= max_rows) {
            break;
        }
    }

    let row_count = smiles.len();
    if row_count == 0 {
        return Err(TrainingError::Dataset(format!(
            "split {split} in {} did not contain any rows",
            data_dir.display()
        )));
    }
    load_progress.finish_with_message(format!("loaded {row_count} {split} rows"));

    let fingerprints = precompute_fingerprints(split, &smiles, &cids)?;

    let pathway_teacher = if include_teacher {
        Some(load_teacher_vectors(
            &data_dir.join(format!("{split}.pathway-vectors.f16.zst")),
            row_count,
            PATHWAY_WIDTH,
            split,
            "pathway",
        )?)
    } else {
        None
    };
    let superclass_teacher = if include_teacher {
        Some(load_teacher_vectors(
            &data_dir.join(format!("{split}.superclass-vectors.f16.zst")),
            row_count,
            SUPERCLASS_WIDTH,
            split,
            "superclass",
        )?)
    } else {
        None
    };
    let class_teacher = if include_teacher {
        Some(load_teacher_vectors(
            &data_dir.join(format!("{split}.class-vectors.f16.zst")),
            row_count,
            CLASS_WIDTH,
            split,
            "class",
        )?)
    } else {
        None
    };

    Ok(Arc::new(TeacherSplitStorage {
        smiles,
        cids,
        pathway_ids,
        superclass_ids,
        class_ids,
        fingerprints,
        pathway_teacher,
        superclass_teacher,
        class_teacher,
    }))
}

/// Builds a Burn dataloader for a previously loaded split.
#[must_use]
pub fn build_dataloader<B: Backend>(
    storage: &Arc<TeacherSplitStorage>,
    batch_size: usize,
    num_workers: usize,
    shuffle_seed: Option<u64>,
    include_teacher: bool,
) -> Arc<dyn DataLoader<B, NpClassifierBatch>> {
    let batcher = NpClassifierBatcher::new(Arc::clone(storage), include_teacher);
    let dataset = SplitDataset::new(storage.len());
    let builder = DataLoaderBuilder::new(batcher)
        .batch_size(batch_size)
        .num_workers(num_workers);

    match shuffle_seed {
        Some(seed) => builder.shuffle(seed).build(dataset),
        None => builder.build(dataset),
    }
}

fn ensure_exists(path: &Path) -> Result<(), TrainingError> {
    if path.exists() {
        Ok(())
    } else {
        Err(TrainingError::MissingFile(path.to_path_buf()))
    }
}

fn ensure_training_dataset(data_dir: &Path) -> Result<(), TrainingError> {
    let missing = missing_distillation_dataset_files(data_dir);
    if !missing.is_empty() {
        eprintln!(
            "distillation dataset is missing {} files; downloading from {DISTILLATION_DATASET_DOI} into {}",
            missing.len(),
            data_dir.display()
        );
        ensure_distillation_dataset(data_dir)?;
    }
    Ok(())
}

fn precompute_fingerprints(
    split: &str,
    smiles: &[String],
    cids: &[i64],
) -> Result<FingerprintCache, TrainingError> {
    let mut dense = vec![0_u16; smiles.len() * FINGERPRINT_INPUT_WIDTH];
    let progress = rows_progress(format!("precompute {split} fingerprints"), smiles.len())?;
    dense
        .par_chunks_mut(FINGERPRINT_INPUT_WIDTH * FINGERPRINT_PROGRESS_CHUNK_ROWS)
        .zip(smiles.par_chunks(FINGERPRINT_PROGRESS_CHUNK_ROWS))
        .zip(cids.par_chunks(FINGERPRINT_PROGRESS_CHUNK_ROWS))
        .try_for_each_init(
            || (DenseFingerprintEncoder::default(), progress.clone()),
            |(encoder, progress), ((row_chunk, smiles_chunk), cids_chunk)| {
                for ((row, smiles), cid) in row_chunk
                    .chunks_exact_mut(FINGERPRINT_INPUT_WIDTH)
                    .zip(smiles_chunk)
                    .zip(cids_chunk)
                {
                    row.fill(0);
                    encoder.encode_into(smiles, *cid, row)?;
                }
                progress.inc(usize_to_u64(
                    smiles_chunk.len(),
                    "fingerprint chunk row count",
                )?);
                Ok::<(), TrainingError>(())
            },
        )?;
    progress.finish_with_message(format!("precomputed {} {split} fingerprints", smiles.len()));

    Ok(FingerprintCache::new(dense))
}

fn load_teacher_vectors(
    path: &Path,
    rows: usize,
    width: usize,
    split: &str,
    head_name: &str,
) -> Result<Vec<f16>, TrainingError> {
    ensure_exists(path)?;
    let file = File::open(path)?;
    let mut decoder = zstd::stream::read::Decoder::new(BufReader::new(file))?;
    let expected_values = rows
        .checked_mul(width)
        .ok_or_else(|| TrainingError::Dataset("teacher vector size overflowed".to_owned()))?;
    let expected_bytes = expected_values
        .checked_mul(2)
        .ok_or_else(|| TrainingError::Dataset("teacher byte size overflowed".to_owned()))?;
    let mut bytes = vec![0_u8; expected_bytes];
    let progress = bytes_progress(format!("load {split} {head_name} teacher"), expected_bytes)?;
    let mut filled = 0_usize;
    while filled < expected_bytes {
        let read = decoder.read(&mut bytes[filled..]).map_err(|error| {
            TrainingError::Dataset(format!(
                "teacher sidecar {} failed while reading bytes: {error}",
                path.display()
            ))
        })?;
        if read == 0 {
            return Err(TrainingError::Dataset(format!(
                "teacher sidecar {} ended before {} bytes could be read",
                path.display(),
                expected_bytes
            )));
        }
        filled += read;
        progress.inc(usize_to_u64(read, "teacher sidecar byte count")?);
    }
    progress.finish_with_message(format!("loaded {split} {head_name} teacher sidecar"));

    let mut values = Vec::with_capacity(expected_values);
    for chunk in bytes.chunks_exact(2) {
        values.push(f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])));
    }
    Ok(values)
}

fn write_active_counts_to_row(
    row: &mut [u16],
    offset: usize,
    fingerprint: &finge_rs::CountFingerprint,
) -> Result<(), TrainingError> {
    for (index, count) in fingerprint.active_counts() {
        let folded_count = u16::try_from(count).map_err(|_| {
            TrainingError::Dataset(format!(
                "fingerprint count {count} at index {} exceeded `u16` storage",
                offset + index
            ))
        })?;
        row[offset + index] = folded_count;
    }

    Ok(())
}

fn column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a dyn Array, TrainingError> {
    batch
        .column_by_name(name)
        .map(std::convert::AsRef::as_ref)
        .ok_or_else(|| TrainingError::Dataset(format!("missing parquet column {name}")))
}

fn label_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<LabelColumnView<'a>, TrainingError> {
    let list = as_list_array(column(batch, name)?);
    let values = list
        .values()
        .as_any()
        .downcast_ref::<UInt16Array>()
        .ok_or_else(|| {
            TrainingError::Dataset(format!("list column {name} did not contain uint16 values"))
        })?;
    Ok(LabelColumnView { list, values })
}

struct LabelColumnView<'a> {
    list: &'a arrow_array::ListArray,
    values: &'a UInt16Array,
}

impl LabelColumnView<'_> {
    fn values(&self, row_index: usize) -> Vec<u16> {
        let offsets = self.list.value_offsets();
        let start = usize::try_from(offsets[row_index]).expect("offset should fit into usize");
        let end = usize::try_from(offsets[row_index + 1]).expect("offset should fit into usize");
        self.values.values()[start..end].to_vec()
    }
}

fn rows_progress(
    label: impl Into<String>,
    total_rows: usize,
) -> Result<ProgressBar, TrainingError> {
    let progress = ProgressBar::new(usize_to_u64(total_rows, "row count")?);
    progress.set_style(progress_style(
        "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>9}/{len:9} {msg}",
    )?);
    progress.set_message(label.into());
    progress.enable_steady_tick(PROGRESS_TICK_INTERVAL);
    Ok(progress)
}

fn bytes_progress(
    label: impl Into<String>,
    total_bytes: usize,
) -> Result<ProgressBar, TrainingError> {
    let progress = ProgressBar::new(usize_to_u64(total_bytes, "byte count")?);
    progress.set_style(progress_style(
        "[{elapsed_precise}] {bar:40.cyan/blue} {bytes:>10}/{total_bytes:10} {msg}",
    )?);
    progress.set_message(label.into());
    progress.enable_steady_tick(PROGRESS_TICK_INTERVAL);
    Ok(progress)
}

fn progress_style(template: &str) -> Result<ProgressStyle, TrainingError> {
    ProgressStyle::with_template(template)
        .map(|style| style.progress_chars("=>-"))
        .map_err(|error| TrainingError::Dataset(format!("invalid progress bar template: {error}")))
}

fn usize_to_u64(value: usize, label: &str) -> Result<u64, TrainingError> {
    u64::try_from(value)
        .map_err(|error| TrainingError::Dataset(format!("{label} does not fit into u64: {error}")))
}

/// Minimal manifest fields needed by the training CLI.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct TrainingManifest {
    /// Stored output widths for the three classifier heads.
    pub vector_widths: TrainingVectorWidths,
}

/// Width metadata copied from the published dataset manifest.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct TrainingVectorWidths {
    /// Pathway width.
    pub pathway: usize,
    /// Superclass width.
    pub superclass: usize,
    /// Class width.
    pub class_: usize,
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{BufWriter, Write};

    use npclassifier_core::{CountedMorganGenerator, FingerprintGenerator};

    use super::*;

    #[test]
    fn load_teacher_vectors_reads_only_the_requested_prefix_rows() {
        let temp_dir = std::env::temp_dir().join(format!(
            "npclassifier-train-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        fs::create_dir_all(&temp_dir).expect("temp dir");
        let path = temp_dir.join("teacher.f16.zst");

        let values = [0.1_f32, 0.2, 0.3, 0.4];
        let mut writer = zstd::stream::write::Encoder::new(
            BufWriter::new(File::create(&path).expect("teacher file")),
            9,
        )
        .expect("encoder");
        for value in values {
            writer
                .write_all(&f16::from_f32(value).to_bits().to_le_bytes())
                .expect("write teacher value");
        }
        writer.finish().expect("finish teacher file");

        let loaded =
            load_teacher_vectors(&path, 1, 2, "test", "pathway").expect("prefix teacher vectors");
        assert_eq!(loaded.len(), 2);
        assert!((loaded[0].to_f32() - 0.1).abs() < 1e-4);
        assert!((loaded[1].to_f32() - 0.2).abs() < 1e-4);

        fs::remove_file(&path).expect("remove teacher file");
        fs::remove_dir(&temp_dir).expect("remove temp dir");
    }

    #[test]
    fn precomputed_dense_u16_fingerprints_match_dense_generator_output() {
        let smiles = vec!["C".to_owned(), "CCO".to_owned()];
        let cids = vec![1_i64, 2_i64];
        let cache = precompute_fingerprints("test", &smiles, &cids).expect("fingerprint cache");
        let generator = CountedMorganGenerator::default();

        for (row_index, smiles) in smiles.iter().enumerate() {
            let expected = generator
                .generate(smiles)
                .expect("dense fingerprint generation")
                .fingerprint()
                .concatenated();
            let counts = cache.row(row_index);
            let mut observed = vec![0.0_f32; FINGERPRINT_INPUT_WIDTH];
            NpClassifierBatcher::append_fingerprint(&mut observed, 0, counts);

            assert_eq!(observed, expected);
        }
    }
}
