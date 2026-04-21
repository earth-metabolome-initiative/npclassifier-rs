//! Distillation dataset curation for retraining a smaller `NPClassifier`.
//!
//! The sibling `npc-labeler` workspace writes row metadata to Parquet and
//! aligned `float16` teacher vectors to compressed sidecars. This module
//! filters that output down to high-confidence rows, stratifies the result by
//! multilabel signature, and materializes train/validation/test splits for
//! Rust-native retraining workflows.

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use arrow_array::builder::{
    BooleanBuilder, Float32Builder, Int64Builder, ListBuilder, StringBuilder, UInt16Builder,
};
use arrow_array::cast::{as_boolean_array, as_list_array, as_primitive_array, as_string_array};
use arrow_array::{
    Array, ArrayRef, PrimitiveArray, RecordBatch, UInt16Array,
    types::{Int64Type, UInt16Type},
};
use arrow_schema::{DataType, Field, Schema};
use half::f16;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::Serialize;

use crate::NpClassifierError;

const PATHWAY_THRESHOLD: f32 = 0.5;
const SUPERCLASS_THRESHOLD: f32 = 0.3;
const CLASS_THRESHOLD: f32 = 0.1;
const PROGRESS_TICK_INTERVAL: Duration = Duration::from_millis(100);

/// Default source directory for old-RDKit teacher outputs.
pub const DEFAULT_COMPLETED_DIR: &str = "/home/luca/github/npc-labeler/work/completed";
/// Default vocabulary file aligned with the teacher outputs.
pub const DEFAULT_VOCABULARY_PATH: &str =
    "/home/luca/github/npc-labeler/work/releases/vocabulary.json";
/// Default output directory for curated splits.
pub const DEFAULT_OUTPUT_DIR: &str = "data/distillation/teacher-splits";

/// One completed `npc-labeler` chunk with aligned row metadata and score sidecars.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CompletedPart {
    /// Chunk stem such as `part-000001`.
    pub part_name: String,
    /// Row metadata file.
    pub rows_path: PathBuf,
    /// Pathway teacher vectors.
    pub pathway_vectors_path: PathBuf,
    /// Superclass teacher vectors.
    pub superclass_vectors_path: PathBuf,
    /// Class teacher vectors.
    pub class_vectors_path: PathBuf,
}

/// Vector widths for the three teacher heads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct VectorWidths {
    /// Number of pathway outputs.
    pub pathway: usize,
    /// Number of superclass outputs.
    pub superclass: usize,
    /// Number of class outputs.
    pub class_: usize,
}

/// Positive and negative confidence margins used during curation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ConfidenceMargins {
    /// Pathway positive margin above threshold.
    pub pathway_positive: f32,
    /// Pathway negative margin below threshold.
    pub pathway_negative: f32,
    /// Superclass positive margin above threshold.
    pub superclass_positive: f32,
    /// Superclass negative margin below threshold.
    pub superclass_negative: f32,
    /// Class positive margin above threshold.
    pub class_positive: f32,
    /// Class negative margin below threshold.
    pub class_negative: f32,
}

impl Default for ConfidenceMargins {
    fn default() -> Self {
        Self {
            pathway_positive: 0.25,
            pathway_negative: 0.20,
            superclass_positive: 0.20,
            superclass_negative: 0.10,
            class_positive: 0.15,
            class_negative: 0.05,
        }
    }
}

/// Fractions used for train / validation / test holdouts.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct SplitFractions {
    /// Training-set fraction.
    pub train: f64,
    /// Validation-set fraction.
    pub validation: f64,
    /// Test-set fraction.
    pub test: f64,
}

impl Default for SplitFractions {
    fn default() -> Self {
        Self {
            train: 0.8,
            validation: 0.1,
            test: 0.1,
        }
    }
}

impl SplitFractions {
    fn normalized(self) -> Result<Self, NpClassifierError> {
        let total = self.train + self.validation + self.test;
        if total <= 0.0 {
            return Err(NpClassifierError::Dataset(
                "split fractions must sum to a positive value".to_owned(),
            ));
        }
        Ok(Self {
            train: self.train / total,
            validation: self.validation / total,
            test: self.test / total,
        })
    }
}

/// Runtime configuration for summary or materialization passes.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CurationConfig {
    /// Source directories containing `part-*.parquet` and vector sidecars.
    pub input_dirs: Vec<PathBuf>,
    /// Vocabulary JSON aligned with the teacher outputs.
    pub vocabulary_path: PathBuf,
    /// Output directory for curated splits.
    pub output_dir: PathBuf,
    /// Streaming batch size used for Parquet and vector decoding.
    pub batch_rows: usize,
    /// Optional global row cap for smoke tests.
    pub max_rows: Option<usize>,
    /// Confidence margins applied to each head.
    pub margins: ConfidenceMargins,
    /// Stratified split fractions.
    pub split_fractions: SplitFractions,
    /// Signatures below this support are routed to train only.
    pub min_signature_count: usize,
    /// Deterministic seed for split shuffling.
    pub seed: u64,
}

impl Default for CurationConfig {
    fn default() -> Self {
        Self {
            input_dirs: vec![PathBuf::from(DEFAULT_COMPLETED_DIR)],
            vocabulary_path: PathBuf::from(DEFAULT_VOCABULARY_PATH),
            output_dir: PathBuf::from(DEFAULT_OUTPUT_DIR),
            batch_rows: 50_000,
            max_rows: None,
            margins: ConfidenceMargins::default(),
            split_fractions: SplitFractions::default(),
            min_signature_count: 10,
            seed: 0,
        }
    }
}

/// Counts collected during the filtering pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct SelectionSummary {
    /// Rows scanned from the teacher dataset.
    pub scanned_rows: u64,
    /// Rows rejected due to parse / `RDKit` / runtime failure flags.
    pub failed_rows: u64,
    /// Rows rejected because at least one head had no labels.
    pub partial_rows: u64,
    /// Rows rejected because at least one teacher vector had non-finite values.
    pub nonfinite_vector_rows: u64,
    /// Rows rejected because margins were too close to the decision thresholds.
    pub indecisive_rows: u64,
    /// Rows accepted for the stratified split stage.
    pub kept_rows: u64,
}

/// Signature-level split summary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct SignatureSummary {
    /// Number of unique multilabel signatures after filtering.
    pub unique_signatures: u64,
    /// Signatures routed to train only.
    pub rare_signatures: u64,
    /// Rows routed to train only due to rare signatures.
    pub rare_signature_rows: u64,
    /// Signatures eligible for stratified splitting.
    pub eligible_signatures: u64,
}

/// Row counts written to each split.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct SplitRowCounts {
    /// Training rows.
    pub train: u64,
    /// Validation rows.
    pub validation: u64,
    /// Test rows.
    pub test: u64,
}

impl SplitRowCounts {
    fn increment(&mut self, split: Split) {
        match split {
            Split::Train => self.train += 1,
            Split::Validation => self.validation += 1,
            Split::Test => self.test += 1,
        }
    }
}

/// Report emitted by summary and curate commands.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DistillationReport {
    /// Source directories scanned by the run.
    pub input_dirs: Vec<String>,
    /// Optional row cap used for the run.
    pub max_rows: Option<usize>,
    /// Confidence margins applied to each head.
    pub margins: ConfidenceMargins,
    /// Selection counters from the filtering pass.
    pub selection_summary: SelectionSummary,
    /// Signature summary from split assignment.
    pub signature_summary: SignatureSummary,
    /// Final split counts.
    pub split_row_counts: SplitRowCounts,
}

/// Manifest written next to the curated dataset.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CuratedManifest {
    /// Manifest creation timestamp in UTC.
    pub created_at: String,
    /// Source directories containing the teacher outputs.
    pub input_dirs: Vec<String>,
    /// Copied vocabulary filename.
    pub vocabulary: String,
    /// Vector widths for the three heads.
    pub vector_widths: VectorWidths,
    /// Original `NPClassifier` thresholds.
    pub thresholds: BTreeMap<String, f32>,
    /// Positive and negative confidence margins.
    pub margins: ConfidenceMargins,
    /// Requested split fractions before normalization.
    pub split_fractions: SplitFractions,
    /// Signatures below this support were routed to train only.
    pub min_signature_count: usize,
    /// Deterministic seed for split shuffling.
    pub seed: u64,
    /// Parquet streaming batch size.
    pub batch_rows: usize,
    /// Optional row cap used for this run.
    pub max_rows: Option<usize>,
    /// Filtering counters.
    pub selection_summary: SelectionSummary,
    /// Signature-level split summary.
    pub signature_summary: SignatureSummary,
    /// Output row counts.
    pub split_row_counts: SplitRowCounts,
    /// Output filenames for each split.
    pub outputs: BTreeMap<String, BTreeMap<String, String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Head {
    Pathway,
    Superclass,
    Class,
}

impl Head {
    const ALL: [Self; 3] = [Self::Pathway, Self::Superclass, Self::Class];

    fn as_str(self) -> &'static str {
        match self {
            Self::Pathway => "pathway",
            Self::Superclass => "superclass",
            Self::Class => "class",
        }
    }

    fn vectors_filename(self, split_name: &str) -> String {
        format!("{split_name}.{}-vectors.f16.zst", self.as_str())
    }

    fn threshold(self) -> f32 {
        match self {
            Self::Pathway => PATHWAY_THRESHOLD,
            Self::Superclass => SUPERCLASS_THRESHOLD,
            Self::Class => CLASS_THRESHOLD,
        }
    }

    fn positive_margin(self, margins: ConfidenceMargins) -> f32 {
        match self {
            Self::Pathway => margins.pathway_positive,
            Self::Superclass => margins.superclass_positive,
            Self::Class => margins.class_positive,
        }
    }

    fn negative_margin(self, margins: ConfidenceMargins) -> f32 {
        match self {
            Self::Pathway => margins.pathway_negative,
            Self::Superclass => margins.superclass_negative,
            Self::Class => margins.class_negative,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SignatureKey {
    pathway_ids: Vec<u16>,
    superclass_ids: Vec<u16>,
    class_ids: Vec<u16>,
    is_glycoside: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AcceptedRow {
    part_index: usize,
    row_index: usize,
    signature_id: usize,
}

enum RowEvaluation {
    Failed,
    Partial,
    Nonfinite,
    Indecisive,
    Accepted(SignatureKey),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Assignment {
    row_index: usize,
    split: Split,
    signature_id: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Split {
    Train,
    Validation,
    Test,
}

impl Split {
    fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "validation",
            Self::Test => "test",
        }
    }
}

#[derive(Debug, Clone)]
struct CuratedRow {
    cid: i64,
    smiles: String,
    pathway_ids: Vec<u16>,
    superclass_ids: Vec<u16>,
    class_ids: Vec<u16>,
    is_glycoside: bool,
    source_part: String,
    source_row: i64,
    signature_id: i64,
    pathway_positive_margin: f32,
    pathway_negative_margin: f32,
    superclass_positive_margin: f32,
    superclass_negative_margin: f32,
    class_positive_margin: f32,
    class_negative_margin: f32,
}

#[derive(Debug, Default)]
struct SplitBatch {
    rows: Vec<CuratedRow>,
    pathway_vectors: Vec<f32>,
    superclass_vectors: Vec<f32>,
    class_vectors: Vec<f32>,
}

#[derive(Clone, Copy)]
struct RowVectors<'a> {
    pathway: &'a [f32],
    superclass: &'a [f32],
    class_: &'a [f32],
}

impl SplitBatch {
    fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

struct SplitWriter {
    rows_writer: ArrowWriter<BufWriter<File>>,
    pathway_writer: zstd::stream::write::Encoder<'static, BufWriter<File>>,
    superclass_writer: zstd::stream::write::Encoder<'static, BufWriter<File>>,
    class_writer: zstd::stream::write::Encoder<'static, BufWriter<File>>,
    widths: VectorWidths,
}

impl SplitWriter {
    fn create(
        out_dir: &Path,
        split: Split,
        widths: VectorWidths,
    ) -> Result<Self, NpClassifierError> {
        fs::create_dir_all(out_dir)?;
        let rows_path = out_dir.join(format!("{}.parquet", split.as_str()));
        let rows_file = BufWriter::new(File::create(rows_path)?);
        let rows_writer = ArrowWriter::try_new(rows_file, curated_schema(), None)
            .map_err(|error| NpClassifierError::Dataset(error.to_string()))?;

        let pathway_writer = zstd::stream::write::Encoder::new(
            BufWriter::new(File::create(
                out_dir.join(Head::Pathway.vectors_filename(split.as_str())),
            )?),
            9,
        )?;
        let superclass_writer = zstd::stream::write::Encoder::new(
            BufWriter::new(File::create(
                out_dir.join(Head::Superclass.vectors_filename(split.as_str())),
            )?),
            9,
        )?;
        let class_writer = zstd::stream::write::Encoder::new(
            BufWriter::new(File::create(
                out_dir.join(Head::Class.vectors_filename(split.as_str())),
            )?),
            9,
        )?;

        Ok(Self {
            rows_writer,
            pathway_writer,
            superclass_writer,
            class_writer,
            widths,
        })
    }

    fn write_batch(&mut self, batch: &SplitBatch) -> Result<(), NpClassifierError> {
        if batch.is_empty() {
            return Ok(());
        }

        let row_count = batch.rows.len();
        if batch.pathway_vectors.len() != row_count * self.widths.pathway
            || batch.superclass_vectors.len() != row_count * self.widths.superclass
            || batch.class_vectors.len() != row_count * self.widths.class_
        {
            return Err(NpClassifierError::Dataset(
                "split vector buffers do not match the buffered row count".to_owned(),
            ));
        }

        let record_batch = build_record_batch(&batch.rows)?;
        self.rows_writer
            .write(&record_batch)
            .map_err(|error| NpClassifierError::Dataset(error.to_string()))?;
        write_f16_slice(&mut self.pathway_writer, &batch.pathway_vectors)?;
        write_f16_slice(&mut self.superclass_writer, &batch.superclass_vectors)?;
        write_f16_slice(&mut self.class_writer, &batch.class_vectors)?;
        Ok(())
    }

    fn close(self) -> Result<(), NpClassifierError> {
        self.rows_writer
            .close()
            .map_err(|error| NpClassifierError::Dataset(error.to_string()))?;
        self.pathway_writer.finish()?;
        self.superclass_writer.finish()?;
        self.class_writer.finish()?;
        Ok(())
    }
}

struct PartDecoders {
    pathway: zstd::stream::read::Decoder<'static, BufReader<File>>,
    superclass: zstd::stream::read::Decoder<'static, BufReader<File>>,
    class_: zstd::stream::read::Decoder<'static, BufReader<File>>,
}

impl PartDecoders {
    fn open(part: &CompletedPart) -> Result<Self, NpClassifierError> {
        Ok(Self {
            pathway: zstd::stream::read::Decoder::new(File::open(&part.pathway_vectors_path)?)?,
            superclass: zstd::stream::read::Decoder::new(File::open(
                &part.superclass_vectors_path,
            )?)?,
            class_: zstd::stream::read::Decoder::new(File::open(&part.class_vectors_path)?)?,
        })
    }
}

struct LabelColumnView<'a> {
    list: &'a arrow_array::ListArray,
    values: &'a PrimitiveArray<UInt16Type>,
}

impl<'a> LabelColumnView<'a> {
    fn is_empty(&self, row: usize) -> bool {
        let offsets = self.list.value_offsets();
        offsets[row] == offsets[row + 1]
    }

    fn values_at(&self, row: usize) -> Result<&'a [u16], NpClassifierError> {
        if self.list.is_null(row) {
            return Err(NpClassifierError::Dataset(format!(
                "unexpected null list in {}",
                self.list.data_type()
            )));
        }
        let offsets = self.list.value_offsets();
        let start = usize::try_from(offsets[row]).map_err(|error| {
            NpClassifierError::Dataset(format!("failed to decode list offset: {error}"))
        })?;
        let end = usize::try_from(offsets[row + 1]).map_err(|error| {
            NpClassifierError::Dataset(format!("failed to decode list offset: {error}"))
        })?;
        Ok(&self.values.values()[start..end])
    }
}

struct BatchView<'a> {
    cids: &'a arrow_array::Int64Array,
    smiles: &'a arrow_array::StringArray,
    pathway_ids: LabelColumnView<'a>,
    superclass_ids: LabelColumnView<'a>,
    class_ids: LabelColumnView<'a>,
    is_glycoside: &'a arrow_array::BooleanArray,
    parse_failed: &'a arrow_array::BooleanArray,
    rdkit_failed: &'a arrow_array::BooleanArray,
    other_failure: &'a arrow_array::BooleanArray,
}

impl<'a> BatchView<'a> {
    fn from_record_batch(batch: &'a RecordBatch) -> Result<Self, NpClassifierError> {
        let pathway_ids = label_column(batch, "pathway_ids")?;
        let superclass_ids = label_column(batch, "superclass_ids")?;
        let class_ids = label_column(batch, "class_ids")?;
        Ok(Self {
            cids: as_primitive_array::<Int64Type>(column(batch, "cid")?),
            smiles: as_string_array(column(batch, "smiles")?),
            pathway_ids,
            superclass_ids,
            class_ids,
            is_glycoside: as_boolean_array(column(batch, "isglycoside")?),
            parse_failed: as_boolean_array(column(batch, "parse_failed")?),
            rdkit_failed: as_boolean_array(column(batch, "rdkit_failed")?),
            other_failure: as_boolean_array(column(batch, "other_failure")?),
        })
    }

    fn row_is_failed(&self, row: usize) -> bool {
        self.parse_failed.value(row)
            || self.rdkit_failed.value(row)
            || self.other_failure.value(row)
    }

    fn row_is_complete(&self, row: usize) -> bool {
        !self.pathway_ids.is_empty(row)
            && !self.superclass_ids.is_empty(row)
            && !self.class_ids.is_empty(row)
    }

    fn signature_key(&self, row: usize) -> Result<SignatureKey, NpClassifierError> {
        Ok(SignatureKey {
            pathway_ids: self.pathway_ids.values_at(row)?.to_vec(),
            superclass_ids: self.superclass_ids.values_at(row)?.to_vec(),
            class_ids: self.class_ids.values_at(row)?.to_vec(),
            is_glycoside: self.is_glycoside.value(row),
        })
    }

    fn build_row(
        &self,
        row: usize,
        part_name: &str,
        source_row: usize,
        signature_id: usize,
        vectors: RowVectors<'_>,
    ) -> Result<CuratedRow, NpClassifierError> {
        let pathway_ids = self.pathway_ids.values_at(row)?;
        let superclass_ids = self.superclass_ids.values_at(row)?;
        let class_ids = self.class_ids.values_at(row)?;
        let (pathway_positive_margin, pathway_negative_margin) =
            head_margins(vectors.pathway, pathway_ids, PATHWAY_THRESHOLD)?;
        let (superclass_positive_margin, superclass_negative_margin) =
            head_margins(vectors.superclass, superclass_ids, SUPERCLASS_THRESHOLD)?;
        let (class_positive_margin, class_negative_margin) =
            head_margins(vectors.class_, class_ids, CLASS_THRESHOLD)?;

        Ok(CuratedRow {
            cid: self.cids.value(row),
            smiles: self.smiles.value(row).to_owned(),
            pathway_ids: pathway_ids.to_vec(),
            superclass_ids: superclass_ids.to_vec(),
            class_ids: class_ids.to_vec(),
            is_glycoside: self.is_glycoside.value(row),
            source_part: part_name.to_owned(),
            source_row: i64::try_from(source_row).map_err(|error| {
                NpClassifierError::Dataset(format!("source row does not fit into i64: {error}"))
            })?,
            signature_id: i64::try_from(signature_id).map_err(|error| {
                NpClassifierError::Dataset(format!("signature id does not fit into i64: {error}"))
            })?,
            pathway_positive_margin,
            pathway_negative_margin,
            superclass_positive_margin,
            superclass_negative_margin,
            class_positive_margin,
            class_negative_margin,
        })
    }
}

/// Discovers completed parts under the configured input directories.
///
/// # Errors
///
/// Returns an error if no parts are found or if any sidecar file is missing.
pub fn discover_parts(completed_dirs: &[PathBuf]) -> Result<Vec<CompletedPart>, NpClassifierError> {
    if completed_dirs.is_empty() {
        return Err(NpClassifierError::Dataset(
            "at least one input directory must be configured".to_owned(),
        ));
    }
    let mut parts = Vec::new();
    for completed_dir in completed_dirs {
        discover_parts_in_dir(completed_dir, completed_dir, &mut parts)?;
    }
    parts.sort_by(|left, right| left.rows_path.cmp(&right.rows_path));
    if parts.is_empty() {
        return Err(NpClassifierError::Dataset(format!(
            "no completed parts found under {}",
            completed_dirs
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    Ok(parts)
}

fn discover_parts_in_dir(
    input_root: &Path,
    current_dir: &Path,
    parts: &mut Vec<CompletedPart>,
) -> Result<(), NpClassifierError> {
    for entry in fs::read_dir(current_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            discover_parts_in_dir(input_root, &path, parts)?;
            continue;
        }
        if path.extension().and_then(std::ffi::OsStr::to_str) != Some("parquet") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(std::ffi::OsStr::to_str) else {
            continue;
        };
        if !stem.starts_with("part-") {
            continue;
        }
        let parent_dir = path.parent().ok_or_else(|| {
            NpClassifierError::Dataset(format!("missing parent directory for {}", path.display()))
        })?;
        let relative_stem = path
            .strip_prefix(input_root)
            .unwrap_or(&path)
            .with_extension("")
            .display()
            .to_string();
        let part_name = format!("{}::{}", input_root.display(), relative_stem);
        let part = CompletedPart {
            part_name,
            rows_path: path.clone(),
            pathway_vectors_path: parent_dir.join(format!("{stem}.pathway-vectors.f16.zst")),
            superclass_vectors_path: parent_dir.join(format!("{stem}.superclass-vectors.f16.zst")),
            class_vectors_path: parent_dir.join(format!("{stem}.class-vectors.f16.zst")),
        };
        for sidecar in [
            &part.pathway_vectors_path,
            &part.superclass_vectors_path,
            &part.class_vectors_path,
        ] {
            if !sidecar.exists() {
                return Err(NpClassifierError::Dataset(format!(
                    "missing vector sidecar {}",
                    sidecar.display()
                )));
            }
        }
        parts.push(part);
    }
    Ok(())
}

/// Loads vector widths from the sibling `vocabulary.json`.
///
/// # Errors
///
/// Returns an error if the vocabulary file cannot be read or decoded.
pub fn load_widths(vocabulary_path: &Path) -> Result<VectorWidths, NpClassifierError> {
    let vocabulary: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(vocabulary_path)?)?;
    let pathway = vocabulary
        .get("pathway")
        .and_then(serde_json::Value::as_array)
        .map(Vec::len)
        .ok_or_else(|| NpClassifierError::Dataset("missing pathway vocabulary".to_owned()))?;
    let superclass = vocabulary
        .get("superclass")
        .and_then(serde_json::Value::as_array)
        .map(Vec::len)
        .ok_or_else(|| NpClassifierError::Dataset("missing superclass vocabulary".to_owned()))?;
    let class_ = vocabulary
        .get("class")
        .and_then(serde_json::Value::as_array)
        .map(Vec::len)
        .ok_or_else(|| NpClassifierError::Dataset("missing class vocabulary".to_owned()))?;
    Ok(VectorWidths {
        pathway,
        superclass,
        class_,
    })
}

/// Runs the filtering and split-assignment pass without materializing outputs.
///
/// # Errors
///
/// Returns an error if the source dataset cannot be streamed.
pub fn summarize_completed(
    config: &CurationConfig,
) -> Result<DistillationReport, NpClassifierError> {
    let parts = discover_parts(&config.input_dirs)?;
    let widths = load_widths(&config.vocabulary_path)?;
    let scan_progress = rows_progress(
        "scan teacher rows",
        total_rows_to_scan(&parts, config.max_rows)?,
    );
    let (accepted, signature_counts, selection_summary) =
        collect_metadata(&parts, widths, config, &scan_progress)?;
    scan_progress.finish_with_message(format!(
        "scanned {} rows, kept {}",
        selection_summary.scanned_rows, selection_summary.kept_rows
    ));
    let (_, signature_summary, split_row_counts) =
        build_split_assignments(&accepted, &signature_counts, config)?;
    Ok(DistillationReport {
        input_dirs: config
            .input_dirs
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        max_rows: config.max_rows,
        margins: config.margins,
        selection_summary,
        signature_summary,
        split_row_counts,
    })
}

/// Runs the full curation pass and writes train / validation / test outputs.
///
/// # Errors
///
/// Returns an error if the source dataset cannot be read or the outputs cannot
/// be materialized.
pub fn curate_completed(config: &CurationConfig) -> Result<DistillationReport, NpClassifierError> {
    let parts = discover_parts(&config.input_dirs)?;
    let widths = load_widths(&config.vocabulary_path)?;
    let scan_progress = rows_progress(
        "scan teacher rows",
        total_rows_to_scan(&parts, config.max_rows)?,
    );
    let (accepted, signature_counts, selection_summary) =
        collect_metadata(&parts, widths, config, &scan_progress)?;
    scan_progress.finish_with_message(format!(
        "scanned {} rows, kept {}",
        selection_summary.scanned_rows, selection_summary.kept_rows
    ));
    let (assignments, signature_summary, split_row_counts) =
        build_split_assignments(&accepted, &signature_counts, config)?;
    materialize_splits(&parts, widths, &assignments, config)?;
    write_manifest(
        widths,
        config,
        selection_summary,
        signature_summary,
        split_row_counts,
    )?;
    Ok(DistillationReport {
        input_dirs: config
            .input_dirs
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        max_rows: config.max_rows,
        margins: config.margins,
        selection_summary,
        signature_summary,
        split_row_counts,
    })
}

fn collect_metadata(
    parts: &[CompletedPart],
    widths: VectorWidths,
    config: &CurationConfig,
    progress: &ProgressBar,
) -> Result<(Vec<AcceptedRow>, Vec<usize>, SelectionSummary), NpClassifierError> {
    let mut accepted = Vec::new();
    let mut summary = SelectionSummary::default();
    let mut signature_ids = HashMap::<SignatureKey, usize>::new();
    let mut signature_counts = Vec::<usize>::new();
    let mut seen_rows = 0_usize;

    for (part_index, part) in parts.iter().enumerate() {
        if config
            .max_rows
            .is_some_and(|max_rows| seen_rows >= max_rows)
        {
            break;
        }
        let part_limit = config
            .max_rows
            .map(|max_rows| max_rows.saturating_sub(seen_rows));
        progress.set_message(format!("scan {}", part.part_name));
        let rows_read = scan_part(
            part,
            widths,
            config.batch_rows,
            part_limit,
            Some(progress),
            |batch, matrices, row_offset| {
                let evaluations = (0..batch.cids.len())
                    .into_par_iter()
                    .map(|row| evaluate_row(&batch, matrices, row, widths, config.margins))
                    .collect::<Vec<_>>();
                for (row, evaluation) in evaluations.into_iter().enumerate() {
                    summary.scanned_rows += 1;
                    match evaluation? {
                        RowEvaluation::Failed => summary.failed_rows += 1,
                        RowEvaluation::Partial => summary.partial_rows += 1,
                        RowEvaluation::Nonfinite => summary.nonfinite_vector_rows += 1,
                        RowEvaluation::Indecisive => summary.indecisive_rows += 1,
                        RowEvaluation::Accepted(signature) => {
                            let signature_id =
                                if let Some(signature_id) = signature_ids.get(&signature) {
                                    *signature_id
                                } else {
                                    let next_id = signature_counts.len();
                                    signature_ids.insert(signature, next_id);
                                    signature_counts.push(0);
                                    next_id
                                };
                            signature_counts[signature_id] += 1;
                            accepted.push(AcceptedRow {
                                part_index,
                                row_index: row_offset + row,
                                signature_id,
                            });
                            summary.kept_rows += 1;
                        }
                    }
                }
                Ok(())
            },
        )?;
        seen_rows += rows_read;
    }

    Ok((accepted, signature_counts, summary))
}

fn evaluate_row(
    batch: &BatchView<'_>,
    matrices: &DecodedMatrices,
    row: usize,
    widths: VectorWidths,
    margins: ConfidenceMargins,
) -> Result<RowEvaluation, NpClassifierError> {
    if batch.row_is_failed(row) {
        return Ok(RowEvaluation::Failed);
    }
    if !batch.row_is_complete(row) {
        return Ok(RowEvaluation::Partial);
    }

    let pathway_vector = row_slice(&matrices.pathway, row, widths.pathway);
    let superclass_vector = row_slice(&matrices.superclass, row, widths.superclass);
    let class_vector = row_slice(&matrices.class_, row, widths.class_);
    if !row_vectors_are_finite(pathway_vector, superclass_vector, class_vector) {
        return Ok(RowEvaluation::Nonfinite);
    }
    if !row_passes_confidence(
        batch,
        row,
        pathway_vector,
        superclass_vector,
        class_vector,
        margins,
    )? {
        return Ok(RowEvaluation::Indecisive);
    }

    Ok(RowEvaluation::Accepted(batch.signature_key(row)?))
}

fn scan_part<F>(
    part: &CompletedPart,
    widths: VectorWidths,
    batch_rows: usize,
    max_rows: Option<usize>,
    progress: Option<&ProgressBar>,
    mut visit: F,
) -> Result<usize, NpClassifierError>
where
    F: FnMut(BatchView<'_>, &DecodedMatrices, usize) -> Result<(), NpClassifierError>,
{
    let file = File::open(&part.rows_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|error| NpClassifierError::Dataset(error.to_string()))?;
    let mut reader = builder
        .with_batch_size(batch_rows)
        .build()
        .map_err(|error| NpClassifierError::Dataset(error.to_string()))?;
    let mut decoders = PartDecoders::open(part)?;
    let mut seen_rows = 0_usize;

    for batch in &mut reader {
        let batch = batch.map_err(|error| NpClassifierError::Dataset(error.to_string()))?;
        let count = batch.num_rows();
        let remaining = max_rows.map(|limit| limit.saturating_sub(seen_rows));
        if remaining == Some(0) {
            break;
        }
        let usable_rows = remaining.map_or(count, |remaining| remaining.min(count));
        let batch = if usable_rows == count {
            batch
        } else {
            batch.slice(0, usable_rows)
        };
        let batch_view = BatchView::from_record_batch(&batch)?;
        let matrices = read_batch_matrices(&mut decoders, widths, usable_rows)?;
        visit(batch_view, &matrices, seen_rows)?;
        if let Some(progress) = progress {
            progress.inc(usize_to_u64(usable_rows, "progress increment")?);
        }
        seen_rows += usable_rows;
        if usable_rows < count {
            break;
        }
    }

    Ok(seen_rows)
}

fn build_split_assignments(
    accepted: &[AcceptedRow],
    signature_counts: &[usize],
    config: &CurationConfig,
) -> Result<(Vec<Vec<Assignment>>, SignatureSummary, SplitRowCounts), NpClassifierError> {
    if accepted.is_empty() {
        return Err(NpClassifierError::Dataset(
            "no rows survived filtering".to_owned(),
        ));
    }

    let fractions = config.split_fractions.normalized()?;
    let mut by_signature = vec![Vec::<usize>::new(); signature_counts.len()];
    for (accepted_index, row) in accepted.iter().enumerate() {
        by_signature[row.signature_id].push(accepted_index);
    }

    let mut assignments_by_part = vec![
        Vec::<Assignment>::new();
        accepted.iter().map(|row| row.part_index).max().unwrap_or(0)
            + 1
    ];
    let mut split_row_counts = SplitRowCounts::default();
    let mut signature_summary = SignatureSummary {
        unique_signatures: u64::try_from(signature_counts.len()).map_err(|error| {
            NpClassifierError::Dataset(format!("signature count does not fit into u64: {error}"))
        })?,
        ..SignatureSummary::default()
    };
    let mut rng = StdRng::seed_from_u64(config.seed);
    let assignment_progress = steps_progress(
        "assign signatures",
        usize_to_u64(signature_counts.len(), "signature count")?,
    );

    for (signature_id, accepted_indices) in by_signature.iter_mut().enumerate() {
        let support = signature_counts[signature_id];
        if support < config.min_signature_count {
            signature_summary.rare_signatures += 1;
            signature_summary.rare_signature_rows += u64::try_from(support).map_err(|error| {
                NpClassifierError::Dataset(format!("row count does not fit into u64: {error}"))
            })?;
            for accepted_index in accepted_indices.iter().copied() {
                let accepted_row = accepted[accepted_index];
                assignments_by_part[accepted_row.part_index].push(Assignment {
                    row_index: accepted_row.row_index,
                    split: Split::Train,
                    signature_id,
                });
                split_row_counts.increment(Split::Train);
            }
            continue;
        }

        signature_summary.eligible_signatures += 1;
        accepted_indices.shuffle(&mut rng);
        let [train_count, validation_count, test_count] =
            assign_group_counts(accepted_indices.len(), fractions);
        for (position, accepted_index) in accepted_indices.iter().copied().enumerate() {
            let split = if position < train_count {
                Split::Train
            } else if position < train_count + validation_count {
                Split::Validation
            } else {
                debug_assert!(position < train_count + validation_count + test_count);
                Split::Test
            };
            let accepted_row = accepted[accepted_index];
            assignments_by_part[accepted_row.part_index].push(Assignment {
                row_index: accepted_row.row_index,
                split,
                signature_id,
            });
            split_row_counts.increment(split);
        }
        assignment_progress.inc(1);
    }

    for assignments in &mut assignments_by_part {
        assignments.sort_unstable_by_key(|assignment| assignment.row_index);
    }
    assignment_progress
        .finish_with_message(format!("assigned {} signatures", signature_counts.len()));

    Ok((assignments_by_part, signature_summary, split_row_counts))
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn assign_group_counts(group_size: usize, fractions: SplitFractions) -> [usize; 3] {
    let raw_counts = [
        group_size as f64 * fractions.train,
        group_size as f64 * fractions.validation,
        group_size as f64 * fractions.test,
    ];
    let mut counts = raw_counts.map(f64::floor).map(|value| value as usize);
    let assigned = counts.iter().sum::<usize>();
    let mut remainders = [
        (0_usize, raw_counts[0] - counts[0] as f64),
        (1_usize, raw_counts[1] - counts[1] as f64),
        (2_usize, raw_counts[2] - counts[2] as f64),
    ];
    remainders.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    for index in 0..group_size.saturating_sub(assigned) {
        counts[remainders[index % remainders.len()].0] += 1;
    }
    counts
}

fn materialize_splits(
    parts: &[CompletedPart],
    widths: VectorWidths,
    assignments_by_part: &[Vec<Assignment>],
    config: &CurationConfig,
) -> Result<(), NpClassifierError> {
    if config.output_dir.exists() {
        fs::remove_dir_all(&config.output_dir)?;
    }
    fs::create_dir_all(&config.output_dir)?;

    let mut train_writer = SplitWriter::create(&config.output_dir, Split::Train, widths)?;
    let mut validation_writer = SplitWriter::create(&config.output_dir, Split::Validation, widths)?;
    let mut test_writer = SplitWriter::create(&config.output_dir, Split::Test, widths)?;
    let mut seen_rows = 0_usize;
    let total_assigned_rows = assignments_by_part.iter().map(Vec::len).sum::<usize>();
    let materialize_progress = rows_progress(
        "write curated splits",
        usize_to_u64(total_assigned_rows, "assigned rows")?,
    );

    for (part_index, part) in parts.iter().enumerate() {
        if config
            .max_rows
            .is_some_and(|max_rows| seen_rows >= max_rows)
        {
            break;
        }
        let part_limit = config
            .max_rows
            .map(|max_rows| max_rows.saturating_sub(seen_rows));
        let assignments = &assignments_by_part[part_index];
        let mut pointer = 0_usize;

        materialize_progress.set_message(format!("write {}", part.part_name));
        let rows_read = scan_part(
            part,
            widths,
            config.batch_rows,
            part_limit,
            None,
            |batch, matrices, row_offset| {
                let batch_end = row_offset + batch.cids.len();
                let mut train_batch = SplitBatch::default();
                let mut validation_batch = SplitBatch::default();
                let mut test_batch = SplitBatch::default();
                let mut written_rows = 0_usize;

                while pointer < assignments.len() && assignments[pointer].row_index < batch_end {
                    let assignment = assignments[pointer];
                    let local_row = assignment.row_index - row_offset;
                    let pathway_vector = row_slice(&matrices.pathway, local_row, widths.pathway);
                    let superclass_vector =
                        row_slice(&matrices.superclass, local_row, widths.superclass);
                    let class_vector = row_slice(&matrices.class_, local_row, widths.class_);
                    let row = batch.build_row(
                        local_row,
                        &part.part_name,
                        assignment.row_index,
                        assignment.signature_id,
                        RowVectors {
                            pathway: pathway_vector,
                            superclass: superclass_vector,
                            class_: class_vector,
                        },
                    )?;
                    let target = match assignment.split {
                        Split::Train => &mut train_batch,
                        Split::Validation => &mut validation_batch,
                        Split::Test => &mut test_batch,
                    };
                    target.rows.push(row);
                    target.pathway_vectors.extend_from_slice(pathway_vector);
                    target
                        .superclass_vectors
                        .extend_from_slice(superclass_vector);
                    target.class_vectors.extend_from_slice(class_vector);
                    pointer += 1;
                    written_rows += 1;
                }

                train_writer.write_batch(&train_batch)?;
                validation_writer.write_batch(&validation_batch)?;
                test_writer.write_batch(&test_batch)?;
                materialize_progress.inc(usize_to_u64(written_rows, "materialization increment")?);
                Ok(())
            },
        )?;
        seen_rows += rows_read;
    }

    train_writer.close()?;
    validation_writer.close()?;
    test_writer.close()?;
    materialize_progress.finish_with_message(format!("wrote {total_assigned_rows} curated rows"));
    Ok(())
}

fn write_manifest(
    widths: VectorWidths,
    config: &CurationConfig,
    selection_summary: SelectionSummary,
    signature_summary: SignatureSummary,
    split_row_counts: SplitRowCounts,
) -> Result<(), NpClassifierError> {
    let vocabulary_name = config
        .vocabulary_path
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .ok_or_else(|| {
            NpClassifierError::Dataset(format!(
                "invalid vocabulary filename {}",
                config.vocabulary_path.display()
            ))
        })?
        .to_owned();
    fs::copy(
        &config.vocabulary_path,
        config.output_dir.join(&vocabulary_name),
    )?;

    let mut thresholds = BTreeMap::new();
    thresholds.insert("pathway".to_owned(), PATHWAY_THRESHOLD);
    thresholds.insert("superclass".to_owned(), SUPERCLASS_THRESHOLD);
    thresholds.insert("class".to_owned(), CLASS_THRESHOLD);

    let mut outputs = BTreeMap::new();
    for split in [Split::Train, Split::Validation, Split::Test] {
        let split_name = split.as_str().to_owned();
        let mut files = BTreeMap::new();
        files.insert("rows".to_owned(), format!("{split_name}.parquet"));
        for head in Head::ALL {
            files.insert(
                format!("{}_vectors", head.as_str()),
                head.vectors_filename(&split_name),
            );
        }
        outputs.insert(split_name, files);
    }

    let manifest = CuratedManifest {
        created_at: chrono_like_timestamp(),
        input_dirs: config
            .input_dirs
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        vocabulary: vocabulary_name,
        vector_widths: widths,
        thresholds,
        margins: config.margins,
        split_fractions: config.split_fractions,
        min_signature_count: config.min_signature_count,
        seed: config.seed,
        batch_rows: config.batch_rows,
        max_rows: config.max_rows,
        selection_summary,
        signature_summary,
        split_row_counts,
        outputs,
    };
    fs::write(
        config.output_dir.join("manifest.json"),
        format!("{}\n", serde_json::to_string_pretty(&manifest)?),
    )?;
    Ok(())
}

fn row_passes_confidence(
    batch: &BatchView<'_>,
    row: usize,
    pathway_vector: &[f32],
    superclass_vector: &[f32],
    class_vector: &[f32],
    margins: ConfidenceMargins,
) -> Result<bool, NpClassifierError> {
    let pathway_ids = batch.pathway_ids.values_at(row)?;
    let superclass_ids = batch.superclass_ids.values_at(row)?;
    let class_ids = batch.class_ids.values_at(row)?;

    let heads = [
        (Head::Pathway, pathway_vector, pathway_ids),
        (Head::Superclass, superclass_vector, superclass_ids),
        (Head::Class, class_vector, class_ids),
    ];
    for (head, vector, labels) in heads {
        let (positive_margin, negative_margin) = head_margins(vector, labels, head.threshold())?;
        if positive_margin < head.positive_margin(margins)
            || negative_margin < head.negative_margin(margins)
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn head_margins(
    vector: &[f32],
    label_ids: &[u16],
    threshold: f32,
) -> Result<(f32, f32), NpClassifierError> {
    if label_ids.is_empty() {
        return Err(NpClassifierError::Dataset(
            "cannot compute confidence margins for an empty label set".to_owned(),
        ));
    }

    let mut positive_margin = f32::INFINITY;
    let mut negative_margin = f32::INFINITY;
    for (index, value) in vector.iter().copied().enumerate() {
        let is_positive = label_ids
            .iter()
            .any(|label_id| usize::from(*label_id) == index);
        if is_positive {
            positive_margin = positive_margin.min(value - threshold);
        } else {
            negative_margin = negative_margin.min(threshold - value);
        }
    }
    if negative_margin == f32::INFINITY {
        negative_margin = f32::INFINITY;
    }
    Ok((positive_margin, negative_margin))
}

fn row_vectors_are_finite(
    pathway_vector: &[f32],
    superclass_vector: &[f32],
    class_vector: &[f32],
) -> bool {
    pathway_vector.iter().all(|value| value.is_finite())
        && superclass_vector.iter().all(|value| value.is_finite())
        && class_vector.iter().all(|value| value.is_finite())
}

fn row_slice(matrix: &[f32], row: usize, width: usize) -> &[f32] {
    let start = row * width;
    let end = start + width;
    &matrix[start..end]
}

struct DecodedMatrices {
    pathway: Vec<f32>,
    superclass: Vec<f32>,
    class_: Vec<f32>,
}

fn read_batch_matrices(
    decoders: &mut PartDecoders,
    widths: VectorWidths,
    rows: usize,
) -> Result<DecodedMatrices, NpClassifierError> {
    Ok(DecodedMatrices {
        pathway: read_f16_matrix(&mut decoders.pathway, rows, widths.pathway)?,
        superclass: read_f16_matrix(&mut decoders.superclass, rows, widths.superclass)?,
        class_: read_f16_matrix(&mut decoders.class_, rows, widths.class_)?,
    })
}

fn read_f16_matrix<R: Read>(
    reader: &mut R,
    rows: usize,
    width: usize,
) -> Result<Vec<f32>, NpClassifierError> {
    let expected_bytes = rows
        .checked_mul(width)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| NpClassifierError::Dataset("vector batch size overflowed".to_owned()))?;
    let mut bytes = vec![0_u8; expected_bytes];
    reader.read_exact(&mut bytes)?;
    let mut values = Vec::with_capacity(rows * width);
    for chunk in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        values.push(f16::from_bits(bits).to_f32());
    }
    Ok(values)
}

fn write_f16_slice<W: Write>(writer: &mut W, values: &[f32]) -> Result<(), NpClassifierError> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        bytes.extend_from_slice(&f16::from_f32(*value).to_bits().to_le_bytes());
    }
    writer.write_all(&bytes)?;
    Ok(())
}

fn curated_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("cid", DataType::Int64, false),
        Field::new("smiles", DataType::Utf8, false),
        Field::new(
            "pathway_ids",
            DataType::List(Arc::new(Field::new("item", DataType::UInt16, true))),
            false,
        ),
        Field::new(
            "superclass_ids",
            DataType::List(Arc::new(Field::new("item", DataType::UInt16, true))),
            false,
        ),
        Field::new(
            "class_ids",
            DataType::List(Arc::new(Field::new("item", DataType::UInt16, true))),
            false,
        ),
        Field::new("isglycoside", DataType::Boolean, false),
        Field::new("source_part", DataType::Utf8, false),
        Field::new("source_row", DataType::Int64, false),
        Field::new("signature_id", DataType::Int64, false),
        Field::new("pathway_positive_margin", DataType::Float32, false),
        Field::new("pathway_negative_margin", DataType::Float32, false),
        Field::new("superclass_positive_margin", DataType::Float32, false),
        Field::new("superclass_negative_margin", DataType::Float32, false),
        Field::new("class_positive_margin", DataType::Float32, false),
        Field::new("class_negative_margin", DataType::Float32, false),
    ]))
}

fn build_record_batch(rows: &[CuratedRow]) -> Result<RecordBatch, NpClassifierError> {
    let mut cid_builder = Int64Builder::new();
    let mut smiles_builder = StringBuilder::new();
    let mut pathway_builder = ListBuilder::new(UInt16Builder::new());
    let mut superclass_builder = ListBuilder::new(UInt16Builder::new());
    let mut class_builder = ListBuilder::new(UInt16Builder::new());
    let mut glycoside_builder = BooleanBuilder::new();
    let mut source_part_builder = StringBuilder::new();
    let mut source_row_builder = Int64Builder::new();
    let mut signature_id_builder = Int64Builder::new();
    let mut pathway_positive_builder = Float32Builder::new();
    let mut pathway_negative_builder = Float32Builder::new();
    let mut superclass_positive_builder = Float32Builder::new();
    let mut superclass_negative_builder = Float32Builder::new();
    let mut class_positive_builder = Float32Builder::new();
    let mut class_negative_builder = Float32Builder::new();

    for row in rows {
        cid_builder.append_value(row.cid);
        smiles_builder.append_value(&row.smiles);
        append_label_values(&mut pathway_builder, &row.pathway_ids);
        append_label_values(&mut superclass_builder, &row.superclass_ids);
        append_label_values(&mut class_builder, &row.class_ids);
        glycoside_builder.append_value(row.is_glycoside);
        source_part_builder.append_value(&row.source_part);
        source_row_builder.append_value(row.source_row);
        signature_id_builder.append_value(row.signature_id);
        pathway_positive_builder.append_value(row.pathway_positive_margin);
        pathway_negative_builder.append_value(row.pathway_negative_margin);
        superclass_positive_builder.append_value(row.superclass_positive_margin);
        superclass_negative_builder.append_value(row.superclass_negative_margin);
        class_positive_builder.append_value(row.class_positive_margin);
        class_negative_builder.append_value(row.class_negative_margin);
    }

    RecordBatch::try_new(
        curated_schema(),
        vec![
            Arc::new(cid_builder.finish()) as ArrayRef,
            Arc::new(smiles_builder.finish()) as ArrayRef,
            Arc::new(pathway_builder.finish()) as ArrayRef,
            Arc::new(superclass_builder.finish()) as ArrayRef,
            Arc::new(class_builder.finish()) as ArrayRef,
            Arc::new(glycoside_builder.finish()) as ArrayRef,
            Arc::new(source_part_builder.finish()) as ArrayRef,
            Arc::new(source_row_builder.finish()) as ArrayRef,
            Arc::new(signature_id_builder.finish()) as ArrayRef,
            Arc::new(pathway_positive_builder.finish()) as ArrayRef,
            Arc::new(pathway_negative_builder.finish()) as ArrayRef,
            Arc::new(superclass_positive_builder.finish()) as ArrayRef,
            Arc::new(superclass_negative_builder.finish()) as ArrayRef,
            Arc::new(class_positive_builder.finish()) as ArrayRef,
            Arc::new(class_negative_builder.finish()) as ArrayRef,
        ],
    )
    .map_err(|error| NpClassifierError::Dataset(error.to_string()))
}

fn append_label_values(builder: &mut ListBuilder<UInt16Builder>, values: &[u16]) {
    builder.values().append_slice(values);
    builder.append(true);
}

fn column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a dyn Array, NpClassifierError> {
    batch
        .column_by_name(name)
        .map(std::convert::AsRef::as_ref)
        .ok_or_else(|| NpClassifierError::Dataset(format!("missing parquet column {name}")))
}

fn label_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<LabelColumnView<'a>, NpClassifierError> {
    let list = as_list_array(column(batch, name)?);
    let values = list
        .values()
        .as_any()
        .downcast_ref::<UInt16Array>()
        .ok_or_else(|| {
            NpClassifierError::Dataset(format!("list column {name} did not contain uint16 values"))
        })?;
    Ok(LabelColumnView { list, values })
}

fn chrono_like_timestamp() -> String {
    let now = std::time::SystemTime::now();
    let seconds = now
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default();
    format!("{seconds}Z")
}

fn total_rows_to_scan(
    parts: &[CompletedPart],
    max_rows: Option<usize>,
) -> Result<u64, NpClassifierError> {
    let mut total = 0_usize;
    for part in parts {
        total = total.saturating_add(part_row_count(part)?);
        if max_rows.is_some_and(|limit| total >= limit) {
            total = max_rows.unwrap_or(total);
            break;
        }
    }

    usize_to_u64(total, "planned row count")
}

fn part_row_count(part: &CompletedPart) -> Result<usize, NpClassifierError> {
    let file = File::open(&part.rows_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|error| NpClassifierError::Dataset(error.to_string()))?;
    usize::try_from(builder.metadata().file_metadata().num_rows()).map_err(|error| {
        NpClassifierError::Dataset(format!(
            "row count for {} does not fit into usize: {error}",
            part.rows_path.display()
        ))
    })
}

fn rows_progress(label: &str, total: u64) -> ProgressBar {
    let progress = ProgressBar::new(total);
    progress.set_style(
        ProgressStyle::with_template(
            "{spinner:.cyan} {msg:<24} [{elapsed_precise}] {wide_bar:.cyan/blue} {human_pos}/{human_len} ({eta})",
        )
        .expect("valid progress bar template")
        .progress_chars("=>-"),
    );
    progress.set_message(label.to_owned());
    progress.enable_steady_tick(PROGRESS_TICK_INTERVAL);
    progress
}

fn steps_progress(label: &str, total: u64) -> ProgressBar {
    let progress = ProgressBar::new(total);
    progress.set_style(
        ProgressStyle::with_template(
            "{spinner:.cyan} {msg:<24} [{elapsed_precise}] {wide_bar:.cyan/blue} {pos}/{len} ({eta})",
        )
        .expect("valid progress bar template")
        .progress_chars("=>-"),
    );
    progress.set_message(label.to_owned());
    progress.enable_steady_tick(PROGRESS_TICK_INTERVAL);
    progress
}

fn usize_to_u64(value: usize, context: &str) -> Result<u64, NpClassifierError> {
    u64::try_from(value).map_err(|error| {
        NpClassifierError::Dataset(format!("{context} does not fit into u64: {error}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufReader;

    use tempfile::tempdir;

    #[test]
    fn assign_group_counts_preserves_size() {
        let counts = assign_group_counts(11, SplitFractions::default());
        assert_eq!(counts.iter().sum::<usize>(), 11);
        assert_eq!(counts, [9, 1, 1]);
    }

    #[test]
    fn build_split_assignments_sorts_rows_within_each_part() {
        let accepted = vec![
            AcceptedRow {
                part_index: 0,
                row_index: 0,
                signature_id: 0,
            },
            AcceptedRow {
                part_index: 0,
                row_index: 1,
                signature_id: 1,
            },
            AcceptedRow {
                part_index: 0,
                row_index: 2,
                signature_id: 0,
            },
            AcceptedRow {
                part_index: 0,
                row_index: 3,
                signature_id: 1,
            },
        ];
        let signature_counts = vec![2, 2];
        let config = CurationConfig {
            split_fractions: SplitFractions {
                train: 1.0,
                validation: 0.0,
                test: 0.0,
            },
            min_signature_count: 1,
            seed: 0,
            ..CurationConfig::default()
        };

        let (assignments, _, split_counts) =
            build_split_assignments(&accepted, &signature_counts, &config).expect("assignments");
        assert_eq!(split_counts.train, 4);
        assert_eq!(
            assignments[0]
                .iter()
                .map(|assignment| assignment.row_index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn discover_parts_keeps_duplicate_stems_unique_across_input_roots() {
        let tempdir = tempdir().expect("tempdir");
        let left = tempdir.path().join("left");
        let right = tempdir.path().join("right");
        fs::create_dir_all(&left).expect("left dir");
        fs::create_dir_all(&right).expect("right dir");

        for dir in [&left, &right] {
            fs::write(dir.join("part-000001.parquet"), b"").expect("rows");
            fs::write(dir.join("part-000001.pathway-vectors.f16.zst"), b"").expect("pathway");
            fs::write(dir.join("part-000001.superclass-vectors.f16.zst"), b"").expect("superclass");
            fs::write(dir.join("part-000001.class-vectors.f16.zst"), b"").expect("class");
        }

        let parts = discover_parts(&[left.clone(), right.clone()]).expect("parts");
        assert_eq!(parts.len(), 2);
        assert_ne!(parts[0].part_name, parts[1].part_name);
        assert!(parts[0].part_name.starts_with(&left.display().to_string()));
        assert!(parts[1].part_name.starts_with(&right.display().to_string()));
    }

    #[test]
    fn head_margins_measure_nearest_positive_and_negative() {
        let vector = [0.8_f32, 0.1, 0.02, 0.95];
        let labels = [0_u16, 3_u16];
        let (positive, negative) = head_margins(&vector, &labels, 0.5).expect("margins");
        assert!((positive - 0.3).abs() < 1e-6);
        assert!((negative - 0.4).abs() < 1e-6);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn summarize_and_curate_small_dataset() {
        let tempdir = tempdir().expect("tempdir");
        let input_dir = tempdir.path().join("input");
        let output_dir = tempdir.path().join("output");
        fs::create_dir_all(&input_dir).expect("input dir");

        let vocabulary_path = input_dir.join("vocabulary.json");
        fs::write(
            &vocabulary_path,
            r#"{"pathway":["p0","p1"],"superclass":["s0","s1"],"class":["c0","c1","c2"]}"#,
        )
        .expect("vocabulary");

        let rows_path = input_dir.join("part-000001.parquet");
        let pathway_path = input_dir.join("part-000001.pathway-vectors.f16.zst");
        let superclass_path = input_dir.join("part-000001.superclass-vectors.f16.zst");
        let class_path = input_dir.join("part-000001.class-vectors.f16.zst");

        let schema = Arc::new(Schema::new(vec![
            Field::new("cid", DataType::Int64, false),
            Field::new("smiles", DataType::Utf8, false),
            Field::new(
                "pathway_ids",
                DataType::List(Arc::new(Field::new("item", DataType::UInt16, true))),
                false,
            ),
            Field::new(
                "superclass_ids",
                DataType::List(Arc::new(Field::new("item", DataType::UInt16, true))),
                false,
            ),
            Field::new(
                "class_ids",
                DataType::List(Arc::new(Field::new("item", DataType::UInt16, true))),
                false,
            ),
            Field::new("isglycoside", DataType::Boolean, false),
            Field::new("parse_failed", DataType::Boolean, false),
            Field::new("rdkit_failed", DataType::Boolean, false),
            Field::new("other_failure", DataType::Boolean, false),
            Field::new("error_message", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow_array::Int64Array::from(vec![1_i64, 2, 3, 4])) as ArrayRef,
                Arc::new(arrow_array::StringArray::from(vec!["A", "B", "C", "D"])) as ArrayRef,
                Arc::new(arrow_array::ListArray::from_iter_primitive::<
                    UInt16Type,
                    _,
                    _,
                >(vec![
                    Some(vec![Some(0_u16)]),
                    Some(vec![Some(0_u16)]),
                    Some(Vec::<Option<u16>>::new()),
                    Some(vec![Some(0_u16)]),
                ])) as ArrayRef,
                Arc::new(arrow_array::ListArray::from_iter_primitive::<
                    UInt16Type,
                    _,
                    _,
                >(vec![
                    Some(vec![Some(0_u16)]),
                    Some(vec![Some(1_u16)]),
                    Some(vec![Some(0_u16)]),
                    Some(vec![Some(0_u16)]),
                ])) as ArrayRef,
                Arc::new(arrow_array::ListArray::from_iter_primitive::<
                    UInt16Type,
                    _,
                    _,
                >(vec![
                    Some(vec![Some(1_u16)]),
                    Some(vec![Some(2_u16)]),
                    Some(vec![Some(1_u16)]),
                    Some(vec![Some(2_u16)]),
                ])) as ArrayRef,
                Arc::new(arrow_array::BooleanArray::from(vec![
                    false, true, false, false,
                ])) as ArrayRef,
                Arc::new(arrow_array::BooleanArray::from(vec![
                    false, false, false, true,
                ])) as ArrayRef,
                Arc::new(arrow_array::BooleanArray::from(vec![
                    false, false, false, false,
                ])) as ArrayRef,
                Arc::new(arrow_array::BooleanArray::from(vec![
                    false, false, false, false,
                ])) as ArrayRef,
                Arc::new(arrow_array::StringArray::from(vec!["", "", "", "boom"])) as ArrayRef,
            ],
        )
        .expect("record batch");
        let rows_file = BufWriter::new(File::create(&rows_path).expect("rows file"));
        let mut rows_writer =
            ArrowWriter::try_new(rows_file, batch.schema(), None).expect("writer");
        rows_writer.write(&batch).expect("write parquet");
        rows_writer.close().expect("close parquet");

        let pathway_vectors = [0.9_f32, 0.1, 0.95, 0.05, 0.91, 0.09, 0.92, 0.08];
        let superclass_vectors = [0.9_f32, 0.1, 0.05, 0.95, 0.8, 0.2, 0.7, 0.3];
        let class_vectors = [
            0.01_f32, 0.95, 0.02, 0.01, 0.02, 0.95, 0.01, 0.99, 0.0, 0.01, 0.02, 0.97,
        ];
        {
            let mut writer = zstd::stream::write::Encoder::new(
                BufWriter::new(File::create(&pathway_path).expect("pathway file")),
                9,
            )
            .expect("pathway encoder");
            write_f16_slice(&mut writer, &pathway_vectors).expect("pathway vectors");
            writer.finish().expect("pathway finish");
        }
        {
            let mut writer = zstd::stream::write::Encoder::new(
                BufWriter::new(File::create(&superclass_path).expect("superclass file")),
                9,
            )
            .expect("superclass encoder");
            write_f16_slice(&mut writer, &superclass_vectors).expect("superclass vectors");
            writer.finish().expect("superclass finish");
        }
        {
            let mut writer = zstd::stream::write::Encoder::new(
                BufWriter::new(File::create(&class_path).expect("class file")),
                9,
            )
            .expect("class encoder");
            write_f16_slice(&mut writer, &class_vectors).expect("class vectors");
            writer.finish().expect("class finish");
        }

        let config = CurationConfig {
            input_dirs: vec![input_dir],
            vocabulary_path,
            output_dir: output_dir.clone(),
            batch_rows: 2,
            split_fractions: SplitFractions {
                train: 1.0,
                validation: 0.0,
                test: 0.0,
            },
            min_signature_count: 10,
            ..CurationConfig::default()
        };

        let summary = summarize_completed(&config).expect("summary");
        assert_eq!(summary.selection_summary.scanned_rows, 4);
        assert_eq!(summary.selection_summary.failed_rows, 1);
        assert_eq!(summary.selection_summary.partial_rows, 1);
        assert_eq!(summary.selection_summary.kept_rows, 2);
        assert_eq!(summary.split_row_counts.train, 2);

        let report = curate_completed(&config).expect("curate");
        assert_eq!(report.split_row_counts.train, 2);
        assert!(output_dir.join("train.parquet").exists());
        assert!(output_dir.join("manifest.json").exists());

        let mut reader = zstd::stream::read::Decoder::new(BufReader::new(
            File::open(output_dir.join("train.pathway-vectors.f16.zst")).expect("train vectors"),
        ))
        .expect("decoder");
        let decoded = read_f16_matrix(&mut reader, 2, 2).expect("decoded");
        for (expected, observed) in pathway_vectors[0..4].iter().zip(decoded.iter()) {
            assert!((expected - observed).abs() < 5e-4);
        }
    }
}
