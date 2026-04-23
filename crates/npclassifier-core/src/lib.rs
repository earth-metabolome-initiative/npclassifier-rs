#![doc = include_str!("../../../README.md")]

/// Classification pipeline traits, thresholding, and voting entrypoints.
pub mod classifier;
/// Finalized distillation dataset download helpers.
#[cfg(feature = "distillation-dataset")]
pub mod distillation;
/// Top-level error type shared by the workspace crates.
pub mod error;
/// Counted Morgan fingerprint generation backed by `finge-rs`.
#[cfg(feature = "fingerprints")]
pub mod finge;
/// Fingerprint input contracts for the recovered model family.
pub mod fingerprint;
/// Embedded mock fingerprint fixtures used before live chemistry support lands.
pub mod mock;
/// Static descriptions of the recovered head architecture.
pub mod model;
/// Ontology loading and label hierarchy helpers.
pub mod ontology;
/// Packed weight loaders and dense inference runtime.
pub mod packed;
/// Reference dataset schemas and comparison helpers.
pub mod reference;
/// Parallel hosted packed-model runner utilities.
#[cfg(feature = "runner")]
pub mod runner;
/// Ontology-aware reconciliation of pathway, superclass, and class hits.
pub mod voting;
/// Browser-facing worker protocol and batch result types.
pub mod web;

pub use classifier::{
    ClassificationOutput, ClassificationThresholds, ClassifierPipeline, InferenceEngine,
    RawPredictions, classify_scores,
};
#[cfg(feature = "distillation-dataset")]
pub use distillation::{
    DEFAULT_DISTILLATION_DATA_DIR, DISTILLATION_DATASET_DOI, DISTILLATION_DATASET_FILES,
    DISTILLATION_DATASET_RECORD_ID, DistillationDatasetDownload, DistillationDatasetDownloadConfig,
    DistillationDatasetFile, download_distillation_dataset, ensure_distillation_dataset,
    missing_distillation_dataset_files,
};
pub use error::NpClassifierError;
#[cfg(feature = "fingerprints")]
pub use finge::CountedMorganGenerator;
pub use fingerprint::{
    DRAFT_MORGAN_RADIUS, FINGERPRINT_FORMULA_BITS, FINGERPRINT_INPUT_WIDTH,
    FINGERPRINT_RADIUS_BITS, FingerprintGenerator, FingerprintInput, FingerprintSpec,
    PreparedInput,
};
pub use mock::{MockExpectedLabels, MockFingerprintGenerator, MockFingerprintRecord};
pub use model::{BACKBONE_LAYERS, DenseLayerSpec, MODEL_HEADS, ModelHead, ModelHeadSpec};
pub use ontology::{ClassHierarchy, EmbeddedOntology, Ontology, OntologyError, SuperHierarchy};
pub use packed::{PackedHeadModel, PackedModelSet, PackedModelVariant};
pub use reference::{
    PUBCHEM_REFERENCE_COMPLETED_KEY, PUBCHEM_REFERENCE_DOI, PUBCHEM_REFERENCE_MANIFEST_KEY,
    PUBCHEM_REFERENCE_RECORD_ID, PredictionComparison, PredictionComparisonReason,
    PredictionLabels, PredictionMismatch, PubchemReferenceChunk, PubchemReferenceManifest,
    PubchemReferenceRow, compare_reference_prediction, compare_reference_rows,
};
#[cfg(feature = "runner")]
pub use runner::{HostedModel, PackedClassifier, PackedClassifierBuilder};
pub use voting::{IndexedLabel, VoteInput, VoteOutcome, vote_classification};
pub use web::{
    WebBatchEntry, WebModelVariant, WebScoredLabel, WebWorkerRequest, WebWorkerResponse,
    classify_web_entry, classify_web_entry_with_thresholds,
};
