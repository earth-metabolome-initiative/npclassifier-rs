use thiserror::Error;

use crate::{model::ModelHead, ontology::OntologyError};

/// Top-level error type for fingerprinting, ontology, and inference failures.
#[derive(Debug, Error)]
pub enum NpClassifierError {
    /// Ontology decoding or validation failed.
    #[error(transparent)]
    Ontology(#[from] OntologyError),
    /// File I/O failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// JSON decoding failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Fingerprint counts did not match the expected width.
    #[error("invalid {section} fingerprint length: expected {expected}, got {actual}")]
    InvalidFingerprintLength {
        /// Fingerprint section name.
        section: &'static str,
        /// Expected vector length.
        expected: usize,
        /// Actual vector length.
        actual: usize,
    },
    /// Raw predictions did not match the expected head width.
    #[error("invalid {head} prediction width: expected {expected}, got {actual}")]
    InvalidPredictionWidth {
        /// Head with the invalid output width.
        head: ModelHead,
        /// Expected vector length.
        expected: usize,
        /// Actual vector length.
        actual: usize,
    },
    /// Fingerprint generation failed.
    #[error("fingerprint generation failed: {0}")]
    Fingerprint(String),
    /// Dataset curation or split materialization failed.
    #[error("dataset curation failed: {0}")]
    Dataset(String),
    /// Model loading or inference failed.
    #[error("model inference failed: {0}")]
    Model(String),
    /// Remote model access failed.
    #[error("remote model access failed: {0}")]
    Remote(String),
    /// A requested pipeline step is not implemented.
    #[error("unsupported pipeline step: {0}")]
    Unsupported(String),
    #[cfg(feature = "reference-dataset")]
    /// Zenodo dataset access failed.
    #[error(transparent)]
    Zenodo(#[from] zenodo_rs::ZenodoError),
}
