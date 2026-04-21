//! Error handling for the Burn training CLI.

use std::path::PathBuf;

use thiserror::Error;

use npclassifier_core::NpClassifierError;

/// Top-level error for training data loading, model training, and evaluation.
#[derive(Debug, Error)]
pub enum TrainingError {
    /// Generic file I/O failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// JSON encoding or decoding failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Parquet decoding failed.
    #[error(transparent)]
    Parquet(#[from] parquet::errors::ParquetError),
    /// Core classifier utilities failed.
    #[error(transparent)]
    Core(#[from] NpClassifierError),
    /// Required curated split files were missing.
    #[error("missing curated split file: {0}")]
    MissingFile(PathBuf),
    /// The curated dataset contents did not match the expected schema.
    #[error("invalid curated dataset: {0}")]
    Dataset(String),
    /// Burn training or recording failed.
    #[error("burn training failed: {0}")]
    Burn(String),
}
