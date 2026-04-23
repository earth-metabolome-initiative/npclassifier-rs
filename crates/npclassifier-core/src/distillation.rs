//! Download helpers for the finalized distillation teacher splits.
//!
//! The distillation corpus is no longer curated inside this repository. The
//! final train / validation / test splits are published as a Zenodo record and
//! can be downloaded into the directory layout expected by the training tools.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use zenodo_rs::{Auth, RecordId, ZenodoClient};

use crate::NpClassifierError;

/// Zenodo record id for the finalized distillation teacher splits.
pub const DISTILLATION_DATASET_RECORD_ID: u64 = 19_701_295;
/// DOI for the finalized distillation teacher splits.
pub const DISTILLATION_DATASET_DOI: &str = "10.5281/zenodo.19701295";
/// Default local dataset directory used by the training tools.
pub const DEFAULT_DISTILLATION_DATA_DIR: &str = "data";

/// Files that make up the finalized distillation teacher split record.
pub const DISTILLATION_DATASET_FILES: &[&str] = &[
    "README.md",
    "LICENSE",
    "manifest.json",
    "SHA256SUMS.txt",
    "vocabulary.json",
    "train.parquet",
    "train.pathway-vectors.f16.zst",
    "train.superclass-vectors.f16.zst",
    "train.class-vectors.f16.zst",
    "validation.parquet",
    "validation.pathway-vectors.f16.zst",
    "validation.superclass-vectors.f16.zst",
    "validation.class-vectors.f16.zst",
    "test.parquet",
    "test.pathway-vectors.f16.zst",
    "test.superclass-vectors.f16.zst",
    "test.class-vectors.f16.zst",
];

/// Download options for the published distillation dataset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistillationDatasetDownloadConfig {
    /// Dataset directory for the split files.
    pub data_dir: PathBuf,
    /// Re-download files even when they already exist locally.
    pub force: bool,
}

impl Default for DistillationDatasetDownloadConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from(DEFAULT_DISTILLATION_DATA_DIR),
            force: false,
        }
    }
}

/// One downloaded or skipped Zenodo file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistillationDatasetFile {
    /// File key inside the Zenodo record.
    pub key: String,
    /// Local destination path.
    pub path: PathBuf,
    /// Bytes written by this invocation.
    pub bytes_written: u64,
    /// Whether the file was already present and left untouched.
    pub skipped: bool,
}

/// Summary returned after a dataset download run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistillationDatasetDownload {
    /// Zenodo record id used for the download.
    pub record_id: u64,
    /// Dataset DOI.
    pub doi: String,
    /// Local dataset directory.
    pub data_dir: PathBuf,
    /// File-level download results.
    pub files: Vec<DistillationDatasetFile>,
}

/// Downloads the finalized distillation dataset from Zenodo.
///
/// Existing files are skipped unless [`DistillationDatasetDownloadConfig::force`]
/// is set. Fresh downloads are written to a temporary `.part` file first, then
/// atomically renamed into place after `zenodo-rs` finishes checksum validation.
///
/// # Errors
///
/// Returns an error if Zenodo access fails, a file cannot be written, or a
/// downloaded checksum does not match the Zenodo record metadata.
pub async fn download_distillation_dataset(
    config: &DistillationDatasetDownloadConfig,
) -> Result<DistillationDatasetDownload, NpClassifierError> {
    fs::create_dir_all(&config.data_dir)?;
    let client = ZenodoClient::builder(Auth::new(
        std::env::var(Auth::TOKEN_ENV_VAR).unwrap_or_default(),
    ))
    .user_agent("npclassifier-rs/distillation-dataset")
    .build()?;

    let mut files = Vec::with_capacity(DISTILLATION_DATASET_FILES.len());
    for key in DISTILLATION_DATASET_FILES {
        let path = config.data_dir.join(key);
        if path.exists() && !config.force {
            files.push(DistillationDatasetFile {
                key: (*key).to_owned(),
                path,
                bytes_written: 0,
                skipped: true,
            });
            continue;
        }

        let temp_path = temporary_download_path(&path);
        if temp_path.exists() {
            fs::remove_file(&temp_path)?;
        }
        let resolved = client
            .download_record_file_by_key_to_path(
                RecordId(DISTILLATION_DATASET_RECORD_ID),
                key,
                &temp_path,
            )
            .await?;
        fs::rename(&temp_path, &path)?;
        files.push(DistillationDatasetFile {
            key: (*key).to_owned(),
            path,
            bytes_written: resolved.bytes_written,
            skipped: false,
        });
    }

    Ok(DistillationDatasetDownload {
        record_id: DISTILLATION_DATASET_RECORD_ID,
        doi: DISTILLATION_DATASET_DOI.to_owned(),
        data_dir: config.data_dir.clone(),
        files,
    })
}

/// Downloads missing finalized distillation dataset files from Zenodo.
///
/// This is the synchronous entrypoint intended for training tools. It returns
/// without network access when all expected files already exist.
///
/// # Errors
///
/// Returns an error if the dataset is incomplete and downloading a missing file
/// from Zenodo fails.
pub fn ensure_distillation_dataset(data_dir: &Path) -> Result<(), NpClassifierError> {
    if missing_distillation_dataset_files(data_dir).is_empty() {
        return Ok(());
    }
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(download_distillation_dataset(
            &DistillationDatasetDownloadConfig {
                data_dir: data_dir.to_path_buf(),
                force: false,
            },
        ))?;
    Ok(())
}

/// Lists finalized distillation dataset files missing from a directory.
#[must_use]
pub fn missing_distillation_dataset_files(data_dir: &Path) -> Vec<&'static str> {
    DISTILLATION_DATASET_FILES
        .iter()
        .copied()
        .filter(|key| !data_dir.join(key).exists())
        .collect()
}

fn temporary_download_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or("download");
    path.with_file_name(format!("{file_name}.part"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finalized_dataset_file_list_contains_all_training_inputs() {
        for key in [
            "manifest.json",
            "vocabulary.json",
            "train.parquet",
            "train.pathway-vectors.f16.zst",
            "train.superclass-vectors.f16.zst",
            "train.class-vectors.f16.zst",
            "validation.parquet",
            "validation.pathway-vectors.f16.zst",
            "validation.superclass-vectors.f16.zst",
            "validation.class-vectors.f16.zst",
            "test.parquet",
            "test.pathway-vectors.f16.zst",
            "test.superclass-vectors.f16.zst",
            "test.class-vectors.f16.zst",
        ] {
            assert!(
                DISTILLATION_DATASET_FILES.contains(&key),
                "missing dataset file key {key}"
            );
        }
    }

    #[test]
    fn default_download_config_uses_training_data_dir() {
        let config = DistillationDatasetDownloadConfig::default();
        assert_eq!(
            config.data_dir,
            PathBuf::from(DEFAULT_DISTILLATION_DATA_DIR)
        );
        assert!(!config.force);
    }

    #[test]
    fn missing_dataset_files_reports_absent_files() {
        let tempdir = std::env::temp_dir().join(format!(
            "npclassifier-distillation-missing-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tempdir);
        std::fs::create_dir_all(&tempdir).expect("tempdir");
        std::fs::write(tempdir.join("manifest.json"), "{}").expect("manifest");
        let missing = missing_distillation_dataset_files(&tempdir);
        assert!(!missing.contains(&"manifest.json"));
        assert!(missing.contains(&"train.parquet"));
        std::fs::remove_dir_all(tempdir).expect("cleanup");
    }
}
