#![allow(missing_docs)]

pub mod data;
pub mod error;
pub mod metric;
pub mod model;
pub mod sync_best;

pub const CLASS_MCC_LOG_NAME: &str = "Class MCC";
pub const SYNC_BEST_MODEL_FILE_STEM: &str = "best-model";
pub const SYNC_BEST_METADATA_FILE_NAME: &str = "best-model.json";
