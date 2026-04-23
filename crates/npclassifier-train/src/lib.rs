#![allow(missing_docs)]

pub mod calibration;
pub mod data;
pub mod error;
pub mod evaluation;
pub mod metric;
pub mod model;
pub mod quantization;
pub mod sync_best;
pub mod web_export;

pub const CLASS_MCC_LOG_NAME: &str = "Class MCC";
pub const SYNC_BEST_MODEL_FILE_STEM: &str = "best-model";
pub const SYNC_BEST_METADATA_FILE_NAME: &str = "best-model.json";
