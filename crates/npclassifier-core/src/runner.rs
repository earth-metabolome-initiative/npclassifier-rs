//! Hosted packed-model runner utilities.

use std::{
    collections::hash_map::DefaultHasher,
    fs,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
};

use rayon::prelude::*;
use reqwest::StatusCode;

use crate::{
    ClassificationThresholds, CountedMorganGenerator, EmbeddedOntology, NpClassifierError,
    Ontology, PackedModelSet, PackedModelVariant, WebBatchEntry,
    classify_web_entry_with_thresholds,
};

/// Hosted packed model bundles available to the native runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HostedModel {
    /// Smaller shared-stem browser-native model.
    Mini,
    /// Larger faithful architecture model.
    Faithful,
}

impl HostedModel {
    /// Stable user-facing label for the hosted model.
    #[must_use]
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Mini => "Mini",
            Self::Faithful => "Faithful",
        }
    }

    const fn base_url(self) -> &'static str {
        match self {
            Self::Mini => {
                "https://huggingface.co/EarthMetabolomeInitiative/npclassifier-rs-models/resolve/main/mini-shared"
            }
            Self::Faithful => {
                "https://huggingface.co/EarthMetabolomeInitiative/npclassifier-rs-models/resolve/main/full"
            }
        }
    }
}

/// Builder for a reusable packed `NPClassifier` runner.
#[derive(Debug, Clone)]
pub struct PackedClassifierBuilder {
    model: HostedModel,
    variant: PackedModelVariant,
    thresholds: Option<ClassificationThresholds>,
    cache_dir: Option<PathBuf>,
    parallelism: Option<usize>,
}

impl Default for PackedClassifierBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PackedClassifierBuilder {
    /// Create a builder with q4 as the default packed variant.
    #[must_use]
    pub fn new() -> Self {
        Self {
            model: HostedModel::Mini,
            variant: PackedModelVariant::Q4Kernel,
            thresholds: None,
            cache_dir: None,
            parallelism: None,
        }
    }

    /// Select which hosted model bundle to download automatically.
    #[must_use]
    pub fn with_model(mut self, model: HostedModel) -> Self {
        self.model = model;
        self
    }

    /// Override the packed model variant.
    #[must_use]
    pub fn with_variant(mut self, variant: PackedModelVariant) -> Self {
        self.variant = variant;
        self
    }

    /// Override the classification thresholds.
    #[must_use]
    pub fn with_thresholds(mut self, thresholds: ClassificationThresholds) -> Self {
        self.thresholds = Some(thresholds);
        self
    }

    /// Override the cache directory used for downloaded model bundles.
    #[must_use]
    pub fn with_cache_dir(mut self, cache_dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(cache_dir.into());
        self
    }

    /// Limit the rayon pool size used by batch classification.
    #[must_use]
    pub fn with_parallelism(mut self, threads: usize) -> Self {
        self.parallelism = Some(threads);
        self
    }

    /// Build the runner.
    ///
    /// # Errors
    ///
    /// Returns an error if the ontology, thresholds, or packed model bundle
    /// cannot be loaded.
    pub fn build(self) -> Result<PackedClassifier, NpClassifierError> {
        let ontology = EmbeddedOntology::load()?;
        let cache_root = self.cache_dir.unwrap_or_else(default_cache_root);
        let model_dir = cache_hosted_bundle(self.model, &cache_root, self.variant)?;
        let loaded_thresholds = load_thresholds_from_dir(&model_dir)?;

        let model = PackedModelSet::from_dir(&model_dir, self.variant)?;

        Ok(PackedClassifier {
            model: self.model,
            variant: self.variant,
            ontology,
            packed_model: model,
            thresholds: self.thresholds.unwrap_or(loaded_thresholds),
            parallelism: self.parallelism,
        })
    }
}

/// Reusable packed classifier for cached hosted bundles.
#[derive(Debug, Clone)]
pub struct PackedClassifier {
    model: HostedModel,
    variant: PackedModelVariant,
    ontology: Ontology,
    packed_model: PackedModelSet,
    thresholds: ClassificationThresholds,
    parallelism: Option<usize>,
}

impl PackedClassifier {
    /// Return the configured hosted model.
    #[must_use]
    pub const fn model(&self) -> HostedModel {
        self.model
    }

    /// Return the packed variant.
    #[must_use]
    pub const fn variant(&self) -> PackedModelVariant {
        self.variant
    }

    /// Return the active thresholds.
    #[must_use]
    pub const fn thresholds(&self) -> ClassificationThresholds {
        self.thresholds
    }

    /// Classify one SMILES string.
    #[must_use]
    pub fn classify(&self, smiles: &str) -> WebBatchEntry {
        let generator = CountedMorganGenerator::new();
        classify_web_entry_with_thresholds(
            smiles,
            &self.ontology,
            &generator,
            &self.packed_model,
            self.thresholds,
        )
    }

    /// Classify one batch of SMILES strings in parallel while preserving order.
    #[must_use]
    pub fn classify_batch_parallel(&self, smiles: &[String]) -> Vec<WebBatchEntry> {
        let run = || {
            smiles
                .par_iter()
                .map(|smiles| self.classify(smiles))
                .collect::<Vec<_>>()
        };

        match self.parallelism {
            Some(threads) if threads > 0 => rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_or_else(|_| run(), |pool| pool.install(run)),
            _ => run(),
        }
    }

    /// Split newline-delimited SMILES, trim empty lines, and classify them in parallel.
    #[must_use]
    pub fn classify_lines_parallel(&self, input: &str) -> Vec<WebBatchEntry> {
        let smiles = input
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_owned)
            .collect::<Vec<_>>();
        self.classify_batch_parallel(&smiles)
    }
}

fn load_thresholds_from_dir(
    model_dir: &std::path::Path,
) -> Result<ClassificationThresholds, NpClassifierError> {
    let path = model_dir.join("thresholds.json");
    if !path.exists() {
        return Ok(ClassificationThresholds::default());
    }
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

fn cache_hosted_bundle(
    hosted_model: HostedModel,
    cache_root: &Path,
    variant: PackedModelVariant,
) -> Result<PathBuf, NpClassifierError> {
    let base_url = hosted_model.base_url();
    let cache_dir = cache_root.join(cache_key(base_url, variant));
    fs::create_dir_all(&cache_dir)?;

    download_if_missing(
        &cache_dir.join("thresholds.json"),
        &join_url(base_url, "thresholds.json"),
        false,
    )?;
    download_if_missing(
        &cache_dir
            .join("pathway")
            .join(format!("pathway.{}.npz", variant.suffix())),
        &join_url(
            base_url,
            &format!("pathway/pathway.{}.npz", variant.suffix()),
        ),
        false,
    )?;
    download_if_missing(
        &cache_dir
            .join("superclass")
            .join(format!("superclass.{}.npz", variant.suffix())),
        &join_url(
            base_url,
            &format!("superclass/superclass.{}.npz", variant.suffix()),
        ),
        false,
    )?;
    download_if_missing(
        &cache_dir
            .join("class")
            .join(format!("class.{}.npz", variant.suffix())),
        &join_url(base_url, &format!("class/class.{}.npz", variant.suffix())),
        false,
    )?;
    download_if_missing(
        &cache_dir
            .join("shared")
            .join(format!("shared.{}.npz", variant.suffix())),
        &join_url(base_url, &format!("shared/shared.{}.npz", variant.suffix())),
        true,
    )?;

    Ok(cache_dir)
}

fn download_if_missing(
    destination: &Path,
    url: &str,
    optional: bool,
) -> Result<(), NpClassifierError> {
    if destination.exists() {
        return Ok(());
    }

    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }

    let response = reqwest::blocking::get(url)
        .map_err(|error| NpClassifierError::Remote(format!("failed to GET {url}: {error}")))?;
    if response.status() == StatusCode::NOT_FOUND && optional {
        return Ok(());
    }
    if !response.status().is_success() {
        return Err(NpClassifierError::Remote(format!(
            "failed to GET {url}: HTTP {}",
            response.status()
        )));
    }
    let bytes = response
        .bytes()
        .map_err(|error| NpClassifierError::Remote(format!("failed to read {url}: {error}")))?;
    fs::write(destination, bytes.as_ref())?;
    Ok(())
}

fn join_url(base_url: &str, suffix: &str) -> String {
    format!("{}/{}", base_url.trim_end_matches('/'), suffix)
}

fn cache_key(base_url: &str, variant: PackedModelVariant) -> String {
    let mut hasher = DefaultHasher::new();
    base_url.hash(&mut hasher);
    variant.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn default_cache_root() -> PathBuf {
    if let Some(path) = std::env::var_os("NPCLASSIFIER_MODEL_CACHE") {
        return PathBuf::from(path);
    }
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(path).join("npclassifier-rs");
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".cache").join("npclassifier-rs");
    }
    std::env::temp_dir().join("npclassifier-rs")
}

#[cfg(test)]
mod tests {
    use super::{HostedModel, PackedClassifierBuilder, cache_key, join_url};
    use crate::PackedModelVariant;

    #[test]
    fn join_url_trims_trailing_slash() {
        assert_eq!(
            join_url("https://example.invalid/models/", "thresholds.json"),
            "https://example.invalid/models/thresholds.json"
        );
    }

    #[test]
    fn cache_key_depends_on_source_and_variant() {
        let first = cache_key("https://example.invalid/a", PackedModelVariant::Q4Kernel);
        let second = cache_key("https://example.invalid/a", PackedModelVariant::Q8Kernel);
        let third = cache_key("https://example.invalid/b", PackedModelVariant::Q4Kernel);

        assert_ne!(first, second);
        assert_ne!(first, third);
    }

    #[test]
    fn builder_defaults_to_q4() {
        let builder = PackedClassifierBuilder::new();
        assert_eq!(builder.variant, PackedModelVariant::Q4Kernel);
    }

    #[test]
    fn builder_defaults_to_mini_model() {
        let builder = PackedClassifierBuilder::new();
        assert_eq!(builder.model, HostedModel::Mini);
    }
}
