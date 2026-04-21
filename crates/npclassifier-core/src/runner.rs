//! Local and remote packed-model runner utilities.

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

/// Source for a packed model bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelBundleSource {
    /// Read the packed bundle from an already materialized local directory.
    Local(PathBuf),
    /// Download the packed bundle from a remote base URL and cache it locally.
    Remote(String),
}

/// Builder for a reusable packed `NPClassifier` runner.
#[derive(Debug, Clone)]
pub struct PackedClassifierBuilder {
    source: Option<ModelBundleSource>,
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
            source: None,
            variant: PackedModelVariant::Q4Kernel,
            thresholds: None,
            cache_dir: None,
            parallelism: None,
        }
    }

    /// Load a packed bundle from a local directory.
    #[must_use]
    pub fn with_local_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.source = Some(ModelBundleSource::Local(path.into()));
        self
    }

    /// Download a packed bundle from a remote base URL.
    #[must_use]
    pub fn with_remote_base_url(mut self, url: impl Into<String>) -> Self {
        self.source = Some(ModelBundleSource::Remote(url.into()));
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

    /// Override the cache directory used for remote model downloads.
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
        let source = self.source.ok_or_else(|| {
            NpClassifierError::Unsupported("model source is not configured".to_owned())
        })?;
        let ontology = EmbeddedOntology::load()?;
        let (model_dir, loaded_thresholds) = match source.clone() {
            ModelBundleSource::Local(path) => {
                let thresholds = load_thresholds_from_dir(&path)?;
                (path, thresholds)
            }
            ModelBundleSource::Remote(base_url) => {
                let cache_root = self.cache_dir.unwrap_or_else(default_cache_root);
                let cached_dir = cache_remote_bundle(&base_url, &cache_root, self.variant)?;
                let thresholds = load_thresholds_from_dir(&cached_dir)?;
                (cached_dir, thresholds)
            }
        };

        let model = PackedModelSet::from_dir(&model_dir, self.variant)?;

        Ok(PackedClassifier {
            source,
            variant: self.variant,
            ontology,
            model,
            thresholds: self.thresholds.unwrap_or(loaded_thresholds),
            parallelism: self.parallelism,
        })
    }
}

/// Reusable packed classifier for local or cached remote bundles.
#[derive(Debug, Clone)]
pub struct PackedClassifier {
    source: ModelBundleSource,
    variant: PackedModelVariant,
    ontology: Ontology,
    model: PackedModelSet,
    thresholds: ClassificationThresholds,
    parallelism: Option<usize>,
}

impl PackedClassifier {
    /// Return the configured source.
    #[must_use]
    pub fn source(&self) -> &ModelBundleSource {
        &self.source
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
            &self.model,
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
    model_dir: &Path,
) -> Result<ClassificationThresholds, NpClassifierError> {
    let path = model_dir.join("thresholds.json");
    if !path.exists() {
        return Ok(ClassificationThresholds::default());
    }
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

fn cache_remote_bundle(
    base_url: &str,
    cache_root: &Path,
    variant: PackedModelVariant,
) -> Result<PathBuf, NpClassifierError> {
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
    use super::{PackedClassifierBuilder, cache_key, join_url};
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
}
