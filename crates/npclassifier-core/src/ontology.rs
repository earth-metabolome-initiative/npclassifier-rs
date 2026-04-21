use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Mapping from one class label to its parent pathway and superclass labels.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassHierarchy {
    /// Pathway indices associated with the class.
    #[serde(rename = "Pathway")]
    pub pathway: Vec<usize>,
    /// Superclass indices associated with the class.
    #[serde(rename = "Superclass")]
    pub superclass: Vec<usize>,
}

/// Mapping from one superclass label to its parent pathway labels.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuperHierarchy {
    /// Pathway indices associated with the superclass.
    #[serde(rename = "Pathway")]
    pub pathway: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawOntology {
    #[serde(rename = "Pathway")]
    pathway: BTreeMap<String, usize>,
    #[serde(rename = "Superclass")]
    superclass: BTreeMap<String, usize>,
    #[serde(rename = "Class")]
    class: BTreeMap<String, usize>,
    #[serde(rename = "Class_hierarchy")]
    class_hierarchy: BTreeMap<usize, ClassHierarchy>,
    #[serde(rename = "Super_hierarchy")]
    super_hierarchy: BTreeMap<usize, SuperHierarchy>,
}

/// Dense ontology recovered from the Python `index_v1.json` file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ontology {
    pathway_labels: Vec<String>,
    superclass_labels: Vec<String>,
    class_labels: Vec<String>,
    class_hierarchy: BTreeMap<usize, ClassHierarchy>,
    super_hierarchy: BTreeMap<usize, SuperHierarchy>,
}

/// Errors raised while decoding or validating the embedded ontology.
#[derive(Debug, Error)]
pub enum OntologyError {
    /// The ontology JSON could not be decoded.
    #[error("failed to decode ontology JSON")]
    InvalidJson(#[from] serde_json::Error),
    /// One ontology group had no labels at all.
    #[error("ontology section {group} is empty")]
    EmptyGroup {
        /// Ontology group name.
        group: &'static str,
    },
    /// Dense index reconstruction found a missing label entry.
    #[error("ontology section {group} is missing an entry at index {index}")]
    MissingDenseEntry {
        /// Ontology group name.
        group: &'static str,
        /// Missing dense index.
        index: usize,
    },
}

impl Ontology {
    /// Decodes the recovered ontology JSON into dense label tables.
    ///
    /// # Errors
    ///
    /// Returns an [`OntologyError`] if the JSON is invalid, a section is empty,
    /// or a dense label index is missing.
    pub fn from_json_str(json: &str) -> Result<Self, OntologyError> {
        let raw = serde_json::from_str::<RawOntology>(json)?;

        Ok(Self {
            pathway_labels: dense_labels(raw.pathway, "Pathway")?,
            superclass_labels: dense_labels(raw.superclass, "Superclass")?,
            class_labels: dense_labels(raw.class, "Class")?,
            class_hierarchy: raw.class_hierarchy,
            super_hierarchy: raw.super_hierarchy,
        })
    }

    /// Returns the number of pathway labels.
    #[must_use]
    pub fn pathway_count(&self) -> usize {
        self.pathway_labels.len()
    }

    /// Returns the number of superclass labels.
    #[must_use]
    pub fn superclass_count(&self) -> usize {
        self.superclass_labels.len()
    }

    /// Returns the number of class labels.
    #[must_use]
    pub fn class_count(&self) -> usize {
        self.class_labels.len()
    }

    /// Returns one pathway label by index.
    #[must_use]
    pub fn pathway_name(&self, index: usize) -> Option<&str> {
        self.pathway_labels.get(index).map(String::as_str)
    }

    /// Returns one superclass label by index.
    #[must_use]
    pub fn superclass_name(&self, index: usize) -> Option<&str> {
        self.superclass_labels.get(index).map(String::as_str)
    }

    /// Returns one class label by index.
    #[must_use]
    pub fn class_name(&self, index: usize) -> Option<&str> {
        self.class_labels.get(index).map(String::as_str)
    }

    /// Returns the class hierarchy entry for one class index.
    #[must_use]
    pub fn class_hierarchy(&self, index: usize) -> Option<&ClassHierarchy> {
        self.class_hierarchy.get(&index)
    }

    /// Returns the superclass hierarchy entry for one superclass index.
    #[must_use]
    pub fn super_hierarchy(&self, index: usize) -> Option<&SuperHierarchy> {
        self.super_hierarchy.get(&index)
    }

    /// Returns the pathway indices associated with one class index.
    #[must_use]
    pub fn class_pathways(&self, index: usize) -> &[usize] {
        self.class_hierarchy(index)
            .map_or(&[], |entry| entry.pathway.as_slice())
    }

    /// Returns the superclass indices associated with one class index.
    #[must_use]
    pub fn class_superclasses(&self, index: usize) -> &[usize] {
        self.class_hierarchy(index)
            .map_or(&[], |entry| entry.superclass.as_slice())
    }

    /// Returns the pathway indices associated with one superclass index.
    #[must_use]
    pub fn superclass_pathways(&self, index: usize) -> &[usize] {
        self.super_hierarchy(index)
            .map_or(&[], |entry| entry.pathway.as_slice())
    }
}

/// Loader for the embedded ontology snapshot bundled with the repository.
pub struct EmbeddedOntology;

impl EmbeddedOntology {
    /// Loads the embedded `index_v1.json` ontology snapshot.
    ///
    /// # Errors
    ///
    /// Returns an [`OntologyError`] if the bundled ontology cannot be decoded
    /// or validated.
    pub fn load() -> Result<Ontology, OntologyError> {
        Ontology::from_json_str(include_str!("assets/index_v1.json"))
    }
}

fn dense_labels(
    labels_by_name: BTreeMap<String, usize>,
    group: &'static str,
) -> Result<Vec<String>, OntologyError> {
    let Some(max_index) = labels_by_name.values().copied().max() else {
        return Err(OntologyError::EmptyGroup { group });
    };

    let mut labels = vec![None; max_index + 1];
    for (name, index) in labels_by_name {
        labels[index] = Some(name);
    }

    labels
        .into_iter()
        .enumerate()
        .map(|(index, label)| label.ok_or(OntologyError::MissingDenseEntry { group, index }))
        .collect()
}
