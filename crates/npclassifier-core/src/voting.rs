use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::ontology::Ontology;

/// Label name paired with its ontology index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexedLabel {
    /// Numeric ontology index.
    pub index: usize,
    /// Human-readable ontology label.
    pub name: String,
}

/// Final ontology-aware labels produced by the voting stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoteOutcome {
    /// Voted pathway labels.
    pub pathways: Vec<IndexedLabel>,
    /// Voted superclass labels.
    pub superclasses: Vec<IndexedLabel>,
    /// Voted class labels.
    pub classes: Vec<IndexedLabel>,
    /// Optional glycoside flag passed through from the fingerprint layer.
    pub is_glycoside: Option<bool>,
}

/// Inputs consumed by the ontology-aware voting stage.
#[derive(Debug, Clone, Copy)]
pub struct VoteInput<'a> {
    /// Pathway indices whose raw scores cleared the threshold.
    pub pathways_above_threshold: &'a [usize],
    /// Class indices whose raw scores cleared the threshold.
    pub classes_above_threshold: &'a [usize],
    /// Superclass indices whose raw scores cleared the threshold.
    pub superclasses_above_threshold: &'a [usize],
    /// Full class score vector.
    pub class_scores: &'a [f32],
    /// Full superclass score vector.
    pub superclass_scores: &'a [f32],
    /// Pathway indices implied by the surviving class hits.
    pub pathways_from_classes: &'a [usize],
    /// Pathway indices implied by the surviving superclass hits.
    pub pathways_from_superclasses: &'a [usize],
    /// Optional glycoside flag to preserve in the final output.
    pub is_glycoside: Option<bool>,
}

/// Reconciles thresholded hits into a final ontology-consistent label set.
#[must_use]
pub fn vote_classification(input: VoteInput<'_>, ontology: &Ontology) -> VoteOutcome {
    let mut pathway_hits = sort_unique(input.pathways_above_threshold.to_vec());
    let mut class_hits = sort_unique(input.classes_above_threshold.to_vec());
    let mut superclass_hits = sort_unique(input.superclasses_above_threshold.to_vec());
    let pathways_from_classes = sort_unique(input.pathways_from_classes.to_vec());
    let pathways_from_superclasses = sort_unique(input.pathways_from_superclasses.to_vec());

    let consensus_pathways = consensus_pathways(
        &pathway_hits,
        &pathways_from_classes,
        &pathways_from_superclasses,
    );

    if consensus_pathways.is_empty() {
        return pathway_only_outcome(&pathway_hits, input.is_glycoside, ontology);
    }

    let mut voted_pathways = consensus_pathways;

    if intersects(&pathway_hits, &voted_pathways) {
        if intersects(&voted_pathways, &pathways_from_superclasses) {
            superclass_hits
                .retain(|index| intersects(ontology.superclass_pathways(*index), &voted_pathways));

            if superclass_hits.is_empty() {
                retain_classes_for_pathways(&mut class_hits, ontology, &voted_pathways);
                superclass_hits = superclasses_from_classes(ontology, &class_hits);
            } else if superclass_hits.len() > 1 {
                retain_classes_for_pathways(&mut class_hits, ontology, &voted_pathways);
                if !class_hits.is_empty() {
                    rebuild_from_classes(
                        ontology,
                        &class_hits,
                        &mut superclass_hits,
                        &mut pathway_hits,
                        &mut voted_pathways,
                    );
                } else if voted_pathways.len() == 1 {
                    superclass_hits = argmax_as_singleton(input.superclass_scores);
                    class_hits = argmax_as_singleton(input.class_scores)
                        .into_iter()
                        .filter(|index| {
                            intersects(&superclass_hits, ontology.class_superclasses(*index))
                        })
                        .collect();
                }
            } else {
                class_hits.retain(|index| {
                    intersects(&superclass_hits, ontology.class_superclasses(*index))
                });
                if class_hits.is_empty() {
                    class_hits = argmax_as_singleton(input.class_scores)
                        .into_iter()
                        .filter(|index| {
                            intersects(&superclass_hits, ontology.class_superclasses(*index))
                        })
                        .collect();
                }
            }
        } else {
            retain_classes_for_pathways(&mut class_hits, ontology, &voted_pathways);
            superclass_hits = superclasses_from_classes(ontology, &class_hits);
        }
    } else {
        superclass_hits
            .retain(|index| intersects(ontology.superclass_pathways(*index), &voted_pathways));
        retain_classes_for_pathways(&mut class_hits, ontology, &voted_pathways);
        rebuild_from_classes(
            ontology,
            &class_hits,
            &mut superclass_hits,
            &mut pathway_hits,
            &mut voted_pathways,
        );
    }

    VoteOutcome {
        pathways: resolve_pathways(ontology, &sort_unique(voted_pathways)),
        superclasses: resolve_superclasses(ontology, &sort_unique(superclass_hits)),
        classes: resolve_classes(ontology, &sort_unique(class_hits)),
        is_glycoside: input.is_glycoside,
    }
}

fn pathway_only_outcome(
    pathway_hits: &[usize],
    is_glycoside: Option<bool>,
    ontology: &Ontology,
) -> VoteOutcome {
    VoteOutcome {
        pathways: resolve_pathways(ontology, pathway_hits),
        superclasses: Vec::new(),
        classes: Vec::new(),
        is_glycoside,
    }
}

fn consensus_pathways(
    pathways: &[usize],
    from_classes: &[usize],
    from_superclasses: &[usize],
) -> Vec<usize> {
    let mut counts = BTreeMap::<usize, usize>::new();
    for index in pathways
        .iter()
        .chain(from_classes.iter())
        .chain(from_superclasses.iter())
        .copied()
    {
        *counts.entry(index).or_default() += 1;
    }

    let mut triple = counts
        .iter()
        .filter_map(|(index, count)| (*count == 3).then_some(*index))
        .collect::<Vec<_>>();
    triple.sort_unstable();

    if !triple.is_empty() {
        return triple;
    }

    let mut double = counts
        .iter()
        .filter_map(|(index, count)| (*count == 2).then_some(*index))
        .collect::<Vec<_>>();
    double.sort_unstable();
    double
}

fn argmax_as_singleton(scores: &[f32]) -> Vec<usize> {
    scores
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| vec![index])
        .unwrap_or_default()
}

fn retain_classes_for_pathways(
    class_hits: &mut Vec<usize>,
    ontology: &Ontology,
    voted_pathways: &[usize],
) {
    class_hits.retain(|index| intersects(ontology.class_pathways(*index), voted_pathways));
}

fn rebuild_from_classes(
    ontology: &Ontology,
    class_hits: &[usize],
    superclass_hits: &mut Vec<usize>,
    pathway_hits: &mut Vec<usize>,
    voted_pathways: &mut Vec<usize>,
) {
    *superclass_hits = superclasses_from_classes(ontology, class_hits);
    *pathway_hits = pathways_from_classes(ontology, class_hits);
    voted_pathways.clone_from(pathway_hits);
}

fn superclasses_from_classes(ontology: &Ontology, class_hits: &[usize]) -> Vec<usize> {
    flatten_unique(
        class_hits
            .iter()
            .flat_map(|index| ontology.class_superclasses(*index).iter().copied()),
    )
}

fn pathways_from_classes(ontology: &Ontology, class_hits: &[usize]) -> Vec<usize> {
    flatten_unique(
        class_hits
            .iter()
            .flat_map(|index| ontology.class_pathways(*index).iter().copied()),
    )
}

fn resolve_pathways(ontology: &Ontology, indices: &[usize]) -> Vec<IndexedLabel> {
    resolve_labels(indices, |index| {
        ontology.pathway_name(index).map(str::to_owned)
    })
}

fn resolve_superclasses(ontology: &Ontology, indices: &[usize]) -> Vec<IndexedLabel> {
    resolve_labels(indices, |index| {
        ontology.superclass_name(index).map(str::to_owned)
    })
}

fn resolve_classes(ontology: &Ontology, indices: &[usize]) -> Vec<IndexedLabel> {
    resolve_labels(indices, |index| {
        ontology.class_name(index).map(str::to_owned)
    })
}

fn resolve_labels<F>(indices: &[usize], mut lookup: F) -> Vec<IndexedLabel>
where
    F: FnMut(usize) -> Option<String>,
{
    indices
        .iter()
        .filter_map(|index| {
            lookup(*index).map(|name| IndexedLabel {
                index: *index,
                name,
            })
        })
        .collect()
}

fn sort_unique(mut values: Vec<usize>) -> Vec<usize> {
    values.sort_unstable();
    values.dedup();
    values
}

fn flatten_unique(iter: impl Iterator<Item = usize>) -> Vec<usize> {
    let mut values = iter.collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    values
}

fn intersects(left: &[usize], right: &[usize]) -> bool {
    left.iter().any(|value| right.contains(value))
}

#[cfg(test)]
mod tests {
    use crate::ontology::Ontology;

    use super::{VoteInput, vote_classification};

    fn fixture_ontology() -> Ontology {
        Ontology::from_json_str(
            r#"{
                "Pathway": {"Path A": 0, "Path B": 1},
                "Superclass": {"Super A": 0, "Super B": 1},
                "Class": {"Class A": 0, "Class B": 1, "Class C": 2},
                "Class_hierarchy": {
                    "0": {"Pathway": [0], "Superclass": [0]},
                    "1": {"Pathway": [1], "Superclass": [1]},
                    "2": {"Pathway": [0], "Superclass": [1]}
                },
                "Super_hierarchy": {
                    "0": {"Pathway": [0]},
                    "1": {"Pathway": [1]}
                }
            }"#,
        )
        .expect("fixture ontology should parse")
    }

    #[test]
    fn returns_only_raw_pathways_without_consensus() {
        let ontology = fixture_ontology();
        let outcome = vote_classification(
            VoteInput {
                pathways_above_threshold: &[0],
                classes_above_threshold: &[],
                superclasses_above_threshold: &[],
                class_scores: &[0.0, 0.0, 0.0],
                superclass_scores: &[0.0, 0.0],
                pathways_from_classes: &[],
                pathways_from_superclasses: &[],
                is_glycoside: Some(false),
            },
            &ontology,
        );

        assert_eq!(outcome.pathways[0].name, "Path A");
        assert!(outcome.superclasses.is_empty());
        assert!(outcome.classes.is_empty());
    }

    #[test]
    fn filters_hits_to_the_consensus_pathway() {
        let ontology = fixture_ontology();
        let outcome = vote_classification(
            VoteInput {
                pathways_above_threshold: &[0],
                classes_above_threshold: &[0, 1, 2],
                superclasses_above_threshold: &[0, 1],
                class_scores: &[0.9, 0.2, 0.6],
                superclass_scores: &[0.7, 0.1],
                pathways_from_classes: &[0],
                pathways_from_superclasses: &[0],
                is_glycoside: None,
            },
            &ontology,
        );

        assert_eq!(
            outcome
                .pathways
                .iter()
                .map(|label| label.name.as_str())
                .collect::<Vec<_>>(),
            vec!["Path A"]
        );
        assert_eq!(
            outcome
                .superclasses
                .iter()
                .map(|label| label.name.as_str())
                .collect::<Vec<_>>(),
            vec!["Super A"]
        );
        assert_eq!(
            outcome
                .classes
                .iter()
                .map(|label| label.name.as_str())
                .collect::<Vec<_>>(),
            vec!["Class A"]
        );
    }

    #[test]
    fn falls_back_to_the_highest_scoring_compatible_class() {
        let ontology = fixture_ontology();
        let outcome = vote_classification(
            VoteInput {
                pathways_above_threshold: &[0],
                classes_above_threshold: &[1],
                superclasses_above_threshold: &[0],
                class_scores: &[0.9, 0.2, 0.4],
                superclass_scores: &[0.8, 0.1],
                pathways_from_classes: &[0],
                pathways_from_superclasses: &[0],
                is_glycoside: Some(true),
            },
            &ontology,
        );

        assert_eq!(
            outcome
                .classes
                .iter()
                .map(|label| label.name.as_str())
                .collect::<Vec<_>>(),
            vec!["Class A"]
        );
        assert_eq!(outcome.is_glycoside, Some(true));
    }
}
