use std::sync::OnceLock;

use dioxus::prelude::*;
use npclassifier_core::MockFingerprintRecord;

use crate::{
    actions::{copy_entries_as_json, download_entries_as_json},
    classifier::use_classifier,
    hooks::{use_entry_keyboard_navigation, use_transient_message},
    ui::{Header, InputPanel, ResultPanel},
};

const STYLE: &str = include_str!("../public/style.css");
const PLACEHOLDER_SMILES: &str = "CCO\nC1=CC=CC=C1\nCC(=O)OC1=CC=CC=C1C(=O)O";
const DEFAULT_FALLBACK_SMILES: &str = "CCO";
const DEFAULT_SAMPLE_POOL_SIZE: usize = 100;
const DEFAULT_EXCLUDED_STARTUP_SMILES: &str = "C1=C(C(C=C(C1O)Cl)Cl)Cl";
const COPY_MESSAGE_CLEAR_MS: i32 = 2420;
const REPOSITORY_URL: &str = "https://github.com/earth-metabolome-initiative/npclassifier-rs";
const SITE_URL: &str = "https://npc.earthmetabolome.org/";
const PAGE_TITLE: &str = "NPClassifier | Browser-side natural product classification";
const PAGE_DESCRIPTION: &str = "Classify natural products from SMILES directly in your browser with Mini and Faithful Rust/WASM NPClassifier models. No structure upload required.";
const MINI_MODEL_TOOLTIP: &str = "Mini: compact distilled NPClassifier variant for routine browser use; usually close to Faithful, but it can differ on individual edge cases, at about 9 MiB in q4.";
const FAITHFUL_MODEL_TOOLTIP: &str = "Faithful: q4 NPClassifier with the original architecture; larger and slower, but the better choice when you want the closest match to NPClassifier behavior, at about 121 MiB.";

#[component]
pub fn App() -> Element {
    let classifier = use_classifier(default_startup_smiles);
    let mut copy_message = use_transient_message(COPY_MESSAGE_CLEAR_MS);
    let view = classifier.view();
    let classifier_for_keyboard_previous = classifier.clone();
    let classifier_for_keyboard_next = classifier.clone();
    use_entry_keyboard_navigation(
        view.entry_count,
        move || classifier_for_keyboard_previous.select_previous(),
        move || classifier_for_keyboard_next.select_next(),
    );

    let current_input = classifier.current_input();
    let current_model = classifier.current_model();
    let copy_entries_disabled = !classifier.has_export_entries();
    let state = view.state.clone();
    let active_entry = view.active_entry.clone();
    let classifier_for_input = classifier.clone();
    let classifier_for_model_select = classifier.clone();
    let classifier_for_previous = classifier.clone();
    let classifier_for_next = classifier.clone();
    let classifier_for_copy = classifier.clone();
    let classifier_for_download = classifier.clone();

    rsx! {
        document::Title { "{PAGE_TITLE}" }
        document::Meta { name: "description", content: "{PAGE_DESCRIPTION}" }
        document::Meta { name: "viewport", content: "width=device-width, initial-scale=1" }
        document::Meta { name: "theme-color", content: "#183746" }
        document::Meta { name: "color-scheme", content: "light" }
        document::Meta { name: "application-name", content: "NPClassifier" }
        document::Meta { name: "author", content: "Earth Metabolome Initiative" }
        document::Meta { name: "robots", content: "index, follow" }
        document::Meta {
            name: "keywords",
            content: "NPClassifier, natural products, metabolomics, cheminformatics, SMILES, classification, Rust, WebAssembly",
        }
        document::Meta { property: "og:type", content: "website" }
        document::Meta { property: "og:site_name", content: "NPClassifier" }
        document::Meta { property: "og:title", content: "{PAGE_TITLE}" }
        document::Meta { property: "og:description", content: "{PAGE_DESCRIPTION}" }
        document::Meta { property: "og:url", content: "{SITE_URL}" }
        document::Meta { name: "twitter:card", content: "summary" }
        document::Meta { name: "twitter:title", content: "{PAGE_TITLE}" }
        document::Meta { name: "twitter:description", content: "{PAGE_DESCRIPTION}" }
        document::Link { rel: "canonical", href: "{SITE_URL}" }
        document::Link {
            rel: "icon",
            href: "favicon.ico",
            r#type: "image/x-icon",
        }
        document::Link {
            rel: "shortcut icon",
            href: "favicon.ico",
            r#type: "image/x-icon",
        }

        style { {STYLE} }

        main { class: "page",
            Header {
                repository_url: REPOSITORY_URL,
            }

            section { class: "layout",
                InputPanel {
                    current_input,
                    placeholder: PLACEHOLDER_SMILES,
                    current_model,
                    mini_tooltip: MINI_MODEL_TOOLTIP,
                    faithful_tooltip: FAITHFUL_MODEL_TOOLTIP,
                    on_input: move |value: String| {
                        classifier_for_input.handle_input(&value);
                        copy_message.set(None);
                    },
                    on_select_model: move |model| {
                        classifier_for_model_select.select_model(model);
                        copy_message.set(None);
                    },
                }

                ResultPanel {
                    state,
                    active_entry,
                    entry_count: view.entry_count,
                    active_index: view.active_index,
                    copy_entries_disabled,
                    copy_message: copy_message(),
                    on_copy: move |()| {
                        let entries = classifier_for_copy.export_entries();
                        let mut copy_message = copy_message;
                        spawn(async move {
                            match copy_entries_as_json(&entries).await {
                                Ok(message) => copy_message.set(Some(message)),
                                Err(error) => {
                                    copy_message.set(Some(error));
                                }
                            }
                        });
                    },
                    on_download: move |()| {
                        let entries = classifier_for_download.export_entries();
                        match download_entries_as_json(&entries) {
                            Ok(message) => copy_message.set(Some(message)),
                            Err(error) => copy_message.set(Some(error)),
                        }
                    },
                    on_select_previous: move |()| classifier_for_previous.select_previous(),
                    on_select_next: move |()| classifier_for_next.select_next(),
                }
            }
        }
    }
}

fn default_startup_smiles() -> String {
    default_smiles_pool()
        .get(choose_default_smiles_index(default_smiles_pool().len()))
        .cloned()
        .unwrap_or_else(|| String::from(DEFAULT_FALLBACK_SMILES))
}

fn default_smiles_pool() -> &'static Vec<String> {
    static POOL: OnceLock<Vec<String>> = OnceLock::new();
    POOL.get_or_init(|| {
        npclassifier_core::MockFingerprintGenerator::reference_128()
            .map(|generator| {
                generator
                    .records()
                    .filter(|record| should_offer_as_default_example(record))
                    .take(DEFAULT_SAMPLE_POOL_SIZE)
                    .map(|record| record.smiles.clone())
                    .collect::<Vec<_>>()
            })
            .ok()
            .filter(|pool| !pool.is_empty())
            .unwrap_or_else(|| vec![String::from(DEFAULT_FALLBACK_SMILES)])
    })
}

fn should_offer_as_default_example(record: &MockFingerprintRecord) -> bool {
    record.smiles != DEFAULT_EXCLUDED_STARTUP_SMILES
        && (!record.expected.pathways.is_empty()
            || !record.expected.superclasses.is_empty()
            || !record.expected.classes.is_empty())
}

#[cfg(target_arch = "wasm32")]
fn choose_default_smiles_index(len: usize) -> usize {
    if len <= 1 {
        0
    } else {
        let random_bits = js_sys::Math::random().to_bits();
        let folded = random_bits ^ (random_bits >> 32);
        usize::try_from(folded).unwrap_or(0) % len
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn choose_default_smiles_index(_len: usize) -> usize {
    0
}

#[cfg(test)]
mod tests {
    use npclassifier_core::{MockExpectedLabels, MockFingerprintRecord};

    use super::{DEFAULT_EXCLUDED_STARTUP_SMILES, should_offer_as_default_example};

    fn mock_record(smiles: &str, labels: MockExpectedLabels) -> MockFingerprintRecord {
        MockFingerprintRecord {
            name: String::from("probe"),
            smiles: String::from(smiles),
            formula_counts: Vec::new(),
            radius_counts: Vec::new(),
            is_glycoside: false,
            expected: labels,
        }
    }

    #[test]
    fn startup_examples_exclude_the_known_problem_case() {
        let record = mock_record(
            DEFAULT_EXCLUDED_STARTUP_SMILES,
            MockExpectedLabels {
                pathways: vec![String::from("Pathway")],
                superclasses: Vec::new(),
                classes: Vec::new(),
                is_glycoside: false,
            },
        );

        assert!(!should_offer_as_default_example(&record));
    }

    #[test]
    fn startup_examples_require_at_least_one_expected_label() {
        let empty = mock_record(
            "CCO",
            MockExpectedLabels {
                pathways: Vec::new(),
                superclasses: Vec::new(),
                classes: Vec::new(),
                is_glycoside: false,
            },
        );
        let labeled = mock_record(
            "CCO",
            MockExpectedLabels {
                pathways: vec![String::from("Pathway")],
                superclasses: Vec::new(),
                classes: Vec::new(),
                is_glycoside: false,
            },
        );

        assert!(!should_offer_as_default_example(&empty));
        assert!(should_offer_as_default_example(&labeled));
    }
}
