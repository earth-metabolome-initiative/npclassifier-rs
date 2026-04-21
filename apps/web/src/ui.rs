use dioxus::prelude::*;
use npclassifier_core::{WebBatchEntry as BatchEntry, WebModelVariant};

use crate::classifier::BatchState;
use crate::presentation::{
    GroupKind, VisibleLabel, WebOverview, format_score, visible_scored_labels,
};

const VISIBLE_PATHWAY_LIMIT: usize = 3;
const VISIBLE_SUPERCLASS_LIMIT: usize = 8;
const VISIBLE_CLASS_LIMIT: usize = 12;

#[component]
pub fn Header(repository_url: &'static str) -> Element {
    rsx! {
        header { class: "hero",
            div { class: "hero-main",
                div { class: "hero-head",
                    p { class: "eyebrow", "Earth Metabolome Initiative" }
                    h1 { "NPClassifier" }
                }
                p { class: "hero-copy",
                    "Classify natural products from SMILES in your browser."
                }
            }
            div { class: "hero-links",
                a {
                    class: "hero-link",
                    href: repository_url,
                    target: "_blank",
                    rel: "noopener noreferrer",
                    title: "Go to the GitHub repository",
                    aria_label: "Go to the GitHub repository",
                    {fa_icon("fa-brands fa-github")}
                    span { "GitHub" }
                }
            }
        }
    }
}

#[component]
pub fn InputPanel(
    current_input: String,
    placeholder: &'static str,
    current_model: WebModelVariant,
    mini_tooltip: &'static str,
    faithful_tooltip: &'static str,
    on_input: EventHandler<String>,
    on_select_model: EventHandler<WebModelVariant>,
) -> Element {
    let mini_button_class = if current_model == WebModelVariant::MiniShared {
        "model-toggle-button is-active"
    } else {
        "model-toggle-button"
    };
    let faithful_button_class = if current_model == WebModelVariant::Full {
        "model-toggle-button is-reference is-active"
    } else {
        "model-toggle-button is-reference"
    };

    rsx! {
        section { class: "panel input-panel",
            div { class: "panel-head",
                div { class: "panel-head-row",
                    h2 { class: "title-with-icon",
                        span { "SMILES, one per line" }
                    }
                    div { class: "model-toggle", role: "group", aria_label: "Classifier model",
                        button {
                            class: mini_button_class,
                            title: "{mini_tooltip}",
                            aria_label: "Mini model. {mini_tooltip}",
                            onclick: move |_| on_select_model.call(WebModelVariant::MiniShared),
                            {fa_icon(model_toggle_icon_class(WebModelVariant::MiniShared))}
                            span { "{WebModelVariant::MiniShared.display_name()}" }
                        }
                        button {
                            class: faithful_button_class,
                            title: "{faithful_tooltip}",
                            aria_label: "Faithful model. {faithful_tooltip}",
                            onclick: move |_| on_select_model.call(WebModelVariant::Full),
                            {fa_icon(model_toggle_icon_class(WebModelVariant::Full))}
                            span { "{WebModelVariant::Full.display_name()}" }
                        }
                    }
                }
            }

            textarea {
                class: "smiles-input",
                value: current_input,
                placeholder,
                oninput: move |event| on_input.call(event.value()),
            }
        }
    }
}

#[component]
pub fn ResultPanel(
    state: BatchState,
    active_entry: Option<BatchEntry>,
    entry_count: usize,
    active_index: usize,
    copy_entries_disabled: bool,
    copy_message: Option<String>,
    on_copy: EventHandler<()>,
    on_download: EventHandler<()>,
    on_select_previous: EventHandler<()>,
    on_select_next: EventHandler<()>,
) -> Element {
    rsx! {
        section { class: "panel result-panel",
            div { class: "result-head",
                div { class: "result-head-copy",
                    p { class: "eyebrow", "Classification" }
                    div { class: "result-head-main",
                        if let Some(entry) = active_entry.as_ref() {
                            h2 {
                                class: "active-smiles",
                                title: "{entry.smiles}",
                                "{entry.smiles}"
                            }
                        } else {
                            h2 { "No SMILES added" }
                        }
                        div { class: "result-actions",
                            button {
                                class: "copy-button",
                                aria_label: "Copy classification JSON",
                                title: "Copy classification JSON",
                                disabled: copy_entries_disabled,
                                onclick: move |_| on_copy.call(()),
                                {fa_icon("fa-solid fa-copy")}
                            }
                            span {
                                class: "result-actions-divider",
                                aria_hidden: "true",
                                "|"
                            }
                            button {
                                class: "copy-button",
                                aria_label: "Download classification JSON",
                                title: "Download classification JSON",
                                disabled: copy_entries_disabled,
                                onclick: move |_| on_download.call(()),
                                {fa_icon("fa-solid fa-download")}
                            }
                        }
                    }
                }
            }

            if let Some(message) = copy_message {
                p { class: "copy-toast", "{message}" }
            }

            div { class: result_body_class(&state, active_entry.as_ref()),
                {render_result_state(&state, active_entry.as_ref())}
            }

            if entry_count > 1 {
                div { class: "result-nav",
                    button {
                        class: "result-nav-button",
                        aria_label: "Show previous entry",
                        title: "Show previous entry",
                        onclick: move |_| on_select_previous.call(()),
                        {fa_icon("fa-solid fa-arrow-left")}
                    }
                    p { class: "result-nav-status",
                        "Showing entry {active_index + 1} of {entry_count}"
                    }
                    button {
                        class: "result-nav-button",
                        aria_label: "Show next entry",
                        title: "Show next entry",
                        onclick: move |_| on_select_next.call(()),
                        {fa_icon("fa-solid fa-arrow-right")}
                    }
                }
            }
        }
    }
}

pub fn fa_icon(class_name: &str) -> Element {
    let class_name = class_name.to_string();
    rsx! {
        i {
            class: "{class_name}",
            aria_hidden: "true",
        }
    }
}

fn render_result_state(state: &BatchState, active_entry: Option<&BatchEntry>) -> Element {
    match state {
        BatchState::Loading(progress) => rsx! {
            div { class: "loading-card",
                div { class: "loading-head",
                    div { class: "state-icon",
                        {fa_icon("fa-solid fa-spinner fa-spin")}
                    }
                    p { class: "empty-title", "{progress.label}" }
                }
                progress {
                    class: "loading-progress",
                    aria_label: "Classifier loading progress",
                    aria_valuetext: "{progress.completed} of {progress.total} complete",
                    max: "{progress.total.max(1)}",
                    value: "{progress.completed}",
                }
                p { class: "loading-meta",
                    "{progress.completed} / {progress.total} complete"
                }
            }
        },
        BatchState::Fatal(error) => rsx! {
            div { class: "empty-state error-state",
                div { class: "state-icon error-icon",
                    {fa_icon("fa-solid fa-triangle-exclamation")}
                }
                p { class: "empty-title", "Classifier could not run" }
                p { class: "panel-copy",
                    "The worker could not load or execute the q4 classifier."
                }
                p { class: "copy-note", "{error}" }
            }
        },
        BatchState::Ready(_) => {
            if let Some(entry) = active_entry {
                if let Some(error) = entry.error.as_deref() {
                    rsx! {
                        div { class: "empty-state error-state",
                            div { class: "state-icon error-icon",
                                {fa_icon("fa-solid fa-circle-exclamation")}
                            }
                            p { class: "empty-title", "Invalid SMILES" }
                            p { class: "panel-copy",
                                "This line could not be parsed as SMILES, so no classification was produced."
                            }
                            p { class: "copy-note", "{error}" }
                        }
                    }
                } else if entry_has_no_labels(entry) {
                    rsx! {
                        div { class: "empty-state",
                            div { class: "state-icon",
                                {fa_icon("fa-solid fa-ban")}
                            }
                            p { class: "empty-title", "No classification for this case" }
                            p { class: "panel-copy",
                                "NPClassifier did not assign any pathway, superclass, or class label to this structure."
                            }
                            p { class: "copy-note",
                                "This can happen for structures outside the model's domain or when no score clears the decision thresholds."
                            }
                        }
                    }
                } else {
                    let overview = WebOverview::load();
                    let visible_pathways = visible_scored_labels(
                        &entry.labels.pathways,
                        &entry.pathway_scores,
                        VISIBLE_PATHWAY_LIMIT,
                    );
                    let visible_superclasses = visible_scored_labels(
                        &entry.labels.superclasses,
                        &entry.superclass_scores,
                        VISIBLE_SUPERCLASS_LIMIT,
                    );
                    let visible_classes = visible_scored_labels(
                        &entry.labels.classes,
                        &entry.class_scores,
                        VISIBLE_CLASS_LIMIT,
                    );
                    rsx! {
                        {label_group(
                            "Pathways",
                            "fa-solid fa-route",
                            &visible_pathways,
                            GroupKind::Pathway,
                            overview,
                        )}
                        if !entry.labels.superclasses.is_empty() {
                            {label_group(
                                "Superclasses",
                                "fa-solid fa-sitemap",
                                &visible_superclasses,
                                GroupKind::Superclass,
                                overview,
                            )}
                        }
                        if !entry.labels.classes.is_empty() {
                            {label_group(
                                "Classes",
                                "fa-solid fa-tags",
                                &visible_classes,
                                GroupKind::Class,
                                overview,
                            )}
                        }
                    }
                }
            } else {
                empty_result_state()
            }
        }
        BatchState::Empty => empty_result_state(),
    }
}

fn result_body_class(state: &BatchState, active_entry: Option<&BatchEntry>) -> &'static str {
    match state {
        BatchState::Loading(_) => "result-body is-loading",
        BatchState::Fatal(_) => "result-body is-compact",
        BatchState::Ready(_) => {
            if let Some(entry) = active_entry {
                if entry.error.is_some() || entry_has_no_labels(entry) {
                    "result-body is-compact"
                } else {
                    "result-body is-ready"
                }
            } else {
                "result-body"
            }
        }
        BatchState::Empty => "result-body",
    }
}

fn entry_has_no_labels(entry: &BatchEntry) -> bool {
    entry.labels.pathways.is_empty()
        && entry.labels.superclasses.is_empty()
        && entry.labels.classes.is_empty()
}

fn empty_result_state() -> Element {
    rsx! {}
}

fn label_group(
    title: &str,
    icon_class: &'static str,
    labels: &[VisibleLabel],
    group_kind: GroupKind,
    overview: &WebOverview,
) -> Element {
    rsx! {
        section { class: "section-card",
            div { class: "section-head",
                h3 { class: "title-with-icon",
                    {fa_icon(icon_class)}
                    span { "{title}" }
                }
            }

            div { class: "predicted-row",
                if labels.is_empty() {
                    span { class: "tone-chip muted-chip", "None" }
                } else {
                    {labels.iter().map(|label| {
                        let tone = overview.tone_for_name(group_kind, &label.name);
                        rsx! {
                            span {
                                class: "tone-chip",
                                style: "{tone}",
                                span { class: "chip-label", "{label.name}" }
                                span { class: "chip-score", "{format_score(label.score)}" }
                            }
                        }
                    })}
                }
            }
        }
    }
}

fn model_toggle_icon_class(model: WebModelVariant) -> &'static str {
    match model {
        WebModelVariant::MiniShared => "fa-solid fa-gauge-high",
        WebModelVariant::Full => "fa-solid fa-scale-balanced",
    }
}

#[cfg(test)]
mod tests {
    use npclassifier_core::{FingerprintGenerator, PredictionLabels, WebBatchEntry};

    use super::entry_has_no_labels;

    #[test]
    fn counted_morgan_rejects_invalid_smiles() {
        let generator = npclassifier_core::CountedMorganGenerator::new();
        let result = generator.generate("CCCCeeeeeedsdsdsààùàùàùàùàù");

        assert!(result.is_err());
    }

    #[test]
    fn no_label_entries_are_treated_as_unclassified() {
        let entry = WebBatchEntry {
            smiles: String::from("C1=C(C(C=C(C1O)Cl)Cl)Cl"),
            error: None,
            labels: PredictionLabels::new(Vec::new(), Vec::new(), Vec::new(), None),
            pathway_scores: Vec::new(),
            superclass_scores: Vec::new(),
            class_scores: Vec::new(),
        };

        assert!(entry_has_no_labels(&entry));
    }
}
