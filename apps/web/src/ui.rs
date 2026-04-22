use dioxus::prelude::*;
use npclassifier_core::{WebBatchEntry as BatchEntry, WebModelVariant};

use crate::classifier::{BatchState, LoadingState};
use crate::presentation::{
    GroupKind, VisibleLabel, WebOverview, format_score, visible_scored_labels,
};

const VISIBLE_PATHWAY_LIMIT: usize = 3;
const VISIBLE_SUPERCLASS_LIMIT: usize = 8;
const VISIBLE_CLASS_LIMIT: usize = 12;

#[derive(Clone, Copy)]
enum IconKind {
    Repository,
    Mini,
    Faithful,
    Copy,
    Download,
    ArrowLeft,
    ArrowRight,
    Spinner,
    Error,
    Warning,
    Ban,
    Pathway,
    Superclass,
    Class,
}

#[component]
pub fn Header(repository_url: &'static str) -> Element {
    rsx! {
        header { class: "hero",
            div { class: "hero-main",
                div { class: "hero-head",
                    p { class: "eyebrow", "Earth Metabolome Initiative" }
                    h1 {
                        "NPClassifier"
                        span { class: "hero-rust-suffix", ".rs" }
                    }
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
                    {app_icon(IconKind::Repository)}
                    span { "GitHub" }
                }
            }
        }
    }
}

#[component]
pub fn InputPanel(
    current_input: String,
    input_notice: Option<String>,
    placeholder: &'static str,
    max_input_bytes: usize,
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
                            {app_icon(model_toggle_icon(WebModelVariant::MiniShared))}
                            span { "{WebModelVariant::MiniShared.display_name()}" }
                        }
                        button {
                            class: faithful_button_class,
                            title: "{faithful_tooltip}",
                            aria_label: "Faithful model. {faithful_tooltip}",
                            onclick: move |_| on_select_model.call(WebModelVariant::Full),
                            {app_icon(model_toggle_icon(WebModelVariant::Full))}
                            span { "{WebModelVariant::Full.display_name()}" }
                        }
                    }
                }
            }

            textarea {
                class: "smiles-input",
                value: current_input,
                placeholder,
                maxlength: "{max_input_bytes}",
                oninput: move |event| on_input.call(event.value()),
            }

            if let Some(input_notice) = input_notice {
                p {
                    class: "panel-copy input-note",
                    role: "alert",
                    "{input_notice}"
                }
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
                                {app_icon(IconKind::Copy)}
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
                                {app_icon(IconKind::Download)}
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
                        {app_icon(IconKind::ArrowLeft)}
                    }
                    p { class: "result-nav-status",
                        "Showing entry {active_index + 1} of {entry_count}"
                    }
                    button {
                        class: "result-nav-button",
                        aria_label: "Show next entry",
                        title: "Show next entry",
                        onclick: move |_| on_select_next.call(()),
                        {app_icon(IconKind::ArrowRight)}
                    }
                }
            }
        }
    }
}

fn app_icon(icon: IconKind) -> Element {
    match icon {
        IconKind::Repository => repository_icon(),
        IconKind::Mini => mini_icon(),
        IconKind::Faithful => faithful_icon(),
        IconKind::Copy => copy_icon(),
        IconKind::Download => download_icon(),
        IconKind::ArrowLeft => arrow_left_icon(),
        IconKind::ArrowRight => arrow_right_icon(),
        IconKind::Spinner => spinner_icon(),
        IconKind::Error => error_icon(),
        IconKind::Warning => warning_icon(),
        IconKind::Ban => ban_icon(),
        IconKind::Pathway => pathway_icon(),
        IconKind::Superclass => superclass_icon(),
        IconKind::Class => class_icon(),
    }
}

fn repository_icon() -> Element {
    rsx! {
        svg { class: "app-icon", view_box: "0 0 24 24", fill: "none",
            circle { cx: "6", cy: "18", r: "2" }
            circle { cx: "10", cy: "6", r: "2" }
            circle { cx: "18", cy: "8", r: "2" }
            path { d: "M8 17c3 0 5-2 5-5V8" }
            path { d: "M12 8h4" }
        }
    }
}

fn mini_icon() -> Element {
    rsx! {
        svg { class: "app-icon", view_box: "0 0 24 24", fill: "none",
            path { d: "M4 14a8 8 0 1 1 16 0" }
            path { d: "M12 14l4-4" }
            path { d: "M12 14h.01" }
        }
    }
}

fn faithful_icon() -> Element {
    rsx! {
        svg { class: "app-icon", view_box: "0 0 24 24", fill: "none",
            path { d: "M12 3v18" }
            path { d: "M6 6h12" }
            path { d: "M7 6l-3 5h6Z" }
            path { d: "M17 6l-3 5h6Z" }
            path { d: "M6 18h12" }
        }
    }
}

fn copy_icon() -> Element {
    rsx! {
        svg { class: "app-icon", view_box: "0 0 24 24", fill: "none",
            rect { x: "9", y: "5", width: "10", height: "14", rx: "2" }
            path { d: "M15 5V4a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v11a2 2 0 0 0 2 2h1" }
        }
    }
}

fn download_icon() -> Element {
    rsx! {
        svg { class: "app-icon", view_box: "0 0 24 24", fill: "none",
            path { d: "M12 4v10" }
            path { d: "m8 10 4 4 4-4" }
            path { d: "M4 18v1a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-1" }
        }
    }
}

fn arrow_left_icon() -> Element {
    rsx! {
        svg { class: "app-icon", view_box: "0 0 24 24", fill: "none",
            path { d: "M19 12H5" }
            path { d: "m12 19-7-7 7-7" }
        }
    }
}

fn arrow_right_icon() -> Element {
    rsx! {
        svg { class: "app-icon", view_box: "0 0 24 24", fill: "none",
            path { d: "M5 12h14" }
            path { d: "m12 5 7 7-7 7" }
        }
    }
}

fn spinner_icon() -> Element {
    rsx! {
        svg { class: "app-icon is-spin", view_box: "0 0 24 24", fill: "none",
            path { d: "M21 12a9 9 0 1 1-9-9" }
        }
    }
}

fn error_icon() -> Element {
    rsx! {
        svg { class: "app-icon", view_box: "0 0 24 24", fill: "none",
            path { d: "M12 3 2.5 20h19Z" }
            path { d: "M12 9v4" }
            path { d: "M12 17h.01" }
        }
    }
}

fn warning_icon() -> Element {
    rsx! {
        svg { class: "app-icon", view_box: "0 0 24 24", fill: "none",
            circle { cx: "12", cy: "12", r: "9" }
            path { d: "M12 8v5" }
            path { d: "M12 16h.01" }
        }
    }
}

fn ban_icon() -> Element {
    rsx! {
        svg { class: "app-icon", view_box: "0 0 24 24", fill: "none",
            circle { cx: "12", cy: "12", r: "9" }
            path { d: "m8 8 8 8" }
        }
    }
}

fn pathway_icon() -> Element {
    rsx! {
        svg { class: "app-icon", view_box: "0 0 24 24", fill: "none",
            circle { cx: "6", cy: "18", r: "2" }
            circle { cx: "10", cy: "6", r: "2" }
            circle { cx: "18", cy: "8", r: "2" }
            path { d: "M8 17c3 0 5-2 5-5V8" }
            path { d: "M12 8h4" }
        }
    }
}

fn superclass_icon() -> Element {
    rsx! {
        svg { class: "app-icon", view_box: "0 0 24 24", fill: "none",
            rect { x: "4", y: "15", width: "4", height: "4", rx: "1" }
            rect { x: "10", y: "15", width: "4", height: "4", rx: "1" }
            rect { x: "16", y: "15", width: "4", height: "4", rx: "1" }
            rect { x: "10", y: "5", width: "4", height: "4", rx: "1" }
            path { d: "M12 9v3" }
            path { d: "M6 12h12" }
            path { d: "M6 12v3" }
            path { d: "M12 12v3" }
            path { d: "M18 12v3" }
        }
    }
}

fn class_icon() -> Element {
    rsx! {
        svg { class: "app-icon", view_box: "0 0 24 24", fill: "none",
            path { d: "M7 7h6l4 4v6l-4 4H7l-4-4v-6Z" }
            path { d: "m13 7 4 4" }
        }
    }
}

fn render_result_state(state: &BatchState, active_entry: Option<&BatchEntry>) -> Element {
    match state {
        BatchState::Loading(progress) => render_loading_state(progress),
        BatchState::Fatal(error) => render_fatal_state(error),
        BatchState::Ready(_) => {
            if let Some(entry) = active_entry {
                if let Some(error) = entry.error.as_deref() {
                    render_invalid_smiles_state(error)
                } else if entry_has_no_labels(entry) {
                    render_unclassified_state()
                } else {
                    render_labeled_result(entry)
                }
            } else {
                empty_result_state()
            }
        }
        BatchState::Empty => empty_result_state(),
    }
}

fn render_loading_state(progress: &LoadingState) -> Element {
    rsx! {
        div { class: "loading-card",
                div { class: "loading-head",
                    div { class: "state-icon",
                    {app_icon(IconKind::Spinner)}
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
            if let Some(eta_seconds) = progress.eta_seconds {
                p { class: "loading-meta",
                    "ETA {format_eta(eta_seconds)}"
                }
            }
            if progress.suggest_native {
                p { class: "panel-copy loading-note",
                    "This browser run is estimated to take over an hour. For large-scale batches, use the native CLI instead."
                }
            }
        }
    }
}

fn render_fatal_state(error: &str) -> Element {
    rsx! {
        div { class: "empty-state error-state",
            div { class: "state-icon error-icon",
                {app_icon(IconKind::Error)}
            }
            p { class: "empty-title", "Classifier could not run" }
            p { class: "panel-copy",
                "The worker could not load or execute the q4 classifier."
            }
            p { class: "copy-note", "{error}" }
        }
    }
}

fn render_invalid_smiles_state(error: &str) -> Element {
    rsx! {
        div { class: "empty-state error-state",
            div { class: "state-icon error-icon",
                {app_icon(IconKind::Warning)}
            }
            p { class: "empty-title", "Invalid SMILES" }
            p { class: "panel-copy",
                "This line could not be parsed as SMILES, so no classification was produced."
            }
            p { class: "copy-note", "{error}" }
        }
    }
}

fn render_unclassified_state() -> Element {
    rsx! {
        div { class: "empty-state",
            div { class: "state-icon",
                {app_icon(IconKind::Ban)}
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
}

fn render_labeled_result(entry: &BatchEntry) -> Element {
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
            IconKind::Pathway,
            &visible_pathways,
            GroupKind::Pathway,
            overview,
        )}
        if !entry.labels.superclasses.is_empty() {
            {label_group(
                "Superclasses",
                IconKind::Superclass,
                &visible_superclasses,
                GroupKind::Superclass,
                overview,
            )}
        }
        if !entry.labels.classes.is_empty() {
            {label_group(
                "Classes",
                IconKind::Class,
                &visible_classes,
                GroupKind::Class,
                overview,
            )}
        }
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

fn format_eta(total_seconds: u64) -> String {
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;

    match (hours, minutes, seconds) {
        (0, 0, seconds) => format!("{seconds}s"),
        (0, minutes, seconds) => format!("{minutes}m {seconds:02}s"),
        (hours, minutes, _) => format!("{hours}h {minutes:02}m"),
    }
}

fn label_group(
    title: &str,
    icon_class: IconKind,
    labels: &[VisibleLabel],
    group_kind: GroupKind,
    overview: &WebOverview,
) -> Element {
    rsx! {
        section { class: "section-card",
            div { class: "section-head",
                h3 { class: "title-with-icon",
                    {app_icon(icon_class)}
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

fn model_toggle_icon(model: WebModelVariant) -> IconKind {
    match model {
        WebModelVariant::MiniShared => IconKind::Mini,
        WebModelVariant::Full => IconKind::Faithful,
    }
}

#[cfg(test)]
mod tests {
    use npclassifier_core::{FingerprintGenerator, PredictionLabels, WebBatchEntry};

    use super::{entry_has_no_labels, format_eta};

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

    #[test]
    fn eta_formatting_is_compact_and_readable() {
        assert_eq!(format_eta(42), "42s");
        assert_eq!(format_eta(125), "2m 05s");
        assert_eq!(format_eta(3_726), "1h 02m");
    }
}
