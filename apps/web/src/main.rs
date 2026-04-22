//! Dioxus web shell for the serverless `NPClassifier` frontend.

mod actions;
mod app;
mod classifier;
mod hooks;
mod presentation;
mod ui;

fn main() {
    console_error_panic_hook::set_once();
    dioxus::launch(app::App);
}
