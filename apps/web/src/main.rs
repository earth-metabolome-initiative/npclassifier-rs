//! Dioxus web shell for the serverless `NPClassifier` frontend.

mod actions;
mod app;
mod classifier;
mod hooks;
mod presentation;
mod ui;

fn main() {
    dioxus::launch(app::App);
}
