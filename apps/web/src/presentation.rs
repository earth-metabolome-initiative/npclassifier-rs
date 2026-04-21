use std::{collections::HashMap, sync::OnceLock};

use npclassifier_core::{EmbeddedOntology, Ontology, WebScoredLabel as ScoredLabel};

const CLASS_COLOR_SEED: u64 = 0x8d0f_84d1_31a7_25b9;

pub(crate) struct WebOverview {
    colors: OntologyColors,
    label_index: LabelIndex,
}

impl WebOverview {
    pub(crate) fn load() -> &'static Self {
        static OVERVIEW: OnceLock<WebOverview> = OnceLock::new();
        OVERVIEW.get_or_init(Self::build)
    }

    fn build() -> Self {
        let ontology = EmbeddedOntology::load().expect("embedded ontology should parse");
        let colors = OntologyColors::build(&ontology);
        let label_index = LabelIndex::build(&ontology);
        Self {
            colors,
            label_index,
        }
    }

    pub(crate) fn tone_for_name(&self, kind: GroupKind, name: &str) -> &str {
        match kind {
            GroupKind::Pathway => self
                .label_index
                .pathways
                .get(name)
                .map_or(self.colors.muted.style.as_str(), |index| {
                    self.colors.pathways[*index].style.as_str()
                }),
            GroupKind::Superclass => self
                .label_index
                .superclasses
                .get(name)
                .map_or(self.colors.muted.style.as_str(), |index| {
                    self.colors.superclasses[*index].style.as_str()
                }),
            GroupKind::Class => self
                .label_index
                .classes
                .get(name)
                .map_or(self.colors.muted.style.as_str(), |index| {
                    self.colors.classes[*index].style.as_str()
                }),
        }
    }
}

struct LabelIndex {
    pathways: HashMap<String, usize>,
    superclasses: HashMap<String, usize>,
    classes: HashMap<String, usize>,
}

impl LabelIndex {
    fn build(ontology: &Ontology) -> Self {
        Self {
            pathways: collect_labels(ontology.pathway_count(), |index| {
                ontology.pathway_name(index)
            }),
            superclasses: collect_labels(ontology.superclass_count(), |index| {
                ontology.superclass_name(index)
            }),
            classes: collect_labels(ontology.class_count(), |index| ontology.class_name(index)),
        }
    }
}

struct OntologyColors {
    pathways: Vec<ChipTone>,
    superclasses: Vec<ChipTone>,
    classes: Vec<ChipTone>,
    muted: ChipTone,
}

impl OntologyColors {
    fn build(ontology: &Ontology) -> Self {
        let class_rgb = build_class_palette(ontology.class_count());
        let mut superclass_children = vec![Vec::new(); ontology.superclass_count()];
        let mut pathway_children = vec![Vec::new(); ontology.pathway_count()];

        for (class_index, color) in class_rgb.iter().copied().enumerate() {
            for &super_index in ontology.class_superclasses(class_index) {
                if let Some(children) = superclass_children.get_mut(super_index) {
                    children.push(color);
                }
            }
        }

        let superclass_rgb = superclass_children
            .iter()
            .enumerate()
            .map(|(index, children)| {
                if children.is_empty() {
                    fallback_palette_color(index, ontology.superclass_count(), 53.0)
                } else {
                    average_color(children)
                }
            })
            .collect::<Vec<_>>();

        for (super_index, color) in superclass_rgb.iter().copied().enumerate() {
            for &pathway_index in ontology.superclass_pathways(super_index) {
                if let Some(children) = pathway_children.get_mut(pathway_index) {
                    children.push(color);
                }
            }
        }

        for (class_index, color) in class_rgb.iter().copied().enumerate() {
            for &pathway_index in ontology.class_pathways(class_index) {
                if let Some(children) = pathway_children.get_mut(pathway_index) {
                    children.push(color);
                }
            }
        }

        let pathway_rgb = pathway_children
            .iter()
            .enumerate()
            .map(|(index, children)| {
                if children.is_empty() {
                    fallback_palette_color(index, ontology.pathway_count(), 97.0)
                } else {
                    average_color(children)
                }
            })
            .collect::<Vec<_>>();

        Self {
            pathways: pathway_rgb.into_iter().map(ChipTone::from_rgb).collect(),
            superclasses: superclass_rgb.into_iter().map(ChipTone::from_rgb).collect(),
            classes: class_rgb.into_iter().map(ChipTone::from_rgb).collect(),
            muted: ChipTone::muted(),
        }
    }
}

#[derive(Clone)]
struct ChipTone {
    style: String,
}

impl ChipTone {
    fn from_rgb(color: Rgb) -> Self {
        let text = color.mix(Rgb::new(12.0, 19.0, 27.0), 0.58);
        let style = format!(
            "background: rgba({:.0}, {:.0}, {:.0}, 0.16); border-color: rgba({:.0}, {:.0}, {:.0}, 0.34); color: rgb({:.0}, {:.0}, {:.0});",
            color.r, color.g, color.b, color.r, color.g, color.b, text.r, text.g, text.b,
        );
        Self { style }
    }

    fn muted() -> Self {
        Self {
            style: String::from(
                "background: rgba(21, 35, 43, 0.06); border-color: rgba(21, 35, 43, 0.1); color: rgb(95, 111, 118);",
            ),
        }
    }
}

#[derive(Clone, Copy)]
struct Rgb {
    r: f64,
    g: f64,
    b: f64,
}

impl Rgb {
    const fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    fn mix(self, other: Self, amount: f64) -> Self {
        Self {
            r: self.r * (1.0 - amount) + other.r * amount,
            g: self.g * (1.0 - amount) + other.g * amount,
            b: self.b * (1.0 - amount) + other.b * amount,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum GroupKind {
    Pathway,
    Superclass,
    Class,
}

#[derive(Clone)]
pub(crate) struct VisibleLabel {
    pub(crate) name: String,
    pub(crate) score: f32,
}

pub(crate) fn visible_scored_labels(
    labels: &[String],
    scores: &[ScoredLabel],
    limit: usize,
) -> Vec<VisibleLabel> {
    if labels.is_empty() || limit == 0 {
        return Vec::new();
    }

    let score_lookup = scores
        .iter()
        .map(|score| (score.name.as_str(), score.score))
        .collect::<HashMap<_, _>>();
    let mut ranked = labels
        .iter()
        .map(|label| VisibleLabel {
            name: label.clone(),
            score: score_lookup
                .get(label.as_str())
                .copied()
                .unwrap_or(f32::NEG_INFINITY),
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.name.cmp(&right.name))
    });
    ranked.truncate(limit);
    ranked
}

pub(crate) fn format_score(score: f32) -> String {
    format!("{score:.3}")
}

fn collect_labels<'a>(
    count: usize,
    mut name_at: impl FnMut(usize) -> Option<&'a str>,
) -> HashMap<String, usize> {
    let mut labels = HashMap::with_capacity(count);
    for index in 0..count {
        if let Some(name) = name_at(index) {
            labels.insert(String::from(name), index);
        }
    }
    labels
}

fn build_class_palette(count: usize) -> Vec<Rgb> {
    (0..count)
        .map(|index| fallback_palette_color(index, count, 11.0))
        .collect()
}

fn fallback_palette_color(index: usize, count: usize, offset: f64) -> Rgb {
    let seed = splitmix64(CLASS_COLOR_SEED ^ u64::try_from(index).expect("index should fit"));
    let jitter = f64::from(u16::try_from(seed % 3600).expect("jitter should fit")) / 10.0;
    let golden_angle = 137.507_77;
    let count = f64::from(u32::try_from(count.max(1)).expect("count should fit"));
    let index = f64::from(u32::try_from(index).expect("index should fit"));
    let saturation_step = f64::from(u8::try_from((seed >> 8) & 0x0f).expect("step should fit"));
    let lightness_step = f64::from(u8::try_from((seed >> 16) & 0x0f).expect("step should fit"));
    let hue = ((index * golden_angle) + offset + jitter / count) % 360.0;
    let saturation = 0.64 + (saturation_step / 100.0) - 0.04;
    let lightness = 0.57 + (lightness_step / 100.0) - 0.05;
    hsl_to_rgb(
        hue,
        saturation.clamp(0.54, 0.72),
        lightness.clamp(0.48, 0.64),
    )
}

fn average_color(colors: &[Rgb]) -> Rgb {
    let count = f64::from(u32::try_from(colors.len()).expect("color count should fit"));
    let (r, g, b) = colors.iter().fold((0.0, 0.0, 0.0), |(r, g, b), color| {
        (r + color.r, g + color.g, b + color.b)
    });
    Rgb::new(r / count, g / count, b / count)
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn hsl_to_rgb(hue: f64, saturation: f64, lightness: f64) -> Rgb {
    let chroma = (1.0 - (2.0 * lightness - 1.0).abs()) * saturation;
    let segment = hue / 60.0;
    let second = chroma * (1.0 - ((segment % 2.0) - 1.0).abs());
    let (r1, g1, b1) = if segment < 1.0 {
        (chroma, second, 0.0)
    } else if segment < 2.0 {
        (second, chroma, 0.0)
    } else if segment < 3.0 {
        (0.0, chroma, second)
    } else if segment < 4.0 {
        (0.0, second, chroma)
    } else if segment < 5.0 {
        (second, 0.0, chroma)
    } else {
        (chroma, 0.0, second)
    };
    let match_value = lightness - chroma / 2.0;
    Rgb::new(
        (r1 + match_value) * 255.0,
        (g1 + match_value) * 255.0,
        (b1 + match_value) * 255.0,
    )
}
