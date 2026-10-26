use crate::storage::{load_project, read_text, save_project_new};
use crate::{Curve, CurveKind, Evaluation, Project};
use gpui::*;
use gpui_miniapp::{MiniApp, MiniAppConfig};
use gpui_ui_kit::{Button, NumberInput, theme::ThemeExt};
use std::{path::PathBuf, sync::Arc};

actions!(speaker_designer, [UndoProject, RedoProject]);

const FIELDS: [(&str, &str); 10] = [
    ("Woofer series R", "Ω"),
    ("Woofer series L", "mH"),
    ("Woofer shunt C", "µF"),
    ("Tweeter series R", "Ω"),
    ("Tweeter series C", "µF"),
    ("Tweeter shunt L", "mH"),
    ("Woofer delay", "ms"),
    ("Tweeter delay", "ms"),
    ("FRD voltage", "V"),
    ("FRD distance", "m"),
];

pub fn run() {
    MiniApp::run(
        MiniAppConfig::new("Sonium · Speaker Designer")
            .size(1360.0, 980.0)
            .min_size(1080.0, 760.0)
            .scrollable(false)
            .with_theme(true),
        create,
    );
}

/// Creates the editor with shortcuts restricted to its focused subtree.
pub fn create(cx: &mut App) -> Entity<Designer> {
    let entity = cx.new(Designer::new);
    cx.bind_keys([
        KeyBinding::new("secondary-z", UndoProject, Some("SpeakerDesigner")),
        KeyBinding::new("secondary-shift-z", RedoProject, Some("SpeakerDesigner")),
    ]);
    entity
}

/// Embeddable passive crossover editor over measured driver responses and impedances.
pub struct Designer {
    focus: FocusHandle,
    project: Project,
    evaluation: Option<Arc<Evaluation>>,
    message: String,
    selected: usize,
    generation: u64,
    undo: Vec<Project>,
    redo: Vec<Project>,
}

impl Designer {
    /// Creates an editor without application-wide keyboard bindings.
    pub fn new(cx: &mut Context<Self>) -> Self {
        Self {
            focus: cx.focus_handle().tab_stop(true),
            project: Project::default(),
            evaluation: None,
            message: "Import both drivers’ FRD and ZMA files to calculate.".into(),
            selected: 1,
            generation: 0,
            undo: Vec::new(),
            redo: Vec::new(),
        }
    }

    /// Returns the current draft; callers must validate it before processing or export.
    pub fn project(&self) -> &Project {
        &self.project
    }

    /// Increases when a passive calculation starts, not on selection or completion.
    pub fn calculation_generation(&self) -> u64 {
        self.generation
    }

    fn replace(&mut self, project: Project, cx: &mut Context<Self>) {
        if let Err(error) = project.validate_draft() {
            self.message = format!("Change rejected: {error}");
            cx.notify();
            return;
        }
        self.undo.push(self.project.clone());
        if self.undo.len() > 20 {
            self.undo.remove(0);
        }
        self.redo.clear();
        self.project = project;
        self.calculate(cx);
    }

    fn history(&mut self, redo: bool, cx: &mut Context<Self>) {
        let previous = if redo {
            self.redo.pop()
        } else {
            self.undo.pop()
        };
        if let Some(project) = previous {
            if redo {
                self.undo.push(self.project.clone());
            } else {
                self.redo.push(self.project.clone());
            }
            self.project = project;
            self.calculate(cx);
        }
    }

    fn calculate(&mut self, cx: &mut Context<Self>) {
        self.generation += 1;
        let generation = self.generation;
        let project = self.project.clone();
        self.evaluation = None;
        self.message = "Calculating electrical / acoustic prediction…".into();
        cx.spawn(async move |this, cx| {
            let result = cx.background_spawn(async move { project.evaluate() }).await;
            let _ = this.update(cx, |this, cx| {
                if this.generation != generation {
                    return;
                }
                match result {
                    Ok(result) => {
                        this.evaluation = Some(Arc::new(result));
                        this.message = "Prediction ready · 256 logarithmic frequency points".into();
                    }
                    Err(error) => this.message = error.to_string(),
                }
                cx.notify();
            });
        })
        .detach();
        cx.notify();
    }

    fn import(&mut self, slot: usize, cx: &mut Context<Self>) {
        let generation = self.generation;
        let prompt = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Import measured FRD / ZMA".into()),
            initial_directory: None,
            extensions: Vec::new(),
        });
        cx.spawn(async move |this, cx| {
            let result = async {
                let paths = prompt.await.map_err(|e| e.to_string())?
                    .map_err(|e| e.to_string())?;
                let Some(path) = paths.and_then(|paths| paths.into_iter().next()) else { return Ok(None); };
                cx.background_spawn(async move {
                    let text = read_text(&path).map_err(|e| e.to_string())?;
                    Curve::parse(&text, if slot.is_multiple_of(2) { CurveKind::Response } else { CurveKind::Impedance },
                        path.display().to_string()).map(Some).map_err(|e| e.to_string())
                }).await
            }.await;
            let _ = this.update(cx, |this, cx| {
                match result {
                    Ok(Some(curve)) => {
                        if this.generation != generation {
                            this.message = "Import cancelled: project changed while chooser was open. Retry import.".into();
                            cx.notify(); return;
                        }
                        let mut project = this.project.clone();
                        match slot {
                            0 => project.woofer.response = Some(curve),
                            1 => project.woofer.impedance = Some(curve),
                            2 => project.tweeter.response = Some(curve),
                            _ => project.tweeter.impedance = Some(curve),
                        }
                        // A partially replaced demo must retain its synthetic warning.
                        this.replace(project, cx);
                    }
                    Err(error) => { this.message = format!("Import rejected: {error}"); cx.notify(); }
                    Ok(None) => {}
                }
            });
        }).detach();
    }

    fn open(&mut self, cx: &mut Context<Self>) {
        let generation = self.generation;
        let prompt = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Open speaker project JSON".into()),
            initial_directory: None,
            extensions: Vec::new(),
        });
        cx.spawn(async move |this, cx| {
            let result = async {
                let paths = prompt
                    .await
                    .map_err(|e| e.to_string())?
                    .map_err(|e| e.to_string())?;
                let Some(path) = paths.and_then(|p| p.into_iter().next()) else {
                    return Ok(None);
                };
                cx.background_spawn(async move {
                    load_project(&path).map(Some).map_err(|e| e.to_string())
                })
                .await
            }
            .await;
            let _ = this.update(cx, |this, cx| match result {
                Ok(Some(project)) => {
                    if this.generation == generation {
                        this.replace(project, cx);
                    } else {
                        this.message =
                            "Open cancelled: project changed while chooser was open. Retry open."
                                .into();
                        cx.notify();
                    }
                }
                Err(error) => {
                    this.message = format!("Open rejected: {error}");
                    cx.notify();
                }
                Ok(None) => {}
            });
        })
        .detach();
    }

    fn save(&mut self, cx: &mut Context<Self>) {
        if let Err(error) = self.project.validate_draft() {
            self.message = format!("Save rejected: {error}");
            cx.notify();
            return;
        }
        let prompt = cx.prompt_for_new_path(&PathBuf::from("."), Some("speaker-project.json"));
        let project = self.project.clone();
        cx.spawn(async move |this, cx| {
            let result: Result<Option<PathBuf>, String> = async {
                let path = prompt
                    .await
                    .map_err(|e| e.to_string())?
                    .map_err(|e| e.to_string())?;
                let Some(path) = path else {
                    return Ok(None);
                };
                cx.background_spawn(async move {
                    save_project_new(&path, &project).map_err(|e| e.to_string())?;
                    Ok(Some(path))
                })
                .await
            }
            .await;
            let _ = this.update(cx, |this, cx| {
                match result {
                    Ok(Some(path)) => this.message = format!("Saved {}", path.display()),
                    Err(error) => this.message = format!("Save failed: {error}"),
                    Ok(None) => return,
                }
                cx.notify();
            });
        })
        .detach();
    }

    fn value(&self, field: usize) -> f64 {
        let c = &self.project.crossover;
        match field {
            0 => c.woofer_resistance_ohm,
            1 => c.woofer_inductance_mh,
            2 => c.woofer_capacitance_uf,
            3 => c.tweeter_resistance_ohm,
            4 => c.tweeter_capacitance_uf,
            5 => c.tweeter_inductance_mh,
            6 => self.project.woofer.delay_ms,
            7 => self.project.tweeter.delay_ms,
            8 => self.project.reference_voltage_v,
            _ => self.project.reference_distance_m,
        }
    }

    fn edit(&mut self, field: usize, value: f64, cx: &mut Context<Self>) {
        let allow_zero = matches!(field, 0 | 3 | 6 | 7);
        if !value.is_finite() || value < 0.0 || (!allow_zero && value == 0.0) {
            self.message =
                "Value rejected: use a finite, positive value (R and delay may be zero).".into();
            cx.notify();
            return;
        }
        if self.value(field) == value {
            return;
        }
        let mut p = self.project.clone();
        match field {
            0 => p.crossover.woofer_resistance_ohm = value,
            1 => p.crossover.woofer_inductance_mh = value,
            2 => p.crossover.woofer_capacitance_uf = value,
            3 => p.crossover.tweeter_resistance_ohm = value,
            4 => p.crossover.tweeter_capacitance_uf = value,
            5 => p.crossover.tweeter_inductance_mh = value,
            6 => p.woofer.delay_ms = value,
            7 => p.tweeter.delay_ms = value,
            8 => p.reference_voltage_v = value,
            _ => p.reference_distance_m = value,
        }
        self.replace(p, cx);
    }

    fn driver_panel(&self, tweeter: bool, cx: &mut Context<Self>) -> impl IntoElement {
        let driver = if tweeter {
            &self.project.tweeter
        } else {
            &self.project.woofer
        };
        let offset = if tweeter { 2 } else { 0 };
        let mut panel =
            div()
                .flex()
                .flex_col()
                .gap_2()
                .child(if tweeter { "TWEETER" } else { "WOOFER" });
        for (index, (label, curve)) in [
            ("Import FRD · SPL + phase", &driver.response),
            ("Import ZMA · impedance", &driver.impedance),
        ]
        .into_iter()
        .enumerate()
        {
            let slot = offset + index;
            panel = panel
                .child(
                    Button::new(format!("import-{slot}"), label)
                        .on_click(click(cx, move |this, _, cx| this.import(slot, cx))),
                )
                .child(
                    div().text_xs().child(
                        curve
                            .as_ref()
                            .map(|c| {
                                let name = std::path::Path::new(&c.source)
                                    .file_name()
                                    .unwrap_or_default()
                                    .to_string_lossy();
                                format!("{} · {} points", name, c.samples.len())
                            })
                            .unwrap_or_else(|| "No file imported".into()),
                    ),
                );
        }
        panel.child(
            Button::new(
                if tweeter { "polarity-t" } else { "polarity-w" },
                if driver.invert {
                    "Polarity: inverted (−)"
                } else {
                    "Polarity: normal (+)"
                },
            )
            .on_click(click(cx, move |this, _, cx| {
                let mut p = this.project.clone();
                let driver = if tweeter {
                    &mut p.tweeter
                } else {
                    &mut p.woofer
                };
                driver.invert = !driver.invert;
                this.replace(p, cx);
            })),
        )
    }
}

impl Focusable for Designer {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}

impl Render for Designer {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let selected = self.selected;
        let (label, unit) = FIELDS[selected];
        let mut inspector = div()
            .flex()
            .flex_col()
            .gap_2()
            .child("COMPONENT INSPECTOR")
            .child(
                NumberInput::new(format!("field-{selected}"))
                    .value(self.value(selected))
                    .decimals(3)
                    .step(0.1)
                    .range(
                        if matches!(selected, 0 | 3 | 6 | 7) {
                            0.0
                        } else {
                            0.001
                        },
                        100_000.0,
                    )
                    .label(label)
                    .unit(unit)
                    .on_change({
                        let entity = cx.entity();
                        move |value, _, cx| {
                            entity.update(cx, |this, cx| this.edit(selected, value, cx))
                        }
                    }),
            )
            .child(div().text_xs().child(
                "Click a circuit component to edit. Values commit on Enter; Escape cancels.",
            ));
        for (index, field) in FIELDS.iter().enumerate().skip(6) {
            inspector = inspector.child(
                Button::new(
                    format!("select-{index}"),
                    format!("{}: {:.3} {}", field.0, self.value(index), field.1),
                )
                .on_click(click(cx, move |this, _, cx| {
                    this.selected = index;
                    cx.notify();
                })),
            );
        }
        let sidebar = div().id("inspector-scroll").overflow_y_scroll().w(px(280.0)).flex_shrink_0().p_4().flex().flex_col().gap_5()
            .border_r_1().border_color(theme.border)
            .child(self.driver_panel(false, cx)).child(self.driver_panel(true, cx)).child(inspector)
            .child(div().text_xs().child("FRDs must share voltage, distance and time origin. Reference values document imports; they do not rescale measurements."));
        let mut circuit = div().flex().flex_col().gap_2().child("FIXED 2-WAY PASSIVE CROSSOVER")
            .child(div().text_xs().child("Top: woofer R–L / shunt C. Bottom: tweeter R–C / shunt L. Right: measured driver load. Common input and ground."))
            .child(schematic(theme.text_primary.into()));
        for branch in 0..2 {
            let indices = if branch == 0 { [0, 1, 2] } else { [3, 4, 5] };
            let mut row = div().flex().items_center().gap_2().child(if branch == 0 {
                "Woofer"
            } else {
                "Tweeter"
            });
            for index in indices {
                row = row.child(
                    Button::new(
                        format!("component-{index}"),
                        format!(
                            "{}  {:.3} {}",
                            [
                                "R",
                                if branch == 0 { "L" } else { "C" },
                                if branch == 0 {
                                    "C ↓ GND"
                                } else {
                                    "L ↓ GND"
                                }
                            ][index % 3],
                            self.value(index),
                            FIELDS[index].1
                        ),
                    )
                    .on_click(click(cx, move |this, _, cx| {
                        this.selected = index;
                        cx.notify();
                    })),
                );
            }
            circuit = circuit.child(row);
        }
        let mut main = div().id("designer-scroll").flex_1().min_w_0().overflow_y_scroll()
            .p_5().flex().flex_col().gap_4().child(circuit)
            .child(div().text_xs().child("Predicted on-axis sum · imported complex driver response × solved terminal voltage. No room, directivity or mutual coupling model."));
        for (index, title) in [
            "SPL · dB   —   Woofer (blue) / Tweeter (orange) / Sum (green)",
            "Summed phase · degrees",
            "Input impedance magnitude · Ω",
        ]
        .into_iter()
        .enumerate()
        {
            main = main.child(plot_panel(
                title,
                self.evaluation.clone(),
                index,
                theme.border.into(),
            ));
        }
        div().id("speaker-designer").key_context("SpeakerDesigner").track_focus(&self.focus)
            .on_mouse_down(MouseButton::Left, cx.listener(|this, _, window, cx| {
                if !this.focus.contains_focused(window, cx) {
                    window.focus(&this.focus, cx);
                }
            }))
            .on_action(cx.listener(|this, _: &UndoProject, _, cx| this.history(false, cx)))
            .on_action(cx.listener(|this, _: &RedoProject, _, cx| this.history(true, cx)))
            .size_full().flex().flex_col().bg(theme.background).text_color(theme.text_primary)
            .child(div().flex().flex_wrap().items_center().gap_3().p_3().border_b_1().border_color(theme.border)
                .child(div().flex_1().text_lg().child("Sonium / Speaker Designer"))
                .child(Button::new("new", "New project").on_click(click(cx, |this, _, cx| this.replace(Project::default(), cx))))
                .child(Button::new("open", "Open JSON").on_click(click(cx, |this, _, cx| this.open(cx))))
                .child(Button::new("save", "Save as JSON").on_click(click(cx, |this, _, cx| this.save(cx))))
                .child(Button::new("undo", "Undo").disabled(self.undo.is_empty()).on_click(click(cx, |this, _, cx| {
                    this.history(false, cx);
                })))
                .child(Button::new("redo", "Redo").disabled(self.redo.is_empty()).on_click(click(cx, |this, _, cx| {
                    this.history(true, cx);
                })))
                .child(Button::new("demo", "Load synthetic demo").on_click(click(cx, |this, _, cx| this.replace(Project::demo(), cx)))))
            .child(div().px_4().py_2().text_sm().child(if self.project.synthetic {
                "SYNTHETIC DATA · demonstration only · electrical prediction, no acoustic verification"
            } else { "MEASURED-INPUT WORKSPACE · prediction only · imported provenance remains user supplied" }))
            .child(div().flex().flex_1().min_h_0().child(sidebar).child(main))
            .child(div().px_4().py_2().border_t_1().border_color(theme.border).text_sm().child(self.message.clone()))
    }
}

fn stroke(
    window: &mut Window,
    points: impl IntoIterator<Item = Point<Pixels>>,
    color: Hsla,
    width: f32,
) {
    let mut path = PathBuilder::stroke(px(width));
    let mut points = points.into_iter();
    if let Some(start) = points.next() {
        path.move_to(start);
        for p in points {
            path.line_to(p);
        }
        if let Ok(path) = path.build() {
            window.paint_path(path, color);
        }
    }
}

fn click(
    cx: &Context<Designer>,
    callback: impl Fn(&mut Designer, &mut Window, &mut Context<Designer>) + 'static,
) -> impl Fn(&mut Window, &mut App) + 'static {
    let entity = cx.entity();
    move |window, cx| entity.update(cx, |this, cx| callback(this, window, cx))
}

fn schematic(color: Hsla) -> impl IntoElement {
    canvas(
        |_, _, _| (),
        move |bounds, (), window, _| {
            let scale = f32::from(bounds.size.width) / 520.0;
            let p = |x: f32, y: f32| bounds.origin + point(px(x * scale), px(y));
            stroke(window, [p(25.0, 25.0), p(25.0, 105.0)], color, 1.5);
            stroke(window, [p(5.0, 65.0), p(5.0, 145.0)], color, 1.5);
            for branch in 0..2 {
                let y = 25.0 + branch as f32 * 80.0;
                stroke(window, [p(25.0, y), p(110.0, y)], color, 1.5);
                stroke(
                    window,
                    (0..=10).map(|i| {
                        p(
                            110.0 + i as f32 * 4.0,
                            y + if i == 0 || i == 10 {
                                0.0
                            } else if i % 2 == 0 {
                                -5.0
                            } else {
                                5.0
                            },
                        )
                    }),
                    color,
                    1.5,
                );
                stroke(window, [p(150.0, y), p(220.0, y)], color, 1.5);
                if branch == 0 {
                    stroke(
                        window,
                        (0..=60).map(|i| {
                            p(
                                220.0 + i as f32,
                                y - (i as f32 * std::f32::consts::PI / 15.0).sin().abs() * 9.0,
                            )
                        }),
                        color,
                        1.5,
                    );
                } else {
                    stroke(window, [p(220.0, y), p(245.0, y)], color, 1.5);
                    stroke(window, [p(245.0, y - 10.0), p(245.0, y + 10.0)], color, 1.5);
                    stroke(window, [p(255.0, y - 10.0), p(255.0, y + 10.0)], color, 1.5);
                    stroke(window, [p(255.0, y), p(280.0, y)], color, 1.5);
                }
                stroke(
                    window,
                    [p(280.0, y), p(455.0, y), p(455.0, y + 10.0)],
                    color,
                    1.5,
                );
                stroke(
                    window,
                    [
                        p(444.0, y + 10.0),
                        p(466.0, y + 10.0),
                        p(466.0, y + 30.0),
                        p(444.0, y + 30.0),
                        p(444.0, y + 10.0),
                    ],
                    color,
                    1.5,
                );
                stroke(
                    window,
                    [p(455.0, y + 30.0), p(455.0, y + 40.0), p(30.0, y + 40.0)],
                    color,
                    1.5,
                );
                // The return crosses the input bus with a bridge, never a junction.
                stroke(
                    window,
                    [
                        p(30.0, y + 40.0),
                        p(28.0, y + 35.0),
                        p(22.0, y + 35.0),
                        p(20.0, y + 40.0),
                        p(5.0, y + 40.0),
                    ],
                    color,
                    1.5,
                );
                stroke(window, [p(345.0, y), p(345.0, y + 12.0)], color, 1.5);
                if branch == 0 {
                    stroke(window, [p(335.0, y + 12.0), p(355.0, y + 12.0)], color, 1.5);
                    stroke(window, [p(335.0, y + 23.0), p(355.0, y + 23.0)], color, 1.5);
                    stroke(window, [p(345.0, y + 23.0), p(345.0, y + 40.0)], color, 1.5);
                } else {
                    stroke(
                        window,
                        (0..=40).map(|i| {
                            p(
                                345.0 + (i as f32 * std::f32::consts::PI / 10.0).sin().abs() * 7.0,
                                y + 12.0 + i as f32 * 0.7,
                            )
                        }),
                        color,
                        1.5,
                    );
                }
            }
        },
    )
    .w_full()
    .h(px(160.0))
}

/// Plots branch and summed responses. Invalid or empty response vectors render an empty plot.
pub fn response_plot(
    title: &'static str,
    evaluation: Option<Arc<Evaluation>>,
    border: Hsla,
) -> impl IntoElement {
    let evaluation = evaluation.filter(|e| {
        let frequencies = &e.frequencies_hz;
        frequencies.len() >= 2
            && frequencies.iter().all(|f| f.is_finite() && *f > 0.0)
            && frequencies.windows(2).all(|pair| pair[0] < pair[1])
            && [&e.woofer_db, &e.tweeter_db, &e.sum_db]
                .iter()
                .all(|values| {
                    values.len() == frequencies.len() && values.iter().all(|v| v.is_finite())
                })
    });
    plot_panel(title, evaluation, 0, border)
}

fn plot_panel(
    title: &'static str,
    evaluation: Option<Arc<Evaluation>>,
    kind: usize,
    border: Hsla,
) -> impl IntoElement {
    let range = evaluation.as_ref().map(|e| {
        let values: Vec<f64> = match kind {
            0 => e
                .woofer_db
                .iter()
                .chain(&e.tweeter_db)
                .chain(&e.sum_db)
                .copied()
                .collect(),
            1 => e.sum_phase_deg.clone(),
            _ => e.input_impedance_ohm.clone(),
        };
        let low = values
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f64::INFINITY, f64::min);
        let high = values
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        let padding = ((high - low) * 0.08).max(1.0);
        (low - padding, high + padding)
    });
    let axis = evaluation.as_ref().map(|e| format!("{:.0} Hz                                     logarithmic frequency                                     {:.0} Hz",
        e.frequencies_hz[0], e.frequencies_hz[e.frequencies_hz.len() - 1]));
    let mut panel = div()
        .border_1()
        .border_color(border)
        .rounded_md()
        .p_3()
        .flex()
        .flex_col()
        .gap_1()
        .child(div().text_sm().child(title));
    if let Some((low, high)) = range {
        panel = panel.child(
            div()
                .text_xs()
                .child(format!("Vertical range: {low:.1} … {high:.1}")),
        );
    }
    panel
        .child(
            canvas(
                |_, _, _| (),
                move |bounds, (), window, _| {
                    let Some(e) = evaluation else {
                        return;
                    };
                    let Some((low, high)) = range else {
                        return;
                    };
                    let width = f32::from(bounds.size.width);
                    let height = f32::from(bounds.size.height);
                    let origin = bounds.origin;
                    for line in 0..=4 {
                        let y = height * line as f32 / 4.0;
                        stroke(
                            window,
                            [
                                origin + point(px(0.0), px(y)),
                                origin + point(px(width), px(y)),
                            ],
                            border,
                            1.0,
                        );
                    }
                    let f0 = e.frequencies_hz[0].ln();
                    let span = e.frequencies_hz[e.frequencies_hz.len() - 1].ln() - f0;
                    for frequency in [
                        20_f64, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0,
                    ] {
                        let x = ((frequency.ln() - f0) / span) as f32 * width;
                        if x >= 0.0 && x <= width {
                            stroke(
                                window,
                                [
                                    origin + point(px(x), px(0.0)),
                                    origin + point(px(x), px(height)),
                                ],
                                border,
                                1.0,
                            );
                        }
                    }
                    let series: Vec<(&[f64], Hsla)> = match kind {
                        0 => vec![
                            (&e.woofer_db, rgb(0x62a8ff).into()),
                            (&e.tweeter_db, rgb(0xf5a257).into()),
                            (&e.sum_db, rgb(0x56c9a2).into()),
                        ],
                        1 => vec![(&e.sum_phase_deg, rgb(0xb18bfa).into())],
                        _ => vec![(&e.input_impedance_ohm, rgb(0x56c9c3).into())],
                    };
                    for (values, color) in series {
                        stroke(
                            window,
                            e.frequencies_hz
                                .iter()
                                .zip(values)
                                .filter(|(_, v)| v.is_finite())
                                .map(|(f, v)| {
                                    origin
                                        + point(
                                            px(((f.ln() - f0) / span) as f32 * width),
                                            px((1.0 - (v - low) / (high - low)) as f32 * height),
                                        )
                                }),
                            color,
                            1.8,
                        );
                    }
                },
            )
            .w_full()
            .h(px(140.0)),
        )
        .child(div().text_xs().child(axis.unwrap_or_else(|| {
            "Import four curves or load synthetic demo to view predictions.".into()
        })))
}
