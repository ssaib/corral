//! «ottimizza»: three presets of one model sized for this computer, from how much memory they
//! take: lite leaves a quarter of the VRAM free, normal is balanced, max takes the longest context
//! that fits. Each is sized by llama-fit-params, measured with llama-bench, then saved as the
//! model's presets "lite", "normal" and "max".

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ratatui::Frame;
use ratatui::crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::Line;
use ratatui::widgets::{BorderType, Paragraph, Wrap};

use crate::installs::{self, Install};
use crate::preset::{self, Fit, Gguf};
use crate::tui::widgets::{Modal, bar, block, click, key, panel, spinner, wrapped};
use crate::tui::{Act, Answer, Ctx, Ev, Key, Screen, Task, config};
use crate::ui::*;
use crate::{bench, hub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tier {
    Lite,
    Normal,
    Max,
}

pub const TIERS: [Tier; 3] = [Tier::Lite, Tier::Normal, Tier::Max];

impl Tier {
    /// Also the name of the preset it's saved as.
    pub fn name(self) -> &'static str {
        match self {
            Tier::Lite => "lite",
            Tier::Normal => "normal",
            Tier::Max => "max",
        }
    }

    pub fn desc(self) -> &'static str {
        match self {
            Tier::Lite => "poca memoria: lascia un quarto della VRAM libero per altro",
            Tier::Normal => "bilanciato: fino a 32k di contesto, tutto in GPU se ci sta",
            Tier::Max => "il contesto più lungo che entra, prompt più veloci",
        }
    }

    fn color(self) -> u32 {
        match self {
            Tier::Lite => CYAN,
            Tier::Normal => LOAD,
            Tier::Max => SRV,
        }
    }
}

/// The preset keys a tier decides; the rest of `base` (alias, port, sampling, device…) stays.
const TUNED: [&str; 15] = [
    "ctx",
    "ngl",
    "ts",
    "ot",
    "fit",
    "fit_target",
    "fit_ctx",
    "fa",
    "ctk",
    "ctv",
    "batch",
    "ubatch",
    "np",
    "cpu_moe",
    "n_cpu_moe",
];

/// A quarter of the VRAM of the biggest GPU, in steps of 256 MiB and at least the auto-fit's own
/// default margin (1024 MiB).
pub fn quarter(vram_mib: u64) -> u64 {
    let steps = (vram_mib / 4 + 128) / 256;
    (steps * 256).max(1024)
}

/// How to size a tier: the presets to try in order (the first that llama-fit-params keeps whole
/// in VRAM wins, else the last one that fits at all) and the longest context it may keep.
#[derive(Clone)]
pub struct Plan {
    pub attempts: Vec<Preset>,
    pub max_ctx: Option<u32>,
}

/// The plan of `tier` over `base`. The attempts leave the context ("0", the model's own) and the
/// layers in VRAM to the auto-fit, which lowers them to what fits; `fit_ctx` is how far it may
/// lower the context before it moves layers to the CPU instead. A full-precision KV cache comes
/// first, then q8_0 (half the memory for the same context, with a negligible loss).
/// `train_ctx`: the model's training context (0 = unknown); `vram_mib`: its biggest GPU.
pub fn plan(tier: Tier, base: &Preset, train_ctx: u32, vram_mib: u64) -> Plan {
    let mut start = base.clone();
    for f in preset::FIELDS.iter().filter(|f| TUNED.contains(&f.key)) {
        start.insert(f.key.into(), f.default.into());
    }
    let capped = |n: u32| if train_ctx > 0 { n.min(train_ctx) } else { n };
    let attempt = |kv: &str, min_ctx: Option<u32>, extra: &[(&str, String)]| {
        let mut p = start.clone();
        p.insert("ctk".into(), kv.into());
        p.insert("ctv".into(), kv.into());
        // a quantized V cache needs flash attention; a full one leaves it to llama.cpp (not every
        // backend has it for every model)
        p.insert("fa".into(), if kv == "f16" { "auto" } else { "on" }.into());
        if let Some(n) = min_ctx {
            p.insert("fit_ctx".into(), n.to_string());
        }
        for (k, v) in extra {
            p.insert(k.to_string(), v.clone());
        }
        p
    };
    match tier {
        // a fixed short context, and a quarter of the VRAM left alone: layers go to the CPU if needed
        Tier::Lite => {
            let mut extra = vec![("ctx", capped(8192).to_string()), ("np", "1".to_string())];
            if vram_mib > 0 {
                extra.push(("fit_target", quarter(vram_mib).to_string()));
            }
            Plan { attempts: vec![attempt("q8_0", None, &extra)], max_ctx: None }
        }
        // 32k, all in VRAM if a quantized cache or a shorter context gets it there
        Tier::Normal => {
            let n = capped(32768);
            let short = capped(8192);
            let attempts =
                vec![attempt("f16", Some(n), &[]), attempt("q8_0", Some(n), &[]), attempt("q8_0", Some(short), &[])];
            Plan { attempts, max_ctx: Some(n) }
        }
        // the longest context that fits, a small margin, bigger batches for faster prompts
        Tier::Max => {
            let extra = [("ubatch", "1024".to_string()), ("fit_target", "512".to_string())];
            let mut attempts = vec![];
            if train_ctx > 0 {
                attempts.push(attempt("f16", Some(train_ctx), &extra)); // the whole context, full precision
            }
            attempts.push(attempt("q8_0", None, &extra));
            Plan { attempts, max_ctx: None }
        }
    }
}

/// llama-fit-params kept the whole model in VRAM: no layer and no tensor moved to the CPU.
/// `layers`: the model's blocks (0 = unknown); llama.cpp counts one more, the output layer.
pub fn all_on_gpu(p: &Preset, layers: u32) -> bool {
    let ot = p.get("ot").map_or("", String::as_str);
    let ngl = p.get("ngl").map_or("", String::as_str);
    let every_layer = match ngl {
        "" | "auto" | "all" => true,
        n => layers > 0 && n.parse::<u32>().is_ok_and(|n| n > layers),
    };
    every_layer && ot.is_empty()
}

static RE_PROJECTED: std::sync::LazyLock<regex::Regex> =
    re!(r"projected to use (\d+) MiB of device memory vs\. (\d+) MiB of free device memory");

/// The VRAM llama-fit-params expects the settings to take, and the free VRAM, in MiB.
pub fn projected(fit: &Fit) -> Option<(u64, u64)> {
    let m = fit.verdict.iter().rev().find_map(|l| RE_PROJECTED.captures(l))?;
    Some((m[1].parse().ok()?, m[2].parse().ok()?))
}

/// Why llama-fit-params gave up, in one line.
fn failure(fit: &Fit) -> String {
    let last = fit.tail.iter().rev().find(|l| !l.trim().is_empty());
    last.map_or("llama-fit-params non riesce a farlo entrare in memoria".into(), |l| clip(l.trim(), 120))
}

/// The values of a tier: each attempt through llama-fit-params (see `plan`), the context cut to
/// the plan's longest, then one more pass with everything fixed for the VRAM it takes.
/// Blocking: about a second per pass.
pub fn fit_tier(
    inst: &Install,
    model: &Path,
    plan: &Plan,
    layers: u32,
) -> Result<(Preset, Option<(u64, u64)>), String> {
    let mut chosen: Option<Preset> = None;
    let mut error = String::from("nessun tentativo");
    for p in &plan.attempts {
        let fit = preset::fit_params(inst, model, p)?;
        if !fit.ok {
            error = failure(&fit);
            continue;
        }
        let mut fitted = p.clone();
        preset::apply_fit(&mut fitted, &fit.fitted);
        let whole = all_on_gpu(&fitted, layers);
        chosen = Some(fitted);
        if whole {
            break;
        }
    }
    let Some(mut fitted) = chosen else { return Err(error) };
    let ctx: u32 = fitted.get("ctx").and_then(|c| c.parse().ok()).unwrap_or(0);
    if let Some(max) = plan.max_ctx
        && (ctx == 0 || ctx > max)
    {
        fitted.insert("ctx".into(), max.to_string()); // less context only frees memory
    }
    fitted.insert("fit_ctx".into(), String::new()); // the context is fixed now
    let check = preset::fit_params(inst, model, &fitted)?;
    if !check.ok {
        return Err(failure(&check));
    }
    Ok((fitted, projected(&check)))
}

/// Store the tiers as presets of `model`, replacing ones with the same name, and select `chosen`.
/// A model with no saved preset yet keeps its "default" next to them.
pub fn save(state: &mut State, model: &Path, tiers: &[(Tier, &Preset)], chosen: Tier) -> Result<(), String> {
    if !state.presets.contains_key(&ps(model)) {
        let current = preset::get_preset(state, model);
        preset::put_named(state, model, DEFAULT_PRESET, &current);
    }
    for (tier, p) in tiers {
        preset::put_named(state, model, tier.name(), p);
    }
    preset::select_preset(state, model, chosen.name());
    save_state(state)
}

// -- the screen ------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Step {
    Waiting,
    Fitting,
    Measuring,
    Done,
}

struct Row {
    tier: Tier,
    plan: Plan,
    preset: Preset, // the first attempt until llama-fit-params has sized it
    step: Step,
    fits: bool,                // llama-fit-params sized it
    vram: Option<(u64, u64)>,  // MiB it takes, MiB free
    speed: Option<(f64, f64)>, // pp512, tg128 tokens/s
    error: String,
}

enum Msg {
    Step(usize, Step),
    Fitted(usize, Result<(Preset, Option<(u64, u64)>), String>),
    Measured(usize, Result<(f64, f64), String>),
    Line(String), // llama-bench's progress
}

pub struct Tune {
    model: PathBuf,
    inst: Option<Install>,
    gguf: Option<Gguf>,
    size: u64,
    rows: Vec<Row>,
    sel: usize,
    task: Option<Task<Msg>>,
    cancel: Arc<AtomicBool>,
    last: String,
    cards: Vec<Rect>,
    launch_after_save: bool,
}

impl Tune {
    pub fn new(ctx: &Ctx, model: &Path) -> Self {
        let mut t = Self {
            model: model.into(),
            inst: ctx.hw.inst.clone(),
            gguf: preset::gguf_info(model),
            size: preset::model_size(model),
            rows: vec![],
            sel: 1, // normal
            task: None,
            cancel: Arc::new(AtomicBool::new(false)),
            last: String::new(),
            cards: vec![],
            launch_after_save: false,
        };
        t.plan(ctx);
        t.run();
        t
    }

    /// The tiers' starting values, from the model's current preset and this computer's GPUs.
    fn plan(&mut self, ctx: &Ctx) {
        let base = preset::get_preset(&ctx.state, &self.model);
        let train = self.gguf.as_ref().map_or(0, |g| g.ctx);
        let vram = ctx.hw.devs.iter().map(|d| d.total).max().unwrap_or(0);
        self.rows = TIERS
            .iter()
            .map(|&tier| {
                let plan = plan(tier, &base, train, vram);
                let preset = plan.attempts[0].clone();
                Row {
                    tier,
                    plan,
                    preset,
                    step: Step::Waiting,
                    fits: false,
                    vram: None,
                    speed: None,
                    error: String::new(),
                }
            })
            .collect();
    }

    /// One thread for all three tiers, one after the other: they would compete for the GPU.
    fn run(&mut self) {
        let Some(inst) = self.inst.clone() else {
            for r in &mut self.rows {
                r.error = "nessuna installazione di llama.cpp attiva".into();
                r.step = Step::Done;
            }
            return;
        };
        self.cancel = Arc::new(AtomicBool::new(false));
        let cancel = self.cancel.clone();
        let model = self.model.clone();
        let plans: Vec<Plan> = self.rows.iter().map(|r| r.plan.clone()).collect();
        let layers = self.gguf.as_ref().map_or(0, |g| g.layers);
        self.task = Some(Task::spawn(move |tx| {
            for (i, plan) in plans.iter().enumerate() {
                if cancel.load(Ordering::Relaxed) {
                    return;
                }
                let _ = tx.send(Msg::Step(i, Step::Fitting));
                let fitted = fit_tier(&inst, &model, plan, layers);
                let ready = fitted.as_ref().ok().map(|(p, _)| p.clone());
                let _ = tx.send(Msg::Fitted(i, fitted));
                let Some(p) = ready else { continue };
                let _ = tx.send(Msg::Step(i, Step::Measuring));
                let mut progress = |l: String| {
                    let _ = tx.send(Msg::Line(l));
                };
                let rec = bench::measure(&inst, &model, &bench::params(&p), &cancel, &mut progress);
                let _ = tx.send(Msg::Measured(i, rec.map(|r| (r.pp, r.tg))));
            }
        }));
    }

    fn running(&self) -> bool {
        self.task.is_some()
    }

    /// Enter / a: save the tiers that fit, asking first when that replaces presets with their
    /// names. true = saved right away (no question needed).
    fn ask_save(&mut self, launch: bool, ctx: &mut Ctx) -> bool {
        if self.running() {
            ctx.toast("aspetta la fine dei test (esc li annulla)", WARN);
            return false;
        }
        if !self.rows[self.sel].fits {
            ctx.error(format!("il profilo {} non è stato verificato", self.rows[self.sel].tier.name()));
            return false;
        }
        self.launch_after_save = launch;
        let saved = preset::preset_names(&ctx.state, &self.model);
        let replaced: Vec<&str> = TIERS.iter().map(|t| t.name()).filter(|n| saved.iter().any(|s| s == n)).collect();
        if replaced.is_empty() {
            self.store(ctx);
            return true;
        }
        let q = format!("sostituisco i preset già salvati {}?", replaced.join(", "));
        ctx.ask("replace", Modal::ask(&q, true));
        false
    }

    fn store(&mut self, ctx: &mut Ctx) {
        let ready: Vec<(Tier, &Preset)> = self.rows.iter().filter(|r| r.fits).map(|r| (r.tier, &r.preset)).collect();
        let names: Vec<&str> = ready.iter().map(|(t, _)| t.name()).collect();
        let chosen = self.rows[self.sel].tier;
        match save(&mut ctx.state, &self.model, &ready, chosen) {
            Ok(()) => ctx.toast(format!("✓ preset {} salvati · in uso: {}", names.join(", "), chosen.name()), OK),
            Err(e) => ctx.error(e),
        }
    }

    fn command(&self, ctx: &Ctx) -> String {
        let Some(i) = &ctx.hw.inst else { return "—".into() };
        let p = &self.rows[self.sel].preset;
        let cmd = shell_join(&preset::build_cmd(&installs::server(i), &self.model, p, None));
        match p.get("api_key") {
            Some(k) if !k.is_empty() => cmd.replace(k.as_str(), "•••"),
            _ => cmd,
        }
    }

    fn header(&self, ctx: &Ctx) -> Vec<Line<'static>> {
        let shape = match &self.gguf {
            Some(g) if g.experts > 0 => format!("  {} · {} layer · MoE {} esperti", g.arch, g.layers, g.experts),
            Some(g) => format!("  {} · {} layer", g.arch, g.layers),
            None => String::new(),
        };
        let train = match self.gguf.as_ref().map_or(0, |g| g.ctx) {
            0 => String::new(),
            n => format!(" · contesto di addestramento {}", hub::thousands(n.into())),
        };
        let gpus: Vec<String> =
            ctx.hw.devs.iter().map(|d| format!("{} {} / {} MiB liberi", d.desc, d.free, d.total)).collect();
        let gpus = if gpus.is_empty() { "nessuna GPU".into() } else { gpus.join(" · ") };
        vec![
            Line::from(vec![
                sp(fname(&ps(&self.model)).to_string(), bold(FG)),
                sp(format!("  {}", size(self.size)), st(DIM)),
                sp(shape, st(CYAN)),
                sp(train, st(DIM)),
            ]),
            Line::from(sp(format!("{gpus} · RAM {} disponibili", size(ctx.hw.ram)), st(DIM))),
            Line::from(sp(
                "llama-fit-params adatta contesto e layer alla memoria, llama-bench misura la velocità: \
                 pp512 = lettura del prompt, tg128 = generazione (token/s)",
                st(DIM),
            )),
        ]
    }

    fn card_lines(&self, r: &Row, width: u16) -> Vec<Line<'static>> {
        let v = |k: &str| r.preset.get(k).cloned().unwrap_or_default();
        let label = |k: &str| sp(format!("{k:<7}"), st(DIM));
        let train = self.gguf.as_ref().map_or(0, |g| g.ctx);
        let longest = match (r.plan.max_ctx, train) {
            (Some(max), _) => max,
            (None, t) => t,
        };
        let ctx_shown = match v("ctx").as_str() {
            "0" | "" if longest > 0 => format!("fino a {}", hub::thousands(longest.into())),
            "0" | "" => "dal modello".into(),
            n => n.parse::<u64>().map_or(n.to_string(), hub::thousands),
        };
        let layers = match (v("ngl").as_str(), &self.gguf) {
            ("auto", _) if r.fits => "tutti".to_string(),
            ("auto", _) => "da stimare".to_string(),
            ("all", _) => "tutti".to_string(),
            (n, Some(g)) => format!("{n} di {}", g.layers + 1), // + the output layer, as llama.cpp counts
            (n, None) => n.to_string(),
        };
        let layers = match layers.as_str() {
            "da stimare" => layers,
            "tutti" => "tutti i layer".into(),
            _ => format!("{layers} layer"),
        };
        let kv = if v("ctk") == v("ctv") { v("ctk") } else { format!("{} / {}", v("ctk"), v("ctv")) };
        let mut lines = vec![
            Line::from(sp(r.tier.desc(), st(DIM))),
            Line::default(),
            Line::from(vec![label("ctx"), sp(ctx_shown, bold(FG))]),
            Line::from(vec![label("GPU"), sp(layers, st(FG))]),
            Line::from(vec![label("KV"), sp(kv, st(FG)), sp(format!("  fa {}", v("fa")), st(DIM))]),
        ];
        match r.vram {
            Some((used, free)) => {
                let frac = used as f64 / free.max(1) as f64;
                let color = if frac > 1.0 { ERR } else { r.tier.color() };
                let w = (width as usize).saturating_sub(9).clamp(4, 24);
                let mut b = vec![label("VRAM")];
                b.extend(bar(frac, w, color));
                lines.push(Line::from(b));
                let text = format!("{} di {} liberi", size(used << 20), size(free << 20));
                lines.push(Line::from(vec![label(""), sp(text, st(DIM))]));
            }
            None => lines.push(Line::from(vec![label("VRAM"), sp("—", st(DIM))])),
        }
        let speed = |x: Option<f64>| x.map_or("—".to_string(), |t| format!("{t:.1} t/s"));
        lines.push(Line::from(vec![label("pp512"), sp(speed(r.speed.map(|s| s.0)), bold(SRV))]));
        lines.push(Line::from(vec![label("tg128"), sp(speed(r.speed.map(|s| s.1)), bold(SLOT))]));
        lines.push(Line::default());
        let status = match r.step {
            Step::Waiting => sp("· in attesa", st(DIM)),
            Step::Fitting => sp(format!("{} stima della memoria…", spinner()), st(LOAD)),
            Step::Measuring => sp(format!("{} benchmark… {}", spinner(), clip(&self.last, 40)), st(WARN)),
            Step::Done if !r.error.is_empty() && !r.fits => sp(format!("✖ {}", r.error), st(ERR)),
            Step::Done if !r.error.is_empty() => sp(format!("⚠ entra, ma {}", r.error), st(WARN)),
            Step::Done => sp("✓ pronto", bold(OK)),
        };
        lines.push(Line::from(status));
        lines
    }
}

impl Drop for Tune {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed); // leaving the screen stops llama-bench
    }
}

impl Screen for Tune {
    fn title(&self) -> String {
        format!("ottimizza · {}", fname(&ps(&self.model)))
    }

    fn keys(&self, _ctx: &Ctx) -> Vec<Key> {
        if self.running() {
            return vec![(KeyCode::Right, "←→", "scegli"), (KeyCode::Esc, "esc", "annulla")];
        }
        vec![
            (KeyCode::Right, "←→", "scegli"),
            (KeyCode::Enter, "invio", "salva i preset"),
            (KeyCode::Char('a'), "a", "salva e avvia"),
            (KeyCode::Char('r'), "r", "ripeti"),
            (KeyCode::Esc, "esc", "indietro"),
        ]
    }

    fn draw(&mut self, f: &mut Frame, area: Rect, ctx: &Ctx) {
        let cmd = short(&self.command(ctx));
        let cmd_rows = wrapped(&[Line::from(cmd.as_str())], area.width.saturating_sub(4)).min(5);
        // the cards as tall as the longest one, the command right under them
        let card_w = area.width.saturating_sub(2) / 3;
        let text_w = card_w.saturating_sub(4);
        let card_rows = self.rows.iter().map(|r| wrapped(&self.card_lines(r, text_w), text_w)).max().unwrap_or(0);
        let [head, cards, cmd_area, _] = Layout::vertical([
            Constraint::Length(4),
            Constraint::Length(card_rows + 2),
            Constraint::Length(cmd_rows + 2),
            Constraint::Fill(1),
        ])
        .areas(area);
        f.render_widget(Paragraph::new(self.header(ctx)).wrap(Wrap { trim: true }), head);
        let slots = Layout::horizontal([Constraint::Fill(1); 3]).spacing(1).split(cards);
        self.cards = slots.to_vec();
        for (i, (r, slot)) in self.rows.iter().zip(slots.iter()).enumerate() {
            let selected = i == self.sel;
            let color = r.tier.color();
            let title = format!(" {} ", r.tier.name());
            let title_style = if selected { bold(BG).bg(rgb(color)) } else { bold(color) };
            let border = if selected { color } else { DIM };
            let mut blk = block(title, title_style, border);
            blk = blk.border_type(if selected { BorderType::Thick } else { BorderType::Rounded });
            let inner = blk.inner(*slot);
            let lines = self.card_lines(r, inner.width);
            f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }).block(blk), *slot);
        }
        let title = format!("comando · {}", self.rows[self.sel].tier.name());
        let p = Paragraph::new(sp(cmd, st(LOAD))).wrap(Wrap { trim: false }).block(panel(&title, DIM));
        f.render_widget(p, cmd_area);
    }

    fn on(&mut self, ev: &Ev, ctx: &mut Ctx) -> Act {
        if let Ev::Answer("replace", a) = ev {
            if *a != Answer::Yes {
                return Act::None;
            }
            self.store(ctx);
            return self.after_save(ctx);
        }
        if let Some(p) = click(ev)
            && let Some(i) = self.cards.iter().position(|r| r.contains(p))
        {
            self.sel = i;
            return Act::None;
        }
        match key(ev) {
            Some(KeyCode::Left) => self.sel = self.sel.saturating_sub(1),
            Some(KeyCode::Right | KeyCode::Tab) => self.sel = (self.sel + 1).min(TIERS.len() - 1),
            Some(KeyCode::Enter) => {
                self.ask_save(false, ctx);
            }
            Some(KeyCode::Char('a')) => {
                if self.ask_save(true, ctx) {
                    return self.after_save(ctx);
                }
            }
            Some(KeyCode::Char('r')) if !self.running() => {
                self.plan(ctx);
                self.run();
            }
            _ => return Act::Pass,
        }
        Act::None
    }

    fn tick(&mut self, _ctx: &mut Ctx) -> Act {
        let Some(t) = self.task.as_mut() else { return Act::None };
        let msgs = t.poll();
        let done = t.done;
        for m in msgs {
            match m {
                Msg::Step(i, s) => self.rows[i].step = s,
                Msg::Fitted(i, Ok((p, vram))) => {
                    let r = &mut self.rows[i];
                    r.preset = p;
                    r.vram = vram;
                    r.fits = true;
                }
                Msg::Fitted(i, Err(e)) => {
                    let r = &mut self.rows[i];
                    r.error = e;
                    r.step = Step::Done;
                }
                Msg::Measured(i, res) => {
                    let r = &mut self.rows[i];
                    r.step = Step::Done;
                    match res {
                        Ok(s) => r.speed = Some(s),
                        Err(e) => r.error = format!("benchmark non riuscito: {e}"),
                    }
                }
                Msg::Line(l) => self.last = l,
            }
        }
        if done {
            self.task = None;
            crate::tui::notify("ottimizzazione finita");
        }
        Act::None
    }
}

impl Tune {
    /// After saving with `a`: start the server with the chosen tier.
    fn after_save(&mut self, ctx: &mut Ctx) -> Act {
        let launch = self.launch_after_save;
        self.launch_after_save = false;
        if !launch {
            return Act::None;
        }
        let p = self.rows[self.sel].preset.clone();
        match config::launch(ctx, &self.model, &p) {
            Ok(s) => Act::Push(s),
            Err(e) => {
                ctx.error(e);
                Act::None
            }
        }
    }
}
