//! llama.cpp installations: list (activate / update / delete / import), prebuilt release
//! (tag → variant → download) and source build with live log.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::Instant;

use ratatui::Frame;
use ratatui::crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::Line;
use ratatui::widgets::Paragraph;
use regex::Regex;

use super::widgets::*;
use super::{Act, Answer, Ctx, Ev, Fetch, Key, Screen, Task};
use crate::installs::{self as inst, Install, Progress, Release, Variant};
use crate::ui::*;

/// Make `i` the active install.
fn activate(ctx: &mut Ctx, i: &Install) {
    ctx.state.install = Some(i.name.clone());
    ctx.save();
    ctx.reload_hw();
}

pub struct Installs {
    list: List,
    insts: Vec<Install>,
    versions: Vec<String>,
    vtask: Task<(usize, String)>,
    latest: Option<String>,
    ltask: Option<Task<Option<String>>>,
}

impl Installs {
    pub fn new() -> Self {
        let insts = inst::all_installs();
        let all = insts.clone();
        let vtask = Task::spawn(move |tx| {
            for (i, x) in all.iter().enumerate() {
                let _ = tx.send((i, inst::version(x)));
            }
        });
        let mut list = List::new(0);
        list.set_len(insts.len());
        let ltask = insts.iter().any(|i| i.kind == "release").then(|| Task::once(inst::latest_tag));
        Self { versions: vec![String::new(); insts.len()], insts, list, vtask, latest: None, ltask }
    }

    /// The newest release tag when it is newer than install `i`.
    fn newer(&self, i: &Install) -> Option<&String> {
        self.latest.as_ref().filter(|l| inst::newer_release(i, l))
    }

    fn selected(&self) -> Option<&Install> {
        self.insts.get(self.list.sel)
    }
}

impl Screen for Installs {
    fn title(&self) -> String {
        "installazioni".into()
    }

    fn keys(&self, _ctx: &Ctx) -> Vec<Key> {
        vec![
            (KeyCode::Enter, "invio", "attiva"),
            (KeyCode::Char('r'), "r", "release"),
            (KeyCode::Char('b'), "b", "build"),
            (KeyCode::Char('i'), "i", "importa"),
            (KeyCode::Char('u'), "u", "aggiorna"),
            (KeyCode::Char('x'), "x", "elimina"),
            (KeyCode::Esc, "esc", "indietro"),
        ]
    }

    fn draw(&mut self, f: &mut Frame, area: Rect, ctx: &Ctx) {
        if self.insts.is_empty() {
            let t = vec![
                Line::from(sp("nessuna installazione di llama.cpp", st(WARN))),
                Line::default(),
                Line::from(sp("r  scarica una release precompilata (consigliato)", st(FG))),
                Line::from(sp("b  compila da sorgente (Vulkan, CUDA, Metal, CPU, flag custom)", st(FG))),
                Line::from(sp("i  importa una cartella llama.cpp che hai già", st(FG))),
            ];
            f.render_widget(Paragraph::new(t), area);
            return;
        }
        let mut area = area;
        let have_latest = self.insts.iter().any(|i| i.git_ref.is_some() && i.git_ref == self.latest);
        let outdated = self.insts.iter().any(|i| self.newer(i).is_some());
        if let Some(l) = self.latest.as_ref().filter(|_| outdated && !have_latest) {
            let [note, rest] = Layout::vertical([Constraint::Length(2), Constraint::Fill(1)]).areas(area);
            let msg = format!("↑ nuova release {l} disponibile: seleziona una release e premi u (la vecchia resta)");
            f.render_widget(Paragraph::new(sp(msg, st(WARN))), note);
            area = rest;
        }
        let active = ctx.state.install.clone().or(self.insts.first().map(|i| i.name.clone()));
        self.list.mark = self.insts.iter().position(|i| Some(&i.name) == active.as_ref());
        let rows = self
            .insts
            .iter()
            .zip(&self.versions)
            .map(|(i, v)| {
                let v = match self.newer(i) {
                    _ if v.is_empty() => sp(format!("{}", spinner()), st(DIM)),
                    Some(l) => sp(format!("{v} ↑ {l}"), st(WARN)),
                    None => cell(v.clone()),
                };
                vec![
                    sp(i.name.clone(), bold(FG)),
                    cell(i.kind.clone()),
                    cell(i.backend.clone()),
                    v,
                    sp(short(&i.bin_dir), st(DIM)),
                ]
            })
            .collect();
        self.list.draw(f, area, &["nome", "tipo", "backend", "versione", "binari"], rows, ctx);
    }

    fn on(&mut self, ev: &Ev, ctx: &mut Ctx) -> Act {
        if let Ev::Answer(tag, a) = ev {
            match (*tag, a) {
                ("import", Answer::Text(t)) if !t.is_empty() => match inst::import_dir(&expand(t)) {
                    Ok(i) => {
                        ctx.toast(format!("✓ importata {} ({})", i.name, i.backend), OK);
                        activate(ctx, &i);
                        self.resume(ctx);
                    }
                    Err(e) => ctx.error(e),
                },
                ("remove", Answer::Yes) => {
                    if let Some(i) = self.selected().cloned() {
                        match inst::remove(&i) {
                            Ok(()) => ctx.toast(format!("eliminata {}", i.name), OK),
                            Err(e) => ctx.error(e),
                        }
                        if ctx.state.install.as_ref() == Some(&i.name) {
                            ctx.state.install = None;
                            ctx.save();
                        }
                        ctx.reload_hw();
                        self.resume(ctx);
                    }
                }
                ("upgrade", Answer::Yes) => {
                    if let Some(i) = self.selected()
                        && let (Some(tag), Some(v)) = (self.latest.clone(), inst::release_variant(i))
                    {
                        return Act::Push(Box::new(ReleaseScreen::upgrade(tag, v)));
                    }
                }
                ("ref", Answer::Text(r)) if !r.is_empty() => {
                    if let Some(mut i) = self.selected().cloned() {
                        i.git_ref = Some(r.clone());
                        return Act::Push(Box::new(Build::update(i)));
                    }
                }
                _ => {}
            }
            return Act::None;
        }
        if let Some(Hit::Open(i)) = self.list.on(ev) {
            let i = self.insts[i].clone();
            ctx.toast(format!("attiva: {}", i.name), OK);
            activate(ctx, &i);
            return Act::None;
        }
        match key(ev) {
            Some(KeyCode::Char('r')) => Act::Push(Box::new(ReleaseScreen::new())),
            Some(KeyCode::Char('b')) => Act::Push(Box::new(Build::new())),
            Some(KeyCode::Char('i')) => {
                let cand = ps(&inst::external_candidate());
                ctx.ask(
                    "import",
                    Modal::prompt("importa cartella llama.cpp", "checkout con build/bin o release estratta", &cand),
                );
                Act::None
            }
            Some(KeyCode::Char('u')) => {
                match self.selected() {
                    Some(i) if i.kind == "source" => {
                        let r = i.git_ref.clone().unwrap_or("master".into());
                        ctx.ask("ref", Modal::prompt(&format!("aggiorna {}", i.name), "branch / tag / commit", &r));
                    }
                    Some(i) if i.kind == "release" => match (self.newer(i), inst::release_variant(i)) {
                        (Some(l), Some(v)) => {
                            let q = format!(
                                "installo {} accanto a {} e la rendo attiva? (la vecchia resta come fallback)",
                                inst::release_name(l, &v),
                                i.name
                            );
                            ctx.ask("upgrade", Modal::ask(&q, true));
                        }
                        (Some(_), None) => ctx.toast("variante sconosciuta: installa la nuova release con r", WARN),
                        (None, _) if self.ltask.is_some() => ctx.toast("controllo l'ultima release…", WARN),
                        (None, _) => match &self.latest {
                            Some(l) => ctx.toast(format!("già all'ultima release ({l})"), OK),
                            None => ctx.toast("ultima release sconosciuta (GitHub non raggiungibile?): usa r", WARN),
                        },
                    },
                    Some(_) => ctx.toast("le cartelle importate si aggiornano a mano, non le tocco", WARN),
                    None => {}
                }
                Act::None
            }
            Some(KeyCode::Char('x')) => {
                if let Some(i) = self.selected() {
                    let q = if i.kind == "external" {
                        format!("tolgo {} dall'elenco? (la cartella originale non viene toccata)", i.name)
                    } else {
                        format!("elimino {} e tutti i suoi file?", inst::install_dir(i).display())
                    };
                    ctx.ask("remove", Modal::ask(&q, i.kind == "external"));
                }
                Act::None
            }
            _ => Act::Pass,
        }
    }

    fn tick(&mut self, _ctx: &mut Ctx) -> Act {
        if let Some(t) = self.ltask.as_mut()
            && let Some(l) = t.poll().pop()
        {
            (self.latest, self.ltask) = (l, None);
        }
        for (i, v) in self.vtask.poll() {
            if let Some(slot) = self.versions.get_mut(i) {
                *slot = v;
            }
        }
        Act::None
    }

    fn resume(&mut self, _ctx: &mut Ctx) {
        let sel = self.list.sel;
        *self = Self::new();
        self.list.sel = sel.min(self.insts.len().saturating_sub(1));
    }
}

// -- release -----------------------------------------------------------------------------------

type Loaded = Result<(Vec<Release>, Option<(u32, u32)>), String>;

pub struct ReleaseScreen {
    load: Option<Task<Loaded>>,
    releases: Vec<Release>,
    drv: Option<(u32, u32)>,
    variants: Vec<Variant>, // non-empty = choosing the variant of releases[tag]
    tag: usize,
    list: List,
    job: Option<(Task<Fetch<Install>>, Arc<AtomicBool>, Transfer)>,
    upgrade: Option<(String, String)>, // (tag, variant) to install right after loading
}

impl ReleaseScreen {
    pub fn new() -> Self {
        Self {
            load: Some(Task::once(|| inst::releases().map(|r| (r, inst::driver_cuda())))),
            releases: vec![],
            drv: None,
            variants: vec![],
            tag: 0,
            list: List::new(0),
            job: None,
            upgrade: None,
        }
    }

    /// Install `variant` of release `tag` straight away (update of a release install).
    pub fn upgrade(tag: String, variant: String) -> Self {
        let mut s = Self::new();
        s.upgrade = Some((tag, variant));
        s
    }

    fn auto_start(&mut self, ctx: &mut Ctx) {
        let Some((tag, variant)) = self.upgrade.take() else { return };
        let Some(t) = self.releases.iter().position(|r| r.tag_name == tag) else {
            return ctx.error(format!("release {tag} non trovata tra le ultime"));
        };
        self.pick_tag(t, ctx);
        match self.variants.iter().position(|v| v.variant == variant) {
            Some(i) => {
                self.list.sel = i;
                self.start();
            }
            None => ctx.toast(format!("{tag} non ha la variante {variant}: scegline una"), WARN),
        }
    }

    fn pick_tag(&mut self, i: usize, ctx: &mut Ctx) {
        let Some((os, arch)) = inst::platform() else {
            let msg = format!(
                "nessuna release precompilata per {}-{}: usa la build da sorgente",
                std::env::consts::OS,
                std::env::consts::ARCH
            );
            return ctx.error(msg);
        };
        let v = inst::release_variants(&self.releases[i], self.drv, os, arch);
        if v.is_empty() {
            return ctx.error(format!("nessun binario {os}-{arch} in questa release"));
        }
        let rec = inst::recommended(&v);
        self.tag = i;
        self.variants = v;
        self.list = List::new(rec);
        self.list.mark = Some(rec);
    }

    fn start(&mut self) {
        let tag = self.releases[self.tag].tag_name.clone();
        let v = self.variants[self.list.sel].clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let stop = cancel.clone();
        let task = Task::spawn(move |tx| {
            let mut last = String::new();
            let r = inst::install_release(&tag, &v, &mut |file, p| {
                if file != last {
                    last = file.to_string();
                    let _ = tx.send(Fetch::File(last.clone()));
                }
                let _ = tx.send(Fetch::Prog(p));
                !stop.load(Ordering::Relaxed)
            });
            let _ = tx.send(Fetch::Done(r));
        });
        self.job = Some((task, cancel, Transfer::new("")));
    }
}

impl Drop for ReleaseScreen {
    fn drop(&mut self) {
        if let Some((_, c, _)) = &self.job {
            c.store(true, Ordering::Relaxed);
        }
    }
}

impl Screen for ReleaseScreen {
    fn title(&self) -> String {
        match self.releases.get(self.tag) {
            Some(r) if !self.variants.is_empty() => format!("release {}", r.tag_name),
            _ => "release precompilata".into(),
        }
    }

    fn keys(&self, _ctx: &Ctx) -> Vec<Key> {
        if self.job.is_some() {
            return vec![(KeyCode::Esc, "esc", "interrompi")];
        }
        let what = if self.variants.is_empty() { "scegli tag" } else { "installa variante" };
        vec![(KeyCode::Enter, "invio", what), (KeyCode::Esc, "esc", "indietro")]
    }

    fn draw(&mut self, f: &mut Frame, area: Rect, ctx: &Ctx) {
        if self.load.is_some() {
            f.render_widget(Paragraph::new(sp(format!("{} interrogo GitHub…", spinner()), st(LOAD))), area);
            return;
        }
        let job_h = if self.job.is_some() { 7 } else { 0 };
        let [info, table, job] =
            Layout::vertical([Constraint::Length(2), Constraint::Fill(1), Constraint::Length(job_h)]).areas(area);
        if self.variants.is_empty() {
            f.render_widget(
                Paragraph::new(sp("ultime release di ggml-org/llama.cpp (● = più recente)", st(DIM))),
                info,
            );
            let rows = self
                .releases
                .iter()
                .map(|r| {
                    vec![sp(r.tag_name.clone(), bold(FG)), cell(r.published_at.get(..10).unwrap_or("").to_string())]
                })
                .collect();
            self.list.mark = Some(0);
            self.list.draw(f, table, &["tag", "data"], rows, ctx);
        } else {
            let drv = if cfg!(target_os = "macos") {
                String::new()
            } else {
                self.drv.map_or("nessun driver NVIDIA rilevato".into(), |(a, b)| format!("driver NVIDIA: CUDA {a}.{b}"))
            };
            f.render_widget(Paragraph::new(sp(format!("{drv}  (● = consigliata)"), st(DIM))), info);
            let rows = self
                .variants
                .iter()
                .map(|v| {
                    vec![
                        sp(v.variant.clone(), bold(FG)),
                        cell(format!("{:.0} MiB", v.size as f64 / 1048576.0)),
                        sp(v.status.clone(), st(if v.ok { OK } else { WARN })),
                    ]
                })
                .collect();
            self.list.draw(f, table, &["variante", "dimensione", "compatibilità"], rows, ctx);
        }
        if let Some((_, _, t)) = &self.job {
            let blk = panel("installazione", LOAD);
            let inner = blk.inner(job);
            f.render_widget(blk, job);
            f.render_widget(Paragraph::new(t.lines(inner.width as usize)), inner);
        }
    }

    fn on(&mut self, ev: &Ev, ctx: &mut Ctx) -> Act {
        match ev {
            Ev::Answer("reinstall", Answer::Yes) => {
                self.start();
                return Act::None;
            }
            Ev::Answer("stop", Answer::Yes) => {
                if let Some((_, c, _)) = &self.job {
                    c.store(true, Ordering::Relaxed);
                }
                return Act::None;
            }
            Ev::Answer(..) => return Act::None,
            _ => {}
        }
        if self.job.is_some() {
            if matches!(key(ev), Some(KeyCode::Esc | KeyCode::Char('q'))) {
                ctx.ask("stop", Modal::ask("interrompo l'installazione?", false));
                return Act::None;
            }
            return if matches!(ev, Ev::Key(_)) { Act::Pass } else { Act::None };
        }
        if let Some(Hit::Open(i)) = self.list.on(ev) {
            if self.variants.is_empty() {
                self.pick_tag(i, ctx);
            } else {
                let name = inst::release_name(&self.releases[self.tag].tag_name, &self.variants[i].variant);
                if inst::exists(&name) {
                    ctx.ask("reinstall", Modal::ask(&format!("{name} esiste già, reinstallo?"), false));
                } else {
                    self.start();
                }
            }
            return Act::None;
        }
        if !self.variants.is_empty() && matches!(key(ev), Some(KeyCode::Esc | KeyCode::Char('q'))) {
            self.variants.clear(); // back to the tags
            self.list = List::new(self.tag);
            return Act::None;
        }
        Act::Pass
    }

    fn tick(&mut self, ctx: &mut Ctx) -> Act {
        if let Some(t) = self.load.as_mut()
            && let Some(r) = t.poll().pop()
        {
            self.load = None;
            match r {
                Ok((rel, _)) if rel.is_empty() => {
                    ctx.error("nessuna release trovata");
                    return Act::Pop;
                }
                Ok((rel, drv)) => {
                    (self.releases, self.drv) = (rel, drv);
                    self.auto_start(ctx);
                }
                Err(e) => {
                    ctx.error(format!("impossibile contattare GitHub: {e}"));
                    return Act::Pop;
                }
            }
        }
        let Some((task, _, bar)) = self.job.as_mut() else { return Act::None };
        for m in task.poll() {
            match m {
                Fetch::File(f) => *bar = Transfer::new(&f),
                Fetch::Prog(p) => bar.update(p),
                Fetch::Done(Ok(i)) => {
                    ctx.toast(format!("✓ installata {} · ora è quella attiva", i.name), OK);
                    super::notify(&format!("installata {}", i.name));
                    activate(ctx, &i);
                    self.job = None;
                    return Act::Pop;
                }
                Fetch::Done(Err(e)) => {
                    super::notify("installazione di llama.cpp non riuscita");
                    ctx.error(format!("installazione fallita: {e}"));
                    self.job = None;
                    return Act::None;
                }
            }
        }
        Act::None
    }
}

// -- source build ------------------------------------------------------------------------------

static RE_PCT: LazyLock<Regex> = re!(r"^\[\s*(\d+)%\]");

enum Phase {
    Pick,
    Running,
    Failed(String),
}

pub struct Build {
    phase: Phase,
    opts: Vec<(&'static str, &'static str, &'static [&'static str])>,
    checks: Vec<(bool, String)>,
    list: List,
    // answers collected by the prompts
    backend: usize,
    git_ref: String,
    extra: String,
    // running
    inst: Option<Install>,
    is_update: bool,
    task: Option<Task<Result<Install, String>>>,
    lines: Option<Task<Progress>>,
    cancel: Arc<AtomicBool>,
    step: &'static str,
    pct: usize,
    log: VecDeque<Line<'static>>,
    t0: Instant,
}

impl Build {
    pub fn new() -> Self {
        let opts = inst::build_options();
        let checks = opts.iter().map(|(b, _, _)| inst::prereqs(b)).collect();
        Self {
            phase: Phase::Pick,
            opts,
            checks,
            list: List::new(0),
            backend: 0,
            git_ref: "master".into(),
            extra: String::new(),
            inst: None,
            is_update: false,
            task: None,
            lines: None,
            cancel: Arc::new(AtomicBool::new(false)),
            step: "",
            pct: 0,
            log: VecDeque::new(),
            t0: Instant::now(),
        }
    }

    /// Rebuild an existing source install at its (new) git_ref.
    pub fn update(i: Install) -> Self {
        let mut b = Self::new();
        b.is_update = true;
        b.run(i);
        b
    }

    fn run(&mut self, mut i: Install) {
        let cancel = self.cancel.clone();
        let (ltx, lrx) = std::sync::mpsc::channel();
        self.lines = Some(Task::from_rx(lrx));
        self.task = Some(Task::once(move || inst::build(&mut i, &cancel, &mut |p| drop(ltx.send(p))).map(|()| i)));
        (self.phase, self.t0, self.pct, self.step) = (Phase::Running, Instant::now(), 0, "");
        self.log.clear();
    }

    fn create(&mut self, ctx: &mut Ctx, name: &str) {
        let (backend, _, flags) = self.opts[self.backend];
        let flags =
            flags.iter().map(|f| f.to_string()).chain(self.extra.split_whitespace().map(String::from)).collect();
        match inst::new_source(name, backend, &self.git_ref, flags) {
            Ok(i) => {
                self.inst = Some(i.clone());
                self.run(i);
            }
            Err(e) => ctx.error(e),
        }
    }

    fn answer(&mut self, tag: &str, a: &Answer, ctx: &mut Ctx) {
        let backend = self.opts[self.backend].0;
        match (tag, a) {
            ("prereq", Answer::Yes) => ctx.ask("ref", Modal::prompt("branch / tag / commit", "", &self.git_ref)),
            ("ref", Answer::Text(r)) => {
                self.git_ref = if r.is_empty() { "master".into() } else { r.clone() };
                ctx.ask(
                    "extra",
                    Modal::prompt("flag cmake extra", "es. -DGGML_CUDA_FA_ALL_QUANTS=ON (vuoto = nessuno)", ""),
                );
            }
            ("extra", Answer::Text(x)) => {
                if backend == "custom" && x.is_empty() {
                    return ctx.error("per 'custom' servono dei flag");
                }
                self.extra = x.clone();
                let def = format!("src-{backend}-{}", self.git_ref);
                ctx.ask("name", Modal::prompt("nome installazione", "", &inst::sanitize(&def)));
            }
            ("name", Answer::Text(n)) if !n.is_empty() => {
                let n = inst::sanitize(n);
                self.inst = Some(Install { name: n.clone(), ..placeholder() });
                if inst::exists(&n) {
                    ctx.ask("overwrite", Modal::ask(&format!("{n} esiste già, la cancello e ricompilo?"), false));
                } else {
                    self.create(ctx, &n);
                }
            }
            ("overwrite", Answer::Yes) => {
                let n = self.inst.as_ref().map(|i| i.name.clone()).unwrap_or_default();
                self.create(ctx, &n);
            }
            ("stop", Answer::Yes) => self.cancel.store(true, Ordering::Relaxed),
            _ => {}
        }
    }
}

fn placeholder() -> Install {
    Install {
        name: String::new(),
        kind: "source".into(),
        backend: String::new(),
        bin_dir: String::new(),
        git_ref: None,
        flags: vec![],
        src_dir: None,
    }
}

impl Drop for Build {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed); // kills cmake if we quit mid-build
    }
}

impl Screen for Build {
    fn title(&self) -> String {
        match &self.inst {
            Some(i) if !matches!(self.phase, Phase::Pick) => format!("build · {}", i.name),
            _ => "build da sorgente".into(),
        }
    }

    fn keys(&self, _ctx: &Ctx) -> Vec<Key> {
        match self.phase {
            Phase::Pick => vec![(KeyCode::Enter, "invio", "scegli backend"), (KeyCode::Esc, "esc", "annulla")],
            Phase::Running => vec![(KeyCode::Esc, "esc", "interrompi build")],
            Phase::Failed(_) => vec![(KeyCode::Esc, "esc", "indietro")],
        }
    }

    fn draw(&mut self, f: &mut Frame, area: Rect, ctx: &Ctx) {
        if let Phase::Pick = self.phase {
            let rows = self
                .opts
                .iter()
                .zip(&self.checks)
                .map(|((b, d, _), (ok, msg))| {
                    vec![sp(*b, bold(FG)), cell(*d), sp(msg.clone(), st(if *ok { OK } else { WARN }))]
                })
                .collect();
            self.list.draw(f, area, &["backend", "descrizione", "prerequisiti"], rows, ctx);
            return;
        }
        let name = self.inst.as_ref().map_or(String::new(), |i| i.name.clone());
        let (title, color) = match &self.phase {
            Phase::Failed(_) => (format!("build {name} · fallita"), ERR),
            _ => (format!("build {name}"), LOAD),
        };
        let blk = panel(&title, color);
        let inner = blk.inner(area);
        f.render_widget(blk, area);
        let [head, logs] = Layout::vertical([Constraint::Length(3), Constraint::Fill(1)]).areas(inner);
        let mut b = bar(self.pct as f64 / 100.0, 40, color).to_vec();
        b.push(sp(format!(" {}%  ·  {:.0}s", self.pct, self.t0.elapsed().as_secs_f64()), st(FG)));
        let status = match &self.phase {
            Phase::Failed(e) => sp(format!("✖ {e}"), bold(ERR)),
            _ => sp(format!("{} {}", spinner(), self.step), bold(WARN)),
        };
        f.render_widget(Paragraph::new(vec![Line::from(status), Line::from(b)]), head);
        let n = logs.height as usize;
        let tail: Vec<Line> = self.log.iter().skip(self.log.len().saturating_sub(n)).cloned().collect();
        f.render_widget(Paragraph::new(tail), logs);
    }

    fn on(&mut self, ev: &Ev, ctx: &mut Ctx) -> Act {
        if let Ev::Answer(tag, a) = ev {
            self.answer(tag, a, ctx);
            return Act::None;
        }
        match self.phase {
            Phase::Running => {
                if matches!(key(ev), Some(KeyCode::Esc | KeyCode::Char('q'))) {
                    ctx.ask("stop", Modal::ask("interrompo la build?", false));
                    return Act::None;
                }
                if matches!(ev, Ev::Key(_)) { Act::Pass } else { Act::None }
            }
            Phase::Failed(_) => Act::Pass,
            Phase::Pick => {
                if let Some(Hit::Open(i)) = self.list.on(ev) {
                    self.backend = i;
                    if self.checks[i].0 {
                        self.answer("prereq", &Answer::Yes, ctx);
                    } else {
                        ctx.ask("prereq", Modal::ask(&format!("{}: continuo lo stesso?", self.checks[i].1), false));
                    }
                    return Act::None;
                }
                Act::Pass
            }
        }
    }

    fn tick(&mut self, ctx: &mut Ctx) -> Act {
        if let Some(l) = self.lines.as_mut() {
            for p in l.poll() {
                match p {
                    Progress::Step(s) => self.step = s,
                    Progress::Line(raw) => {
                        if let Some(m) = RE_PCT.captures(&raw) {
                            self.pct = m[1].parse::<usize>().unwrap_or(self.pct).min(100);
                        }
                        if raw.is_empty() {
                            continue;
                        }
                        let style = st(if raw.to_lowercase().contains("error") { ERR } else { FG });
                        self.log.push_back(Line::from(sp(raw, style)));
                        if self.log.len() > 500 {
                            self.log.pop_front();
                        }
                    }
                }
            }
        }
        let Some(t) = self.task.as_mut() else { return Act::None };
        let Some(r) = t.poll().pop() else { return Act::None };
        self.task = None;
        match r {
            Ok(i) => {
                super::notify(&format!("build finita: {}", i.name));
                if self.is_update {
                    ctx.toast(format!("✓ aggiornata {}", i.name), OK);
                    ctx.reload_hw();
                } else {
                    ctx.toast(format!("✓ compilata {} · ora è quella attiva", i.name), OK);
                    activate(ctx, &i);
                }
                Act::Pop
            }
            Err(e) => {
                super::notify("build di llama.cpp non riuscita");
                if !self.is_update
                    && let Some(i) = &self.inst
                {
                    inst::discard(&i.name);
                }
                self.phase = Phase::Failed(e);
                Act::None
            }
        }
    }
}
