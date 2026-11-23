//! Hugging Face: search / suggestions, then quant + mmproj choice and the download.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ratatui::Frame;
use ratatui::crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::Line;
use ratatui::widgets::Paragraph;

use super::widgets::*;
use super::{Act, Answer, Ctx, Ev, Fetch, Key, Screen, Task};
use crate::hub::{self, OVERHEAD, Repo, RepoFiles};
use crate::ui::*;

type Found = Result<Vec<(&'static str, Repo)>, String>;

pub struct Search {
    suggest: bool,
    input: TextInput,
    typing: bool, // focus on the search field (else on the results)
    task: Option<Task<Found>>,
    results: Vec<(&'static str, Repo)>,
    list: List,
    shown: String, // what the results are for
}

impl Search {
    pub fn new() -> Self {
        Self {
            suggest: false,
            input: TextInput::default(),
            typing: true,
            task: None,
            results: vec![],
            list: List::new(0),
            shown: String::new(),
        }
    }

    pub fn suggest(ctx: &Ctx) -> Self {
        let (budget, ram) = (ctx.hw.budget(), ctx.hw.ram);
        Self {
            suggest: true,
            typing: false,
            task: Some(Task::once(move || hub::suggest(budget, ram))),
            shown: "i più scaricati che girano qui".into(),
            ..Self::new()
        }
    }

    fn run(&mut self) {
        let q = self.input.value.trim().to_string();
        if q.is_empty() {
            return;
        }
        self.shown = format!("risultati per '{q}'");
        self.task = Some(Task::once(move || hub::search(&q).map(|v| v.into_iter().map(|r| ("", r)).collect())));
        self.typing = false;
    }
}

impl Screen for Search {
    fn title(&self) -> String {
        if self.suggest { "suggeriti".into() } else { "cerca su Hugging Face".into() }
    }

    fn keys(&self, _ctx: &Ctx) -> Vec<Key> {
        let mut k = vec![(KeyCode::Enter, "invio", if self.typing { "cerca" } else { "apri repo" })];
        if !self.suggest {
            k.push((KeyCode::Tab, "tab", if self.typing { "risultati" } else { "campo di ricerca" }));
        }
        k.push((KeyCode::Esc, "esc", "indietro"));
        k
    }

    fn draw(&mut self, f: &mut Frame, area: Rect, ctx: &Ctx) {
        let input_h = if self.suggest { 0 } else { 3 };
        let [top, info, body] =
            Layout::vertical([Constraint::Length(input_h), Constraint::Length(2), Constraint::Fill(1)]).areas(area);
        if !self.suggest {
            let blk = if self.typing { lit("cerca", SIDE) } else { panel("cerca", DIM) };
            let inner = blk.inner(top);
            f.render_widget(blk, top);
            if self.input.value.is_empty() && !self.typing {
                f.render_widget(Paragraph::new(sp("es. 'qwen2.5 vl 7b', 'gemma coder'", st(DIM))), inner);
            }
            self.input.draw(f, inner, self.typing);
        }
        let (budget, ram) = (ctx.hw.budget(), ctx.hw.ram);
        let mut l = vec![Line::from(sp(
            format!(
                "VRAM libera {} · RAM disponibile {} · 'Q4 su' = dove sta la versione Q4_K_M",
                size(budget),
                size(ram)
            ),
            st(DIM),
        ))];
        if self.task.is_some() {
            l.push(Line::from(sp(format!("{} interrogo Hugging Face…", spinner()), st(LOAD))));
        } else if !self.shown.is_empty() {
            l.push(Line::from(sp(format!("{} · {}", self.shown, self.results.len()), st(FG))));
        }
        f.render_widget(Paragraph::new(l), info);
        if self.results.is_empty() {
            if !self.shown.is_empty() && self.task.is_none() {
                f.render_widget(Paragraph::new(sp("nessun risultato", st(WARN))), body);
            }
            return;
        }
        let rows = self
            .results
            .iter()
            .map(|(cat, m)| {
                let (fit, color) = hub::repo_need(m).map_or(("?", DIM), |n| hub::fit_label(n, budget, ram));
                let mut row = if self.suggest { vec![sp(*cat, st(SLOT))] } else { vec![] };
                row.extend([
                    sp(m.id.clone(), bold(FG)),
                    cell(m.params.map_or("?".into(), |p| format!("{:.1}B", p as f64 / 1e9))),
                    cell(m.arch.clone().unwrap_or("?".into())),
                    cell(m.ctx.map_or("?".into(), |c| c.to_string())),
                    cell(hub::thousands(m.downloads)),
                    sp(fit, st(color)),
                ]);
                row
            })
            .collect();
        let mut head = if self.suggest { vec!["categoria"] } else { vec![] };
        head.extend(["repo", "parametri", "arch", "ctx max", "download/30gg", "Q4 su"]);
        self.list.draw(f, body, &head, rows, ctx);
    }

    fn on(&mut self, ev: &Ev, _ctx: &mut Ctx) -> Act {
        if !self.suggest && click(ev).is_some_and(|_| self.input.on(ev).is_some()) {
            self.typing = true;
            return Act::None;
        }
        if self.typing {
            match self.input.on(ev) {
                Some(Edit::Submit) => {
                    self.run();
                    return Act::None;
                }
                Some(Edit::Cancel) => return Act::Pop,
                Some(Edit::Changed) => return Act::None,
                None if matches!(key(ev), Some(KeyCode::Tab | KeyCode::Down)) && !self.results.is_empty() => {
                    self.typing = false;
                    return Act::None;
                }
                // a click outside the field leaves it and goes on to the results below
                None if click(ev).is_some() => self.typing = false,
                None if matches!(ev, Ev::Key(_)) => return Act::Pass,
                None => return Act::None,
            }
        }
        if let Some(Hit::Open(i)) = self.list.on(ev) {
            return Act::Push(Box::new(RepoScreen::new(&self.results[i].1.id)));
        }
        match key(ev) {
            Some(KeyCode::Tab | KeyCode::Char('/')) if !self.suggest => {
                self.typing = true;
                Act::None
            }
            _ => Act::Pass,
        }
    }

    fn tick(&mut self, ctx: &mut Ctx) -> Act {
        if let Some(t) = self.task.as_mut()
            && let Some(r) = t.poll().pop()
        {
            self.task = None;
            match r {
                Ok(v) => {
                    self.results = v;
                    self.list = List::new(0);
                }
                Err(e) => {
                    ctx.error(format!("errore Hugging Face: {e}"));
                    self.typing = !self.suggest;
                }
            }
        }
        Act::None
    }
}

// -- one repo: quant, mmproj, download --------------------------------------------------

struct Download {
    task: Task<Fetch<()>>,
    cancel: Arc<AtomicBool>,
    files: Vec<(String, u64)>,
    current: usize,
    bar: Transfer,
    model: PathBuf,
    mmproj: Option<PathBuf>,
}

pub struct RepoScreen {
    repo: String,
    task: Option<Task<Result<RepoFiles, String>>>,
    files: Option<RepoFiles>,
    list: List,
    mm_default: Option<usize>,
    quant: usize,
    mmproj: Option<usize>,
    dl: Option<Download>,
}

impl RepoScreen {
    pub fn new(repo: &str) -> Self {
        let r = repo.to_string();
        Self {
            repo: repo.into(),
            task: Some(Task::once(move || hub::repo_files(&r))),
            files: None,
            list: List::new(0),
            mm_default: None,
            quant: 0,
            mmproj: None,
            dl: None,
        }
    }

    fn todo(&self) -> Vec<(String, u64)> {
        let Some(rf) = &self.files else { return vec![] };
        rf.quants[self.quant].files.iter().chain(self.mmproj.map(|i| &rf.mmprojs[i])).cloned().collect()
    }

    fn check_disk(&mut self, ctx: &mut Ctx) {
        let total: u64 = self.todo().iter().map(|f| f.1).sum();
        let dest = hub::repo_dir(&self.repo);
        let _ = std::fs::create_dir_all(&dest);
        match disk_free(&dest) {
            Some(free) if total > free => ctx.ask(
                "disk",
                Modal::ask(&format!("servono {} ma il disco ha {} liberi, continuo?", size(total), size(free)), false),
            ),
            _ => self.start(),
        }
    }

    fn start(&mut self) {
        let Some(rf) = &self.files else { return };
        let files = self.todo();
        let dest = hub::repo_dir(&self.repo);
        let cancel = Arc::new(AtomicBool::new(false));
        let (repo, list, sha, stop) = (self.repo.clone(), files.clone(), rf.sha.clone(), cancel.clone());
        let task = Task::spawn(move |tx| {
            for (f, n) in &list {
                let _ = tx.send(Fetch::File(f.clone()));
                let r = hub::fetch(&repo, f, *n, sha.get(f).map(String::as_str), &mut |p| {
                    let _ = tx.send(Fetch::Prog(p));
                    !stop.load(Ordering::Relaxed)
                });
                if let Err(e) = r {
                    let _ = tx.send(Fetch::Done(Err(e)));
                    return;
                }
            }
            let _ = tx.send(Fetch::Done(Ok(())));
        });
        let model = hub::local(&dest, &rf.quants[self.quant].files[0].0);
        let mmproj = self.mmproj.map(|i| hub::local(&dest, &rf.mmprojs[i].0));
        self.dl =
            Some(Download { task, cancel, current: 0, bar: Transfer::new(fname(&files[0].0)), files, model, mmproj });
    }
}

impl Drop for RepoScreen {
    fn drop(&mut self) {
        if let Some(d) = &self.dl {
            d.cancel.store(true, Ordering::Relaxed);
        }
    }
}

impl Screen for RepoScreen {
    fn title(&self) -> String {
        self.repo.clone()
    }

    fn keys(&self, _ctx: &Ctx) -> Vec<Key> {
        if self.dl.is_some() {
            return vec![(KeyCode::Esc, "esc", "interrompi download")];
        }
        vec![(KeyCode::Enter, "invio", "scarica questa quantizzazione"), (KeyCode::Esc, "esc", "indietro")]
    }

    fn draw(&mut self, f: &mut Frame, area: Rect, ctx: &Ctx) {
        let Some(rf) = &self.files else {
            let msg = format!("{} leggo i file di {}…", spinner(), self.repo);
            f.render_widget(Paragraph::new(sp(msg, st(LOAD))), area);
            return;
        };
        let dl_h = if self.dl.is_some() { 7 } else { 0 };
        let [info, table, dl] =
            Layout::vertical([Constraint::Length(2), Constraint::Fill(1), Constraint::Length(dl_h)]).areas(area);
        let budget = ctx.hw.budget();
        let mm_size = self.mm_default.map_or(0, |i| rf.mmprojs[i].1);
        let mut l = vec![Line::from(sp(
            format!(
                "VRAM libera {} · stima = file{} + {} (KV ~8k ctx)",
                size(budget),
                if rf.mmprojs.is_empty() { "" } else { " + mmproj" },
                size(OVERHEAD)
            ),
            st(DIM),
        ))];
        if rf.gated {
            l.push(Line::from(sp("repo gated: serve un token (HF_TOKEN o `hf auth login`)", st(WARN))));
        }
        f.render_widget(Paragraph::new(l), info);
        let rows = rf
            .quants
            .iter()
            .map(|q| {
                let need = q.size + mm_size + OVERHEAD;
                let (label, color) = hub::fit_label(need, budget, ctx.hw.ram);
                let parts = if q.files.len() > 1 { format!("{} parti", q.files.len()) } else { String::new() };
                vec![
                    sp(q.quant.clone(), bold(FG)),
                    cell(size(q.size)),
                    cell(size(need)),
                    sp(label, st(color)),
                    cell(parts),
                ]
            })
            .collect();
        self.list.draw(f, table, &["quant", "file", "stima memoria", "gira su", ""], rows, ctx);
        if let Some(d) = &self.dl {
            let title = format!("download {}/{}", d.current + 1, d.files.len());
            let blk = panel(&title, LOAD);
            let inner = blk.inner(dl);
            f.render_widget(blk, dl);
            f.render_widget(Paragraph::new(d.bar.lines(inner.width as usize)), inner);
        }
    }

    fn on(&mut self, ev: &Ev, ctx: &mut Ctx) -> Act {
        match ev {
            Ev::Answer("mmproj", Answer::Pick(i)) => {
                let n = self.files.as_ref().map_or(0, |r| r.mmprojs.len());
                self.mmproj = (*i < n).then_some(*i);
                self.check_disk(ctx);
                return Act::None;
            }
            Ev::Answer("disk", Answer::Yes) => {
                self.start();
                return Act::None;
            }
            Ev::Answer("stop", Answer::Yes) => {
                if let Some(d) = &self.dl {
                    d.cancel.store(true, Ordering::Relaxed);
                }
                return Act::None;
            }
            Ev::Answer(..) => return Act::None,
            _ => {}
        }
        if self.dl.is_some() {
            if matches!(key(ev), Some(KeyCode::Esc | KeyCode::Char('q'))) {
                ctx.ask("stop", Modal::ask("interrompo il download? (riprende da dove era rimasto)", false));
                return Act::None;
            }
            return if matches!(ev, Ev::Key(_)) { Act::Pass } else { Act::None }; // Ctrl+C still quits
        }
        let Some(rf) = &self.files else { return Act::Pass };
        if let Some(Hit::Open(i)) = self.list.on(ev) {
            self.quant = i;
            if rf.mmprojs.is_empty() {
                self.mmproj = None;
                self.check_disk(ctx);
            } else {
                let mut rows: Vec<_> =
                    rf.mmprojs.iter().map(|(f, n)| vec![cell(f.clone()), sp(size(*n), st(DIM))]).collect();
                rows.push(vec![sp("nessuno", st(DIM)), cell("")]);
                let sel = self.mm_default.unwrap_or(0);
                ctx.ask("mmproj", Modal::choose("proiettore vision (mmproj)", &["file", "dimensione"], rows, sel));
            }
            return Act::None;
        }
        Act::Pass
    }

    fn tick(&mut self, ctx: &mut Ctx) -> Act {
        if let Some(t) = self.task.as_mut()
            && let Some(r) = t.poll().pop()
        {
            self.task = None;
            match r {
                Ok(rf) => {
                    let names: Vec<&String> = rf.mmprojs.iter().map(|m| &m.0).collect();
                    self.mm_default = hub::best_mmproj(&names);
                    let mm = self.mm_default.map_or(0, |i| rf.mmprojs[i].1);
                    let rec = hub::recommend(&rf.quants, ctx.hw.budget(), mm);
                    self.list = List::new(rec.unwrap_or(0));
                    self.list.mark = rec;
                    self.files = Some(rf);
                }
                Err(e) => {
                    ctx.error(e);
                    return Act::Pop;
                }
            }
        }
        let Some(d) = self.dl.as_mut() else { return Act::None };
        for m in d.task.poll() {
            match m {
                Fetch::File(f) => {
                    d.current = d.files.iter().position(|x| x.0 == f).unwrap_or(0);
                    d.bar = Transfer::new(fname(&f));
                }
                Fetch::Prog(p) => d.bar.update(p),
                Fetch::Done(Ok(())) => {
                    ctx.toast(format!("✓ scaricato in {}", short(&ps(&hub::repo_dir(&self.repo)))), OK);
                    super::notify(&format!("scaricato {}", fname(&ps(&d.model))));
                    let (model, mm) = (d.model.clone(), d.mmproj.clone());
                    self.dl = None;
                    return Act::Replace(Box::new(super::config::Config::new(ctx, &model, mm)));
                }
                Fetch::Done(Err(e)) => {
                    super::notify(&format!("download non riuscito: {e}"));
                    ctx.error(e);
                    self.dl = None;
                    return Act::None;
                }
            }
        }
        Act::None
    }
}
