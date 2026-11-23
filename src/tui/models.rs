//! Local models (every source, size, last use, delete, update check, extra folders, orphaned
//! presets) and the named presets of one model (pick, create, rename, duplicate, delete, export, import).

use std::path::{Path, PathBuf};

use ratatui::Frame;
use ratatui::crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::Line;
use ratatui::widgets::Paragraph;

use super::widgets::*;
use super::{Act, Answer, Ctx, Ev, Key, Screen, Task, config, hf};
use crate::models::{self, Local, Update};
use crate::preset;
use crate::ui::*;

type Checked = Result<(String, Update), String>;

pub struct Models {
    list: List,
    rows: Vec<Local>,
    vision: Vec<bool>,
    check: Option<Task<Checked>>,
    plan: Vec<PathBuf>, // files the pending delete removes
    repo: String,       // repo of the pending update
    dir: usize,         // extra folder pending removal
}

impl Models {
    pub fn new(ctx: &Ctx) -> Self {
        let rows = models::scan(&ctx.state);
        let vision = rows.iter().map(|r| !preset::mmprojs_near(&r.path).is_empty()).collect();
        let last = rows.iter().position(|r| Some(ps(&r.path)) == ctx.state.last_model);
        let mut list = List::new(last.unwrap_or(0));
        list.mark = last;
        list.set_len(rows.len());
        Self { list, rows, vision, check: None, plan: vec![], repo: String::new(), dir: 0 }
    }

    fn reload(&mut self, ctx: &Ctx) {
        let sel = self.rows.get(self.list.sel).map(|r| r.path.clone());
        let check = self.check.take();
        *self = Self::new(ctx);
        self.check = check;
        if let Some(i) = sel.and_then(|p| self.rows.iter().position(|r| r.path == p)) {
            self.list.sel = i;
        }
    }

    fn selected(&self) -> Option<&Local> {
        self.rows.get(self.list.sel)
    }

    /// Enter: a model with several presets opens their list, otherwise the editor.
    fn open(&self, i: usize, ctx: &Ctx) -> Act {
        let m = &self.rows[i].path;
        if preset::preset_names(&ctx.state, m).len() > 1 {
            Act::Push(Box::new(Presets::new(m)))
        } else {
            Act::Push(Box::new(config::Config::new(ctx, m, None)))
        }
    }

    fn ask_delete(&mut self, ctx: &mut Ctx) {
        let Some((path, managed, source)) = self.selected().map(|m| (m.path.clone(), m.managed, m.source)) else {
            return;
        };
        if !managed {
            return ctx.toast(format!("lo gestisce {}: cancellalo da lì (qui lo leggo soltanto)", source), WARN);
        }
        self.plan = models::delete_plan(&ctx.state, &path, &self.rows);
        let total: u64 = self.plan.iter().map(|f| std::fs::metadata(f).map_or(0, |x| x.len())).sum();
        let mut body: Vec<Line<'static>> =
            self.plan.iter().map(|f| Line::from(cell(format!("  {}", short(&ps(f)))))).collect();
        let n = preset::preset_names(&ctx.state, &path).len();
        let saved = ctx.state.presets.contains_key(&ps(&path));
        body.push(Line::default());
        if saved {
            body.push(Line::from(sp(format!("e i suoi preset ({n})"), st(DIM))));
        }
        body.push(Line::from(sp(format!("libera {} · non si può annullare. Elimino?", size(total)), st(FG))));
        ctx.ask("delete", Modal::confirm("elimina modello", body, false));
    }

    fn ask_dirs(&self, ctx: &mut Ctx) {
        let mut rows: Vec<_> =
            ctx.state.model_dirs.iter().map(|d| vec![cell(d.clone()), sp("rimuovi", st(DIM))]).collect();
        rows.push(vec![sp("+ aggiungi cartella…", st(OK)), cell("")]);
        let sel = rows.len() - 1;
        ctx.ask("dirs", Modal::choose("cartelle modelli extra", &["cartella", ""], rows, sel));
    }

    fn answer(&mut self, tag: &str, a: &Answer, ctx: &mut Ctx) -> Act {
        match (tag, a) {
            ("delete", Answer::Yes) => {
                let Some(m) = self.selected().map(|m| m.path.clone()) else { return Act::None };
                match models::delete(&mut ctx.state, &m, &self.plan) {
                    Ok(()) => ctx.toast(format!("eliminato {}", fname(&ps(&m))), OK),
                    Err(e) => ctx.error(e),
                }
                self.reload(ctx);
            }
            ("update", Answer::Yes) => return Act::Push(Box::new(hf::RepoScreen::new(&self.repo))),
            ("dirs", Answer::Pick(i)) if *i < ctx.state.model_dirs.len() => {
                self.dir = *i;
                let q = format!("smetto di cercare modelli in {}? (i file restano)", ctx.state.model_dirs[*i]);
                ctx.ask("rmdir", Modal::ask(&q, true));
            }
            ("dirs", Answer::Pick(_)) => ctx.ask(
                "adddir",
                Modal::prompt("aggiungi cartella modelli", "cartella con file .gguf (anche in sottocartelle)", "~/"),
            ),
            ("adddir", Answer::Text(t)) if !t.is_empty() => {
                let p = expand(t);
                if !p.is_dir() {
                    ctx.error(format!("{} non è una cartella", p.display()));
                } else if !ctx.state.model_dirs.contains(&ps(&p)) {
                    ctx.state.model_dirs.push(ps(&p));
                    ctx.save();
                    self.reload(ctx);
                    ctx.toast(format!("aggiunta {}", p.display()), OK);
                }
            }
            ("rmdir", Answer::Yes) if self.dir < ctx.state.model_dirs.len() => {
                ctx.state.model_dirs.remove(self.dir);
                ctx.save();
                self.reload(ctx);
            }
            ("orphans", Answer::Yes) => match models::drop_orphans(&mut ctx.state) {
                Ok(n) => ctx.toast(format!("tolti i dati dei modelli non più presenti: {n}"), OK),
                Err(e) => ctx.error(e),
            },
            _ => {}
        }
        Act::None
    }
}

impl Screen for Models {
    fn title(&self) -> String {
        "modelli locali".into()
    }

    fn keys(&self, _ctx: &Ctx) -> Vec<Key> {
        vec![
            (KeyCode::Enter, "invio", "configura e avvia"),
            (KeyCode::Char('p'), "p", "preset"),
            (KeyCode::Char('u'), "u", "aggiornamenti"),
            (KeyCode::Char('x'), "x", "elimina"),
            (KeyCode::Char('f'), "f", "cartelle"),
            (KeyCode::Char('o'), "o", "orfani"),
            (KeyCode::Char('c'), "c", "cerca su HF"),
            (KeyCode::Esc, "esc", "indietro"),
        ]
    }

    fn draw(&mut self, f: &mut Frame, area: Rect, ctx: &Ctx) {
        if self.rows.is_empty() {
            let t = vec![
                Line::from(sp(
                    "nessun modello in models/ né nelle cache note (llama.cpp, Hugging Face, LM Studio)",
                    st(WARN),
                )),
                Line::from(sp("premi c per cercarne uno su Hugging Face, f per aggiungere una cartella", st(DIM))),
            ];
            f.render_widget(Paragraph::new(t), area);
            return;
        }
        let [info, table] = Layout::vertical([Constraint::Length(2), Constraint::Fill(1)]).areas(area);
        let mine: u64 = self.rows.iter().filter(|r| r.managed).map(|r| r.size).sum();
        let status = if self.check.is_some() {
            sp(format!("   {} confronto con Hugging Face…", spinner()), st(LOAD))
        } else {
            sp("", st(DIM))
        };
        let head = Line::from(vec![
            sp(format!("{} modelli · {} su disco", self.rows.len(), size(models::total_size(&self.rows))), st(FG)),
            sp(format!(" (di cui {} nelle cartelle di Corral/tue, il resto nelle cache)", size(mine)), st(DIM)),
            status,
        ]);
        f.render_widget(Paragraph::new(head), info);
        let tick = |b: bool| sp(if b { "✓" } else { "" }, st(OK));
        let rows = self
            .rows
            .iter()
            .zip(&self.vision)
            .map(|(r, v)| {
                let key = ps(&r.path);
                let names = preset::preset_names(&ctx.state, &r.path);
                let saved =
                    ctx.state.presets.get(&key).is_some_and(|n| n.values().any(|p| !p.is_empty()) || n.len() > 1);
                let pre = match names.len() {
                    _ if !saved => String::new(),
                    1 => names[0].clone(),
                    n => format!("{} (+{})", preset::preset_name(&ctx.state, &r.path), n - 1),
                };
                let used =
                    ctx.state.last_used.get(&key).map_or(String::new(), |s| s.get(..16).unwrap_or(s).to_string());
                vec![
                    sp(fname(&key).to_string(), bold(FG)),
                    sp(r.source, st(if r.managed { SIDE } else { CYAN })),
                    sp(r.folder.clone(), st(DIM)),
                    cell(size(r.size)),
                    cell(used),
                    cell(pre),
                    tick(*v),
                ]
            })
            .collect();
        let head = ["modello", "sorgente", "cartella", "dimensione", "ultimo uso", "preset", "vision"];
        self.list.draw(f, table, &head, rows, ctx);
    }

    fn on(&mut self, ev: &Ev, ctx: &mut Ctx) -> Act {
        if let Ev::Answer(tag, a) = ev {
            return self.answer(tag, a, ctx);
        }
        if let Some(Hit::Open(i)) = self.list.on(ev) {
            return self.open(i, ctx);
        }
        match key(ev) {
            Some(KeyCode::Char('c')) => return Act::Push(Box::new(hf::Search::new())),
            Some(KeyCode::Char('p')) => {
                if let Some(m) = self.selected() {
                    return Act::Push(Box::new(Presets::new(&m.path)));
                }
            }
            Some(KeyCode::Char('u')) if self.check.is_none() => {
                if let Some(m) = self.selected() {
                    if models::hf_origin(&m.path).is_none() {
                        ctx.toast("posso verificare solo i modelli scaricati da Corral (cartella models/)", WARN);
                    } else {
                        let p = m.path.clone();
                        self.check = Some(Task::once(move || models::check_update(&p)));
                    }
                }
            }
            Some(KeyCode::Char('x')) => self.ask_delete(ctx),
            Some(KeyCode::Char('f')) => self.ask_dirs(ctx),
            Some(KeyCode::Char('o')) => {
                let o = models::orphans(&ctx.state);
                if o.is_empty() {
                    ctx.toast("nessun preset orfano", OK);
                } else {
                    let mut body: Vec<Line<'static>> =
                        o.iter().take(12).map(|m| Line::from(cell(format!("  {}", short(m))))).collect();
                    if o.len() > 12 {
                        body.push(Line::from(sp(format!("  … e altri {}", o.len() - 12), st(DIM))));
                    }
                    body.push(Line::default());
                    body.push(Line::from(sp(
                        "questi file non esistono più: tolgo i loro preset e l'ultimo uso?",
                        st(FG),
                    )));
                    ctx.ask("orphans", Modal::confirm("preset orfani", body, false));
                }
            }
            _ => return Act::Pass,
        }
        Act::None
    }

    fn tick(&mut self, ctx: &mut Ctx) -> Act {
        if let Some(t) = self.check.as_mut()
            && let Some(r) = t.poll().pop()
        {
            self.check = None;
            match r {
                Ok((repo, Update::Same)) => ctx.toast(format!("✓ aggiornato: uguale a {repo} su Hugging Face"), OK),
                Ok((repo, Update::Changed(files))) => {
                    let mut body: Vec<Line<'static>> =
                        files.iter().map(|f| Line::from(cell(format!("  {f}")))).collect();
                    body.push(Line::default());
                    body.push(Line::from(sp(
                        "apro il repo per riscaricarlo? (il file attuale resta finché il nuovo non è completo)",
                        st(FG),
                    )));
                    self.repo = repo.clone();
                    ctx.ask("update", Modal::confirm(&format!("nuova versione su {repo}"), body, true));
                }
                Err(e) => ctx.error(format!("verifica aggiornamenti: {e}")),
            }
        }
        Act::None
    }

    fn resume(&mut self, ctx: &mut Ctx) {
        self.reload(ctx); // a download or a launch changes the list / last use
    }
}

// -- named presets ------------------------------------------------------------------------------

pub struct Presets {
    model: PathBuf,
    list: List,
    names: Vec<String>,
}

impl Presets {
    pub fn new(model: &Path) -> Self {
        Self { model: model.into(), list: List::new(0), names: vec![] }
    }

    fn name(&self) -> String {
        self.names.get(self.list.sel).cloned().unwrap_or_default()
    }

    fn values(&self, ctx: &Ctx, name: &str) -> Preset {
        ctx.state.presets.get(&ps(&self.model)).and_then(|n| n.get(name)).cloned().unwrap_or_default()
    }

    fn done(&mut self, ctx: &mut Ctx, r: Result<(), String>, ok: String) {
        match r {
            Ok(()) => {
                ctx.save();
                ctx.toast(ok, OK);
            }
            Err(e) => ctx.error(e),
        }
        let sel = preset::preset_name(&ctx.state, &self.model);
        self.names = preset::preset_names(&ctx.state, &self.model);
        self.list.sel = self.names.iter().position(|n| *n == sel).unwrap_or(0);
    }

    fn answer(&mut self, tag: &str, a: &Answer, ctx: &mut Ctx) {
        let (m, cur) = (self.model.clone(), self.name());
        let s = &mut ctx.state;
        match (tag, a) {
            ("new", Answer::Text(t)) if !t.is_empty() => {
                let r = preset::add_preset(s, &m, t, Preset::new());
                self.done(ctx, r, format!("creato \"{t}\" (valori di default)"));
            }
            ("dup", Answer::Text(t)) if !t.is_empty() => {
                let values = self.values(ctx, &cur);
                let r = preset::add_preset(&mut ctx.state, &m, t, values);
                self.done(ctx, r, format!("\"{cur}\" duplicato in \"{t}\""));
            }
            ("rename", Answer::Text(t)) if !t.is_empty() && *t != cur => {
                let r = preset::rename_preset(s, &m, &cur, t);
                self.done(ctx, r, format!("\"{cur}\" ora si chiama \"{t}\""));
            }
            ("delete", Answer::Yes) => {
                let r = preset::delete_preset(s, &m, &cur);
                self.done(ctx, r, format!("eliminato \"{cur}\""));
            }
            ("export", Answer::Text(t)) if !t.is_empty() => {
                let dest = expand(t);
                match models::export_preset(s, &m, &cur, &dest) {
                    Ok(()) => ctx.toast(format!("✓ esportato in {}", dest.display()), OK),
                    Err(e) => ctx.error(e),
                }
            }
            ("import", Answer::Text(t)) if !t.is_empty() => match models::import_preset(s, &m, &expand(t)) {
                Ok((name, dropped)) => {
                    let skipped = if dropped > 0 { format!(" · {dropped} valori ignorati") } else { String::new() };
                    self.done(ctx, Ok(()), format!("✓ importato \"{name}\"{skipped}"));
                }
                Err(e) => ctx.error(e),
            },
            _ => {}
        }
    }

    fn select(&self, ctx: &mut Ctx) {
        preset::select_preset(&mut ctx.state, &self.model, &self.name());
        ctx.save();
    }
}

impl Screen for Presets {
    fn title(&self) -> String {
        format!("preset · {}", fname(&ps(&self.model)))
    }

    fn keys(&self, _ctx: &Ctx) -> Vec<Key> {
        vec![
            (KeyCode::Enter, "invio", "usa e modifica"),
            (KeyCode::Char('a'), "a", "avvia"),
            (KeyCode::Char('n'), "n", "nuovo"),
            (KeyCode::Char('d'), "d", "duplica"),
            (KeyCode::Char('r'), "r", "rinomina"),
            (KeyCode::Char('x'), "x", "elimina"),
            (KeyCode::Char('e'), "e", "esporta"),
            (KeyCode::Char('i'), "i", "importa"),
            (KeyCode::Esc, "esc", "indietro"),
        ]
    }

    fn draw(&mut self, f: &mut Frame, area: Rect, ctx: &Ctx) {
        self.names = preset::preset_names(&ctx.state, &self.model);
        let sel = preset::preset_name(&ctx.state, &self.model);
        self.list.mark = self.names.iter().position(|n| *n == sel);
        let [info, table] = Layout::vertical([Constraint::Length(2), Constraint::Fill(1)]).areas(area);
        let hint =
            "● = quello usato da invio/avvia e da `corral -y`; i valori mostrati sono quelli diversi dal default";
        f.render_widget(Paragraph::new(sp(hint, st(DIM))), info);
        let rows = self
            .names
            .iter()
            .map(|n| {
                let s = models::summary(&self.values(ctx, n));
                vec![sp(n.clone(), bold(FG)), sp(if s.is_empty() { "(tutto default)".into() } else { s }, st(DIM))]
            })
            .collect();
        self.list.draw(f, table, &["nome", "valori cambiati"], rows, ctx);
    }

    fn on(&mut self, ev: &Ev, ctx: &mut Ctx) -> Act {
        if let Ev::Answer(tag, a) = ev {
            self.answer(tag, a, ctx);
            return Act::None;
        }
        if let Some(Hit::Open(_)) = self.list.on(ev) {
            self.select(ctx);
            return Act::Push(Box::new(config::Config::new(ctx, &self.model, None)));
        }
        let cur = self.name();
        match key(ev) {
            Some(KeyCode::Char('a')) => {
                self.select(ctx);
                let p = preset::get_preset(&ctx.state, &self.model);
                match config::launch(ctx, &self.model, &p) {
                    Ok(s) => return Act::Push(s),
                    Err(e) => ctx.error(e),
                }
            }
            Some(KeyCode::Char('n')) => ctx.ask("new", Modal::prompt("nuovo preset", "nome, es. \"coding 32k\"", "")),
            Some(KeyCode::Char('d')) => {
                let def = models::unique(&format!("{cur} copia"), &self.names);
                ctx.ask("dup", Modal::prompt(&format!("duplica \"{cur}\""), "nome della copia", &def));
            }
            Some(KeyCode::Char('r')) => {
                ctx.ask("rename", Modal::prompt(&format!("rinomina \"{cur}\""), "nuovo nome", &cur))
            }
            Some(KeyCode::Char('x')) => ctx.ask("delete", Modal::ask(&format!("elimino il preset \"{cur}\"?"), false)),
            Some(KeyCode::Char('e')) => {
                let def = ps(&models::export_path(&self.model, &cur));
                ctx.ask("export", Modal::prompt(&format!("esporta \"{cur}\""), "file JSON", &def));
            }
            Some(KeyCode::Char('i')) => ctx.ask(
                "import",
                Modal::prompt("importa preset", "file JSON esportato da Corral (o {chiave: valore})", "~/"),
            ),
            _ => return Act::Pass,
        }
        Act::None
    }
}
