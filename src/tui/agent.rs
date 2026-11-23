//! Agent screens: the recipes (a role → model wizard that makes a multi-model profile) and the
//! export of a profile's client configs, with a test of each of its models.

use std::fs;
use std::path::PathBuf;

use ratatui::Frame;
use ratatui::crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::Line;
use ratatui::widgets::{Paragraph, Wrap};

use super::router::{Editor, running};
use super::widgets::*;
use super::{Act, Answer, Ctx, Ev, Key, Screen, Task};
use crate::agent::{self, FORMATS, RECIPES, Recipe};
use crate::dashboard::copy;
use crate::server::Server;
use crate::ui::*;
use crate::{preset, router};

/// The role → model choice in progress.
struct Wizard {
    recipe: usize,
    models: Vec<PathBuf>,
    order: Vec<usize>, // rows of the open chooser -> models
    picks: Vec<PathBuf>,
}

pub struct Agents {
    list: List,
    wiz: Option<Wizard>,
}

impl Agents {
    pub fn new() -> Self {
        Self { list: List::new(0), wiz: None }
    }

    fn start(&mut self, i: usize, ctx: &mut Ctx) {
        let models = preset::local_models();
        if models.is_empty() {
            return ctx.error("nessun modello locale: scaricane uno (menu › cerca su Hugging Face)");
        }
        self.wiz = Some(Wizard { recipe: i, models, order: vec![], picks: vec![] });
        self.ask_role(ctx);
    }

    fn ask_role(&mut self, ctx: &mut Ctx) {
        let Some(w) = self.wiz.as_mut() else { return };
        let r = &RECIPES[w.recipe];
        let role = &r.roles[w.picks.len()];
        let sug = agent::suggest(role, &w.models);
        w.order = sug.iter().map(|s| s.0).collect();
        let rows = sug
            .iter()
            .map(|&(i, good)| {
                let p = &w.models[i];
                vec![
                    cell(fname(&ps(p)).to_string()),
                    sp(size(preset::model_size(p)), st(DIM)),
                    if good { sp("● adatto", st(OK)) } else { cell("") },
                ]
            })
            .collect();
        let title = format!("{} · {} ({}/{}): {}", r.name, role.id, w.picks.len() + 1, r.roles.len(), role.what);
        ctx.ask("role", Modal::choose(&title, &["modello", "dimensione", ""], rows, 0));
    }

    fn picked(&mut self, k: usize, ctx: &mut Ctx) -> Act {
        let Some(w) = self.wiz.as_mut() else { return Act::None };
        let Some(&m) = w.order.get(k) else { return Act::None };
        w.picks.push(w.models[m].clone());
        let r = &RECIPES[w.recipe];
        if w.picks.len() < r.roles.len() {
            self.ask_role(ctx);
            return Act::None;
        }
        let picks = std::mem::take(&mut w.picks);
        self.wiz = None;
        let name = agent::apply(&mut ctx.state, r, &picks);
        ctx.save();
        ctx.toast(format!("✓ profilo «{name}» pronto: controlla i preset e avvia con a"), OK);
        Act::Push(Box::new(Editor::new(&name)))
    }

    /// The recipe under the cursor.
    fn recipe(&self) -> &'static Recipe {
        &RECIPES[self.list.sel.min(RECIPES.len() - 1)]
    }

    fn detail(&self, ctx: &Ctx) -> Vec<Line<'static>> {
        let r = self.recipe();
        let mut t = vec![Line::from(sp(r.desc, st(FG))), Line::default()];
        for role in r.roles {
            let set: Vec<String> = role.set.iter().map(|(k, v)| format!("{k} {v}")).collect();
            t.push(Line::from(vec![
                sp(format!("• {:<10}", role.id), bold(SLOT)),
                sp(format!("{} ", role.what), st(FG)),
                sp(format!("→ {}", set.join(" · ")), st(DIM)),
            ]));
        }
        if !r.opts.is_empty() {
            let o: Vec<String> = r.opts.iter().map(|(k, v)| format!("{k} {v}")).collect();
            t.push(Line::from(vec![sp("• router     ", bold(WARN)), sp(o.join(" · "), st(DIM))]));
        }
        t.push(Line::default());
        let made = ctx.state.routers.contains_key(r.key);
        t.push(Line::from(sp(
            if made {
                format!("profilo «{}» già creato: invio lo rifà coi modelli che scegli · p lo apre · e esporta", r.key)
            } else {
                format!("invio: scegli un modello per ruolo → nasce il profilo multi-modello «{}»", r.key)
            },
            st(CYAN),
        )));
        t.push(Line::from(sp(
            "poi: config per OpenCode, Qwen Code, Python e variabili d'ambiente, e un test di ogni modello",
            st(DIM),
        )));
        t
    }
}

impl Screen for Agents {
    fn title(&self) -> String {
        "agent e flussi".into()
    }

    fn keys(&self, ctx: &Ctx) -> Vec<Key> {
        let mut k = vec![(KeyCode::Enter, "invio", "usa la ricetta")];
        if ctx.state.routers.contains_key(self.recipe().key) {
            k.extend([(KeyCode::Char('p'), "p", "apri profilo"), (KeyCode::Char('e'), "e", "esporta")]);
        }
        k.push((KeyCode::Esc, "esc", "indietro"));
        k
    }

    fn draw(&mut self, f: &mut Frame, area: Rect, ctx: &Ctx) {
        let h = RECIPES.len() as u16 + 4;
        let [table, detail] = Layout::vertical([Constraint::Length(h), Constraint::Fill(1)]).areas(area);
        let rows = RECIPES
            .iter()
            .map(|r| {
                let prof = match (ctx.state.routers.contains_key(r.key), running(ctx, r.key)) {
                    (_, Some(_)) => sp(format!("● {} attivo", r.key), bold(OK)),
                    (true, None) => sp(format!("profilo {}", r.key), st(FG)),
                    _ => sp("—", st(DIM)),
                };
                vec![sp(r.name, bold(FG)), cell(r.roles.len().to_string()), sp(r.what, st(DIM)), prof]
            })
            .collect();
        let blk = panel("ricette", SIDE);
        let inner = blk.inner(table);
        f.render_widget(blk, table);
        self.list.draw(f, inner, &["ricetta", "modelli", "cosa imposta", "profilo"], rows, ctx);
        f.render_widget(
            Paragraph::new(self.detail(ctx)).wrap(Wrap { trim: false }).block(panel("dettaglio", DIM)),
            detail,
        );
    }

    fn on(&mut self, ev: &Ev, ctx: &mut Ctx) -> Act {
        match ev {
            Ev::Answer("role", Answer::Pick(k)) => return self.picked(*k, ctx),
            Ev::Answer("role", _) => {
                self.wiz = None;
                return Act::None;
            }
            Ev::Answer(..) => return Act::None,
            _ => {}
        }
        if let Some(Hit::Open(i)) = self.list.on(ev) {
            self.start(i, ctx);
            return Act::None;
        }
        let key_name = self.recipe().key;
        let made = ctx.state.routers.contains_key(key_name);
        match key(ev) {
            Some(KeyCode::Char('p')) if made => Act::Push(Box::new(Editor::new(key_name))),
            Some(KeyCode::Char('e')) if made => Act::Push(Box::new(Export::new(ctx, key_name))),
            _ => Act::Pass,
        }
    }
}

// -- export ----------------------------------------------------------------------------------------

/// A profile's client configs (tabs), copy / save, and a test of each model of the running router.
pub struct Export {
    name: String,
    /// a router on another computer (its Server::id): the configs follow it, not a profile here
    server: Option<u64>,
    fmt: usize,
    texts: Vec<(String, String, String)>, // per format: file, where it goes, text
    ids: Vec<String>,
    scroll: u16,
    tabs: Zones<usize>,
    results: Vec<Option<Result<String, String>>>,
    task: Option<Task<(usize, Result<String, String>)>>,
    view: Rect,
}

impl Export {
    pub fn new(ctx: &Ctx, name: &str) -> Self {
        Self::make(ctx, name.into(), None)
    }

    /// The configs of the router `id` of `ctx.servers`, on another computer: they point at its
    /// tunnel here, with the models its host listed.
    pub fn remote(ctx: &Ctx, id: u64) -> Self {
        let s = ctx.servers.iter().find(|s| s.id == id);
        let name = s.map(|s| format!("{}@{}", s.dash.alias(), s.host().unwrap_or(""))).unwrap_or_default();
        Self::make(ctx, name, Some(id))
    }

    fn make(ctx: &Ctx, name: String, server: Option<u64>) -> Self {
        let mut s = Self {
            name,
            server,
            fmt: 0,
            texts: vec![],
            ids: vec![],
            scroll: 0,
            tabs: Zones::default(),
            results: vec![],
            task: None,
            view: Rect::default(),
        };
        s.rebuild(ctx);
        s
    }

    /// The running router the configs are for.
    fn live<'a>(&self, ctx: &'a Ctx) -> Option<&'a Server> {
        match self.server {
            Some(id) => ctx.servers.iter().find(|s| s.id == id && s.running()),
            None => running(ctx, &self.name),
        }
    }

    /// Where the configs point and what they list. The profile here: at its running router when
    /// there is one (its port may have moved).
    fn target(&self, ctx: &Ctx) -> agent::Target {
        if let Some(s) = self.server.and_then(|id| ctx.servers.iter().find(|s| s.id == id)) {
            let models = s.served().to_vec();
            return agent::Target { name: self.name.clone(), base: s.base.clone(), key: s.api_key.clone(), models };
        }
        let prof = ctx.state.routers.get(&self.name).cloned().unwrap_or_default();
        let base = running(ctx, &self.name).map(|s| s.base.clone());
        agent::target(&ctx.state, &self.name, &prof, base.as_deref())
    }

    fn rebuild(&mut self, ctx: &Ctx) {
        let t = self.target(ctx);
        self.ids = t.models.iter().map(|m| m.id.clone()).collect();
        self.texts = (0..FORMATS.len()).map(|i| agent::render(i, &t)).collect();
    }

    fn test(&mut self, ctx: &mut Ctx) {
        let Some(s) = self.live(ctx) else {
            return ctx.error("il router non è attivo: avvialo e riprova");
        };
        let (base, key) = (s.base.clone(), s.api_key.clone());
        let t = self.target(ctx);
        self.results = vec![None; t.models.len()];
        self.task = Some(Task::spawn(move |tx| {
            for (i, m) in t.models.iter().enumerate() {
                if tx.send((i, agent::test(&base, &key, m))).is_err() {
                    return;
                }
            }
        }));
    }

    fn save(&self, ctx: &mut Ctx) {
        let (file, _, text) = &self.texts[self.fmt];
        let dir = BASE.join("export").join(crate::installs::sanitize(&self.name));
        let path = dir.join(file);
        // private: the configs hold the API key
        match fs::create_dir_all(&dir).and_then(|()| write_private(&path, text)) {
            Ok(()) => ctx.toast(format!("✓ salvato in {}", short(&ps(&path))), OK),
            Err(e) => ctx.error(format!("{}: {e}", path.display())),
        }
    }
}

impl Screen for Export {
    fn title(&self) -> String {
        format!("esporta · {}", self.name)
    }

    fn keys(&self, _ctx: &Ctx) -> Vec<Key> {
        vec![
            (KeyCode::Tab, "tab", "formato"),
            (KeyCode::Char('c'), "c", "copia"),
            (KeyCode::Char('w'), "w", "salva"),
            (KeyCode::Char('t'), "t", "prova i modelli"),
            (KeyCode::PageDown, "pgsu/giù", "scorri"),
            (KeyCode::Esc, "esc", "indietro"),
        ]
    }

    fn draw(&mut self, f: &mut Frame, area: Rect, ctx: &Ctx) {
        let tests: Vec<Line> = self
            .results
            .iter()
            .zip(&self.ids)
            .map(|(r, id)| {
                let (mark, msg, color) = match r {
                    None => (spinner().to_string(), "in corso…".to_string(), WARN),
                    Some(Ok(m)) => ("✓".into(), m.clone(), OK),
                    Some(Err(e)) => ("✖".into(), e.clone(), ERR),
                };
                Line::from(vec![sp(format!("{mark} {id}  "), bold(color)), sp(msg, st(FG))])
            })
            .collect();
        let tests_h = if tests.is_empty() { 0 } else { (wrapped(&tests, area.width.saturating_sub(4)) + 2).min(12) };
        let [tabs, body, hint, tests_area] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Fill(1),
            Constraint::Length(1),
            Constraint::Length(tests_h),
        ])
        .areas(area);
        let items: Vec<(&str, bool)> = FORMATS.iter().enumerate().map(|(i, n)| (*n, i == self.fmt)).collect();
        self.tabs.clear();
        for (i, r) in buttons(f, tabs, &items, SIDE, ctx).into_iter().enumerate() {
            self.tabs.add(r, i);
        }
        let (file, place, text) = &self.texts[self.fmt];
        // the API key (the profile's, or the one the running router got) never shows; c copies it
        let here = ctx.state.routers.get(&self.name).filter(|_| self.server.is_none());
        let profile_key = here.map(|p| router::opts(p)["api_key"].clone());
        let live_key = self.live(ctx).map(|s| s.api_key.clone());
        let mut shown = text.clone();
        for key in [profile_key, live_key].into_iter().flatten().filter(|k| !k.is_empty()) {
            shown = shown.replace(&key, "•••");
        }
        let lines: Vec<Line> = shown.lines().map(|l| Line::from(sp(l.to_string(), st(FG)))).collect();
        let max = (lines.len() as u16).saturating_sub(body.height.saturating_sub(2));
        self.scroll = self.scroll.min(max);
        self.view = body;
        let p = Paragraph::new(lines).scroll((self.scroll, 0)).block(panel(file, SLOT));
        f.render_widget(p, body);
        let live = if self.live(ctx).is_some() { "router attivo" } else { "router non avviato" };
        f.render_widget(Paragraph::new(sp(format!("dove: {place} · {live}"), st(DIM))), hint);
        if !tests.is_empty() {
            let blk = panel("prova dei modelli", CYAN);
            f.render_widget(Paragraph::new(tests).wrap(Wrap { trim: true }).block(blk), tests_area);
        }
    }

    fn on(&mut self, ev: &Ev, ctx: &mut Ctx) -> Act {
        if let Some(p) = click(ev)
            && let Some(i) = self.tabs.at(p)
        {
            (self.fmt, self.scroll) = (i, 0);
            return Act::None;
        }
        if let Some((d, p)) = wheel(ev)
            && self.view.contains(p)
        {
            self.scroll = self.scroll.saturating_add_signed(d as i16 * 3);
            return Act::None;
        }
        let page = self.view.height.saturating_sub(2).max(1);
        match key(ev) {
            Some(KeyCode::Tab | KeyCode::Right) => (self.fmt, self.scroll) = ((self.fmt + 1) % FORMATS.len(), 0),
            Some(KeyCode::BackTab | KeyCode::Left) => {
                (self.fmt, self.scroll) = ((self.fmt + FORMATS.len() - 1) % FORMATS.len(), 0)
            }
            Some(KeyCode::Down) => self.scroll = self.scroll.saturating_add(1),
            Some(KeyCode::Up) => self.scroll = self.scroll.saturating_sub(1),
            Some(KeyCode::PageDown) => self.scroll = self.scroll.saturating_add(page),
            Some(KeyCode::PageUp) => self.scroll = self.scroll.saturating_sub(page),
            Some(KeyCode::Char('c')) => {
                let text = self.texts[self.fmt].2.clone();
                copy(ctx, &text);
            }
            Some(KeyCode::Char('w')) => self.save(ctx),
            Some(KeyCode::Char('t')) => self.test(ctx),
            _ => return Act::Pass,
        }
        Act::None
    }

    fn tick(&mut self, _ctx: &mut Ctx) -> Act {
        if let Some(t) = self.task.as_mut() {
            for (i, r) in t.poll() {
                if let Some(slot) = self.results.get_mut(i) {
                    *slot = Some(r);
                }
            }
        }
        Act::None
    }

    fn resume(&mut self, ctx: &mut Ctx) {
        self.rebuild(ctx);
    }
}
