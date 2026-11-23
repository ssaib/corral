//! Multi-model screens: the profiles, a profile's editor (its models with their presets, the
//! router options) and the live view of a running router (its models, load/unload, the log).

use std::path::{Path, PathBuf};

use ratatui::Frame;
use ratatui::crossterm::event::{KeyCode, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::Line;
use ratatui::widgets::{Paragraph, Wrap};

use super::config::Config;
use super::widgets::*;
use super::{Act, Answer, Ctx, Ev, Key, Screen, Task, agent};
use crate::dashboard::{Test, background, copy, open};
use crate::preset::{self, Choices};
use crate::router::{self, Entry, OPTS, Profile};
use crate::server::Server;
use crate::ui::*;
use crate::{agent as recipes, installs, server};

/// The running router of profile `name` on this computer.
pub fn running<'a>(ctx: &'a Ctx, name: &str) -> Option<&'a Server> {
    ctx.servers.iter().find(|s| s.dash.is_router() && s.dash.alias() == name && s.running() && s.host().is_none())
}

/// Start profile `name` (or show it when it already runs); the port moves if it's taken.
pub fn launch(ctx: &mut Ctx, name: &str) -> Result<Box<dyn Screen>, String> {
    if let Some(s) = running(ctx, name) {
        return Ok(Box::new(RouterLive::attach(s.id)));
    }
    let inst = ctx.hw.inst.clone().ok_or("nessuna installazione di llama.cpp attiva")?;
    let launch = server::profile_launch(&ctx.state, &inst, &ctx.hw.version, name)?;
    // profile_launch found the profile
    let models: Vec<String> = ctx.state.routers[name].models.iter().map(|e| e.model.clone()).collect();
    for model in models {
        ctx.state.last_used.insert(model, stamp());
    }
    ctx.save();
    for note in &launch.notes {
        ctx.toast(format!("⚠ {note}"), WARN);
    }
    let s = Server::start(&launch, vec![])?;
    let id = s.id;
    ctx.servers.push(s);
    Ok(Box::new(RouterLive::attach(id)))
}

fn launched(ctx: &mut Ctx, name: &str) -> Act {
    match launch(ctx, name) {
        Ok(s) => Act::Push(s),
        Err(e) => {
            ctx.error(e);
            Act::None
        }
    }
}

// -- profiles ------------------------------------------------------------------------------------

pub struct Profiles {
    list: List,
    target: String, // profile the open rename/delete question is about
}

impl Profiles {
    pub fn new() -> Self {
        Self { list: List::new(0), target: String::new() }
    }
}

impl Screen for Profiles {
    fn title(&self) -> String {
        "multi-modello".into()
    }

    fn keys(&self, ctx: &Ctx) -> Vec<Key> {
        let mut k = vec![(KeyCode::Enter, "invio", "apri"), (KeyCode::Char('n'), "n", "nuovo")];
        if !ctx.state.routers.is_empty() {
            k.extend([
                (KeyCode::Char('a'), "a", "avvia"),
                (KeyCode::Char('r'), "r", "rinomina"),
                (KeyCode::Char('x'), "x", "elimina"),
            ]);
        }
        k.extend([(KeyCode::Char('g'), "g", "ricette agent"), (KeyCode::Esc, "esc", "indietro")]);
        k
    }

    fn draw(&mut self, f: &mut Frame, area: Rect, ctx: &Ctx) {
        let [intro, table] = Layout::vertical([Constraint::Length(3), Constraint::Fill(1)]).areas(area);
        let text = "Più modelli su un solo endpoint: llama-server in modalità router li carica all'avvio o alla prima \
                    richiesta, ognuno col suo preset di Corral. I client scelgono il modello col campo \"model\". \
                    g: profili pronti per agent, multi-agent e RAG.";
        f.render_widget(Paragraph::new(sp(text, st(DIM))).wrap(Wrap { trim: true }), intro);
        let mut rows: Vec<_> = ctx
            .state
            .routers
            .iter()
            .map(|(name, p)| {
                let o = router::opts(p);
                let state = match running(ctx, name) {
                    Some(s) => sp(format!("● attivo · pid {}", s.pid), bold(OK)),
                    None => sp("—", st(DIM)),
                };
                vec![
                    sp(name.clone(), bold(FG)),
                    cell(p.models.len().to_string()),
                    cell(format!("{}:{}", o["host"], o["port"])),
                    state,
                    sp(size(router::total_size(p)), st(DIM)),
                ]
            })
            .collect();
        rows.push(vec![sp("+ nuovo profilo…", st(CYAN))]);
        self.list.draw(f, table, &["profilo", "modelli", "endpoint", "stato", "su disco"], rows, ctx);
    }

    fn on(&mut self, ev: &Ev, ctx: &mut Ctx) -> Act {
        let names: Vec<String> = ctx.state.routers.keys().cloned().collect();
        if let Ev::Answer(tag, a) = ev {
            let t = self.target.clone();
            match (*tag, a) {
                ("new", Answer::Text(n)) if !n.is_empty() => {
                    if ctx.state.routers.contains_key(n) {
                        ctx.error(format!("esiste già un profilo «{n}»"));
                    } else {
                        ctx.state.routers.insert(n.clone(), Profile::default());
                        ctx.save();
                        self.list.sel = ctx.state.routers.keys().position(|k| k == n).unwrap_or(0);
                        return Act::Push(Box::new(Editor::new(n)));
                    }
                }
                ("rename", Answer::Text(n)) if !n.is_empty() && *n != t => {
                    if ctx.state.routers.contains_key(n) {
                        ctx.error(format!("esiste già un profilo «{n}»"));
                    } else if let Some(p) = ctx.state.routers.remove(&t) {
                        ctx.state.routers.insert(n.clone(), p);
                        let _ = std::fs::remove_file(router::ini_path(&t));
                        ctx.save();
                    }
                }
                ("delete", Answer::Yes) => {
                    ctx.state.routers.remove(&t);
                    let _ = std::fs::remove_file(router::ini_path(&t));
                    ctx.save();
                }
                _ => {}
            }
            return Act::None;
        }
        if let Some(Hit::Open(i)) = self.list.on(ev) {
            if let Some(n) = names.get(i) {
                return Act::Push(Box::new(Editor::new(n)));
            }
            ctx.ask("new", Modal::prompt("nuovo profilo multi-modello", "un nome, es. \"coding\"", ""));
            return Act::None;
        }
        let cur = names.get(self.list.sel).cloned();
        let busy = cur.as_deref().is_some_and(|n| running(ctx, n).is_some());
        match (key(ev), cur) {
            (Some(KeyCode::Char('n')), _) => {
                ctx.ask("new", Modal::prompt("nuovo profilo multi-modello", "un nome, es. \"coding\"", ""))
            }
            (Some(KeyCode::Char('a')), Some(n)) => return launched(ctx, &n),
            (Some(KeyCode::Char('r' | 'x')), Some(_)) if busy => ctx.error("il profilo è in esecuzione: fermalo prima"),
            (Some(KeyCode::Char('r')), Some(n)) => {
                ctx.ask("rename", Modal::prompt("rinomina il profilo", "", &n));
                self.target = n;
            }
            (Some(KeyCode::Char('x')), Some(n)) => {
                ctx.ask(
                    "delete",
                    Modal::ask(&format!("elimino il profilo «{n}»? (i preset dei modelli restano)"), false),
                );
                self.target = n;
            }
            (Some(KeyCode::Char('g')), _) => return Act::Push(Box::new(agent::Agents::new())),
            _ => return Act::Pass,
        }
        Act::None
    }
}

// -- one profile ---------------------------------------------------------------------------------

fn opt_color(sec: &str) -> u32 {
    match sec {
        "router" => SLOT,
        "agent" => WARN,
        _ => DIM,
    }
}

pub struct Editor {
    name: String,
    models: List,
    opts: List,
    on_opts: bool, // keyboard focus: the router options (else the models)
    editing: Option<TextInput>,
    picks: Vec<PathBuf>, // models offered by the open chooser
    names: Vec<String>,  // presets offered by the open chooser
    act: usize,          // entry the open action menu is about
    /// the editor edits the model's selected preset: the selection to put back afterwards
    restore: Option<(PathBuf, Option<String>)>,
}

impl Editor {
    pub fn new(name: &str) -> Self {
        let mut s = Self {
            name: name.into(),
            models: List::new(0),
            opts: List::new(0),
            on_opts: false,
            editing: None,
            picks: vec![],
            names: vec![],
            act: 0,
            restore: None,
        };
        s.opts.set_len(OPTS.len());
        s
    }

    fn prof(&self, ctx: &Ctx) -> Profile {
        ctx.state.routers.get(&self.name).cloned().unwrap_or_default()
    }

    fn put(&self, ctx: &mut Ctx, p: Profile) {
        ctx.state.routers.insert(self.name.clone(), p);
        ctx.save();
    }

    fn add(&mut self, ctx: &mut Ctx) {
        let models = preset::local_models();
        if models.is_empty() {
            return ctx.error("nessun modello locale: scaricane uno (menu › cerca su Hugging Face)");
        }
        let rows = models
            .iter()
            .map(|p| {
                let dir = p.parent().map_or(String::new(), |d| short(&ps(d)));
                vec![cell(fname(&ps(p)).to_string()), sp(size(preset::model_size(p)), st(DIM)), sp(dir, st(DIM))]
            })
            .collect();
        self.picks = models;
        ctx.ask(
            "add",
            Modal::choose("aggiungi un modello al profilo", &["modello", "dimensione", "cartella"], rows, 0),
        );
    }

    fn actions(&mut self, i: usize, ctx: &mut Ctx) {
        let prof = self.prof(ctx);
        let Some(e) = prof.models.get(i) else { return self.add(ctx) };
        self.act = i;
        let pname =
            if e.preset.is_empty() { preset::preset_name(&ctx.state, Path::new(&e.model)) } else { e.preset.clone() };
        let others = preset::preset_names(&ctx.state, Path::new(&e.model)).join(", ");
        let rows = vec![
            vec![cell(format!("modifica il preset «{pname}»")), sp("lo stesso editor del server singolo", st(DIM))],
            vec![cell("usa un altro preset…"), sp(others, st(DIM))],
            vec![cell("cambia id…"), sp("il nome che i client mandano nel campo model", st(DIM))],
            vec![
                cell(if e.startup { "caricalo alla prima richiesta" } else { "caricalo all'avvio del router" }),
                sp(if e.startup { "ora: all'avvio" } else { "ora: alla prima richiesta" }, st(DIM)),
            ],
            vec![cell("togli dal profilo"), sp("il file e i suoi preset restano", st(DIM))],
        ];
        ctx.ask("act", Modal::choose(&e.id, &["azione", ""], rows, 0));
    }

    /// Open the preset editor on entry `i`'s preset.
    fn configure(&mut self, i: usize, ctx: &mut Ctx) -> Act {
        let Some(e) = self.prof(ctx).models.get(i).cloned() else { return Act::None };
        let model = PathBuf::from(&e.model);
        if !model.is_file() {
            ctx.error(format!("{} non esiste più", e.model));
            return Act::None;
        }
        if !e.preset.is_empty() {
            let before = ctx.state.preset_sel.get(&e.model).cloned();
            self.restore = Some((model.clone(), before));
            // a preset deleted elsewhere comes back empty (= the defaults), as the router runs it
            ctx.state.presets.entry(e.model.clone()).or_default().entry(e.preset.clone()).or_default();
            preset::select_preset(&mut ctx.state, &model, &e.preset);
        }
        Act::Push(Box::new(Config::new(ctx, &model, None)))
    }

    fn answer(&mut self, tag: &str, a: &Answer, ctx: &mut Ctx) -> Act {
        let mut prof = self.prof(ctx);
        let i = self.act;
        match (tag, a) {
            ("add", Answer::Pick(k)) if *k < self.picks.len() => {
                let model = self.picks[*k].clone();
                let taken: Vec<&str> = prof.models.iter().map(|e| e.id.as_str()).collect();
                let id = router::new_id(&preset::base_alias(&model), &taken);
                let pname = preset::preset_name(&ctx.state, &model);
                let startup = prof.models.is_empty();
                prof.models.push(Entry { id, model: ps(&model), preset: pname, startup });
                self.models.sel = prof.models.len() - 1;
                self.put(ctx, prof);
            }
            ("act", Answer::Pick(0)) => return self.configure(i, ctx),
            ("act", Answer::Pick(1)) => {
                let Some(e) = prof.models.get(i) else { return Act::None };
                self.names = preset::preset_names(&ctx.state, Path::new(&e.model));
                let rows = self.names.iter().map(|n| vec![cell(n.clone())]).collect();
                let sel = self.names.iter().position(|n| *n == e.preset).unwrap_or(0);
                ctx.ask("preset", Modal::choose(&format!("preset di {}", e.id), &["preset"], rows, sel));
            }
            ("preset", Answer::Pick(k)) if *k < self.names.len() && i < prof.models.len() => {
                prof.models[i].preset = self.names[*k].clone();
                self.put(ctx, prof);
            }
            ("act", Answer::Pick(2)) if i < prof.models.len() => {
                ctx.ask("id", Modal::prompt("id del modello", "lettere, cifre, . _ - (es. coder)", &prof.models[i].id))
            }
            ("id", Answer::Text(t)) if i < prof.models.len() => {
                if !router::id_ok(t) {
                    ctx.error(format!("id «{t}» non valido: solo lettere, cifre, . _ -"));
                } else if prof.models.iter().enumerate().any(|(j, e)| j != i && e.id == *t) {
                    ctx.error(format!("l'id «{t}» è già usato"));
                } else {
                    prof.models[i].id = t.clone();
                    self.put(ctx, prof);
                }
            }
            ("act", Answer::Pick(3)) if i < prof.models.len() => {
                prof.models[i].startup = !prof.models[i].startup;
                self.put(ctx, prof);
            }
            ("act", Answer::Pick(4)) if i < prof.models.len() => {
                prof.models.remove(i);
                self.put(ctx, prof);
            }
            ("opt", Answer::Pick(k)) => {
                if let Choices::List(c) = OPTS[self.opts.sel].choices
                    && let Some(v) = c.get(*k)
                {
                    self.set_opt(ctx, v.to_string());
                }
            }
            _ => {}
        }
        Act::None
    }

    fn set_opt(&mut self, ctx: &mut Ctx, v: String) {
        let f = &OPTS[self.opts.sel];
        let v = if v == "-" { f.default.to_string() } else { v.trim().to_string() };
        if let Err(e) = preset::check(f.key, &v) {
            return ctx.error(e);
        }
        self.editing = None;
        let mut prof = self.prof(ctx);
        let mut o = router::opts(&prof);
        o.insert(f.key.into(), v);
        prof.opts = router::opts_diff(&o);
        self.put(ctx, prof);
    }

    fn edit_opt(&mut self, ctx: &mut Ctx) {
        let f = &OPTS[self.opts.sel];
        let cur = router::opts(&self.prof(ctx))[f.key].clone();
        match f.choices {
            Choices::List(c) => {
                let rows = c.iter().map(|v| vec![cell(if v.is_empty() { "(nessuno)" } else { v })]).collect();
                let sel = c.iter().position(|v| *v == cur).unwrap_or(0);
                ctx.ask("opt", Modal::choose(f.label, &["valore"], rows, sel));
            }
            _ => self.editing = Some(TextInput::new(&cur)),
        }
    }

    fn preview(&self, ctx: &Ctx) -> Vec<Line<'static>> {
        let prof = self.prof(ctx);
        let flags = ctx.hw.inst.as_ref().and_then(|i| preset::cached_flags(&installs::server(i)));
        match router::ini(&ctx.state, &prof, flags.as_deref()) {
            Ok(t) => t
                .lines()
                .map(|l| Line::from(sp(l.to_string(), st(if l.starts_with('[') { SLOT } else { FG }))))
                .collect(),
            Err(e) => vec![Line::from(sp(format!("✖ {e}"), st(ERR)))],
        }
    }
}

impl Screen for Editor {
    fn title(&self) -> String {
        self.name.clone()
    }

    fn resume(&mut self, ctx: &mut Ctx) {
        if let Some((model, before)) = self.restore.take() {
            match before {
                Some(name) => preset::select_preset(&mut ctx.state, &model, &name),
                None => {
                    ctx.state.preset_sel.remove(&ps(&model));
                }
            }
            ctx.save();
        }
    }

    fn keys(&self, _ctx: &Ctx) -> Vec<Key> {
        if self.editing.is_some() {
            return vec![(KeyCode::Enter, "invio", "conferma"), (KeyCode::Esc, "esc", "annulla")];
        }
        let mut k = vec![(KeyCode::Enter, "invio", if self.on_opts { "modifica" } else { "azioni" })];
        if !self.on_opts {
            k.extend([
                (KeyCode::Char('n'), "n", "aggiungi"),
                (KeyCode::Char('p'), "p", "preset"),
                (KeyCode::Char('x'), "x", "togli"),
            ]);
        }
        k.extend([
            (KeyCode::Char('a'), "a", "avvia"),
            (KeyCode::Char('v'), "v", "INI"),
            (KeyCode::Char('e'), "e", "esporta"),
            (KeyCode::Tab, "tab", if self.on_opts { "modelli" } else { "opzioni" }),
            (KeyCode::Char('?'), "?", "aiuto"),
            (KeyCode::Esc, "esc", "indietro"),
        ]);
        k
    }

    fn draw(&mut self, f: &mut Frame, area: Rect, ctx: &Ctx) {
        let prof = self.prof(ctx);
        let o = router::opts(&prof);
        let cmd = match &ctx.hw.inst {
            Some(i) => {
                let srv = installs::server(i);
                let c =
                    router::command(&srv, &router::ini_path(&self.name), &prof, preset::cached_flags(&srv).as_deref());
                let s = shell_join(&c);
                if o["api_key"].is_empty() { s } else { s.replace(&o["api_key"], "•••") }
            }
            None => "—".into(),
        };
        let cmd = short(&cmd);
        let n = wrapped(&[Line::from(cmd.as_str())], area.width.saturating_sub(4));
        let models_h = (prof.models.len() as u16 + 5).clamp(6, 14);
        let [info, models, opts, hint, cmd_area] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Length(models_h),
            Constraint::Fill(1),
            Constraint::Length(1),
            Constraint::Length(n.min(4) + 2 + u16::from(n > 1)),
        ])
        .areas(area);
        let free: u64 = ctx.hw.devs.iter().map(|d| d.free).sum::<u64>() << 20;
        let mut l1 = vec![
            sp(format!("http://{}:{}/v1", o["host"], o["port"]), bold(FG)),
            sp(
                format!("  ·  {} · {} su disco", router::count(prof.models.len()), size(router::total_size(&prof))),
                st(DIM),
            ),
        ];
        if free > 0 {
            l1.push(sp(format!(" · VRAM libera {}", size(free)), st(DIM)));
        }
        if let Some(s) = running(ctx, &self.name) {
            l1.push(sp(format!("  ● in esecuzione (pid {})", s.pid), bold(OK)));
        }
        let l2 = sp(
            format!(
                "fino a {} modelli caricati insieme · gli altri si caricano alla prima richiesta e scaricano i meno usati",
                if o["models_max"] == "0" { "∞".into() } else { o["models_max"].clone() }
            ),
            st(DIM),
        );
        f.render_widget(Paragraph::new(vec![Line::from(l1), Line::from(l2)]), info);

        // models
        let blk = if self.on_opts { panel("modelli", DIM) } else { lit("modelli", SIDE) };
        let inner = blk.inner(models);
        f.render_widget(blk, models);
        let mut rows: Vec<_> = prof
            .models
            .iter()
            .map(|e| {
                let m = Path::new(&e.model);
                let p = router::entry_preset(&ctx.state, e);
                let pname = if e.preset.is_empty() { preset::preset_name(&ctx.state, m) } else { e.preset.clone() };
                let file = sp(fname(&e.model).to_string(), if m.is_file() { st(FG) } else { st(ERR) });
                vec![
                    sp(e.id.clone(), bold(FG)),
                    file,
                    cell(pname),
                    if e.startup { sp("all'avvio", st(OK)) } else { sp("a richiesta", st(DIM)) },
                    cell(p["usage"].clone()),
                    sp(format!("ctx {}", if p["ctx"] == "0" { "modello".into() } else { p["ctx"].clone() }), st(DIM)),
                    sp(size(preset::model_size(m)), st(DIM)),
                ]
            })
            .collect();
        rows.push(vec![sp("+ aggiungi modello…", st(CYAN))]);
        let head = ["id", "file", "preset", "carica", "uso", "contesto", "dimensione"];
        self.models.draw(f, inner, &head, rows, ctx);

        // router options
        let blk = if self.on_opts { lit("router", SIDE) } else { panel("router", DIM) };
        let inner = blk.inner(opts);
        f.render_widget(blk, opts);
        let rows = OPTS
            .iter()
            .map(|fl| {
                let v = &o[fl.key];
                let shown = match (fl.key, v.is_empty()) {
                    ("api_key", false) => "••••••".into(),
                    (_, true) => "—".into(),
                    _ => v.clone(),
                };
                let style = if *v != fl.default { bold(FG) } else { st(DIM) };
                vec![sp(fl.sec, st(opt_color(fl.sec))), cell(fl.label), sp(shown, style)]
            })
            .collect();
        self.opts.draw(f, inner, &["sezione", "opzione", "valore"], rows, ctx);
        let note = if let Some(input) = self.editing.as_mut() {
            if let Some(r) = self.opts.cell_rect(self.opts.sel, 2) {
                input.draw(f, r, true);
            }
            let fl = &OPTS[self.opts.sel];
            let def = if fl.default.is_empty() { "nessuno" } else { fl.default };
            sp(format!("✎ invio conferma · esc annulla · ctrl+u svuota · - = {def}"), st(CYAN))
        } else if self.on_opts && OPTS[self.opts.sel].sec == "agent" {
            sp("tool e MCP valgono per la Web UI del router: agiscono sul tuo computer, tienilo su 127.0.0.1", st(WARN))
        } else if self.on_opts {
            let short = crate::help::of(OPTS[self.opts.sel].key).map_or("", |h| h.0);
            sp(format!("{short} · ? spiegazione"), st(FG))
        } else {
            sp("p: i parametri del modello nell'editor dei preset · v: il file INI che riceve llama-server", st(DIM))
        };
        f.render_widget(Paragraph::new(note), hint);
        f.render_widget(
            Paragraph::new(sp(cmd, st(LOAD))).wrap(Wrap { trim: false }).block(panel("comando", DIM)),
            cmd_area,
        );
    }

    fn on(&mut self, ev: &Ev, ctx: &mut Ctx) -> Act {
        if let Ev::Answer(tag, a) = ev {
            return self.answer(tag, a, ctx);
        }
        if let Some(input) = self.editing.as_mut() {
            return match input.on(ev) {
                Some(Edit::Submit) => {
                    let v = input.value.clone();
                    self.set_opt(ctx, v);
                    Act::None
                }
                Some(Edit::Cancel) => {
                    self.editing = None;
                    Act::None
                }
                Some(Edit::Changed) => Act::None,
                None if click(ev).is_some() => {
                    self.editing = None;
                    self.on(ev, ctx)
                }
                None if matches!(ev, Ev::Key(_)) => Act::Pass,
                None => Act::None,
            };
        }
        // the mouse picks the list under it; keys go to the focused one
        let mouse = matches!(ev, Ev::Mouse(_));
        if mouse || !self.on_opts {
            match self.models.on(ev) {
                Some(Hit::Open(i)) => {
                    self.on_opts = false;
                    self.actions(i, ctx);
                    return Act::None;
                }
                Some(Hit::Moved) => {
                    self.on_opts = false;
                    return Act::None;
                }
                None => {}
            }
        }
        if mouse || self.on_opts {
            match self.opts.on(ev) {
                Some(Hit::Open(_)) => {
                    self.on_opts = true;
                    self.edit_opt(ctx);
                    return Act::None;
                }
                Some(Hit::Moved) => {
                    self.on_opts = true;
                    return Act::None;
                }
                None => {}
            }
        }
        let n = self.prof(ctx).models.len();
        let sel = self.models.sel;
        match key(ev) {
            Some(KeyCode::Char('?')) if self.on_opts => {
                let f = &OPTS[self.opts.sel];
                ctx.ask("info", Modal::info(f.label, CYAN, crate::help::explain(f)));
            }
            Some(KeyCode::Tab | KeyCode::BackTab) => self.on_opts = !self.on_opts,
            Some(KeyCode::Char('n')) => self.add(ctx),
            Some(KeyCode::Char('p')) if sel < n => return self.configure(sel, ctx),
            Some(KeyCode::Char('x')) if !self.on_opts && sel < n => {
                let mut prof = self.prof(ctx);
                let gone = prof.models.remove(sel);
                self.put(ctx, prof);
                ctx.toast(format!("tolto {} dal profilo", gone.id), OK);
            }
            Some(KeyCode::Char('a')) => return launched(ctx, &self.name),
            Some(KeyCode::Char('v')) => {
                let title = format!("INI per llama-server · {}", short(&ps(&router::ini_path(&self.name))));
                ctx.ask("info", Modal::info(&title, SLOT, self.preview(ctx)));
            }
            Some(KeyCode::Char('e')) => return Act::Push(Box::new(agent::Export::new(ctx, &self.name))),
            _ => return Act::Pass,
        }
        Act::None
    }
}

// -- the running router ------------------------------------------------------------------------

/// A running router: its models (load / unload, a test prompt to one), its log.
pub struct RouterLive {
    id: u64, // the router's Server::id
    list: List,
    log_area: Rect,
    test: Option<Test>,
    leave: bool, // back as soon as it's down
    job: Option<Task<Result<String, String>>>,
}

impl RouterLive {
    pub fn attach(id: u64) -> Self {
        Self { id, list: List::new(0), log_area: Rect::default(), test: None, leave: false, job: None }
    }

    fn idx(&self, ctx: &Ctx) -> Option<usize> {
        ctx.servers.iter().position(|s| s.id == self.id)
    }

    /// (id, state, usage) of each model: the profile's order, then any other the router lists.
    fn rows(&self, ctx: &Ctx, s: &Server) -> Vec<(String, String, String)> {
        // (id, usage) in the profile's order: this computer's profile, or what the router's host said
        let profile: Vec<(String, String)> = match s.host() {
            Some(_) => s.served().iter().map(|m| (m.id.clone(), m.usage.clone())).collect(),
            None => {
                let prof = ctx.state.routers.get(s.dash.alias()).cloned().unwrap_or_default();
                let usage = |e: &Entry| router::entry_preset(&ctx.state, e)["usage"].clone();
                prof.models.iter().map(|e| (e.id.clone(), usage(e))).collect()
            }
        };
        let up = s.running(); // down: the last poll's states no longer hold
        let st_of = |id: &str| s.dash.m.models.iter().find(|m| m.id == id).filter(|_| up);
        let none = if up { "…" } else { "down" };
        let mut out: Vec<(String, String, String)> = profile
            .into_iter()
            .map(|(id, usage)| {
                let state =
                    st_of(&id).map_or(none.into(), |m| if m.failed { "failed".into() } else { m.state.clone() });
                (id, state, usage)
            })
            .collect();
        for m in s.dash.m.models.iter().filter(|_| up) {
            if !out.iter().any(|r| r.0 == m.id) {
                out.push((m.id.clone(), m.state.clone(), "chat".into()));
            }
        }
        out
    }

    fn forget(&self, ctx: &mut Ctx) {
        ctx.servers.retain(|s| s.id != self.id);
    }
}

impl Screen for RouterLive {
    fn title(&self) -> String {
        "router".into()
    }

    fn keys(&self, ctx: &Ctx) -> Vec<Key> {
        let Some(s) = self.idx(ctx).map(|i| &ctx.servers[i]) else { return vec![] };
        let mut k = if s.running() {
            vec![
                (KeyCode::Enter, "invio", "carica/scarica"),
                (KeyCode::Char('t'), "t", "prova"),
                (KeyCode::Char('e'), "e", "esporta"),
                (KeyCode::Char('c'), "c", "copia URL"),
                (KeyCode::Char('o'), "o", "browser"),
                (KeyCode::Char('s'), "s", if s.stops == 0 { "ferma" } else { "kill" }),
                (KeyCode::Char('b'), "b", "background"),
            ]
        } else {
            vec![(KeyCode::Esc, "esc", "indietro"), (KeyCode::Char('m'), "m", "menu")]
        };
        k.extend([
            (KeyCode::Char('f'), "f", if s.dash.only_warn { "tutto" } else { "warning" }),
            (KeyCode::Char('/'), "/", "cerca"),
            (KeyCode::Char('p'), "p", if s.dash.paused { "riprendi" } else { "pausa" }),
        ]);
        if s.running() {
            let esc = if self.test.is_some() {
                "chiudi prova"
            } else if s.host().is_some() {
                "indietro" // it stays on its host
            } else {
                "esci"
            };
            k.push((KeyCode::Esc, "esc", esc));
        }
        k
    }

    fn draw(&mut self, f: &mut Frame, area: Rect, ctx: &Ctx) {
        let Some(i) = self.idx(ctx) else { return };
        let s = &ctx.servers[i];
        let d = &s.dash;
        let rows = self.rows(ctx, s);
        let list_h = (rows.len() as u16 + 4).clamp(5, 12);
        let [info, models, log] =
            Layout::vertical([Constraint::Length(3), Constraint::Length(list_h), Constraint::Min(5)]).areas(area);
        let (state, color) = d.state();
        let up = d.uptime();
        let loaded = rows.iter().filter(|r| r.1 == "loaded").count();
        let gpu = |g: &crate::server::Gpu| {
            let load = g.load();
            let memory = format!("{} {}/{} MiB", g.name, g.used, g.total);
            if load.is_empty() { memory } else { format!("{memory} {load}") }
        };
        let gpus: Vec<String> = d.m.gpus.iter().map(gpu).collect();
        let lines = vec![
            // like the single server's: the state first, the name, where it answers
            Line::from(vec![
                sp(format!("● {state}"), bold(color)),
                sp(format!("   {}", d.alias()), bold(WOOL)),
                sp(format!("   http://{}/v1", d.endpoint()), st(DIM)),
                sp("   uptime ", st(DIM)),
                sp(format!("{:02}:{:02}:{:02}", up / 3600, up % 3600 / 60, up % 60), st(FG)),
            ]),
            Line::from(sp(
                format!("{loaded}/{} modelli caricati · pid {} · {}", rows.len(), s.pid, gpus.join(" · ")),
                st(DIM),
            )),
            Line::from(sp(
                "nel log le righe di ogni modello portano il suo id · invio sul modello: carica o scarica",
                st(DIM),
            )),
        ];
        f.render_widget(Paragraph::new(lines), info);
        let blk = panel("modelli", SLOT);
        let inner = blk.inner(models);
        f.render_widget(blk, models);
        let table = rows
            .iter()
            .map(|(id, state, usage)| {
                let (label, style) = match state.as_str() {
                    "loaded" => ("● caricato".to_string(), bold(OK)),
                    "loading" => (format!("{} caricamento", spinner()), bold(WARN)),
                    "unloaded" => ("○ a richiesta".into(), st(DIM)),
                    "sleeping" => ("☾ sospeso".into(), st(CYAN)),
                    "downloading" => (format!("{} download", spinner()), st(WARN)),
                    "down" => ("— fermo".into(), st(DIM)),
                    "failed" => {
                        let code = d.m.models.iter().find(|m| m.id == *id).and_then(|m| m.exit);
                        (format!("✖ fallito{}", code.map_or(String::new(), |c| format!(" (codice {c})"))), bold(ERR))
                    }
                    s => (s.to_string(), st(DIM)),
                };
                let prom = d.m.models.iter().find(|m| m.id == *id).map(|m| m.prom.clone()).unwrap_or_default();
                let p = |k: &str| prom.get(k).copied();
                vec![
                    sp(id.clone(), bold(FG)),
                    sp(label, style),
                    cell(usage.clone()),
                    cell(p("requests_processing").map_or("—".into(), |v| format!("{v:.0}"))),
                    cell(d.model_tps.get(id).map_or("—".into(), |v| format!("{v:.1}"))),
                    cell(p("tokens_predicted_total").map_or("—".into(), |v| format!("{v:.0}"))),
                ]
            })
            .collect();
        let head = ["id", "stato", "uso", "in corso", "gen tok/s", "token generati"];
        self.list.draw(f, inner, &head, table, ctx);
        let [log, extra] = if self.test.is_some() {
            Layout::horizontal([Constraint::Fill(1); 2]).areas(log)
        } else {
            [log, Rect::default()]
        };
        self.log_area = log;
        f.render_widget(d.log_panel(log), log);
        if let Some(t) = &self.test {
            t.draw(f, extra);
        }
        if let Some(j) = &self.job
            && !j.done
        {
            let r = Rect {
                y: models.bottom().saturating_sub(1),
                height: 1,
                x: models.x + 2,
                width: models.width.saturating_sub(4),
            };
            f.render_widget(Paragraph::new(sp(format!(" {} richiesta al router… ", spinner()), bold(WARN))), r);
        }
    }

    fn on(&mut self, ev: &Ev, ctx: &mut Ctx) -> Act {
        let Some(i) = self.idx(ctx) else { return Act::Pop };
        let running = ctx.servers[i].running();
        let rows = self.rows(ctx, &ctx.servers[i]);
        if ctx.servers[i].dash.log_event(ev, self.log_area) {
            return Act::None;
        }
        let ctrl_c =
            matches!(ev, Ev::Key(k) if k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL));
        let (base, api_key) = (ctx.servers[i].base.clone(), ctx.servers[i].api_key.clone());
        let cur = rows.get(self.list.sel).cloned();
        match ev {
            Ev::Answer("leave", Answer::Yes) => {
                self.leave = true;
                if ctx.servers[i].stops == 0 {
                    ctx.servers[i].stop();
                }
                return Act::None;
            }
            Ev::Answer("prompt", Answer::Text(t)) if !t.is_empty() => {
                if let Some((id, ..)) = &cur {
                    self.test = Some(Test::start(&ctx.servers[i], id, t.clone()));
                }
                return Act::None;
            }
            Ev::Answer(..) => return Act::None,
            _ if ctrl_c && running => {
                ctx.servers[i].stop();
                return Act::None;
            }
            _ => {}
        }
        if let Some(hit) = self.list.on(ev) {
            let busy = self.job.as_ref().is_some_and(|j| !j.done);
            if let Hit::Open(k) = hit
                && running
                && !busy
                && let Some((id, state, _)) = rows.get(k).cloned()
            {
                let unload = matches!(state.as_str(), "loaded" | "loading" | "sleeping");
                self.job = Some(Task::once(move || {
                    let r =
                        if unload { router::unload(&base, &api_key, &id) } else { router::load(&base, &api_key, &id) };
                    r.map(|()| format!("{id}: {}", if unload { "scaricato" } else { "caricamento avviato" }))
                }));
            }
            return Act::None;
        }
        let s = &mut ctx.servers[i];
        match key(ev) {
            Some(KeyCode::Char('/')) => {
                let cur = s.dash.search.clone();
                ctx.ask("search", Modal::prompt("filtra il log", "testo da cercare (vuoto = tutto)", &cur));
            }
            Some(KeyCode::Char('s')) if running => s.stop(),
            // a router on another computer: leaving its screen never stops it
            Some(KeyCode::Char('b')) if running => return background(ctx, i),
            Some(KeyCode::Esc | KeyCode::Char('q')) if running && self.test.is_none() && s.host().is_some() => {
                return background(ctx, i);
            }
            Some(KeyCode::Char('t')) if running => match cur {
                Some((id, _, usage)) if usage == "chat" => ctx.ask(
                    "prompt",
                    Modal::prompt(&format!("prompt di prova · {id}"), "risposta in streaming accanto al log", ""),
                ),
                Some((id, _, usage)) => {
                    let m = recipes::Served { id, file: String::new(), ctx: 0, vision: false, usage };
                    self.job = Some(Task::once(move || recipes::test(&base, &api_key, &m)));
                }
                None => {}
            },
            Some(KeyCode::Char('e')) if s.host().is_some() => {
                let id = s.id;
                return Act::Push(Box::new(agent::Export::remote(ctx, id)));
            }
            Some(KeyCode::Char('e')) => {
                let name = s.dash.alias().to_string();
                return Act::Push(Box::new(agent::Export::new(ctx, &name)));
            }
            Some(KeyCode::Char('o')) if running => open(ctx, &base),
            Some(KeyCode::Char('c')) if running => copy(ctx, &format!("{base}/v1")),
            Some(KeyCode::Esc) if self.test.is_some() => self.test = None,
            Some(KeyCode::Esc | KeyCode::Char('q')) if running => {
                ctx.ask("leave", Modal::ask("fermo il router (e i suoi modelli) e torno indietro?", true))
            }
            Some(KeyCode::Esc | KeyCode::Char('q')) => {
                self.forget(ctx);
                return Act::Pop;
            }
            Some(KeyCode::Char('m')) if !running => {
                self.forget(ctx);
                return Act::Home;
            }
            _ => return Act::Pass,
        }
        Act::None
    }

    fn tick(&mut self, ctx: &mut Ctx) -> Act {
        if let Some(t) = self.test.as_mut() {
            t.poll();
        }
        if let Some(j) = self.job.as_mut() {
            for r in j.poll() {
                match r {
                    Ok(m) => ctx.toast(format!("✓ {m}"), OK),
                    Err(e) => ctx.error(e),
                }
            }
        }
        let Some(i) = self.idx(ctx) else { return Act::Pop };
        if self.leave && !ctx.servers[i].running() {
            self.forget(ctx);
            return Act::Pop;
        }
        Act::None
    }
}
