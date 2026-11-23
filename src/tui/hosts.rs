//! Other computers with Corral, reached over ssh (the logic is in `remote`): the hosts of
//! ~/.ssh/config, and one host's servers (attach one through a tunnel, stop it), its models and
//! profiles (start one), its own Corral in this terminal, a copy of this one.

use ratatui::Frame;
use ratatui::crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::Line;
use ratatui::widgets::{Paragraph, Wrap};

use super::widgets::*;
use super::{Act, Answer, Ctx, Ev, Key, Screen, Task};
use crate::dashboard::screen_of;
use crate::remote::{self, Link, Models, Problem, Running, Status};
use crate::server::Server;
use crate::ui::*;

// -- the hosts ------------------------------------------------------------------------------------

pub struct Hosts {
    list: List,
    hosts: Vec<String>,
}

impl Hosts {
    pub fn new() -> Self {
        Self { list: List::new(0), hosts: remote::config_hosts() }
    }
}

impl Screen for Hosts {
    fn title(&self) -> String {
        "host remoti".into()
    }

    fn keys(&self, _ctx: &Ctx) -> Vec<Key> {
        vec![(KeyCode::Enter, "invio", "apri"), (KeyCode::Esc, "esc", "indietro")]
    }

    fn draw(&mut self, f: &mut Frame, area: Rect, ctx: &Ctx) {
        if self.hosts.is_empty() {
            let example = |s: &str| Line::from(sp(s.to_string(), bold(CYAN)));
            let t = vec![
                Line::from(sp("nessun host in ~/.ssh/config", st(WARN))),
                Line::default(),
                Line::from(sp("Corral usa il tuo ssh: aggiungi lì il computer, per esempio", st(DIM))),
                example("  Host workstation"),
                example("      HostName 192.168.1.20"),
                example("      User ada"),
                Line::default(),
                Line::from(sp("e accedi con una chiave, senza password: ssh-copy-id workstation", st(DIM))),
            ];
            f.render_widget(Paragraph::new(t), area);
            return;
        }
        let [list, help] = Layout::vertical([Constraint::Fill(1), Constraint::Length(3)]).areas(area);
        let rows = self
            .hosts
            .iter()
            .map(|h| {
                let linked = ctx.servers.iter().filter(|s| s.host() == Some(h.as_str()) && s.running()).count();
                let lending = ctx.servers.iter().any(|s| s.running() && s.lenders().contains(&h.as_str()));
                let here = match (linked, lending) {
                    (0, false) => sp("—", st(DIM)),
                    (0, true) => sp("presta le GPU", st(OK)),
                    (n, false) => sp(format!("{n} collegati"), st(OK)),
                    (n, true) => sp(format!("{n} collegati · presta le GPU"), st(OK)),
                };
                vec![sp(h.clone(), bold(FG)), here]
            })
            .collect();
        self.list.draw(f, list, &["host (da ~/.ssh/config)", "server qui"], rows, ctx);
        let t = vec![
            Line::from(sp("invio: i suoi server (collegali qui con un tunnel), i modelli da avviare", st(DIM))),
            Line::from(sp(
                "serve l'accesso con una chiave, senza password, e Corral sull'host: puoi copiarlo da qui",
                st(DIM),
            )),
        ];
        f.render_widget(Paragraph::new(t).wrap(Wrap { trim: true }), help);
    }

    fn on(&mut self, ev: &Ev, _ctx: &mut Ctx) -> Act {
        if let Some(Hit::Open(i)) = self.list.on(ev)
            && let Some(h) = self.hosts.get(i)
        {
            return Act::Push(Box::new(Host::new(h)));
        }
        Act::Pass
    }
}

// -- one host -------------------------------------------------------------------------------------

/// What a host said when asked: its Corral and servers, and what it can start.
type Seen = Result<(Status, Models), Problem>;

/// A row of a host's list.
#[derive(Clone)]
enum Row {
    Server(Box<Running>),
    Profile(remote::Profile),
    Model(remote::Model),
}

/// The answer of a one-shot task once it's there (the task is then over).
fn answer<T: Send + 'static>(task: &mut Option<Task<T>>) -> Option<T> {
    let t = task.as_mut()?;
    let out = t.poll().pop();
    if out.is_some() || t.done {
        *task = None;
    }
    out
}

pub struct Host {
    name: String,
    list: List,
    seen: Option<Seen>, // None = no answer yet
    asking: Option<Task<Seen>>,
    /// a stop or a copy under way: a message for the end, or why not
    doing: Option<Task<Result<String, String>>>,
    /// a start under way: the server it started, or why not
    starting: Option<Task<Result<Running, String>>>,
    /// what's under way, for the status line
    doing_what: String,
    linking: Option<Task<Result<(Running, Link), String>>>,
    /// the model whose preset the open chooser picks
    choosing: Option<remote::Model>,
}

impl Host {
    pub fn new(name: &str) -> Self {
        let mut h = Self {
            name: name.into(),
            list: List::new(0),
            seen: None,
            asking: None,
            doing: None,
            starting: None,
            doing_what: String::new(),
            linking: None,
            choosing: None,
        };
        h.refresh();
        h
    }

    fn refresh(&mut self) {
        let host = self.name.clone();
        self.asking = Some(Task::once(move || {
            let status = remote::status(&host)?;
            let models = remote::models(&host)?;
            Ok((status, models))
        }));
    }

    fn busy(&self) -> bool {
        self.doing.is_some() || self.starting.is_some() || self.linking.is_some()
    }

    fn rows(&self) -> Vec<Row> {
        let Some(Ok((status, models))) = &self.seen else { return vec![] };
        let mut rows: Vec<Row> = status.servers.iter().map(|r| Row::Server(Box::new(r.clone()))).collect();
        rows.extend(models.profiles.iter().cloned().map(Row::Profile));
        rows.extend(models.models.iter().cloned().map(Row::Model));
        rows
    }

    /// The server this computer already has a link to for `r`.
    fn linked<'a>(&self, ctx: &'a Ctx, r: &Running) -> Option<&'a Server> {
        ctx.servers.iter().find(|s| s.host() == Some(self.name.as_str()) && s.pid == r.pid && s.running())
    }

    /// Run `work` in the background, showing `what` meanwhile.
    fn start_doing(&mut self, what: String, work: impl FnOnce() -> Result<String, String> + Send + 'static) {
        self.doing_what = what;
        self.doing = Some(Task::once(work));
    }

    /// The dashboard of server `r`: through a new link, unless there is one already.
    fn attach(&mut self, r: Running, ctx: &mut Ctx) -> Act {
        if let Some(s) = self.linked(ctx, &r) {
            return Act::Push(screen_of(s));
        }
        let host = self.name.clone();
        self.linking = Some(Task::once(move || remote::link(&host, &r).map(|link| (r, link))));
        Act::None
    }

    /// `corral start <words>` on the host; its server opens here once it's ready.
    fn start(&mut self, what: &str, words: Vec<String>) {
        let host = self.name.clone();
        self.doing_what = format!("avvio {what} su {host}… (un modello grande può volerci qualche minuto)");
        self.starting = Some(Task::once(move || {
            remote::start(&host, &words).map_err(|e| e.to_string())?;
            // the newest server there is the one just started
            let status = remote::status(&host).map_err(|e| e.to_string())?;
            let newest = status.servers.into_iter().max_by_key(|r| r.started);
            newest.ok_or_else(|| "partito, ma l'host non lo elenca più: guarda il suo log".to_string())
        }));
    }

    fn stop(&mut self, r: &Running) {
        let (host, pid) = (self.name.clone(), r.pid);
        let what = format!("fermo {} su {host}…", r.info.alias);
        self.start_doing(what, move || remote::stop(&host, pid).map(|()| "fermato".into()).map_err(|e| e.to_string()));
    }

    fn copy(&mut self) {
        let host = self.name.clone();
        let what = format!("copio Corral su {host}…");
        self.start_doing(what, move || {
            remote::copy_corral(&host)
                .map(|()| "Corral copiato in ~/.local/bin/corral".into())
                .map_err(|e| e.to_string())
        });
    }

    /// The first lines: the host's Corral, system and GPUs, or why it didn't answer.
    fn header(&self) -> Vec<Line<'static>> {
        let name = sp(format!("{}  ", self.name), bold(SLOT));
        let Some(seen) = &self.seen else {
            return vec![Line::from(vec![name, sp(format!("{} mi collego…", spinner()), st(WARN))])];
        };
        let (status, _) = match seen {
            Ok(s) => s,
            Err(p) => {
                let mut t = vec![Line::from(vec![name, sp(format!("✖ {p}"), bold(ERR))])];
                let hint = p.hint(&self.name);
                if !hint.is_empty() {
                    t.push(Line::from(sp(hint, st(DIM))));
                }
                return t;
            }
        };
        let mut corral = format!("Corral {} · {} {}", status.version, status.os, status.arch);
        let same = status.os == std::env::consts::OS && status.arch == std::env::consts::ARCH;
        if remote::older(&status.version, remote::VERSION) && same {
            corral += &format!(" · qui c'è la {}: u la copia sull'host", remote::VERSION);
        }
        let mut t = vec![Line::from(vec![name, sp(corral, st(FG))])];
        for g in &status.gpus {
            let load = g.load();
            let memory = format!("{} · {}/{} MiB", g.name, g.used, g.total);
            let text = if load.is_empty() { memory } else { format!("{memory} · {load}") };
            t.push(Line::from(sp(format!("GPU  {text}"), st(DIM))));
        }
        t
    }

    fn row_cells(&self, row: &Row, ctx: &Ctx) -> Vec<ratatui::text::Span<'static>> {
        match row {
            Row::Server(r) => {
                let health = match r.health.as_str() {
                    "ok" => ("pronto", OK),
                    "loading" => ("caricamento", WARN),
                    "error" => ("errore", ERR),
                    _ => ("non risponde", WARN),
                };
                let here = match self.linked(ctx, r) {
                    Some(s) => format!("qui su {}", s.dash.endpoint()),
                    None => "invio: collegalo qui".into(),
                };
                let kind = if r.info.router { "● router" } else { "● attivo" };
                let mut details = format!("{} · {}", r.url, fname(&r.info.model));
                if !r.keep {
                    details += " · si ferma alla chiusura del Corral dell'host"; // started there, not kept
                }
                vec![
                    sp(kind, bold(OK)),
                    sp(r.info.alias.clone(), bold(FG)),
                    sp(health.0, bold(health.1)),
                    cell(details),
                    sp(here, st(CYAN)),
                ]
            }
            Row::Profile(p) => vec![
                sp("profilo", st(SIDE)),
                sp(p.name.clone(), bold(FG)),
                cell(""),
                cell(crate::router::count(p.models)),
                sp("invio: avvia", st(DIM)),
            ],
            Row::Model(m) => {
                let presets = if m.presets.len() > 1 { format!(" (+{})", m.presets.len() - 1) } else { String::new() };
                vec![
                    sp("modello", st(DIM)),
                    sp(fname(&m.path).to_string(), st(FG)),
                    cell(""),
                    cell(format!("{} · preset {}{presets}", size(m.size), m.preset)),
                    sp("invio: avvia", st(DIM)),
                ]
            }
        }
    }

    fn open(&mut self, row: Row, ctx: &mut Ctx) -> Act {
        match row {
            Row::Server(r) => return self.attach(*r, ctx),
            Row::Profile(p) => self.start(&p.name, vec!["--profile".into(), p.name.clone()]),
            Row::Model(m) if m.presets.len() > 1 => {
                let rows =
                    m.presets.iter().map(|p| vec![sp(p.clone(), if *p == m.preset { bold(FG) } else { st(FG) })]);
                let sel = m.presets.iter().position(|p| *p == m.preset).unwrap_or(0);
                ctx.ask("preset", Modal::choose("con quale preset?", &["preset"], rows.collect(), sel));
                self.choosing = Some(m);
            }
            Row::Model(m) => {
                let words = vec![m.path.clone(), "--preset".into(), m.preset.clone()];
                self.start(fname(&m.path), words);
            }
        }
        Act::None
    }
}

impl Screen for Host {
    fn title(&self) -> String {
        self.name.clone()
    }

    fn keys(&self, _ctx: &Ctx) -> Vec<Key> {
        vec![
            (KeyCode::Enter, "invio", "collega/avvia"),
            (KeyCode::Char('s'), "s", "ferma"),
            (KeyCode::Char('o'), "o", "Corral dell'host qui"),
            (KeyCode::Char('u'), "u", "copia Corral"),
            (KeyCode::Char('r'), "r", "aggiorna"),
            (KeyCode::Esc, "esc", "indietro"),
        ]
    }

    fn draw(&mut self, f: &mut Frame, area: Rect, ctx: &Ctx) {
        let header = self.header();
        let [top, list, bottom] =
            Layout::vertical([Constraint::Length(header.len() as u16 + 1), Constraint::Fill(1), Constraint::Length(2)])
                .areas(area);
        f.render_widget(Paragraph::new(header), top);
        let rows: Vec<Vec<_>> = self.rows().iter().map(|r| self.row_cells(r, ctx)).collect();
        match &self.seen {
            Some(Ok(_)) if rows.is_empty() => {
                let t = vec![
                    Line::from(sp("nessun server attivo e nessun modello sull'host", st(WARN))),
                    Line::from(sp(
                        "o: apri qui il suo Corral, per scaricare un modello o installare llama.cpp",
                        st(DIM),
                    )),
                ];
                f.render_widget(Paragraph::new(t), list);
            }
            Some(Ok(_)) => self.list.draw(f, list, &["", "nome", "stato", "dettagli", ""], rows, ctx),
            _ => {} // no answer yet, or the problem the header shows
        }
        let status = if self.busy() {
            let what = if self.linking.is_some() {
                format!("apro il tunnel verso {}…", self.name)
            } else {
                self.doing_what.clone()
            };
            sp(format!("{} {what}", spinner()), bold(WARN))
        } else if self.asking.is_some() && self.seen.is_some() {
            sp(format!("{} aggiorno…", spinner()), st(DIM))
        } else {
            sp("i server collegati restano sull'host: uscire da Corral chiude solo i tunnel", st(DIM))
        };
        f.render_widget(Paragraph::new(Line::from(status)).wrap(Wrap { trim: true }), bottom);
    }

    fn on(&mut self, ev: &Ev, ctx: &mut Ctx) -> Act {
        let rows = self.rows();
        let current = rows.get(self.list.sel).cloned();
        match ev {
            Ev::Answer("preset", Answer::Pick(i)) => {
                if let Some(m) = self.choosing.take()
                    && let Some(p) = m.presets.get(*i)
                {
                    let words = vec![m.path.clone(), "--preset".into(), p.clone()];
                    self.start(fname(&m.path), words);
                }
                return Act::None;
            }
            Ev::Answer("stop", Answer::Yes) => {
                if let Some(Row::Server(r)) = current {
                    self.stop(&r);
                }
                return Act::None;
            }
            Ev::Answer(..) => {
                self.choosing = None;
                return Act::None;
            }
            _ => {}
        }
        if let Some(Hit::Open(i)) = self.list.on(ev) {
            if self.busy() {
                ctx.toast("aspetta che finisca quello in corso", WARN);
                return Act::None;
            }
            return match rows.get(i).cloned() {
                Some(row) => self.open(row, ctx),
                None => Act::None,
            };
        }
        match key(ev) {
            Some(KeyCode::Char('s')) => match current {
                Some(Row::Server(r)) if !self.busy() => {
                    let question = format!("fermo {} su {}? (chi lo usa resta senza)", r.info.alias, self.name);
                    ctx.ask("stop", Modal::ask(&question, false));
                }
                _ => ctx.toast("scegli un server attivo", WARN),
            },
            Some(KeyCode::Char('o')) => ctx.foreground = Some(remote::session(&self.name)),
            Some(KeyCode::Char('u')) if !self.busy() => self.copy(),
            Some(KeyCode::Char('r')) => self.refresh(),
            _ => return Act::Pass,
        }
        Act::None
    }

    fn tick(&mut self, ctx: &mut Ctx) -> Act {
        if let Some(seen) = answer(&mut self.asking) {
            self.seen = Some(seen);
        }
        if let Some(result) = answer(&mut self.doing) {
            match result {
                Ok(msg) => ctx.toast(format!("✓ {}: {msg}", self.name), OK),
                Err(e) => ctx.error(format!("{}: {e}", self.name)),
            }
            self.refresh(); // it lists what the action changed
        }
        if let Some(result) = answer(&mut self.starting) {
            self.refresh();
            match result {
                Ok(r) => return self.attach(r, ctx),
                Err(e) => ctx.error(format!("{}: {e}", self.name)),
            }
        }
        if let Some(result) = answer(&mut self.linking) {
            match result {
                Ok((r, link)) => {
                    let s = Server::remote(&self.name, &r, link);
                    let screen = screen_of(&s);
                    ctx.servers.push(s);
                    return Act::Push(screen);
                }
                Err(e) => ctx.error(format!("{}: {e}", self.name)),
            }
        }
        Act::None
    }

    fn resume(&mut self, _ctx: &mut Ctx) {
        self.refresh(); // back from a dashboard, or from the host's own Corral
    }
}
