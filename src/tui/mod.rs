//! Full-screen terminal UI: one event loop, a stack of screens, modals and toasts.
//!
//! # Adding a screen
//!
//! 1. A struct with the screen's state and widgets (`widgets::List`, `TextInput`, …), and
//!    `impl Screen` for it:
//!    - `title()`: its breadcrumb in the header;
//!    - `keys()`: footer shortcuts; each is a clickable button that sends that key;
//!    - `draw(f, area, ctx)`: render into `area` (between header and footer). Widgets remember
//!      where they were drawn, so mouse hit-testing is just `widget.on(&ev)` later;
//!    - `on(ev, ctx)`: key presses, mouse events and modal answers. Return `Act::Pass` for what
//!      you don't handle: Esc / q / right click then go back, Ctrl+C quits;
//!    - `tick(ctx)`: every frame (~12/s) while on top: poll your `Task`s here;
//!    - `resume(ctx)`: back on top after the screen above closed (reload what it may have changed).
//! 2. Open it from another screen: `Act::Push(Box::new(MyScreen::new(..)))`.
//! 3. Never block in draw/on/tick: HTTP, processes and downloads go in
//!    `Task::spawn(move |tx| …)`, sending progress through `tx`; drain it with `task.poll()`.
//! 4. Questions: `ctx.ask("tag", Modal::confirm(..))`; the answer arrives in `on()` as
//!    `Ev::Answer("tag", Answer::…)`. Notices: `ctx.toast(..)`, `ctx.error(..)`.
//! 5. Logic with no terminal I/O stays in installs / hub / preset; screens only call it.
//!
//! The terminal is restored on every exit path: normal return, error, and panic (hook below;
//! screens are dropped while unwinding, so the dashboard still stops its server).

pub mod agent;
pub mod config;
pub mod hf;
pub mod hosts;
pub mod installs;
pub mod logs;
pub mod menu;
pub mod models;
pub mod news;
pub mod router;
pub mod splash;
#[cfg(test)]
pub mod tests;
pub mod widgets;

use std::io::{self, stdout};
use std::process::{Command, Stdio};
use std::sync::LazyLock;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};

use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton,
    MouseEvent, MouseEventKind,
};
use ratatui::crossterm::{cursor, execute, terminal};
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Padding, Paragraph, Wrap};
use ratatui::{Frame, Terminal};

use crate::installs::{self as inst, Device, Install};
use crate::ui::*;
use widgets::*;

pub static START: LazyLock<Instant> = LazyLock::new(Instant::now);

/// A desktop notification, for what ends while the user may be looking elsewhere (a download, a
/// build, a crash). On a local Linux desktop notify-send; else the terminal's own notification,
/// which also travels over ssh: OSC 9 for iTerm2, OSC 777 for the others that have one (foot,
/// WezTerm, Ghostty, GNOME's). A terminal that knows neither ignores it.
pub(crate) fn notify(msg: &str) {
    if cfg!(test) {
        return; // not on the desktop of whoever runs the tests
    }
    let desktop = cfg!(target_os = "linux") && !over_ssh() && which("notify-send").is_some();
    if desktop {
        let mut c = Command::new("notify-send");
        c.args(["Corral", msg]).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
        if let Ok(mut child) = c.spawn() {
            thread::spawn(move || child.wait()); // reap it
            return;
        }
    }
    // control characters would end the sequence early (or start another one); ; splits OSC 777
    let text: String = msg.chars().map(|c| if c.is_control() || c == ';' { ' ' } else { c }).collect();
    let iterm = std::env::var("TERM_PROGRAM").is_ok_and(|t| t == "iTerm.app");
    let sequence = if iterm { format!("\x1b]9;{text}\x07") } else { format!("\x1b]777;notify;Corral;{text}\x07") };
    let mut out = stdout();
    let _ = io::Write::write_all(&mut out, sequence.as_bytes());
    let _ = io::Write::flush(&mut out);
}

/// What a screen receives.
#[derive(Debug)]
pub enum Ev {
    Key(KeyEvent),
    Mouse(MouseEvent),
    Answer(&'static str, Answer),
}

/// How a modal was closed.
#[derive(Clone, Debug, PartialEq)]
pub enum Answer {
    Yes,
    No,
    Text(String),
    Pick(usize),
    Cancel,
}

/// What a screen wants the loop to do next.
pub enum Act {
    None,
    /// not handled: apply the defaults (back on Esc/q/right click)
    Pass,
    Push(Box<dyn Screen>),
    Pop,
    Replace(Box<dyn Screen>),
    /// back to the main menu
    Home,
}

/// Footer shortcut: (key it sends, label, description).
pub type Key = (KeyCode, &'static str, &'static str);

pub trait Screen {
    fn title(&self) -> String;
    fn keys(&self, _ctx: &Ctx) -> Vec<Key> {
        vec![]
    }
    fn draw(&mut self, f: &mut Frame, area: Rect, ctx: &Ctx);
    fn on(&mut self, ev: &Ev, ctx: &mut Ctx) -> Act;
    fn tick(&mut self, _ctx: &mut Ctx) -> Act {
        Act::None
    }
    fn resume(&mut self, _ctx: &mut Ctx) {}
    /// false = draws the whole screen itself (no Corral header); the footer stays.
    fn chrome(&self) -> bool {
        true
    }
}

// -- background work ---------------------------------------------------------------------

/// A thread sending messages to the UI; `poll()` drains them without blocking.
pub struct Task<M> {
    rx: Receiver<M>,
    pub done: bool,
}

impl<M: Send + 'static> Task<M> {
    pub fn spawn(f: impl FnOnce(Sender<M>) + Send + 'static) -> Self {
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || f(tx));
        Self { rx, done: false }
    }

    /// Wrap a channel whose sender lives in some other task.
    pub fn from_rx(rx: Receiver<M>) -> Self {
        Self { rx, done: false }
    }

    /// One result, e.g. `Task::once(move || hub::search(&q))`.
    pub fn once(f: impl FnOnce() -> M + Send + 'static) -> Self {
        Self::spawn(move |tx| {
            let _ = tx.send(f());
        })
    }

    pub fn poll(&mut self) -> Vec<M> {
        let mut out = vec![];
        loop {
            match self.rx.try_recv() {
                Ok(m) => out.push(m),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.done = true;
                    break;
                }
            }
        }
        out
    }
}

/// Messages of a download task: next file, progress, end.
pub enum Fetch<T> {
    File(String),
    Prog(Dl),
    Done(Result<T, String>),
}

// -- shared context ------------------------------------------------------------------------

/// Host facts gathered at startup (and again after the active install changes).
#[derive(Default)]
pub struct Hw {
    pub step: usize, // how many of HW_STEPS are done
    pub inst: Option<Install>,
    pub ram: u64,
    pub version: String,
    pub devs: Vec<Device>,
}

pub const HW_STEPS: [&str; 4] =
    ["installazione attiva", "RAM disponibile", "versione llama.cpp", "GPU · --list-devices"];

impl Hw {
    pub fn ready(&self) -> bool {
        self.step >= HW_STEPS.len()
    }
    /// Free VRAM of the largest GPU, in bytes.
    pub fn budget(&self) -> u64 {
        self.devs.iter().map(|x| x.free).max().unwrap_or(0) << 20
    }
}

enum HwMsg {
    Inst(Option<Install>),
    Ram(u64),
    Version(String),
    Devs(Vec<Device>),
}

fn hw_task(state: &State) -> Task<HwMsg> {
    let want = state.install.clone();
    Task::spawn(move |tx| {
        let inst = inst::active(&State { install: want, ..State::default() });
        let _ = tx.send(HwMsg::Inst(inst.clone()));
        let _ = tx.send(HwMsg::Ram(ram_available()));
        let _ = tx.send(HwMsg::Version(inst.as_ref().map_or("—".into(), inst::version)));
        let _ = tx.send(HwMsg::Devs(inst.as_ref().map_or(vec![], inst::devices)));
    })
}

struct Toast {
    msg: String,
    color: u32,
    until: Instant,
}

/// Everything screens share: saved state, host facts, the open modal, toasts, the pointer.
pub struct Ctx {
    pub state: State,
    pub hw: Hw,
    mouse: Option<Position>,
    modal: Option<(&'static str, Modal)>,
    toasts: Vec<Toast>,
    hw_task: Option<Task<HwMsg>>,
    /// llama-server processes (dropping one stops it, unless it's kept running after exit)
    pub servers: Vec<crate::server::Server>,
    /// a program to give the terminal to, until it ends (a host's Corral over ssh -t)
    pub foreground: Option<Command>,
}

impl Ctx {
    pub fn new(state: State) -> Self {
        let mut c = Self::bare(state);
        c.reload_hw();
        c.servers = crate::server::adopt();
        if !c.servers.is_empty() {
            c.toast(format!("{} server ancora attivi ripresi: menu › server attivi", c.servers.len()), OK);
        }
        c
    }

    /// No background detection (tests).
    pub fn bare(state: State) -> Self {
        Self {
            state,
            hw: Hw::default(),
            mouse: None,
            modal: None,
            toasts: vec![],
            hw_task: None,
            servers: vec![],
            foreground: None,
        }
    }

    pub fn set_mouse(&mut self, p: Option<Position>) {
        self.mouse = p;
    }

    pub fn ask(&mut self, tag: &'static str, m: Modal) {
        self.modal = Some((tag, m));
    }

    pub fn toast(&mut self, msg: impl Into<String>, color: u32) {
        let until = Instant::now() + Duration::from_secs(if color == ERR { 8 } else { 4 });
        self.toasts.push(Toast { msg: msg.into(), color, until });
    }

    pub fn error(&mut self, msg: impl Into<String>) {
        self.toast(format!("✖ {}", msg.into()), ERR);
    }

    pub fn save(&mut self) {
        if let Err(e) = save_state(&self.state) {
            self.error(e);
        }
    }

    /// A server that crashed comes back by itself, a few times (see `Server::restart`).
    fn restart_crashed(&mut self) {
        use crate::server::{RESTART_WINDOW, RESTARTS, Restart, last_error};
        for i in 0..self.servers.len() {
            let Some(restart) = self.servers[i].restart() else { continue };
            let s = &self.servers[i];
            let what = format!("{} si è fermato (codice {})", s.dash.alias(), s.dash.exit_code.unwrap_or(-1));
            let error = last_error(s.log_path());
            let why = if error.is_empty() { String::new() } else { format!(" · {error}") };
            match restart {
                Restart::Started(fresh) => {
                    self.toast(format!("⚠ {what}: lo riavvio{why}"), WARN);
                    notify(&format!("{what}: lo riavvio"));
                    self.servers[i] = *fresh;
                }
                Restart::GaveUp => {
                    let minutes = RESTART_WINDOW.as_secs() / 60;
                    self.error(format!("{what} {RESTARTS} volte in {minutes} minuti: resta fermo{why}"));
                    notify(&format!("{what} {RESTARTS} volte in {minutes} minuti: resta fermo"));
                }
                Restart::Failed(e) => {
                    self.error(format!("{what} e non riparte: {e}"));
                    notify(&format!("{what} e non riparte"));
                }
            }
        }
    }

    /// Re-detect install/version/GPUs in the background (values stay until replaced).
    pub fn reload_hw(&mut self) {
        self.hw.step = 0;
        self.hw_task = Some(hw_task(&self.state));
    }

    pub fn hovered(&self, r: Rect) -> bool {
        self.mouse.is_some_and(|p| r.contains(p))
    }

    fn pump(&mut self) {
        for s in &mut self.servers {
            s.pump();
        }
        self.restart_crashed();
        self.toasts.retain(|t| t.until > Instant::now());
        let Some(t) = self.hw_task.as_mut() else { return };
        for m in t.poll() {
            match m {
                HwMsg::Inst(i) => self.hw.inst = i,
                HwMsg::Ram(r) => self.hw.ram = r,
                HwMsg::Version(v) => self.hw.version = v,
                HwMsg::Devs(d) => self.hw.devs = d,
            }
            self.hw.step += 1;
        }
        if t.done {
            self.hw_task = None;
        }
    }
}

// -- terminal --------------------------------------------------------------------------------

fn enter() -> io::Result<Terminal<CrosstermBackend<io::Stdout>>> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        leave();
        hook(info);
    }));
    take_screen()?;
    Terminal::new(CrosstermBackend::new(stdout()))
}

/// Raw keys, the alternate screen, the mouse: the terminal as the UI wants it.
fn take_screen() -> io::Result<()> {
    terminal::enable_raw_mode()?;
    execute!(stdout(), terminal::EnterAlternateScreen, EnableMouseCapture)
}

/// Give the terminal back (idempotent: also called from the panic hook).
pub fn leave() {
    let _ = execute!(stdout(), DisableMouseCapture, terminal::LeaveAlternateScreen, cursor::Show);
    let _ = terminal::disable_raw_mode();
}

/// Run the UI until the last screen closes. `problem`: why the saved state couldn't be read
/// (shown as an error); `yes` = start the last model right away.
pub fn run(state: State, problem: Option<String>, yes: bool) -> io::Result<()> {
    LazyLock::force(&START);
    let mut term = enter()?;
    // SIGTERM/SIGHUP/console close: the quit thread stops the servers (their own guards), then this
    let _quit = on_quit(leave);
    let mut app = App {
        stack: vec![Box::new(menu::Menu::new()), Box::new(splash::Splash::new(yes))],
        ctx: Ctx::new(state),
        footer: Zones::default(),
    };
    if let Some(p) = problem {
        app.ctx.error(p);
    }
    let res = app.run(&mut term);
    if quit_requested() {
        // the quit thread (ui::on_quit) stops the servers, restores the terminal and exits; dropping
        // the screens here would race it, and their second SIGINT would turn the stop into a kill
        loop {
            thread::park();
        }
    }
    drop(app); // stops a running server before the terminal goes back to the shell
    leave();
    res
}

struct App {
    stack: Vec<Box<dyn Screen>>,
    ctx: Ctx,
    footer: Zones<KeyCode>,
}

impl App {
    fn run(&mut self, term: &mut Terminal<CrosstermBackend<io::Stdout>>) -> io::Result<()> {
        // a quit signal ends the loop (when it isn't stuck: see ui::on_quit); `run` takes it from there
        while !self.stack.is_empty() && !quit_requested() {
            self.ctx.pump();
            if let Some(top) = self.stack.last_mut() {
                let a = top.tick(&mut self.ctx);
                self.apply(a);
            }
            if self.stack.is_empty() || quit_requested() {
                break;
            }
            term.draw(|f| self.draw(f))?;
            if event::poll(Duration::from_millis(80))? {
                // drain the burst (mouse motion, paste) before the next frame
                while self.stack.last().is_some() {
                    let e = event::read()?;
                    self.event(e);
                    if !event::poll(Duration::ZERO)? {
                        break;
                    }
                }
            }
            if let Some(cmd) = self.ctx.foreground.take() {
                self.hand_over(term, cmd)?;
            }
        }
        Ok(())
    }

    /// Give the terminal to `cmd` until it ends (the screen of the shell comes back meanwhile),
    /// then take it back. A failure stays on screen until Enter: ssh says why there.
    fn hand_over(&mut self, term: &mut Terminal<CrosstermBackend<io::Stdout>>, mut cmd: Command) -> io::Result<()> {
        leave();
        let status = without_interrupt(|| cmd.status());
        let ok = status.as_ref().is_ok_and(|s| s.success());
        if let Err(e) = &status {
            println!("impossibile avviare ssh: {e}");
        }
        if !ok {
            println!("\n(invio per tornare a Corral)");
            let _ = io::stdin().read_line(&mut String::new());
        }
        take_screen()?;
        term.clear()?;
        if let Some(top) = self.stack.last_mut() {
            top.resume(&mut self.ctx);
        }
        Ok(())
    }

    fn event(&mut self, e: Event) {
        match e {
            // Windows reports releases too
            Event::Key(k) if k.kind != KeyEventKind::Release => self.dispatch(Ev::Key(k)),
            Event::Mouse(m) => {
                let p = Position::new(m.column, m.row);
                match m.kind {
                    MouseEventKind::Moved | MouseEventKind::Drag(_) => self.ctx.set_mouse(Some(p)),
                    MouseEventKind::Down(MouseButton::Left) if self.ctx.modal.is_none() => match self.footer.at(p) {
                        Some(code) => self.dispatch(Ev::Key(KeyEvent::new(code, KeyModifiers::NONE))),
                        None => self.dispatch(Ev::Mouse(m)),
                    },
                    _ => self.dispatch(Ev::Mouse(m)),
                }
            }
            _ => {} // resize: the next frame redraws at the new size
        }
    }

    fn dispatch(&mut self, ev: Ev) {
        if let Some((tag, m)) = self.ctx.modal.as_mut() {
            if is_ctrl_c(&ev) {
                self.ctx.modal = None; // Ctrl+C always gets out of a modal
                return;
            }
            if let Some(ans) = m.on(&ev) {
                let tag = *tag;
                self.ctx.modal = None;
                self.dispatch(Ev::Answer(tag, ans));
            }
            return;
        }
        let Some(top) = self.stack.last_mut() else { return };
        match top.on(&ev, &mut self.ctx) {
            Act::Pass => {
                // back: q, Esc, right click; at the main menu only q (and Ctrl+C) quit
                let root = self.stack.len() == 1;
                let back = match &ev {
                    Ev::Key(k) => k.code == KeyCode::Char('q') || (k.code == KeyCode::Esc && !root),
                    Ev::Mouse(m) => m.kind == MouseEventKind::Down(MouseButton::Right) && !root,
                    Ev::Answer(..) => false,
                };
                if is_ctrl_c(&ev) {
                    self.stack.clear(); // everything closes (the dashboard handles Ctrl+C itself)
                } else if back {
                    self.apply(Act::Pop);
                }
            }
            a => self.apply(a),
        }
    }

    fn apply(&mut self, a: Act) {
        let resumed = match a {
            Act::None | Act::Pass => false,
            Act::Push(s) => {
                self.stack.push(s);
                false
            }
            Act::Pop => {
                self.stack.pop();
                true
            }
            Act::Replace(s) => {
                self.stack.pop();
                self.stack.push(s);
                false
            }
            Act::Home => {
                self.stack.truncate(1);
                true
            }
        };
        if resumed && let Some(top) = self.stack.last_mut() {
            top.resume(&mut self.ctx);
        }
    }

    fn draw(&mut self, f: &mut Frame) {
        let area = f.area();
        f.render_widget(Block::new().style(bg(BG)), area);
        let [main, foot] = Layout::vertical([Constraint::Fill(1), Constraint::Length(1)]).areas(area);
        let Some(top) = self.stack.last_mut() else { return };
        let keys = top.keys(&self.ctx);
        if top.chrome() {
            let [head, body] = Layout::vertical([Constraint::Length(2), Constraint::Fill(1)]).areas(main);
            let crumbs: Vec<String> = self.stack.iter().map(|s| s.title()).collect();
            draw_header(f, head, &crumbs, &self.ctx);
            let body = Rect { x: body.x + 1, width: body.width.saturating_sub(2), ..body };
            if let Some(top) = self.stack.last_mut() {
                top.draw(f, body, &self.ctx);
            }
        } else {
            top.draw(f, main, &self.ctx);
        }
        self.footer = draw_footer(f, foot, &keys, &self.ctx);
        if let Some((tag, mut m)) = self.ctx.modal.take() {
            m.draw(f, main, &self.ctx);
            self.ctx.modal = Some((tag, m));
        }
        draw_toasts(f, main, &self.ctx);
    }
}

fn is_ctrl_c(ev: &Ev) -> bool {
    matches!(ev, Ev::Key(k) if k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL))
}

// -- chrome ------------------------------------------------------------------------------------

/// The bar on top: the wordmark, where you are, and on the right the active llama.cpp and how many
/// servers are running. A blank row under it separates it from the screen.
fn draw_header(f: &mut Frame, area: Rect, crumbs: &[String], ctx: &Ctx) {
    let bar_row = Rect { height: 1, ..area };
    f.render_widget(Block::new().style(bg(SURFACE)), bar_row);
    let mut spans = vec![sp(" ◆ CORRAL ", bold(BG).bg(rgb(SIDE))), sp("  ", st(FG))];
    for (i, c) in crumbs.iter().enumerate() {
        if i > 0 {
            spans.push(sp(" › ", st(FAINT)));
        }
        let last = i + 1 == crumbs.len();
        spans.push(sp(c.clone(), if last { bold(FG) } else { st(DIM) }));
    }
    let mut right: Vec<Span> = vec![];
    let running = ctx.servers.iter().filter(|s| s.running()).count();
    if running > 0 {
        let label = if running == 1 { "1 server attivo".to_string() } else { format!("{running} server attivi") };
        right.push(sp("● ", st(OK)));
        right.push(sp(label, st(FG)));
        right.push(sp("   ", st(FG)));
    }
    match &ctx.hw.inst {
        Some(i) => {
            right.push(sp("llama.cpp ", st(FAINT)));
            right.push(sp(i.name.clone(), st(DIM)));
        }
        None if ctx.hw.step == 0 => right.push(sp("rilevamento…", st(FAINT))),
        None => right.push(sp("nessuna installazione", st(WARN))),
    }
    right.push(sp(" ", st(FG)));
    let used: usize = spans.iter().map(|s| len(&s.content)).sum();
    let right_w: usize = right.iter().map(|s| len(&s.content)).sum();
    let gap = (area.width as usize).saturating_sub(used + right_w);
    if gap > 1 {
        spans.push(sp(" ".repeat(gap), st(FG)));
        spans.extend(right);
    }
    f.render_widget(Paragraph::new(Line::from(spans)).style(bg(SURFACE)), bar_row);
}

/// Shortcut buttons; returns their click zones. Too narrow: the first word of each description.
pub(crate) fn draw_footer(f: &mut Frame, area: Rect, keys: &[Key], ctx: &Ctx) -> Zones<KeyCode> {
    let mut zones = Zones::default();
    let mut spans: Vec<Span> = vec![sp(" ", st(FG))];
    let mut x = area.x + 1;
    let full: usize = keys.iter().map(|(_, l, d)| len(l) + len(d) + 6).sum();
    let short = full + 1 > area.width as usize;
    for &(code, label, desc) in keys {
        let desc = if short { desc.split(' ').next().unwrap_or(desc) } else { desc };
        let w = (len(label) + 2 + 1 + len(desc)) as u16;
        if x + w > area.right() {
            break;
        }
        let r = Rect::new(x, area.y, w, 1);
        let hot = ctx.hovered(r);
        let badge = if hot { bold(BG).bg(rgb(SIDE)) } else { bold(SIDE).bg(rgb(SEL)) };
        spans.extend([sp(format!(" {label} "), badge), sp(format!(" {desc}"), st(if hot { FG } else { DIM }))]);
        spans.push(sp("   ", st(FG)));
        zones.add(r, code);
        x += w + 3;
    }
    f.render_widget(Paragraph::new(Line::from(spans)).style(bg(SURFACE)), area);
    zones
}

/// Notices, newest at the bottom right: a raised strip with a stripe in the notice's colour.
pub(crate) fn draw_toasts(f: &mut Frame, area: Rect, ctx: &Ctx) {
    let mut y = area.bottom();
    for t in ctx.toasts.iter().rev() {
        // as wide as the message (at least 20 columns), but never wider than the screen
        let w = (len(&t.msg) as u16 + 5).max(20).min(area.width.saturating_sub(2));
        let text = vec![Line::from(sp(t.msg.clone(), st(FG)))];
        let h = wrapped(&text, w.saturating_sub(5)) + 2;
        let p = Paragraph::new(text).wrap(Wrap { trim: true });
        if y < area.y + h {
            break;
        }
        y -= h;
        let r = Rect::new(area.right().saturating_sub(w + 1), y, w, h);
        let stripe = Block::new()
            .borders(Borders::LEFT)
            .border_type(BorderType::Thick)
            .border_style(st(t.color).bg(rgb(SURFACE)))
            .style(bg(SURFACE))
            .padding(Padding::new(2, 2, 1, 1));
        f.render_widget(ratatui::widgets::Clear, r);
        f.render_widget(p.block(stripe), r);
        y = y.saturating_sub(1); // a row of air between two notices
    }
}
