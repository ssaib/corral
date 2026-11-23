//! Recent logs (logs/llama-*.log of the servers, logs/build-*.log of the builds) and a viewer for one.

use std::path::PathBuf;
use std::time::SystemTime;

use ratatui::Frame;
use ratatui::crossterm::event::KeyCode;
use ratatui::layout::Rect;
use ratatui::text::Line;
use ratatui::widgets::Paragraph;

use super::widgets::*;
use super::{Act, Ctx, Ev, Key, Screen};
use crate::ui::*;

const TAIL: usize = 5000; // lines kept by the viewer

fn ago(t: SystemTime) -> String {
    let s = SystemTime::now().duration_since(t).map_or(0, |d| d.as_secs());
    match s {
        0..60 => "ora".into(),
        60..3600 => format!("{} min fa", s / 60),
        3600..86400 => format!("{} h fa", s / 3600),
        _ => format!("{} g fa", s / 86400),
    }
}

pub struct Logs {
    list: List,
    files: Vec<(PathBuf, u64, SystemTime)>,
}

impl Logs {
    pub fn new() -> Self {
        let files = log_files();
        let mut list = List::new(0);
        list.set_len(files.len());
        Self { list, files }
    }
}

impl Screen for Logs {
    fn title(&self) -> String {
        "log".into()
    }

    fn keys(&self, _ctx: &Ctx) -> Vec<Key> {
        vec![(KeyCode::Enter, "invio", "apri"), (KeyCode::Esc, "esc", "indietro")]
    }

    fn draw(&mut self, f: &mut Frame, area: Rect, ctx: &Ctx) {
        if self.files.is_empty() {
            f.render_widget(Paragraph::new(sp(format!("nessun log in {}", logs().display()), st(DIM))), area);
            return;
        }
        let rows = self
            .files
            .iter()
            .map(|(p, n, t)| vec![sp(fname(&ps(p)).to_string(), bold(FG)), cell(size(*n)), sp(ago(*t), st(DIM))])
            .collect();
        self.list.draw(f, area, &["file", "dimensione", "modificato"], rows, ctx);
    }

    fn on(&mut self, ev: &Ev, _ctx: &mut Ctx) -> Act {
        match self.list.on(ev) {
            Some(Hit::Open(i)) => Act::Push(Box::new(Viewer::new(self.files[i].0.clone()))),
            Some(Hit::Moved) => Act::None,
            None => Act::Pass,
        }
    }

    fn resume(&mut self, _ctx: &mut Ctx) {
        let sel = self.list.sel;
        *self = Self::new();
        self.list.sel = sel.min(self.files.len().saturating_sub(1));
    }
}

/// The last TAIL lines of a log, opened at the bottom; arrows / PgUp / PgDn / Home / End / wheel scroll.
pub struct Viewer {
    path: PathBuf,
    lines: Vec<String>,
    top: Option<usize>, // first line shown; None = follow the end
    page: usize,
}

impl Viewer {
    pub fn new(path: PathBuf) -> Self {
        let text = std::fs::read(&path).map(|b| String::from_utf8_lossy(&b).into_owned()).unwrap_or_default();
        let all: Vec<&str> = text.lines().collect();
        let lines = all[all.len().saturating_sub(TAIL)..].iter().map(|l| l.to_string()).collect();
        Self { path, lines, top: None, page: 1 }
    }

    fn max_top(&self) -> usize {
        self.lines.len().saturating_sub(self.page)
    }

    fn scroll(&mut self, d: isize) {
        let cur = self.top.unwrap_or(self.max_top()) as isize;
        let t = (cur + d).clamp(0, self.max_top() as isize) as usize;
        self.top = (t < self.max_top()).then_some(t);
    }
}

impl Screen for Viewer {
    fn title(&self) -> String {
        fname(&ps(&self.path)).to_string()
    }

    fn keys(&self, _ctx: &Ctx) -> Vec<Key> {
        vec![
            (KeyCode::PageUp, "pgsu", "pagina su"),
            (KeyCode::PageDown, "pggiù", "pagina giù"),
            (KeyCode::Home, "home", "inizio"),
            (KeyCode::End, "fine", "fine"),
            (KeyCode::Esc, "esc", "indietro"),
        ]
    }

    fn draw(&mut self, f: &mut Frame, area: Rect, _ctx: &Ctx) {
        let blk = panel(&short(&ps(&self.path)), DIM);
        let inner = blk.inner(area);
        f.render_widget(blk, area);
        self.page = inner.height.max(1) as usize;
        let top = self.top.unwrap_or(self.max_top()).min(self.max_top());
        let w = inner.width as usize;
        let shown: Vec<Line> = self.lines[top..(top + self.page).min(self.lines.len())]
            .iter()
            .map(|l| {
                let low = l.to_lowercase();
                let c = if low.contains("error") || low.contains("failed") {
                    ERR
                } else if low.contains("warn") {
                    WARN
                } else {
                    FG
                };
                Line::from(sp(clip(l, w), st(c)))
            })
            .collect();
        f.render_widget(Paragraph::new(shown), inner);
    }

    fn on(&mut self, ev: &Ev, _ctx: &mut Ctx) -> Act {
        if let Some((d, _)) = wheel(ev) {
            self.scroll(d as isize * 3);
            return Act::None;
        }
        let page = self.page as isize;
        match key(ev) {
            Some(KeyCode::Up) => self.scroll(-1),
            Some(KeyCode::Down) => self.scroll(1),
            Some(KeyCode::PageUp) => self.scroll(-page),
            Some(KeyCode::PageDown) => self.scroll(page),
            Some(KeyCode::Home) => self.top = Some(0),
            Some(KeyCode::End) => self.top = None,
            _ => return Act::Pass,
        }
        Act::None
    }
}
