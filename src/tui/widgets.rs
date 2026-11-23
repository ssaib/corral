//! Reusable pieces for screens: selectable `List`, `TextInput`, `Modal`, click `Zones`,
//! panels and bars. Widgets remember where they were drawn, so mouse hit-testing is
//! `widget.on(&ev)` with no extra bookkeeping in the screen.

use std::time::{Duration, Instant};

use ratatui::Frame;
use ratatui::crossterm::event::{KeyCode, KeyModifiers, MouseButton, MouseEventKind};
use ratatui::layout::{Position, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, BorderType, Clear, Padding, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap,
};

use super::{Answer, Ctx, Ev};
use crate::ui::*;

/// Selected-row background, the lighter one under the mouse, a text field.
pub const SEL: u32 = 0x2b2e38;
pub const HOVER: u32 = 0x23252d;
pub const FIELD: u32 = 0x282b34;

pub fn bg(hex: u32) -> Style {
    Style::new().bg(rgb(hex))
}

/// Rounded panel titled "╭─ title ─". A neutral border (DIM or LINE) is drawn in LINE: the
/// border takes a colour only when that colour means something (focus, live, an error).
pub fn block(title: String, title_style: Style, border: u32) -> Block<'static> {
    let border = if border == DIM { LINE } else { border };
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(st(border))
        .padding(Padding::horizontal(1))
        .title(Line::from(vec![sp("─", st(border)), sp(title, title_style)]))
}

/// A panel with a quiet border and its title in `color`.
pub fn panel(title: &str, color: u32) -> Block<'static> {
    block(format!(" {title} "), bold(color), LINE)
}

/// A panel whose border is lit too: the focused or active one.
pub fn lit(title: &str, color: u32) -> Block<'static> {
    block(format!(" {title} "), bold(color), color)
}

/// Thin gauge: the filled part in `color`, the rest of the track in LINE.
pub fn bar(frac: f64, width: usize, color: u32) -> [Span<'static>; 2] {
    let n = (frac.clamp(0.0, 1.0) * width as f64).round() as usize;
    [sp("━".repeat(n), st(color)), sp("━".repeat(width - n), st(LINE))]
}

pub fn spinner() -> char {
    let t = super::START.elapsed().as_millis() / 80;
    ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'][(t % 10) as usize]
}

/// Rows `lines` take once wrapped at `w` columns (word wrap can need a little more).
pub fn wrapped(lines: &[Line], w: u16) -> u16 {
    let rows = |l: &Line| match (l.width() as u16).div_ceil(w.max(1)) {
        0 => 1,
        1 => 1,
        n => n + 1, // word breaks waste some columns: one row of slack
    };
    lines.iter().map(rows).sum()
}

/// `w` columns centred in `area` (clamped to it), `h` rows.
pub fn centered(area: Rect, w: u16, h: u16) -> Rect {
    let (w, h) = (w.min(area.width), h.min(area.height));
    Rect::new(area.x + (area.width - w) / 2, area.y + (area.height - h) / 2, w, h)
}

/// Left-button press position.
pub fn click(ev: &Ev) -> Option<Position> {
    match ev {
        Ev::Mouse(m) if m.kind == MouseEventKind::Down(MouseButton::Left) => Some(Position::new(m.column, m.row)),
        _ => None,
    }
}

/// Wheel: -1 up, +1 down, with the pointer position.
pub fn wheel(ev: &Ev) -> Option<(i32, Position)> {
    match ev {
        Ev::Mouse(m) if m.kind == MouseEventKind::ScrollUp => Some((-1, Position::new(m.column, m.row))),
        Ev::Mouse(m) if m.kind == MouseEventKind::ScrollDown => Some((1, Position::new(m.column, m.row))),
        _ => None,
    }
}

/// Plain key press (no Ctrl/Alt).
pub fn key(ev: &Ev) -> Option<KeyCode> {
    match ev {
        Ev::Key(k) if !k.modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) => Some(k.code),
        _ => None,
    }
}

/// Clickable rectangles recorded while drawing, looked up on click.
pub struct Zones<T>(Vec<(Rect, T)>);

impl<T: Clone> Default for Zones<T> {
    fn default() -> Self {
        Self(vec![])
    }
}

impl<T: Clone> Zones<T> {
    pub fn clear(&mut self) {
        self.0.clear();
    }
    pub fn add(&mut self, r: Rect, v: T) {
        self.0.push((r, v));
    }
    pub fn at(&self, p: Position) -> Option<T> {
        self.0.iter().find(|(r, _)| r.contains(p)).map(|(_, v)| v.clone())
    }
}

/// A row of buttons drawn left to right from `at`; returns the rects (for `Zones`).
pub fn buttons(f: &mut Frame, at: Rect, items: &[(&str, bool)], color: u32, ctx: &Ctx) -> Vec<Rect> {
    let mut x = at.x;
    let mut out = vec![];
    for (label, focused) in items {
        let w = len(label) as u16 + 4;
        let r = Rect::new(x, at.y, w.min(at.right().saturating_sub(x)), 1);
        let style = if *focused {
            bold(BG).bg(rgb(color))
        } else if ctx.hovered(r) {
            bold(FG).bg(rgb(HOVER))
        } else {
            st(FG).bg(rgb(SEL))
        };
        f.render_widget(Paragraph::new(Line::from(sp(format!("  {label}  "), style))), r);
        out.push(r);
        x += w + 2;
    }
    out
}

/// Download progress for one file: bar, bytes, speed, ETA (or the resume check).
pub struct Transfer {
    label: String,
    done: u64,
    total: u64,
    verify: bool,
    t0: Instant,
    base: Option<u64>, // bytes already there when receiving started (for the speed)
}

impl Transfer {
    pub fn new(label: &str) -> Self {
        Self { label: label.into(), done: 0, total: 0, verify: false, t0: Instant::now(), base: None }
    }

    pub fn update(&mut self, p: Dl) {
        match p {
            Dl::Verify(d, t) => (self.verify, self.done, self.total) = (true, d, t),
            Dl::Get(d, t) => {
                if self.base.is_none() {
                    (self.base, self.t0) = (Some(d), Instant::now());
                }
                (self.verify, self.done, self.total) = (false, d, t);
            }
        }
    }

    pub fn lines(&self, w: usize) -> Vec<Line<'static>> {
        let frac = if self.total > 0 { self.done as f64 / self.total as f64 } else { 0.0 };
        let mut b = bar(frac, w.saturating_sub(8).min(60), if self.verify { WARN } else { LOAD }).to_vec();
        b.push(sp(format!(" {:>3.0}%", frac * 100.0), st(FG)));
        let status = if self.verify {
            vec![sp(format!("{} riprendo: verifico la parte già scaricata…", spinner()), st(WARN))]
        } else {
            let fresh = self.done - self.base.unwrap_or(0);
            let speed = fresh as f64 / self.t0.elapsed().as_secs_f64().max(1e-3);
            let eta = if speed > 0.0 && self.total > self.done {
                ((self.total - self.done) as f64 / speed) as u64
            } else {
                0
            };
            vec![
                sp(format!("{} / {}", size(self.done), size(self.total)), st(OK)),
                sp(format!("  ·  {}/s", size(speed as u64)), st(SRV)),
                sp(format!("  ·  {}:{:02}:{:02}", eta / 3600, eta / 60 % 60, eta % 60), st(CYAN)),
            ]
        };
        vec![Line::from(sp(format!("↓ {}", self.label), bold(FG))), Line::default(), Line::from(b), Line::from(status)]
    }
}

// -- List ------------------------------------------------------------------------------

/// What a `List` did with an event.
#[derive(Debug, PartialEq)]
pub enum Hit {
    Moved,
    Open(usize),
}

/// Numbered, selectable table: arrows/PgUp/PgDn/Home/End, Enter, digits for quick picks,
/// wheel, click (first click selects, a click on the selected row opens; `single_click`
/// opens at once). Rows are passed to `draw` each frame, so the list never goes stale.
#[derive(Default)]
pub struct List {
    pub sel: usize,
    pub mark: Option<usize>, // row shown with ● (active install, last model, recommended quant)
    pub single_click: bool,
    len: usize,
    offset: usize,
    area: Rect,     // whole widget, from the last draw
    body: Rect,     // rows only
    cols: Vec<u16>, // x of each column (after the number), relative to body.x
    digits: String,
    digit_at: Option<Instant>,
}

impl List {
    pub fn new(sel: usize) -> Self {
        Self { sel, ..Self::default() }
    }

    /// Set the row count without drawing (lists filled before their first frame).
    pub fn set_len(&mut self, n: usize) {
        self.len = n;
        self.sel = self.sel.min(n.saturating_sub(1));
    }

    /// Screen rect of row `i`'s column `c` up to the right edge, if visible (inline editors).
    pub fn cell_rect(&self, i: usize, c: usize) -> Option<Rect> {
        let y = self.body.y + (i.checked_sub(self.offset)? as u16);
        let x = self.body.x + *self.cols.get(c)?;
        (y < self.body.bottom()).then(|| Rect::new(x, y, self.body.right().saturating_sub(x + 1), 1))
    }

    fn row_at(&self, p: Position) -> Option<usize> {
        let i = self.offset + p.y.checked_sub(self.body.y)? as usize;
        (self.body.contains(p) && i < self.len).then_some(i)
    }

    fn go(&mut self, i: isize) -> Option<Hit> {
        if self.len > 0 {
            self.sel = i.clamp(0, self.len as isize - 1) as usize;
        }
        Some(Hit::Moved)
    }

    /// None = not a list event (the screen handles it).
    pub fn on(&mut self, ev: &Ev) -> Option<Hit> {
        if let Some(p) = click(ev) {
            let i = self.row_at(p)?;
            if self.single_click || i == self.sel {
                self.sel = i;
                return Some(Hit::Open(i));
            }
            self.sel = i;
            return Some(Hit::Moved);
        }
        if let Some((d, p)) = wheel(ev) {
            return self.area.contains(p).then(|| self.go(self.sel as isize + d as isize)).flatten();
        }
        let page = self.body.height.max(1) as isize;
        let s = self.sel as isize;
        let code = key(ev)?;
        if !matches!(code, KeyCode::Char('0'..='9')) {
            self.digits.clear();
        }
        match code {
            KeyCode::Up => self.go(s - 1),
            KeyCode::Down => self.go(s + 1),
            KeyCode::PageUp => self.go(s - page),
            KeyCode::PageDown => self.go(s + page),
            KeyCode::Home => self.go(0),
            KeyCode::End => self.go(self.len as isize - 1),
            KeyCode::Enter if self.len > 0 => Some(Hit::Open(self.sel)),
            KeyCode::Char(c @ '0'..='9') => self.digit(c),
            _ => None,
        }
    }

    /// Digits typed within a second build a row number; it opens as soon as no longer
    /// number could exist (with ≤ 9 rows: at once), otherwise Enter opens it.
    fn digit(&mut self, c: char) -> Option<Hit> {
        if self.digit_at.is_none_or(|t| t.elapsed() > Duration::from_secs(1)) {
            self.digits.clear();
        }
        self.digits.push(c);
        self.digit_at = Some(Instant::now());
        let n: usize = self.digits.parse().unwrap_or(0);
        if n == 0 || n > self.len {
            self.digits.clear();
            return Some(Hit::Moved);
        }
        self.sel = n - 1;
        if n * 10 > self.len {
            self.digits.clear();
            return Some(Hit::Open(n - 1));
        }
        Some(Hit::Moved)
    }

    pub fn draw(&mut self, f: &mut Frame, area: Rect, head: &[&str], rows: Vec<Vec<Span<'static>>>, ctx: &Ctx) {
        self.area = area;
        self.set_len(rows.len());
        let num_w = rows.len().to_string().len().max(1) + 2; // "● 12"
        let mut cells: Vec<Vec<usize>> = vec![head.iter().map(|h| len(h)).collect()];
        cells.extend(rows.iter().map(|r| r.iter().map(|c| len(&c.content)).collect()));
        let room = (area.width as usize).saturating_sub(num_w + 3 + 1); // marker + number + scrollbar
        let w = widths(&cells, room);
        self.body = Rect { y: area.y + 2, height: area.height.saturating_sub(2), ..area };
        let h = self.body.height as usize;
        if self.sel < self.offset {
            self.offset = self.sel;
        } else if h > 0 && self.sel >= self.offset + h {
            self.offset = self.sel + 1 - h;
        }
        self.offset = self.offset.min(self.len.saturating_sub(h));
        // where each column starts, after the marker and the row number
        self.cols.clear();
        let mut x = num_w as u16 + 3;
        for cw in &w {
            self.cols.push(x);
            x += *cw as u16 + 2;
        }
        let line = |cells: &[Span<'static>]| -> Vec<Span<'static>> {
            let mut out = vec![];
            for (i, c) in cells.iter().enumerate().take(w.len()) {
                let t = clip(&c.content, w[i]);
                let pad = " ".repeat(w[i] - len(&t) + 2);
                out.push(Span::styled(t, c.style));
                out.push(Span::raw(pad));
            }
            out
        };
        let mut top = vec![sp(format!(" {:>num_w$}  ", "#"), st(FAINT))];
        top.extend(line(&head.iter().map(|h| sp(*h, st(DIM))).collect::<Vec<_>>()));
        let rule = sp("─".repeat(area.width.saturating_sub(1) as usize), st(LINE));
        let mut lines = vec![Line::from(top), Line::from(rule)];
        for (i, r) in rows.iter().enumerate().skip(self.offset).take(h) {
            let y = self.body.y + (i - self.offset) as u16;
            let row_rect = Rect::new(area.x, y, area.width, 1);
            let selected = i == self.sel;
            let (marker, bgc) = if selected {
                (sp("▌", st(SIDE)), Some(SEL))
            } else if ctx.hovered(row_rect) {
                (sp(" ", st(FG)), Some(HOVER))
            } else {
                (sp(" ", st(FG)), None)
            };
            // the row number is there for typing it: quiet, except on the selected row
            let number_style = if selected { bold(SIDE) } else { st(FAINT) };
            let number = (i + 1).to_string();
            let mut spans = vec![marker];
            if self.mark == Some(i) {
                let pad = num_w.saturating_sub(number.len() + 2);
                spans.push(sp(format!("{}● ", " ".repeat(pad)), st(SIDE)));
                spans.push(sp(number, number_style));
            } else {
                spans.push(sp(format!("{number:>num_w$}"), number_style));
            }
            spans.push(sp("  ", st(FG)));
            spans.extend(line(r));
            let mut l = Line::from(spans);
            if let Some(c) = bgc {
                l = l.style(bg(c));
                let used: usize = l.spans.iter().map(|s| len(&s.content)).sum();
                l.spans.push(Span::raw(" ".repeat((area.width as usize).saturating_sub(used + 1))));
            }
            lines.push(l);
        }
        f.render_widget(Paragraph::new(lines), area);
        if self.len > h && h > 0 {
            let mut s = ScrollbarState::new(self.len.saturating_sub(h)).position(self.offset);
            let sb = Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None)
                .track_symbol(Some("│"))
                .thumb_symbol("┃")
                .track_style(st(LINE))
                .thumb_style(st(DIM));
            f.render_stateful_widget(sb, self.body, &mut s);
        }
    }
}

/// Column widths: content width, shrinking the widest column (never below 8) until all fit
/// in `room` (2 columns of gap each). `cells[row][col]` = text lengths, header first.
pub fn widths(cells: &[Vec<usize>], room: usize) -> Vec<usize> {
    let n = cells.iter().map(Vec::len).max().unwrap_or(0);
    let mut w = vec![0; n];
    for r in cells {
        for (i, c) in r.iter().enumerate() {
            w[i] = w[i].max(*c);
        }
    }
    let room = room.saturating_sub(2 * n);
    while w.iter().sum::<usize>() > room {
        let i = (0..n).rev().max_by_key(|&i| w[i]).unwrap_or(0);
        if w[i] <= 8 {
            break;
        }
        w[i] -= 1;
    }
    w
}

// -- TextInput -------------------------------------------------------------------------

pub enum Edit {
    Changed,
    Submit,
    Cancel,
}

/// One-line editor: arrows, Home/End, Backspace/Delete, Ctrl+U clears, Enter/Esc.
#[derive(Default)]
pub struct TextInput {
    pub value: String,
    cur: usize, // in chars
    area: Rect,
}

impl TextInput {
    pub fn new(v: &str) -> Self {
        Self { value: v.into(), cur: v.chars().count(), area: Rect::default() }
    }

    fn byte(&self, ci: usize) -> usize {
        self.value.char_indices().nth(ci).map_or(self.value.len(), |(b, _)| b)
    }

    pub fn on(&mut self, ev: &Ev) -> Option<Edit> {
        if let Some(p) = click(ev) {
            if !self.area.contains(p) {
                return None;
            }
            self.cur = (self.scroll() + (p.x - self.area.x) as usize).min(self.value.chars().count());
            return Some(Edit::Changed);
        }
        let Ev::Key(k) = ev else { return None };
        let n = self.value.chars().count();
        if k.modifiers.contains(KeyModifiers::CONTROL) {
            return match k.code {
                KeyCode::Char('u') => {
                    self.value.clear();
                    self.cur = 0;
                    Some(Edit::Changed)
                }
                _ => None,
            };
        }
        match k.code {
            KeyCode::Enter => return Some(Edit::Submit),
            KeyCode::Esc => return Some(Edit::Cancel),
            KeyCode::Char(c) => {
                let b = self.byte(self.cur);
                self.value.insert(b, c);
                self.cur += 1;
            }
            KeyCode::Backspace if self.cur > 0 => {
                self.cur -= 1;
                let b = self.byte(self.cur);
                self.value.remove(b);
            }
            KeyCode::Delete if self.cur < n => {
                let b = self.byte(self.cur);
                self.value.remove(b);
            }
            KeyCode::Left => self.cur = self.cur.saturating_sub(1),
            KeyCode::Right => self.cur = (self.cur + 1).min(n),
            KeyCode::Home => self.cur = 0,
            KeyCode::End => self.cur = n,
            KeyCode::Backspace | KeyCode::Delete => {}
            _ => return None,
        }
        Some(Edit::Changed)
    }

    /// First visible char so the cursor stays in view.
    fn scroll(&self) -> usize {
        let w = self.area.width.max(1) as usize;
        (self.cur + 1).saturating_sub(w)
    }

    pub fn draw(&mut self, f: &mut Frame, area: Rect, focused: bool) {
        self.area = area;
        let s = self.scroll();
        let text: String = self.value.chars().skip(s).take(area.width as usize).collect();
        let style = if focused { st(FG).bg(rgb(FIELD)) } else { st(DIM).bg(rgb(FIELD)) };
        f.render_widget(Paragraph::new(Line::from(sp(text, style))).style(bg(FIELD)), area);
        if focused {
            f.set_cursor_position(Position::new(area.x + (self.cur - s) as u16, area.y));
        }
    }
}

// -- Modal -----------------------------------------------------------------------------

/// Fade what's already drawn in `area`, so the dialog on top is the only thing in focus.
fn dim_behind(f: &mut Frame, area: Rect) {
    let buf = f.buffer_mut();
    for pos in area.positions() {
        let cell = &mut buf[pos];
        cell.set_fg(rgb(FAINT));
        cell.set_bg(rgb(BG));
    }
}

enum Kind {
    Confirm(bool), // focused button: true = yes
    Info,
    Prompt(TextInput),
    Choose(List, Vec<String>, Vec<Vec<Span<'static>>>),
}

/// Centred dialog over the current screen; opened with `ctx.ask(tag, modal)`, its
/// `Answer` reaches the screen's `on()` as `Ev::Answer(tag, answer)`.
pub struct Modal {
    title: String,
    color: u32,
    body: Vec<Line<'static>>,
    kind: Kind,
    zones: Zones<Answer>,
}

impl Modal {
    fn new(title: &str, color: u32, body: Vec<Line<'static>>, kind: Kind) -> Self {
        Self { title: title.into(), color, body, kind, zones: Zones::default() }
    }

    /// Sì/No question; Enter = the focused button (starts on `default`), Esc = No.
    pub fn confirm(title: &str, body: Vec<Line<'static>>, default: bool) -> Self {
        Self::new(title, WARN, body, Kind::Confirm(default))
    }

    pub fn ask(question: &str, default: bool) -> Self {
        Self::confirm("conferma", vec![Line::from(sp(question, st(FG)))], default)
    }

    pub fn info(title: &str, color: u32, body: Vec<Line<'static>>) -> Self {
        Self::new(title, color, body, Kind::Info)
    }

    pub fn prompt(title: &str, hint: &str, initial: &str) -> Self {
        let body = if hint.is_empty() { vec![] } else { vec![Line::from(sp(hint, st(DIM)))] };
        Self::new(title, SLOT, body, Kind::Prompt(TextInput::new(initial)))
    }

    pub fn choose(title: &str, head: &[&str], rows: Vec<Vec<Span<'static>>>, sel: usize) -> Self {
        let head = head.iter().map(|h| h.to_string()).collect();
        Self::new(title, SLOT, vec![], Kind::Choose(List::new(sel), head, rows))
    }

    /// Some = closed with that answer.
    pub fn on(&mut self, ev: &Ev) -> Option<Answer> {
        if let Some(p) = click(ev)
            && let Some(a) = self.zones.at(p)
        {
            return Some(match (a, &self.kind) {
                (Answer::Yes, Kind::Prompt(i)) => Answer::Text(i.value.clone()),
                (a, _) => a,
            });
        }
        let code = match ev {
            Ev::Key(k) => k.code,
            _ => KeyCode::Null,
        };
        match &mut self.kind {
            Kind::Confirm(yes) => match code {
                KeyCode::Char('s' | 'S' | 'y' | 'Y') => Some(Answer::Yes),
                KeyCode::Char('n' | 'N') | KeyCode::Esc => Some(Answer::No),
                KeyCode::Enter => Some(if *yes { Answer::Yes } else { Answer::No }),
                KeyCode::Left | KeyCode::Right | KeyCode::Tab | KeyCode::BackTab => {
                    *yes = !*yes;
                    None
                }
                _ => None,
            },
            Kind::Info => {
                matches!(code, KeyCode::Enter | KeyCode::Esc | KeyCode::Char(' ' | 'q')).then_some(Answer::Yes)
            }
            Kind::Prompt(input) => match input.on(ev) {
                Some(Edit::Submit) => Some(Answer::Text(input.value.trim().to_string())),
                Some(Edit::Cancel) => Some(Answer::Cancel),
                _ => None,
            },
            Kind::Choose(list, ..) => match list.on(ev) {
                Some(Hit::Open(i)) => Some(Answer::Pick(i)),
                _ => matches!(code, KeyCode::Esc | KeyCode::Char('q')).then_some(Answer::Cancel),
            },
        }
    }

    pub fn draw(&mut self, f: &mut Frame, area: Rect, ctx: &Ctx) {
        let body_w = self.body.iter().map(|l| l.width()).max().unwrap_or(0) as u16;
        let (extra_w, extra_h) = match &self.kind {
            Kind::Choose(_, head, rows) => {
                let mut cells: Vec<Vec<usize>> = vec![head.iter().map(|h| len(h)).collect()];
                cells.extend(rows.iter().map(|r| r.iter().map(|c| len(&c.content)).collect()));
                let w: usize = widths(&cells, 200).iter().map(|w| w + 2).sum();
                (w as u16 + 10, rows.len() as u16 + 3)
            }
            Kind::Prompt(_) => (60, 3),
            _ => (0, 2),
        };
        // at least 40 columns, but never wider than the screen (a narrow terminal wins)
        let w = (body_w.max(extra_w).max(len(&self.title) as u16 + 6) + 6).max(40).min(area.width.saturating_sub(4));
        let body_h = wrapped(&self.body, w.saturating_sub(6));
        let h = (body_h + extra_h + 4).min(area.height.saturating_sub(2));
        let r = centered(area, w, h);
        dim_behind(f, area);
        f.render_widget(Clear, r);
        let blk = lit(&self.title, self.color).style(bg(SURFACE)).padding(Padding::new(2, 2, 1, 1));
        let inner = blk.inner(r);
        f.render_widget(blk, r);
        f.render_widget(Paragraph::new(self.body.clone()).wrap(Wrap { trim: false }), inner);
        let below = Rect { y: inner.y + body_h, height: inner.height.saturating_sub(body_h), ..inner };
        let last = Rect { y: inner.bottom().saturating_sub(1), height: 1, ..inner };
        self.zones.clear();
        match &mut self.kind {
            Kind::Confirm(yes) => {
                let b = buttons(f, last, &[("Sì (s)", *yes), ("No (n)", !*yes)], self.color, ctx);
                self.zones.add(b[0], Answer::Yes);
                self.zones.add(b[1], Answer::No);
            }
            Kind::Info => {
                let b = buttons(f, last, &[("OK", true)], self.color, ctx);
                self.zones.add(b[0], Answer::Yes);
            }
            Kind::Prompt(input) => {
                input.draw(f, Rect { height: 1, y: below.y, ..below }, true);
                let b = buttons(f, last, &[("OK (invio)", true), ("Annulla (esc)", false)], self.color, ctx);
                self.zones.add(b[0], Answer::Yes);
                self.zones.add(b[1], Answer::Cancel);
            }
            Kind::Choose(list, head, rows) => {
                let heads: Vec<&str> = head.iter().map(String::as_str).collect();
                list.draw(f, Rect { height: below.height.saturating_sub(1), ..below }, &heads, rows.clone(), ctx);
                let b = buttons(f, last, &[("Annulla (esc)", false)], self.color, ctx);
                self.zones.add(b[0], Answer::Cancel);
            }
        }
    }
}
