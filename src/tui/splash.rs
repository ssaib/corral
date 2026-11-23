//! Startup screen: the Corral logo while the real initialisation (Ctx::reload_hw) runs.
//! Leaves when it is done (at least MIN on screen, so it doesn't flash), or on any key/click.

use std::time::{Duration, Instant};

use ratatui::Frame;
use ratatui::crossterm::event::{KeyCode, KeyModifiers};
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::widgets::*;
use super::{Act, Ctx, Ev, HW_STEPS, Screen};
use crate::ui::*;

const MIN: Duration = Duration::from_millis(600);
const MIN_YES: Duration = Duration::from_millis(250);

// ANSI Shadow letters, 6 rows each
const C: [&str; 6] = [" ██████╗", "██╔════╝", "██║     ", "██║     ", "╚██████╗", " ╚═════╝"];
const O: [&str; 6] = [" ██████╗ ", "██╔═══██╗", "██║   ██║", "██║   ██║", "╚██████╔╝", " ╚═════╝ "];
const R: [&str; 6] = ["██████╗ ", "██╔══██╗", "██████╔╝", "██╔══██╗", "██║  ██║", "╚═╝  ╚═╝"];
const A: [&str; 6] = [" █████╗ ", "██╔══██╗", "███████║", "██╔══██║", "██║  ██║", "╚═╝  ╚═╝"];
const L: [&str; 6] = ["██╗     ", "██║     ", "██║     ", "██║     ", "███████╗", "╚══════╝"];

// a llama in profile, looking left, standing in its pen
const LLAMA: [&str; 7] = [
    "   ▟▙▟▙      ",
    " ▄███▛▘      ",
    "  ▝██▌       ",
    "   ██▌       ",
    "   ███▄▄▄▄▄▄ ",
    "   █████████▌",
    "   ▐▌▐▌  ▐▌▐▌",
];

pub struct Splash {
    t0: Instant,
    yes: bool,
}

impl Splash {
    pub fn new(yes: bool) -> Self {
        Self { t0: Instant::now(), yes }
    }

    fn finish(&self, ctx: &mut Ctx) -> Act {
        if !self.yes {
            return Act::Pop;
        }
        match super::config::launch_last(ctx) {
            Ok(dash) => Act::Replace(dash),
            Err(e) => {
                ctx.error(e);
                Act::Pop
            }
        }
    }
}

fn title_rows() -> Vec<String> {
    (0..6).map(|r| [C[r], O[r], R[r], R[r], A[r], L[r]].join("")).collect()
}

/// Shimmer: a light band sweeping across the letters, otherwise the sidebar violet.
fn shimmer(row: &str, t: f64, y: usize) -> Line<'static> {
    let n = len(row) as f64;
    let band = (t * 1.6 % 1.6) * (n + 20.0) - 10.0;
    Line::from(
        row.chars()
            .enumerate()
            .map(|(i, ch)| {
                let d = (i as f64 + y as f64 * 1.5 - band).abs();
                let c = if ch == '█' {
                    if d < 3.0 { FG } else { SIDE }
                } else if d < 3.0 {
                    SIDE
                } else {
                    0x5b4a7a
                };
                Span::styled(ch.to_string(), Style::new().fg(rgb(c)))
            })
            .collect::<Vec<_>>(),
    )
}

/// Fence rails and posts, drawn left to right as `frac` grows.
fn fence(w: usize, frac: f64) -> [Line<'static>; 3] {
    let n = ((w as f64) * frac.clamp(0.0, 1.0)) as usize;
    let post = |i: usize| i % 8 == 3;
    let rail: String = (0..n).map(|i| if post(i) { '╋' } else { '━' }).collect();
    let mid: String = (0..n).map(|i| if post(i) { '┃' } else { ' ' }).collect();
    [rail.clone(), mid, rail].map(|s| Line::from(sp(s, st(WARN))))
}

impl Screen for Splash {
    fn title(&self) -> String {
        "avvio".into()
    }

    fn chrome(&self) -> bool {
        false
    }

    fn keys(&self, _ctx: &Ctx) -> Vec<super::Key> {
        vec![(KeyCode::Enter, "invio", "salta")]
    }

    fn draw(&mut self, f: &mut Frame, area: Rect, ctx: &Ctx) {
        let t = self.t0.elapsed().as_secs_f64();
        let rows = title_rows();
        let logo_w = (len(&rows[0]) + 4 + len(LLAMA[0])) as u16;
        let h = 7 + 3 + 2 + HW_STEPS.len() as u16 + 4;
        let box_ = centered(area, logo_w.max(56) + 4, h);
        // llama + title side by side, title vertically centred on the llama
        let mut lines: Vec<Line> = vec![];
        for (y, l) in LLAMA.iter().enumerate() {
            let mut spans = vec![sp(*l, st(if y < 2 { 0xf0e0c4 } else { WOOL })), sp("    ", st(FG))];
            if let Some(r) = y.checked_sub(1).and_then(|r| rows.get(r)) {
                spans.extend(shimmer(r, t, y).spans);
            }
            lines.push(Line::from(spans));
        }
        let fw = box_.width as usize;
        lines.extend(fence(fw, t / 0.5));
        lines.push(Line::from(vec![
            sp("il recinto dei tuoi lama", st(DIM)),
            sp("  ·  llama.cpp + GGUF da Hugging Face", st(DIM)),
        ]));
        lines.push(Line::default());
        // the real init steps, in the order the background task runs them
        let hw = &ctx.hw;
        lines.push(Line::from(vec![
            sp("✓ ", st(OK)),
            sp(format!("{:<22}", "cartella dati"), st(FG)),
            sp(short_home(), st(DIM)),
        ]));
        for (i, label) in HW_STEPS.iter().enumerate() {
            let value = match i {
                0 => hw.inst.as_ref().map_or("nessuna".into(), |x| format!("{} · {}", x.name, x.backend)),
                1 => size(hw.ram),
                2 => hw.version.clone(),
                _ if hw.devs.is_empty() => "nessuna GPU".into(),
                _ => hw.devs.iter().map(|d| format!("{} {} MiB", d.name, d.total)).collect::<Vec<_>>().join(" · "),
            };
            let (mark, color, value) = if i < hw.step {
                ("✓ ", OK, value)
            } else if i == hw.step {
                ("", WARN, String::new())
            } else {
                ("○ ", DIM, String::new())
            };
            let mark = if mark.is_empty() { format!("{} ", spinner()) } else { mark.into() };
            lines.push(Line::from(vec![
                sp(mark, st(color)),
                sp(format!("{label:<22}"), st(if i <= hw.step { FG } else { DIM })),
                sp(clip(&value, fw.saturating_sub(26)), st(DIM)),
            ]));
        }
        lines.push(Line::default());
        let frac = (hw.step + 1) as f64 / (HW_STEPS.len() + 1) as f64;
        let mut b = bar(frac, fw.saturating_sub(8).min(48), LOAD).to_vec();
        b.push(sp(format!(" {:>3.0}%", frac * 100.0), st(FG)));
        lines.push(Line::from(b));
        lines.push(Line::from(sp("un tasto o un click per saltare", st(DIM))));
        f.render_widget(Paragraph::new(lines), box_);
    }

    fn on(&mut self, ev: &Ev, ctx: &mut Ctx) -> Act {
        match ev {
            Ev::Key(k) if k.modifiers.contains(KeyModifiers::CONTROL) => Act::Pass, // Ctrl+C quits
            Ev::Key(_) => self.finish(ctx),
            Ev::Mouse(_) if click(ev).is_some() => self.finish(ctx),
            _ => Act::None,
        }
    }

    fn tick(&mut self, ctx: &mut Ctx) -> Act {
        // -y only needs the install and its version: the dashboard finds the GPUs itself
        let (done, min) = if self.yes { (ctx.hw.step >= 3, MIN_YES) } else { (ctx.hw.ready(), MIN) };
        if done && self.t0.elapsed() >= min { self.finish(ctx) } else { Act::None }
    }
}

fn short_home() -> String {
    let home = ps(&home());
    let base = ps(&BASE);
    match base.strip_prefix(&home) {
        Some(rest) if !home.is_empty() => format!("~{rest}"),
        _ => base,
    }
}
