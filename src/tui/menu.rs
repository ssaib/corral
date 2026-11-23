//! Main menu: the hardware strip on top, then the actions as cards, grouped by what they're about
//! (models, servers, the system). Arrows, Tab, the letter of a card, or a click open them.

use std::path::PathBuf;

use ratatui::Frame;
use ratatui::crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Padding, Paragraph, Wrap};

use super::models::Models;
use super::widgets::*;
use super::{Act, Answer, Ctx, Ev, Key, Screen, agent, hf, hosts, installs, logs, news, router};
use crate::installs as inst;
use crate::ui::*;
use crate::{models, tune};

pub struct Item {
    pub key: char,
    icon: &'static str,
    title: &'static str,
    desc: &'static str,
}

struct Group {
    name: &'static str,
    items: &'static [Item],
}

const fn item(key: char, icon: &'static str, title: &'static str, desc: &'static str) -> Item {
    Item { key, icon, title, desc }
}

const GROUPS: [Group; 3] = [
    Group {
        name: "modelli",
        items: &[
            item('a', "▶", "avvia server", "scegli un modello locale, configuralo e avvialo"),
            item('o', "◈", "ottimizza", "lite · normal · max: i parametri giusti per questo computer"),
            item('c', "⌕", "cerca su HF", "trova un GGUF su Hugging Face per nome e scaricalo"),
            item('s', "★", "suggeriti", "i modelli più scaricati che girano sul tuo hardware"),
        ],
    },
    Group {
        name: "server",
        items: &[
            item('v', "◉", "server attivi", "dashboard dei server in esecuzione: ferma, lascia attivo"),
            item('m', "◫", "multi-modello", "più modelli su un solo endpoint (router di llama.cpp)"),
            item('g', "✦", "agent e flussi", "ricette per agent, multi-agent e RAG, config per i client"),
            item('h', "⇄", "host remoti", "Corral su altri computer via SSH, collegati con un tunnel"),
        ],
    },
    Group {
        name: "sistema",
        items: &[
            item('i', "↓", "installazioni", "llama.cpp: release, build da sorgente, cartelle importate"),
            item('n', "✉", "news", "build di llama.cpp, nuovi modelli GGUF, notizie IA, Corral"),
            item('l', "≡", "log", "output dei server e delle build recenti"),
        ],
    },
];

/// Every card in reading order: the selection is an index into this.
pub fn items() -> impl Iterator<Item = &'static Item> {
    GROUPS.iter().flat_map(|g| g.items.iter())
}

fn item_count() -> usize {
    items().count()
}

/// The numbers the cards show, read when the menu comes back on top (reading them every frame
/// would walk the model folders 12 times a second).
struct Counts {
    models: usize,
    installs: usize,
    hosts: usize,
    logs: usize,
    news: usize, // items newer than the last visit to the news
    hf_token: bool,
}

impl Counts {
    fn read(ctx: &Ctx) -> Self {
        Self {
            models: models::scan(&ctx.state).len(),
            installs: inst::all_installs().len(),
            hosts: crate::remote::config_hosts().len(),
            logs: log_files().len(),
            news: crate::news::unseen(ctx.state.news_seen),
            hf_token: crate::hub::token().is_some(),
        }
    }
}

/// "1 modello" / "3 modelli".
fn count(n: usize, one: &str, many: &str) -> String {
    if n == 1 { format!("1 {one}") } else { format!("{n} {many}") }
}

pub struct Menu {
    pub sel: usize,
    cards: Vec<Rect>, // where each card was drawn (empty = not on screen)
    counts: Option<Counts>,
    picks: Vec<PathBuf>, // models offered by the "ottimizza" chooser
    first_run_checked: bool,
    after_splash: bool,
}

impl Menu {
    pub fn new() -> Self {
        Self {
            sel: 0,
            cards: vec![Rect::default(); item_count()],
            counts: None,
            picks: vec![],
            first_run_checked: false,
            after_splash: true,
        }
    }

    fn open(&mut self, i: usize, ctx: &mut Ctx) -> Act {
        self.sel = i;
        let Some(item) = items().nth(i) else { return Act::None };
        let needs_install = matches!(item.key, 'a' | 'm' | 'o');
        if needs_install && ctx.hw.inst.is_none() {
            ctx.toast("serve prima un'installazione di llama.cpp", WARN);
            return Act::Push(Box::new(installs::Installs::new()));
        }
        match item.key {
            'a' => Act::Push(Box::new(Models::new(ctx))),
            'o' => {
                self.choose_model(ctx);
                Act::None
            }
            'm' => Act::Push(Box::new(router::Profiles::new())),
            'g' => Act::Push(Box::new(agent::Agents::new())),
            'c' => Act::Push(Box::new(hf::Search::new())),
            's' => Act::Push(Box::new(hf::Search::suggest(ctx))),
            'n' => {
                self.counts = None; // the news count goes back to 0 once seen
                Act::Push(Box::new(news::News::new(ctx)))
            }
            'v' => Act::Push(Box::new(crate::dashboard::Servers::new())),
            'h' => Act::Push(Box::new(hosts::Hosts::new())),
            'l' => Act::Push(Box::new(logs::Logs::new())),
            _ => Act::Push(Box::new(installs::Installs::new())),
        }
    }

    /// "ottimizza" first asks which model (the last one used is preselected).
    fn choose_model(&mut self, ctx: &mut Ctx) {
        let found = models::scan(&ctx.state);
        if found.is_empty() {
            return ctx.toast("nessun modello locale: cercane uno su Hugging Face (c)", WARN);
        }
        let rows = found
            .iter()
            .map(|m| vec![cell(fname(&ps(&m.path)).to_string()), sp(size(m.size), st(DIM)), sp(m.source, st(DIM))])
            .collect();
        let last = ctx.state.last_model.clone();
        let sel = found.iter().position(|m| Some(ps(&m.path)) == last).unwrap_or(0);
        self.picks = found.into_iter().map(|m| m.path).collect();
        let title = "ottimizza · quale modello?";
        ctx.ask("tune", Modal::choose(title, &["modello", "dimensione", "fonte"], rows, sel));
    }

    /// What a card says under its description: a live count, or what's missing.
    fn status(&self, key: char, ctx: &Ctx) -> (String, u32) {
        let Some(n) = &self.counts else { return (format!("{} …", spinner()), DIM) };
        let no_install = ctx.hw.step > 0 && ctx.hw.inst.is_none();
        // short: a card is about 25 columns wide
        match key {
            'a' | 'm' | 'o' if no_install => ("serve llama.cpp (i)".into(), WARN),
            'a' | 'o' if n.models == 0 => ("nessun modello locale".into(), WARN),
            'a' => (format!("● {}", count(n.models, "modello", "modelli")), OK),
            'o' => ("stima + benchmark".into(), CYAN),
            'c' if n.hf_token => ("✓ token HF".into(), OK),
            'c' => ("senza token: pubblici".into(), DIM),
            's' if !ctx.hw.ready() => (format!("{} rilevo la GPU…", spinner()), DIM),
            's' if ctx.hw.budget() == 0 => (format!("solo CPU · {} RAM", size(ctx.hw.ram)), CYAN),
            's' => (format!("per {} di VRAM", size(ctx.hw.budget())), CYAN),
            'v' if ctx.servers.is_empty() => ("nessuno attivo".into(), DIM),
            'v' => (format!("● {} attivi", ctx.servers.len()), OK),
            'm' => (count(ctx.state.routers.len(), "profilo", "profili"), DIM),
            'g' => ("OpenCode · Qwen · Python".into(), DIM),
            'h' => (count(n.hosts, "host ssh", "host ssh"), DIM),
            'i' if n.installs == 0 => ("nessuna: installala".into(), WARN),
            'i' => (count(n.installs, "installata", "installate"), DIM),
            'n' if n.news > 0 => (format!("● {} da leggere", n.news), SRV),
            'n' => ("tutto letto".into(), DIM),
            'l' => (count(n.logs, "file", "file"), DIM),
            _ => (String::new(), DIM),
        }
    }

    /// `wide`: room for the GPUs' full names.
    fn hardware(&self, ctx: &Ctx, wide: bool) -> Vec<Line<'static>> {
        let hw = &ctx.hw;
        let wait = || sp(format!("{} rilevamento…", spinner()), st(WARN));
        let label = |k: &str| sp(format!("{k:<11}"), st(DIM));
        let mut lines = vec![];
        if hw.step <= 3 && hw.devs.is_empty() {
            lines.push(Line::from(vec![label("GPU"), wait()]));
        } else if hw.devs.is_empty() {
            lines.push(Line::from(vec![label("GPU"), sp("nessuna (solo CPU)", st(DIM))]));
        }
        for d in &hw.devs {
            let used = d.total.saturating_sub(d.free) as f64 / d.total.max(1) as f64;
            let mut spans = vec![label("GPU"), sp(format!("{} ", d.name), bold(FG))];
            if wide {
                spans.push(sp(format!("{}  ", d.desc), st(FG)));
            }
            spans.extend(bar(used, 16, SLOT));
            spans.push(sp(format!(" {} / {} MiB liberi", d.free, d.total), st(DIM)));
            lines.push(Line::from(spans));
        }
        let mut last: Vec<Span<'static>> = vec![label("llama.cpp")];
        match &hw.inst {
            Some(i) => {
                last.push(sp(i.name.clone(), bold(FG)));
                last.push(sp(format!(" · {} · ", i.backend), st(DIM)));
                last.push(if hw.step > 2 || !hw.version.is_empty() { sp(hw.version.clone(), st(FG)) } else { wait() });
            }
            None if hw.step == 0 => last.push(wait()),
            None => last.push(sp("nessuna installazione (i)", st(WARN))),
        }
        last.push(sp("    RAM ", st(DIM)));
        last.push(sp(format!("{} disponibili", size(hw.ram)), st(FG)));
        if let Some(l) = &ctx.state.last_model {
            last.push(sp("    ultimo ", st(DIM)));
            last.push(sp(fname(l).to_string(), st(FG)));
        }
        lines.push(Line::from(last));
        lines
    }

    fn draw_card(&self, f: &mut Frame, r: Rect, item: &Item, selected: bool, ctx: &Ctx) {
        let narrow = r.width < 26; // no room for the icon in the title
        let hovered = ctx.hovered(r);
        let border = if selected {
            SIDE
        } else if hovered {
            DIM
        } else {
            LINE
        };
        let key_style = if selected { bold(BG).bg(rgb(SIDE)) } else { bold(SIDE).bg(rgb(SEL)) };
        let title_style = if selected { bold(SIDE) } else { bold(FG) };
        let mut title = vec![sp(format!(" {} ", item.key), key_style), sp(" ", title_style)];
        if !narrow {
            title.push(sp(format!("{} ", item.icon), st(if selected { SIDE } else { DIM })));
        }
        title.push(sp(format!("{} ", item.title), title_style));
        let mut block = Block::bordered()
            .border_type(if selected { BorderType::Thick } else { BorderType::Rounded })
            .border_style(st(border))
            .padding(Padding::horizontal(1))
            .title(Line::from(title));
        if selected {
            block = block.style(bg(SEL));
        } else if hovered {
            block = block.style(bg(HOVER));
        }
        let inner = block.inner(r);
        f.render_widget(block, r);
        if inner.height == 0 {
            return;
        }
        // the description on the first rows, the status on the last one
        let desc_rows = inner.height - 1;
        if desc_rows > 0 {
            let desc = Paragraph::new(item.desc).style(st(if selected { FG } else { DIM })).wrap(Wrap { trim: true });
            f.render_widget(desc, Rect { height: desc_rows, ..inner });
        }
        let (status, status_color) = self.status(item.key, ctx);
        let status_row = Rect { y: inner.bottom() - 1, height: 1, ..inner };
        f.render_widget(Paragraph::new(sp(status, st(status_color))), status_row);
    }

    /// The card above or below the selected one: the nearest row that way, then the nearest column.
    fn vertical(&self, down: bool) -> Option<usize> {
        let cur = self.cards.get(self.sel)?;
        let mut best: Option<(u16, u16, usize)> = None; // (rows away, columns away, card)
        for (i, r) in self.cards.iter().enumerate() {
            let that_way = if down { r.y > cur.y } else { r.y < cur.y };
            if r.is_empty() || !that_way {
                continue;
            }
            let dy = r.y.abs_diff(cur.y);
            let dx = r.x.abs_diff(cur.x);
            let closer = best.is_none_or(|(by, bx, _)| (dy, dx) < (by, bx));
            if closer {
                best = Some((dy, dx, i));
            }
        }
        best.map(|(_, _, i)| i)
    }
}

impl Screen for Menu {
    fn title(&self) -> String {
        "menu".into()
    }

    fn keys(&self, _ctx: &Ctx) -> Vec<Key> {
        vec![
            (KeyCode::Enter, "invio", "apri"),
            (KeyCode::Char('a'), "a", "avvia"),
            (KeyCode::Char('o'), "o", "ottimizza"),
            (KeyCode::Char('c'), "c", "cerca"),
            (KeyCode::Char('v'), "v", "server attivi"),
            (KeyCode::Char('m'), "m", "multi"),
            (KeyCode::Char('i'), "i", "installazioni"),
            (KeyCode::Char('q'), "q", "esci"),
        ]
    }

    fn draw(&mut self, f: &mut Frame, area: Rect, ctx: &Ctx) {
        let cols = (area.width / 19).clamp(1, 4) as usize; // 4 from 80 columns up
        let card_rows: u16 = GROUPS.iter().map(|g| g.items.len().div_ceil(cols) as u16).sum();
        let titles = GROUPS.len() as u16;
        let hw_lines = self.hardware(ctx, area.width >= 100);
        let hw_height = hw_lines.len() as u16 + 2;
        // cards as tall as the longest description needs (borders + description + status); when
        // the screen is too short for that, the status alone (half a description reads badly)
        let card_w = area.width.saturating_sub(cols as u16 - 1) / cols as u16;
        let text_w = card_w.saturating_sub(4) as usize; // borders and padding
        let desc_rows = items().map(|it| crate::help::wrap(it.desc, text_w).len()).max().unwrap_or(1);
        let mut card_h = desc_rows as u16 + 3;
        if hw_height + titles + card_rows * card_h > area.height {
            card_h = 3;
        }
        let show_hw = hw_height + titles + card_rows * card_h <= area.height;
        let spare = area.height.saturating_sub(hw_height + titles + card_rows * card_h);
        let gap = u16::from(spare > titles); // a blank row before each group, when there's room
        let mut y = area.y;
        if show_hw {
            let blk = panel("hardware", FG);
            let r = Rect { y, height: hw_height, ..area };
            f.render_widget(Paragraph::new(hw_lines).block(blk), r);
            y += hw_height;
        }
        self.cards = vec![Rect::default(); item_count()];
        let mut index = 0;
        for g in &GROUPS {
            y += gap;
            if y >= area.bottom() {
                break;
            }
            let rule = "─".repeat((area.width as usize).saturating_sub(len(g.name) + 3));
            let head = Line::from(vec![sp(format!(" {} ", g.name), bold(DIM)), sp(rule, st(LINE))]);
            f.render_widget(Paragraph::new(head), Rect { y, height: 1, ..area });
            y += 1;
            for row in g.items.chunks(cols) {
                if y + card_h > area.bottom() {
                    index += row.len(); // no room: these cards aren't on screen
                    continue;
                }
                let line = Rect { y, height: card_h, ..area };
                let slots = Layout::horizontal(vec![Constraint::Fill(1); cols]).spacing(1).split(line);
                for (it, slot) in row.iter().zip(slots.iter()) {
                    self.draw_card(f, *slot, it, index == self.sel, ctx);
                    self.cards[index] = *slot;
                    index += 1;
                }
                y += card_h;
            }
        }
    }

    fn on(&mut self, ev: &Ev, ctx: &mut Ctx) -> Act {
        match ev {
            Ev::Answer("import", a) => return self.answer_import(a, ctx),
            Ev::Answer("tune", Answer::Pick(i)) => {
                return match self.picks.get(*i) {
                    Some(model) => Act::Push(Box::new(tune::Tune::new(ctx, model))),
                    None => Act::None,
                };
            }
            Ev::Answer(..) => return Act::None,
            _ => {}
        }
        if let Some(p) = click(ev)
            && let Some(i) = self.cards.iter().position(|r| r.contains(p))
        {
            return self.open(i, ctx);
        }
        let n = item_count();
        let moved = match key(ev) {
            Some(KeyCode::Right | KeyCode::Tab) => Some((self.sel + 1) % n),
            Some(KeyCode::Left | KeyCode::BackTab) => Some((self.sel + n - 1) % n),
            Some(KeyCode::Down) => self.vertical(true),
            Some(KeyCode::Up) => self.vertical(false),
            Some(KeyCode::Home) => Some(0),
            Some(KeyCode::End) => Some(n - 1),
            Some(KeyCode::Enter) => return self.open(self.sel, ctx),
            Some(KeyCode::Char(c)) => match items().position(|it| it.key == c) {
                Some(i) => return self.open(i, ctx),
                None => return Act::Pass,
            },
            _ => return Act::Pass,
        };
        if let Some(i) = moved {
            self.sel = i;
        }
        Act::None
    }

    fn tick(&mut self, ctx: &mut Ctx) -> Act {
        if self.counts.is_none() {
            self.counts = Some(Counts::read(ctx));
        }
        // first run: offer the llama.cpp checkout next to the data dir, else the installs screen
        if self.first_run_checked || ctx.hw.step == 0 {
            return Act::None;
        }
        self.first_run_checked = true;
        if ctx.hw.inst.is_some() {
            return Act::None;
        }
        let cand = inst::external_candidate();
        if inst::find_server(&cand).is_some() {
            ctx.ask("import", Modal::ask(&format!("trovato llama.cpp in {}, lo importo?", cand.display()), true));
            return Act::None;
        }
        ctx.toast("nessuna installazione di llama.cpp: scegline una", WARN);
        Act::Push(Box::new(installs::Installs::new()))
    }

    fn resume(&mut self, ctx: &mut Ctx) {
        self.counts = None; // models, installs, logs may have changed on the screen that closed
        if std::mem::take(&mut self.after_splash) {
            return; // the splash just did the detection
        }
        ctx.reload_hw(); // VRAM in use changes after a server run, the install after the installs screen
    }
}

impl Menu {
    fn answer_import(&mut self, a: &Answer, ctx: &mut Ctx) -> Act {
        if *a != Answer::Yes {
            return Act::Push(Box::new(installs::Installs::new()));
        }
        match inst::import_dir(&inst::external_candidate()) {
            Ok(i) => {
                ctx.toast(format!("✓ importata {}", i.name), OK);
                ctx.state.install = Some(i.name);
                ctx.save();
                self.counts = None;
                ctx.reload_hw();
            }
            Err(e) => ctx.error(e),
        }
        Act::None
    }
}
