//! News screen: llama.cpp builds, new GGUF models, AI news and Corral's changelog in one list,
//! newest first, with a source filter, the selected item below and an action for each kind
//! (install the build, download the model, open the link).

use ratatui::Frame;
use ratatui::crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::Line;
use ratatui::widgets::{Paragraph, Wrap};

use super::widgets::*;
use super::{Act, Answer, Ctx, Ev, Key, Screen, Task, hf, installs};
use crate::hub;
use crate::installs as inst;
use crate::news::{self, Item, SOURCES};
use crate::ui::*;

pub struct News {
    data: news::News,
    corral: Vec<Item>,
    task: Option<Task<news::News>>,
    filter: usize, // 0 = all, i = SOURCES[i - 1]
    tabs: Zones<usize>,
    list: List,
    seen: u64, // last visit before this one: newer items get a ●
    feed: usize,
}

impl News {
    pub fn new(ctx: &mut Ctx) -> Self {
        let seen = ctx.state.news_seen.unwrap_or(0);
        ctx.state.news_seen = Some(unix_now());
        ctx.save();
        let data = news::cached().unwrap_or_default();
        let mut s = Self {
            data,
            corral: news::corral(),
            task: None,
            filter: 0,
            tabs: Zones::default(),
            list: List::new(0),
            seen,
            feed: 0,
        };
        if news::stale(&s.data) {
            s.refresh(ctx);
        }
        s
    }

    fn feeds(ctx: &Ctx) -> Vec<String> {
        ctx.state.feeds.clone().unwrap_or_else(|| news::DEFAULT_FEEDS.iter().map(|s| s.to_string()).collect())
    }

    fn refresh(&mut self, ctx: &Ctx) {
        if self.task.is_none() {
            let feeds = Self::feeds(ctx);
            self.task = Some(Task::once(move || news::fetch(&feeds)));
        }
    }

    /// Too big for this machine (once the hardware is known).
    fn too_big(i: &Item, ctx: &Ctx) -> bool {
        i.need > 0 && ctx.hw.ram > 0 && hub::fit_label(i.need, ctx.hw.budget(), ctx.hw.ram).0 == "troppo grande"
    }

    fn items(&self, ctx: &Ctx, filter: usize) -> Vec<&Item> {
        let mut v: Vec<&Item> = self
            .data
            .items
            .iter()
            .chain(&self.corral)
            .filter(|i| filter == 0 || SOURCES[filter - 1].0 == i.src)
            .filter(|i| !Self::too_big(i, ctx))
            .collect();
        v.sort_by_key(|a| std::cmp::Reverse(a.date));
        v
    }

    fn selected<'a>(&'a self, ctx: &Ctx) -> Option<&'a Item> {
        self.items(ctx, self.filter).get(self.list.sel).copied()
    }

    /// What an item has to say besides its title (installed build, fits the GPU, …).
    fn extra(i: &Item, ctx: &Ctx) -> ratatui::text::Span<'static> {
        match i.src.as_str() {
            "llama" => {
                let mine = ctx.hw.inst.as_ref().and_then(|x| x.git_ref.as_deref()).and_then(inst::tag_no);
                match (inst::tag_no(&i.tag), mine) {
                    (None, _) => sp("stabile", bold(OK)),
                    (Some(t), Some(m)) if t == m => sp("● installata", bold(OK)),
                    (Some(t), Some(m)) if t > m => sp("più nuova", st(WARN)),
                    _ => sp("", st(DIM)),
                }
            }
            "models" if i.need > 0 => {
                let (label, color) = hub::fit_label(i.need, ctx.hw.budget(), ctx.hw.ram);
                sp(format!("{label} ~{}", size(i.need)), st(color))
            }
            "corral" if i.tag == env!("CARGO_PKG_VERSION") => sp("questa versione", bold(OK)),
            _ => sp("", st(DIM)),
        }
    }

    fn install(i: &Item, ctx: &mut Ctx) -> Act {
        if inst::tag_no(&i.tag).is_none() {
            ctx.toast("una release stabile non ha binari suoi: scegli una build bNNNN", WARN);
            return Act::Push(Box::new(installs::ReleaseScreen::new()));
        }
        let variant = ctx.hw.inst.as_ref().filter(|x| x.kind == "release").and_then(inst::release_variant);
        match variant {
            Some(v) => Act::Push(Box::new(installs::ReleaseScreen::upgrade(i.tag.clone(), v))),
            None => Act::Push(Box::new(installs::ReleaseScreen::new())),
        }
    }

    fn manage_feeds(&self, ctx: &mut Ctx) {
        let mut rows: Vec<_> =
            Self::feeds(ctx).into_iter().map(|u| vec![cell(u), sp("invio = togli", st(DIM))]).collect();
        rows.push(vec![sp("+ aggiungi un feed RSS/Atom…", st(CYAN)), cell("")]);
        rows.push(vec![sp("ripristina i feed predefiniti", st(DIM)), cell("")]);
        ctx.ask("feeds", Modal::choose("feed delle notizie IA", &["indirizzo", ""], rows, 0));
    }

    fn answer(&mut self, tag: &str, a: &Answer, ctx: &mut Ctx) {
        let mut feeds = Self::feeds(ctx);
        match (tag, a) {
            ("feeds", Answer::Pick(i)) if *i < feeds.len() => {
                self.feed = *i;
                ctx.ask("rmfeed", Modal::ask(&format!("tolgo {}?", feeds[*i]), false));
            }
            ("feeds", Answer::Pick(i)) if *i == feeds.len() => ctx.ask(
                "addfeed",
                Modal::prompt("nuovo feed", "indirizzo RSS o Atom, es. https://…/feed.xml", "https://"),
            ),
            ("feeds", Answer::Pick(_)) => {
                ctx.state.feeds = None;
                ctx.save();
                self.refresh(ctx);
            }
            ("rmfeed", Answer::Yes) if self.feed < feeds.len() => {
                feeds.remove(self.feed);
                ctx.state.feeds = Some(feeds);
                ctx.save();
                self.refresh(ctx);
            }
            ("addfeed", Answer::Text(u)) if u.starts_with("http://") || u.starts_with("https://") => {
                if !feeds.contains(u) {
                    feeds.push(u.clone());
                }
                ctx.state.feeds = Some(feeds);
                ctx.save();
                self.refresh(ctx);
            }
            ("addfeed", Answer::Text(u)) if !u.is_empty() => ctx.error(format!("{u}: serve un indirizzo http(s)")),
            _ => {}
        }
    }
}

impl Screen for News {
    fn title(&self) -> String {
        "news".into()
    }

    fn keys(&self, ctx: &Ctx) -> Vec<Key> {
        let mut k = vec![(KeyCode::Enter, "invio", "apri link")];
        match self.selected(ctx).map(|i| i.src.as_str()) {
            Some("llama") => k.push((KeyCode::Char('i'), "i", "installa")),
            Some("models") => k.push((KeyCode::Char('d'), "d", "scarica")),
            _ => {}
        }
        k.extend([
            (KeyCode::Tab, "tab", "fonte"),
            (KeyCode::Char('r'), "r", "aggiorna"),
            (KeyCode::Char('f'), "f", "feed"),
            (KeyCode::Esc, "esc", "indietro"),
        ]);
        k
    }

    fn draw(&mut self, f: &mut Frame, area: Rect, ctx: &Ctx) {
        let errs = u16::from(!self.data.errors.is_empty());
        let [tabs, table, detail, err] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Fill(1),
            Constraint::Length(9),
            Constraint::Length(errs),
        ])
        .areas(area);
        let labels: Vec<String> = std::iter::once("tutte")
            .chain(SOURCES.iter().map(|s| s.1))
            .enumerate()
            .map(|(i, l)| format!("{l} {}", self.items(ctx, i).len()))
            .collect();
        let items: Vec<(&str, bool)> = labels.iter().enumerate().map(|(i, l)| (l.as_str(), i == self.filter)).collect();
        self.tabs.clear();
        let mut tabs_end = tabs.x;
        for (i, r) in buttons(f, tabs, &items, SIDE, ctx).into_iter().enumerate() {
            tabs_end = r.right();
            self.tabs.add(r, i);
        }
        let status = if self.task.is_some() {
            format!("{} aggiorno le fonti…", spinner())
        } else if self.data.at > 0 {
            format!("aggiornate {}", news::ago(unix_now(), self.data.at))
        } else {
            String::new()
        };
        // right-aligned beside the tabs, or below them when it would cover them
        let w = (len(&status) as u16).min(tabs.width);
        let y = if tabs_end + 2 + w <= tabs.right() { tabs.y } else { tabs.y + 1 };
        let r = Rect { x: tabs.right() - w, y, width: w, height: 1 };
        f.render_widget(Paragraph::new(sp(status, st(DIM))), r);
        let now = unix_now();
        let shown = self.items(ctx, self.filter);
        let rows = shown
            .iter()
            .map(|i| {
                let new = self.seen > 0 && i.date > self.seen;
                vec![
                    if new { sp("●", bold(WARN)) } else { cell("") },
                    sp(news::ago(now, i.date), st(DIM)),
                    sp(clip(&i.source, 20), st(SLOT)),
                    sp(i.title.clone(), if new { bold(FG) } else { st(FG) }),
                    Self::extra(i, ctx),
                ]
            })
            .collect();
        let body = match shown.get(self.list.sel) {
            Some(i) => {
                let mut t = vec![
                    Line::from(sp(i.title.clone(), bold(FG))),
                    Line::from(sp(
                        [i.source.as_str(), &news::ago(now, i.date), &i.url]
                            .iter()
                            .filter(|s| !s.is_empty())
                            .cloned()
                            .collect::<Vec<_>>()
                            .join(" · "),
                        st(DIM),
                    )),
                ];
                t.extend(i.text.lines().map(|l| Line::from(sp(l.to_string(), st(FG)))));
                t
            }
            None if self.task.is_some() => vec![Line::from(sp("scarico le novità…", st(DIM)))],
            None => vec![Line::from(sp("niente da mostrare per questa fonte", st(DIM)))],
        };
        drop(shown);
        self.list.draw(f, table, &["", "quando", "fonte", "titolo", ""], rows, ctx);
        f.render_widget(Paragraph::new(body).wrap(Wrap { trim: true }).block(panel("dettaglio", DIM)), detail);
        if let Some(e) = self.data.errors.first() {
            let n = self.data.errors.len();
            let msg = format!("⚠ {n} fonti non raggiungibili · {}", clip(e, area.width as usize));
            f.render_widget(Paragraph::new(sp(msg, st(WARN))), err);
        }
    }

    fn on(&mut self, ev: &Ev, ctx: &mut Ctx) -> Act {
        if let Ev::Answer(tag, a) = ev {
            self.answer(tag, a, ctx);
            return Act::None;
        }
        if let Some(p) = click(ev)
            && let Some(i) = self.tabs.at(p)
        {
            (self.filter, self.list.sel) = (i, 0);
            return Act::None;
        }
        let n = SOURCES.len() + 1;
        let open = |i: &Item, ctx: &mut Ctx| {
            if i.url.is_empty() {
                return;
            }
            crate::dashboard::open(ctx, &i.url);
        };
        if let Some(Hit::Open(k)) = self.list.on(ev) {
            if let Some(i) = self.items(ctx, self.filter).get(k).map(|i| (*i).clone()) {
                open(&i, ctx);
            }
            return Act::None;
        }
        let cur = self.selected(ctx).cloned();
        match (key(ev), cur) {
            (Some(KeyCode::Tab), _) => (self.filter, self.list.sel) = ((self.filter + 1) % n, 0),
            (Some(KeyCode::BackTab), _) => (self.filter, self.list.sel) = ((self.filter + n - 1) % n, 0),
            (Some(KeyCode::Char('r')), _) => self.refresh(ctx),
            (Some(KeyCode::Char('f')), _) => self.manage_feeds(ctx),
            (Some(KeyCode::Char('o')), Some(i)) => open(&i, ctx),
            (Some(KeyCode::Char('i')), Some(i)) if i.src == "llama" => return Self::install(&i, ctx),
            (Some(KeyCode::Char('d')), Some(i)) if i.src == "models" => {
                return Act::Push(Box::new(hf::RepoScreen::new(&i.tag)));
            }
            _ => return Act::Pass,
        }
        Act::None
    }

    fn tick(&mut self, _ctx: &mut Ctx) -> Act {
        if let Some(t) = self.task.as_mut() {
            if let Some(d) = t.poll().pop() {
                self.data = d;
                self.task = None;
            } else if t.done {
                self.task = None; // the fetch died: keep what was shown
            }
        }
        Act::None
    }
}
