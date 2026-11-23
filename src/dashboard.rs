//! Live full-screen dashboard for llama-server: startup steps,
//! slots, throughput, live metrics, a filterable colored log and a test prompt.
//! `Dashboard` parses and renders, `Live` is the screen of one `server::Server`,
//! `Servers` lists the running ones.

use std::collections::{BTreeMap, VecDeque};
use std::io::Write;
use std::path::PathBuf;
use std::sync::LazyLock;
use std::time::Instant;

use ratatui::Frame;
use ratatui::crossterm::event::{KeyCode, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::Line;
use ratatui::widgets::{Paragraph, Wrap};
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::remote::RpcLink;
use crate::server::{self, Chunk, Launch, Metrics, Server};
use crate::tui::widgets::{Hit, List, Modal, bar, block, key, panel, wheel, wrapped};
use crate::tui::{Act, Answer, Ctx, Ev, Key, Screen, Task};
use crate::ui::*;

const SIDE_W: u16 = 34;
const KEEP_LINES: usize = 3000;
static START: LazyLock<Instant> = LazyLock::new(Instant::now);

static RE_LINE: LazyLock<Regex> = re!(r"^[\d.]+ ([IWEDT]) (\S+)\s+(.*)$");
static RE_SLOTS: LazyLock<Regex> = re!(r"n_slots = (\d+), n_ctx_slot = (\d+)");
static RE_LAUNCH: LazyLock<Regex> = re!(r"launch_slot_: id\s+(\d+) \| task (\d+)");
static RE_RELEASE: LazyLock<Regex> = re!(r"release: id\s+(\d+) .*n_tokens = (\d+)");
static RE_PROMPT: LazyLock<Regex> =
    re!(r"prompt eval time =\s+[\d.]+ ms /\s+(\d+) tokens.*?([\d.]+) tokens per second");
static RE_EVAL: LazyLock<Regex> = re!(r"\|\s+eval time =\s+[\d.]+ ms /\s+(\d+) tokens.*?([\d.]+) tokens per second");
static RE_LISTEN: LazyLock<Regex> = re!(r"listening on (\S+)");
// router mode: which model each child port is, and a child's own line forwarded as "[port] line"
static RE_SPAWN: LazyLock<Regex> = re!(r"spawning server instance with name=(\S+) on port (\d+)");
static RE_CHILD: LazyLock<Regex> = re!(r"^\[\s*(\d+)\] (.*)$");
// startup steps: (label, any of these marks it done); a later match implies the earlier ones
const STEPS: [(&str, &[&str]); 5] = [
    ("avvio processo", &["loading model"]),
    ("pesi modello", &["loaded multimodal", "n_slots ="]),
    ("proiettore vision", &["loaded multimodal"]),
    ("slot / contesto", &["n_slots ="]),
    ("server HTTP", &["listening on"]),
];

fn spin() -> char {
    ['◐', '◓', '◑', '◒'][(START.elapsed().as_millis() / 166 % 4) as usize]
}

fn spark(vals: &VecDeque<f64>) -> String {
    let top = vals.iter().copied().fold(0.0, f64::max);
    let top = if top > 0.0 { top } else { 1.0 };
    vals.iter().map(|v| ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'][((v / top * 7.99) as usize).min(7)]).collect()
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Info {
    pub alias: String,
    pub model: String,
    pub mmproj: String,
    pub endpoint: String,
    pub install: String,
    pub backend: String,
    pub version: String,
    /// (name, value) pairs shown in the sidebar
    pub settings: Vec<(String, String)>,
    /// a multi-model router (its screen is `tui::router::RouterLive`)
    #[serde(default)]
    pub router: bool,
    /// the computer it runs on, as ssh names it ("" = this one)
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub host: String,
}

pub struct LogLine {
    pub level: char,
    pub(crate) text: String, // lowercase "actor message", for the text filter
    line: Line<'static>,
}

pub struct Dashboard {
    info: Info,
    log_path: PathBuf,
    pub t0: Instant,
    end: Option<Instant>,
    pub lines: VecDeque<LogLine>,
    pub done: usize, // startup steps completed
    status: String,
    status_style: Style,
    pub url: Option<String>,
    pub exit_code: Option<i32>,
    lost: bool, // a server on another computer: the link dropped (it may still run there)
    pub n_slots: usize,
    pub n_ctx: u64,
    pub busy: BTreeMap<usize, (u64, Instant)>,
    pub requests: u64,
    pub tok_prompt: u64,
    pub tok_gen: u64,
    pub warns: u64,
    errors: u64,
    last_pp: f64,
    pub last_tg: f64,
    tg_hist: VecDeque<f64>,
    pub scroll: usize, // shown log lines scrolled back from the end
    pub paused: bool,  // the view stays put while lines arrive
    pub only_warn: bool,
    pub search: String, // lowercase; "" = no text filter
    pub m: Metrics,
    pub live: VecDeque<f64>,           // generated tok/s between polls
    generated: Option<(f64, Instant)>, // tokens generated so far (finished + running) at the last poll
    /// router: generated tok/s of each loaded model between polls, and its counter at the last poll
    pub model_tps: BTreeMap<String, f64>,
    model_gen: BTreeMap<String, (f64, Instant)>,
    children: BTreeMap<String, String>, // router: child port -> model id
}

impl Dashboard {
    pub fn new(info: Info, log_path: PathBuf) -> Self {
        Self {
            info,
            log_path,
            t0: Instant::now(),
            end: None,
            lines: VecDeque::new(),
            done: 0,
            status: "avvio…".into(),
            status_style: st(CYAN),
            url: None,
            exit_code: None,
            lost: false,
            n_slots: 0,
            n_ctx: 0,
            busy: BTreeMap::new(),
            requests: 0,
            tok_prompt: 0,
            tok_gen: 0,
            warns: 0,
            errors: 0,
            last_pp: 0.0,
            last_tg: 0.0,
            tg_hist: VecDeque::new(),
            scroll: 0,
            paused: false,
            only_warn: false,
            search: String::new(),
            m: Metrics::default(),
            live: VecDeque::new(),
            generated: None,
            model_tps: BTreeMap::new(),
            model_gen: BTreeMap::new(),
            children: BTreeMap::new(),
        }
    }

    // -- state ---------------------------------------------------------------

    pub fn finished(&mut self, code: Option<i32>) {
        self.exit_code = Some(code.unwrap_or(-1));
        self.end = Some(Instant::now());
        self.status = code.map_or("terminato".into(), |c| format!("terminato · codice {c}"));
        self.status_style = bold(if code.is_none_or(|c| c == 0) { OK } else { ERR });
    }

    /// The link to a server on another computer dropped, `why` (ssh's last words, or "").
    pub fn lost(&mut self, why: &str) {
        self.exit_code = Some(-1);
        self.end = Some(Instant::now());
        self.lost = true;
        self.status = if why.is_empty() { "collegamento perso".into() } else { format!("collegamento perso: {why}") };
        self.status_style = bold(WARN);
    }

    pub fn stopping(&mut self, n: u32) {
        let s = if n == 1 && cfg!(unix) { "arresto… (di nuovo = kill)" } else { "kill…" };
        (self.status, self.status_style) = (s.into(), bold(WARN));
    }

    /// A poll of the server's API. Live generation speed = growth of the tokens generated so far:
    /// the /metrics counter (finished tasks) plus what the busy slots decoded (running ones).
    pub fn metrics(&mut self, m: Metrics) {
        let running: u64 = m.slots.iter().filter(|s| s.busy).map(|s| s.decoded).sum();
        let total = m.prom.get("tokens_predicted_total").copied().unwrap_or(0.0) + running as f64;
        if m.health == "ok" {
            if let Some((prev, at)) = self.generated {
                self.live.push_back((total - prev).max(0.0) / at.elapsed().as_secs_f64().max(0.2));
                if self.live.len() > 30 {
                    self.live.pop_front();
                }
            }
            self.generated = Some((total, Instant::now()));
        }
        if m.health == "ok" && self.info.router {
            // ponytail: finished requests only (a router's /metrics per model, no per-slot progress)
            for s in &m.models {
                let Some(&tot) = s.prom.get("tokens_predicted_total") else {
                    self.model_tps.remove(&s.id);
                    self.model_gen.remove(&s.id);
                    continue;
                };
                if let Some((prev, at)) = self.model_gen.get(&s.id) {
                    let rate = (tot - prev).max(0.0) / at.elapsed().as_secs_f64().max(0.2);
                    self.model_tps.insert(s.id.clone(), rate);
                }
                self.model_gen.insert(s.id.clone(), (tot, Instant::now()));
            }
        }
        self.m = m;
    }

    pub fn is_router(&self) -> bool {
        self.info.router
    }

    pub fn state(&self) -> (&'static str, u32) {
        if self.lost {
            ("scollegato", WARN)
        } else if self.exit_code.is_some() {
            ("terminato", ERR)
        } else if self.url.is_some() || self.m.health == "ok" {
            ("pronto", OK)
        } else {
            ("caricamento", WARN)
        }
    }

    pub fn uptime(&self) -> u64 {
        self.end.unwrap_or_else(Instant::now).duration_since(self.t0).as_secs()
    }

    pub fn alias(&self) -> &str {
        &self.info.alias
    }

    pub fn endpoint(&self) -> &str {
        &self.info.endpoint
    }

    fn shown(&self, l: &LogLine) -> bool {
        (!self.only_warn || matches!(l.level, 'W' | 'E')) && (self.search.is_empty() || l.text.contains(&self.search))
    }

    // -- parsing -------------------------------------------------------------

    pub fn feed(&mut self, raw: &str) {
        let raw = raw.trim_end();
        if raw.is_empty() {
            return;
        }
        if let Some(m) = RE_SPAWN.captures(raw) {
            self.children.insert(m[2].to_string(), m[1].to_string());
        }
        // a router's child: its own line, under the model's id
        let (child, raw) = match RE_CHILD.captures(raw) {
            Some(m) => {
                let port = &m[1];
                let id = self.children.get(port).cloned().unwrap_or_else(|| port.to_string());
                (Some(id), m.get(2).map_or("", |x| x.as_str()))
            }
            None => (None, raw),
        };
        let (level, actor, msg) = match RE_LINE.captures(raw) {
            Some(m) => (m[1].to_string(), m[2].trim_end_matches(':').to_string(), m[3].to_string()),
            None => ("I".into(), "server".into(), raw.to_string()),
        };
        let (actor, msg) = match child {
            Some(id) => (id, format!("{actor} {msg}")),
            None => (actor, msg),
        };
        for (i, (_, marks)) in STEPS.iter().enumerate() {
            if i >= self.done && marks.iter().any(|k| msg.contains(k)) {
                self.done = i + 1;
                if let Some((next, _)) = STEPS.get(i + 1) {
                    self.status = format!("caricamento · {next}");
                }
            }
        }
        let num = |s: &str| s.parse::<u64>().unwrap_or(0);
        if let Some(m) = RE_SLOTS.captures(&msg) {
            (self.n_slots, self.n_ctx) = (num(&m[1]) as usize, num(&m[2]));
        } else if let Some(m) = RE_LISTEN.captures(&msg) {
            self.url = Some(m[1].into());
            (self.status, self.status_style) = ("pronto".into(), bold(OK));
        } else if let Some(m) = RE_LAUNCH.captures(&msg) {
            self.busy.insert(num(&m[1]) as usize, (num(&m[2]), Instant::now()));
        } else if let Some(m) = RE_RELEASE.captures(&msg) {
            self.busy.remove(&(num(&m[1]) as usize));
        } else if let Some(m) = RE_PROMPT.captures(&msg) {
            self.requests += 1;
            self.tok_prompt += num(&m[1]);
            self.last_pp = m[2].parse().unwrap_or(0.0);
        } else if let Some(m) = RE_EVAL.captures(&msg) {
            self.tok_gen += num(&m[1]);
            self.last_tg = m[2].parse().unwrap_or(0.0);
            self.tg_hist.push_back(self.last_tg);
            if self.tg_hist.len() > 24 {
                self.tg_hist.pop_front();
            }
        } else if msg.contains("cleaning up before exit") {
            (self.status, self.status_style) = ("arresto…".into(), bold(WARN));
        }
        self.warns += u64::from(level == "W");
        self.errors += u64::from(level == "E");
        let style = match level.as_str() {
            "W" => st(WARN),
            "E" => bold(ERR),
            _ => st(FG),
        };
        let color = match actor.as_str() {
            "srv" => SRV,
            "slot" => SLOT,
            "load" => LOAD,
            "cmn" => DIM,
            _ => SIDE,
        };
        let actor: String = actor.chars().take(11).collect();
        let text = format!("{actor} {msg}").to_lowercase();
        // ponytail: the clock is when Corral read the line (a re-adopted server's history gets "now")
        let line = Line::from(vec![
            sp(format!("{}  ", clock()), st(DIM)),
            sp(format!("{actor:<12}"), bold(color)),
            sp(msg, style),
        ]);
        let l = LogLine { level: level.chars().next().unwrap_or('I'), text, line };
        if (self.scroll > 0 || self.paused) && self.shown(&l) {
            self.scroll += 1; // keep the view still
        }
        self.lines.push_back(l);
        if self.lines.len() > KEEP_LINES {
            self.lines.pop_front();
        }
    }

    // -- rendering ---------------------------------------------------------------

    fn startup(&self) -> Paragraph<'static> {
        let alive = self.exit_code.is_none();
        let w = STEPS.iter().map(|s| len(s.0)).max().unwrap_or(0);
        let mut t: Vec<Line> = STEPS
            .iter()
            .enumerate()
            .map(|(i, (label, _))| {
                let state = if i == 2 && self.info.mmproj.is_empty() {
                    sp("– saltato", st(DIM))
                } else if i < self.done {
                    sp("✓ fatto", st(OK))
                } else if i == self.done && alive {
                    sp(format!("{} in corso", spin()), bold(WARN))
                } else if i == self.done {
                    sp("✖ fallito", bold(ERR))
                } else {
                    sp("○ in attesa", st(DIM))
                };
                Line::from(vec![sp(format!("{label:<w$}  "), st(if i <= self.done { FG } else { DIM })), state])
            })
            .collect();
        let frac = self.done as f64 / STEPS.len() as f64;
        t.push(Line::default());
        t.push(Line::from([bar(frac, 30, LOAD).to_vec(), vec![sp(format!(" {:.0}%", frac * 100.0), st(FG))]].concat()));
        // lit while loading, red if it never got there, quiet once ready
        let border = if self.url.is_some() {
            DIM
        } else if alive {
            LOAD
        } else {
            ERR
        };
        Paragraph::new(t).block(block(" avvio ".into(), bold(LOAD), border))
    }

    fn slots(&self) -> Paragraph<'static> {
        let w = format!("slot {}", self.n_slots.max(self.m.slots.len()).saturating_sub(1)).len();
        let label = |i: usize| format!("{:<w$} ", format!("slot {i}"));
        let t: Vec<Line> = if !self.m.slots.is_empty() {
            // from /slots: how full each slot's context is
            let row = |s: &server::Slot| {
                let frac = s.used as f64 / s.n_ctx.max(1) as f64;
                let (state, style) = if s.busy {
                    (format!(" {} {}/{} tok", spin(), s.used, s.n_ctx), bold(WARN))
                } else {
                    (" ○ libero".into(), st(DIM))
                };
                let head = sp(label(s.id), st(if s.busy { FG } else { DIM }));
                Line::from([vec![head], bar(frac, 12, SLOT).to_vec(), vec![sp(state, style)]].concat())
            };
            self.m.slots.iter().map(row).collect()
        } else if self.n_slots == 0 {
            vec![Line::from(sp("—", st(DIM)))]
        } else {
            (0..self.n_slots)
                .map(|i| {
                    Line::from(match self.busy.get(&i) {
                        Some((task, since)) => [
                            vec![sp(label(i), st(FG))],
                            bar(1.0, 12, SLOT).to_vec(),
                            vec![sp(
                                format!(" {} task {task} · {:.0}s", spin(), since.elapsed().as_secs_f64()),
                                bold(WARN),
                            )],
                        ]
                        .concat(),
                        None => {
                            [vec![sp(label(i), st(DIM))], bar(0.0, 12, SLOT).to_vec(), vec![sp(" ○ libero", st(DIM))]]
                                .concat()
                        }
                    })
                })
                .collect()
        };
        let n_ctx = self.m.slots.first().map_or(self.n_ctx, |s| s.n_ctx);
        let ctx = if n_ctx > 0 { n_ctx.to_string() } else { "…".into() };
        let busy = !self.busy.is_empty() || self.m.slots.iter().any(|s| s.busy);
        Paragraph::new(t).block(block(format!(" slot · ctx {ctx} "), bold(SLOT), if busy { SLOT } else { DIM }))
    }

    fn throughput(&self) -> Paragraph<'static> {
        let hist = &self.tg_hist;
        let avg = if hist.is_empty() { 0.0 } else { hist.iter().sum::<f64>() / hist.len() as f64 };
        let s = spark(hist);
        let row = |label: &str, rate: f64, color: u32, tot: u64| {
            Line::from(vec![
                sp(format!("{label:<11}  "), st(DIM)),
                sp(format!("{rate:8.1} tok/s"), bold(color)),
                sp(format!("  tot {tot} tok"), st(FG)),
            ])
        };
        let t = vec![
            row("prompt", self.last_pp, SRV, self.tok_prompt),
            row("generazione", self.last_tg, SLOT, self.tok_gen),
            Line::from(vec![
                sp("gen tok/s  ", st(DIM)),
                sp(if s.is_empty() { "—".into() } else { s }, st(SLOT)),
                sp(format!("  media {avg:.1}"), st(FG)),
            ]),
        ];
        let border = if self.requests > 0 { SRV } else { DIM };
        Paragraph::new(t).block(block(" velocità · ultima richiesta ".into(), bold(SRV), border))
    }

    /// The LIVE panel: HTTP state and queue, generation speed now, GPU memory.
    fn live_lines(&self) -> Vec<Line<'static>> {
        let m = &self.m;
        let k = |s: &str| sp(format!("{s:<10}"), st(DIM));
        let prom = |n: &str| m.prom.get(n).copied().unwrap_or(0.0);
        let health = match m.health.as_str() {
            "" if self.exit_code.is_some() => sp("—", st(DIM)),
            "" => sp("nessuna risposta", st(WARN)),
            "ok" => sp("ok", bold(OK)),
            h => sp(h.to_string(), st(WARN)),
        };
        let queued = prom("requests_deferred");
        let mut t = vec![if m.prom.is_empty() {
            Line::from(vec![k("http"), health, sp("  · /metrics non attivo", st(DIM))])
        } else {
            Line::from(vec![
                k("http"),
                health,
                sp(format!("  · {:.0} in corso · ", prom("requests_processing")), st(FG)),
                sp(format!("{queued:.0} in coda"), st(if queued > 0.0 { WARN } else { FG })),
            ])
        }];
        let s = spark(&self.live);
        let now = self.live.back().copied().unwrap_or(0.0);
        t.push(Line::from(vec![
            k("gen live"),
            sp(if s.is_empty() { "—".into() } else { s }, st(SLOT)),
            sp(format!(" {now:.1} tok/s"), bold(SLOT)),
        ]));
        if !m.prom.is_empty() {
            // the server's own "average" gauges reset at every /metrics read: use the counters
            let rate = |tok: &str, secs: &str| prom(tok) / prom(secs).max(1e-9);
            let avg = format!(
                "pp {:.1} · tg {:.1} tok/s",
                rate("prompt_tokens_total", "prompt_seconds_total"),
                rate("tokens_predicted_total", "tokens_predicted_seconds_total")
            );
            t.push(Line::from(vec![k("medie"), sp(avg, st(FG))]));
        }
        for g in &m.gpus {
            let mut l = vec![k(&clip(&g.name.replace("NVIDIA GeForce ", ""), 9))];
            l.extend(bar(g.used as f64 / g.total.max(1) as f64, 10, LOAD));
            l.push(sp(format!(" {}/{} MiB", g.used, g.total), st(FG)));
            t.push(Line::from(l));
            // its own line, the panel is narrow; a GPU lent over RPC says whose it is
            let mut detail = g.load();
            if !g.host.is_empty() {
                detail =
                    if detail.is_empty() { format!("su {}", g.host) } else { format!("su {} · {detail}", g.host) };
            }
            if !detail.is_empty() {
                t.push(Line::from(vec![k(""), sp(detail, st(DIM))]));
            }
        }
        if let Some(v) = m.vram {
            t.push(Line::from(vec![k("VRAM srv"), sp(format!("{v} MiB"), bold(LOAD))]));
        }
        t
    }

    fn sidebar(&self) -> Vec<Line<'static>> {
        let (info, up) = (&self.info, self.uptime());
        let mut t = vec![];
        let sec =
            |t: &mut Vec<Line<'static>>, name: &str| t.push(Line::from(vec![sp("◇ ", st(FAINT)), sp(name, bold(DIM))]));
        let kv = |t: &mut Vec<Line<'static>>, k: &str, v: &str, style: Style| {
            t.push(Line::from(vec![
                sp("│ ", st(LINE)),
                sp(format!("{k:<10}"), st(DIM)),
                sp(clip(v, SIDE_W as usize - 16), style),
            ]))
        };
        let full = |t: &mut Vec<Line<'static>>, v: &str| {
            let n = SIDE_W as usize - 6; // too long: keep the end, it's the distinctive part of a file name
            let v = if len(v) <= n {
                v.to_string()
            } else {
                format!("…{}", v.chars().skip(len(v) + 1 - n).collect::<String>())
            };
            t.push(Line::from(vec![sp("│ ", st(LINE)), sp(v, st(FG))]))
        };
        sec(&mut t, "modello");
        kv(&mut t, "alias", &info.alias, bold(FG));
        full(&mut t, fname(&info.model));
        if !info.mmproj.is_empty() {
            full(&mut t, &format!("+ {}", fname(&info.mmproj)));
        }
        sec(&mut t, "llama.cpp");
        kv(&mut t, "install", &info.install, st(FG));
        kv(&mut t, "backend", &format!("{} · {}", info.backend, info.version.split(" ·").next().unwrap_or("")), st(FG));
        kv(&mut t, "endpoint", &info.endpoint, st(FG));
        sec(&mut t, "parametri");
        for (k, v) in &info.settings {
            kv(&mut t, k, v, st(FG));
        }
        kv(&mut t, "uptime", &format!("{:02}:{:02}:{:02}", up / 3600, up % 3600 / 60, up % 60), st(FG));
        sec(&mut t, "stato");
        let spinning = self.exit_code.is_none() && self.url.is_none();
        let status = if spinning { format!("{} {}", spin(), self.status) } else { self.status.clone() };
        t.push(Line::from(vec![sp("│ ", st(LINE)), sp(clip(&status, SIDE_W as usize - 6), self.status_style)]));
        sec(&mut t, "sessione");
        kv(&mut t, "richieste", &self.requests.to_string(), bold(OK));
        kv(&mut t, "token", &format!("{}+{}", self.tok_prompt, self.tok_gen), st(FG));
        kv(&mut t, "warning", &self.warns.to_string(), st(if self.warns > 0 { WARN } else { FG }));
        kv(&mut t, "errori", &self.errors.to_string(), if self.errors > 0 { bold(ERR) } else { st(FG) });
        t
    }

    /// The line on top: how the server is, which one, where it answers; its counters on the right.
    fn status_line(&self, w: u16) -> Line<'static> {
        let (word, color) = self.state();
        let loading = self.exit_code.is_none() && self.url.is_none();
        let dot = if loading { spin().to_string() } else { "●".into() };
        let mut left = vec![
            sp(format!("{dot} {word}"), bold(color)),
            sp("   ", st(FG)),
            sp(self.info.alias.clone(), bold(WOOL)),
            sp(self.on_host(), bold(CYAN)),
        ];
        let address = self.url.clone().unwrap_or_else(|| self.info.endpoint.clone());
        if !address.is_empty() {
            left.push(sp(format!("   {address}"), st(DIM)));
        }
        let up = self.uptime();
        let busy = self.busy.len().max(self.m.slots.iter().filter(|s| s.busy).count());
        let mut right = vec![
            sp("slot ", st(DIM)),
            sp(format!("{busy}/{}", self.n_slots.max(self.m.slots.len())), st(FG)),
            sp("   richieste ", st(DIM)),
            sp(self.requests.to_string(), st(FG)),
        ];
        if self.errors > 0 {
            right.push(sp("   errori ", st(DIM)));
            right.push(sp(self.errors.to_string(), bold(ERR)));
        }
        right.push(sp("   uptime ", st(DIM)));
        right.push(sp(format!("{:02}:{:02}:{:02}", up / 3600, up % 3600 / 60, up % 60), st(FG)));
        let used: usize = left.iter().chain(&right).map(|s| len(&s.content)).sum();
        if used < w as usize {
            left.push(sp(" ".repeat(w as usize - used), st(FG)));
            left.extend(right);
        }
        Line::from(left)
    }

    /// " su <host>" for a server on another computer, else "".
    fn on_host(&self) -> String {
        if self.info.host.is_empty() { String::new() } else { format!(" su {}", self.info.host) }
    }

    /// Move the log view by `d` lines (positive = back in time).
    pub fn scroll(&mut self, d: i32) {
        let n = self.lines.iter().filter(|l| self.shown(l)).count();
        self.scroll = (self.scroll as i64 + d as i64).clamp(0, n as i64) as usize;
    }

    /// Back to following the end (and out of pause).
    pub fn follow(&mut self) {
        (self.scroll, self.paused) = (0, false);
    }

    /// Log view input shared by the dashboards: wheel over `area` (the log panel), PgUp/PgDn,
    /// Home/End, p, f and the answer to the "search" prompt (the screen opens it on '/').
    /// true = handled.
    pub(crate) fn log_event(&mut self, ev: &Ev, area: Rect) -> bool {
        let page = area.height.saturating_sub(2).max(1) as i32;
        if let Some((d, p)) = wheel(ev) {
            if area.contains(p) {
                self.scroll(-d * 3);
            }
            return area.contains(p);
        }
        if let Ev::Answer("search", Answer::Text(t)) = ev {
            self.search = t.to_lowercase();
            self.scroll = 0;
            return true;
        }
        match key(ev) {
            Some(KeyCode::PageUp) => self.scroll(page),
            Some(KeyCode::PageDown) => self.scroll(-page),
            Some(KeyCode::Home) => self.scroll(i32::MAX / 2),
            Some(KeyCode::End) => self.follow(),
            Some(KeyCode::Char('p')) => {
                if self.paused {
                    self.follow()
                } else {
                    self.paused = true
                }
            }
            Some(KeyCode::Char('f')) => {
                self.only_warn = !self.only_warn;
                self.scroll = 0;
            }
            _ => return false,
        }
        true
    }

    pub(crate) fn log_panel(&self, area: Rect) -> Paragraph<'static> {
        let n = (area.height as usize).saturating_sub(2).max(1);
        let shown: Vec<&LogLine> = self.lines.iter().filter(|l| self.shown(l)).collect();
        let back = self.scroll.min(shown.len().saturating_sub(n));
        let skip = shown.len().saturating_sub(n + back);
        let lines: Vec<Line> = shown.iter().skip(skip).take(n).map(|l| l.line.clone()).collect();
        let rel = self.log_path.strip_prefix(&*BASE).unwrap_or(&self.log_path).display().to_string();
        let empty = if self.lines.is_empty() { "—" } else { "nessuna riga corrisponde al filtro" };
        let body = if lines.is_empty() { vec![Line::from(sp(empty, st(DIM)))] } else { lines };
        let mut tags = String::new();
        if self.only_warn {
            tags += "· solo warning/errori ";
        }
        if !self.search.is_empty() {
            tags += &format!("· “{}” ", self.search);
        }
        if self.paused {
            tags += "· ‖ pausa ";
        }
        if back > 0 {
            tags += &format!("· ↑ {back} righe ");
        }
        let held = back > 0 || self.paused;
        // what the colour of a line's source means, on the right of the border
        let mut legend = vec![];
        for (color, label) in [(SRV, "server"), (SLOT, "richieste"), (LOAD, "modello"), (SIDE, "altro")] {
            legend.push(sp(" ■ ", st(color)));
            legend.push(sp(label, st(DIM)));
        }
        legend.push(sp(" ", st(DIM)));
        let title = format!(" log del server · {rel} {tags}");
        let mut blk = block(title.clone(), bold(FG), if held { CYAN } else { DIM });
        let legend = Line::from(legend);
        let room = (area.width as usize).saturating_sub(len(&title) + 4); // corners and a gap
        if legend.width() <= room {
            blk = blk.title(legend.right_aligned());
        }
        Paragraph::new(body).block(blk)
    }

    /// Draw into `area`; returns the log panel's rect (for wheel scrolling) and, with `split`,
    /// the rect beside it (for the test prompt).
    pub fn render(&self, f: &mut Frame, area: Rect, split: bool) -> (Rect, Rect) {
        let side = self.sidebar();
        let live = self.live_lines();
        let body_h = side.len() as u16 + 2;
        let [status, _, body, log] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(body_h),
            Constraint::Min(5),
        ])
        .areas(area);
        let [left, right] = Layout::horizontal([Constraint::Length(SIDE_W), Constraint::Fill(1)]).areas(body);
        let bottom_h = live.len().max(3) as u16 + 2;
        let [top, bottom] = Layout::vertical([Constraint::Fill(1), Constraint::Length(bottom_h)]).areas(right);
        let [start, slots] = Layout::horizontal([Constraint::Fill(1); 2]).areas(top);
        let [thr, met] = Layout::horizontal([Constraint::Fill(1); 2]).areas(bottom);
        let [log, extra] =
            if split { Layout::horizontal([Constraint::Fill(1); 2]).areas(log) } else { [log, Rect::default()] };
        f.render_widget(Paragraph::new(self.status_line(status.width)), status);
        f.render_widget(Paragraph::new(side).block(panel("server", FG)), left);
        f.render_widget(self.startup(), start);
        f.render_widget(self.slots(), slots);
        f.render_widget(self.throughput(), thr);
        let border = if self.m.prom.is_empty() { DIM } else { CYAN };
        f.render_widget(Paragraph::new(live).block(block(" live · /metrics /slots ".into(), bold(CYAN), border)), met);
        f.render_widget(self.log_panel(log), log);
        (log, extra)
    }
}

// -- the screen of one server ----------------------------------------------------------------

/// Test prompt sent from the dashboard, its streamed answer and speed.
pub(crate) struct Test {
    prompt: String,
    out: String,
    tokens: u64,
    t0: Instant,
    end: Option<Result<(u64, f64), String>>,
    task: Task<Chunk>,
}

impl Test {
    /// `model`: the router's model to ask ("" = a single-model server).
    pub(crate) fn start(s: &Server, model: &str, prompt: String) -> Self {
        let (base, key, m, p) = (s.base.clone(), s.api_key.clone(), model.to_string(), prompt.clone());
        let task = Task::spawn(move |tx| server::chat(&base, &key, &m, &p, &tx));
        Self { prompt, out: String::new(), tokens: 0, t0: Instant::now(), end: None, task }
    }

    pub(crate) fn poll(&mut self) {
        for c in self.task.poll() {
            match c {
                Chunk::Text(t) => {
                    self.out += &t;
                    self.tokens += 1;
                }
                Chunk::Done(n, tps) => self.end = Some(Ok((n, tps))),
                Chunk::Err(e) => self.end = Some(Err(e)),
            }
        }
        if self.task.done && self.end.is_none() {
            self.end = Some(Err("interrotto".into()));
        }
    }

    pub(crate) fn draw(&self, f: &mut Frame, area: Rect) {
        let mut t = vec![Line::from(sp(format!("› {}", self.prompt), bold(CYAN))), Line::default()];
        t.extend(self.out.split('\n').map(|l| Line::from(sp(l.to_string(), st(FG)))));
        t.push(Line::default());
        t.push(match &self.end {
            None => {
                let rate = self.tokens as f64 / self.t0.elapsed().as_secs_f64().max(1e-3);
                Line::from(sp(format!("{} {} token · {rate:.1} tok/s", spin(), self.tokens), bold(WARN)))
            }
            Some(Ok((n, tps))) => Line::from(sp(format!("✓ {n} token · {tps:.1} tok/s · t = nuova prova"), bold(OK))),
            Some(Err(e)) => Line::from(sp(format!("✖ {e}"), bold(ERR))),
        });
        let inner = (area.width.saturating_sub(4), area.height.saturating_sub(2));
        let over = wrapped(&t, inner.0).saturating_sub(inner.1); // keep the end in view
        let blk = block(" prova · /v1/chat/completions · esc chiude ".into(), bold(CYAN), CYAN);
        f.render_widget(Paragraph::new(t).wrap(Wrap { trim: false }).scroll((over, 0)).block(blk), area);
    }
}

/// One server's dashboard. s / Ctrl+C = graceful stop (twice = kill); b = back, the server keeps
/// running; Esc asks, stops and goes back; once it's down Esc goes back, m to the menu.
pub struct Live {
    id: u64,     // the server's Server::id
    leave: bool, // go back as soon as the server is down
    log_area: Rect,
    test: Option<Test>,
}

impl Live {
    /// Start `launch` (with the links to the hosts whose GPUs it uses) and register it in
    /// `ctx.servers`.
    pub fn start(ctx: &mut Ctx, launch: &Launch, rpc: Vec<RpcLink>) -> Result<Self, String> {
        let s = Server::start(launch, rpc)?;
        let id = s.id;
        ctx.servers.push(s);
        Ok(Self::attach(id))
    }

    /// The dashboard of a server already in `ctx.servers` (by `Server::id`).
    pub fn attach(id: u64) -> Self {
        Self { id, leave: false, log_area: Rect::default(), test: None }
    }

    fn idx(&self, ctx: &Ctx) -> Option<usize> {
        ctx.servers.iter().position(|s| s.id == self.id)
    }

    fn forget(&self, ctx: &mut Ctx) {
        ctx.servers.retain(|s| s.id != self.id);
    }
}

/// Back from a running server's dashboard, leaving it running (b).
pub(crate) fn background(ctx: &mut Ctx, i: usize) -> Act {
    let msg = format!("{} resta attivo: menu › server attivi", ctx.servers[i].dash.alias());
    ctx.toast(msg, OK);
    Act::Pop
}

/// Copy `text`: the system clipboard tool, else the terminal (OSC 52, also over ssh).
pub(crate) fn copy(ctx: &mut Ctx, text: &str) {
    let shown = clip(text.lines().next().unwrap_or(""), 60);
    // over ssh a clipboard tool would fill this machine's clipboard: the terminal reaches the user's
    if !over_ssh() && server::clipboard(text) {
        ctx.toast(format!("✓ copiato: {shown}"), OK);
    } else {
        let mut out = std::io::stdout();
        let _ = write!(out, "\x1b]52;c;{}\x07", server::base64(text.as_bytes()));
        let _ = out.flush();
        ctx.toast(format!("copiato via terminale (OSC 52): {shown}"), OK);
    }
}

/// Open `url` in the browser. Over ssh that browser would be on this machine: copy the address
/// and explain how to reach it from the user's computer.
pub(crate) fn open(ctx: &mut Ctx, url: &str) {
    if !over_ssh() {
        match server::open_url(url) {
            Ok(()) => ctx.toast(format!("apro {} nel browser", clip(url, 60)), OK),
            Err(e) => ctx.error(format!("impossibile aprire il browser: {e} · {url}")),
        }
        return;
    }
    copy(ctx, url);
    let mut body = vec![Line::from(sp("il browser si aprirebbe su questa macchina: ho copiato l'indirizzo.", st(FG)))];
    body.push(Line::default());
    match server::local_address(url) {
        Some((host, port)) => {
            let connection = std::env::var("SSH_CONNECTION").unwrap_or_default();
            let user = std::env::var("USER").unwrap_or_default();
            body.push(Line::from(sp("dal tuo computer apri un tunnel, in un altro terminale:", st(DIM))));
            body.push(Line::from(sp(server::tunnel_command(host, port, &connection, &user), bold(CYAN))));
            body.push(Line::from(sp(format!("poi apri http://localhost:{port}"), st(DIM))));
        }
        None => body.push(Line::from(sp(format!("aprilo nel browser del tuo computer: {url}"), st(DIM)))),
    }
    ctx.ask("info", Modal::info("collegato via SSH", CYAN, body));
}

impl Screen for Live {
    fn title(&self) -> String {
        "server".into()
    }

    fn keys(&self, ctx: &Ctx) -> Vec<Key> {
        let Some(s) = self.idx(ctx).map(|i| &ctx.servers[i]) else { return vec![] };
        let mut k = if s.running() {
            vec![
                (KeyCode::Char('s'), "s", if s.stops == 0 { "ferma" } else { "kill" }),
                (KeyCode::Char('b'), "b", "background"),
                (KeyCode::Char('t'), "t", "prova"),
                (KeyCode::Char('o'), "o", "browser"),
                (KeyCode::Char('c'), "c", "copia URL"),
            ]
        } else {
            vec![(KeyCode::Esc, "esc", "indietro"), (KeyCode::Char('m'), "m", "menu")]
        };
        k.extend([
            (KeyCode::Char('f'), "f", if s.dash.only_warn { "tutto" } else { "warning" }),
            (KeyCode::Char('/'), "/", "cerca"),
            (KeyCode::Char('p'), "p", if s.dash.paused { "riprendi" } else { "pausa" }),
            (KeyCode::PageUp, "pgsu", "scorri"),
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
        let (log, extra) = ctx.servers[i].dash.render(f, area, self.test.is_some());
        self.log_area = log;
        if let Some(t) = &self.test {
            t.draw(f, extra);
        }
    }

    fn on(&mut self, ev: &Ev, ctx: &mut Ctx) -> Act {
        let Some(i) = self.idx(ctx) else { return Act::Pop };
        let running = ctx.servers[i].running();
        let s = &mut ctx.servers[i];
        if s.dash.log_event(ev, self.log_area) {
            return Act::None;
        }
        let ctrl_c =
            matches!(ev, Ev::Key(k) if k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL));
        match ev {
            Ev::Answer("leave", Answer::Yes) => {
                self.leave = true;
                if s.stops == 0 {
                    s.stop();
                }
                return Act::None;
            }
            Ev::Answer("prompt", Answer::Text(t)) if !t.is_empty() => {
                self.test = Some(Test::start(s, "", t.clone()));
                return Act::None;
            }
            Ev::Answer(..) => return Act::None,
            _ if ctrl_c && running => {
                s.stop();
                return Act::None;
            }
            _ => {}
        }
        match key(ev) {
            Some(KeyCode::Up) => s.dash.scroll(1),
            Some(KeyCode::Down) => s.dash.scroll(-1),
            Some(KeyCode::Char('/')) => {
                let cur = s.dash.search.clone();
                ctx.ask("search", Modal::prompt("filtra il log", "testo da cercare (vuoto = tutto)", &cur));
            }
            Some(KeyCode::Char('s')) if running => s.stop(),
            // a server on another computer: leaving its dashboard never stops it
            Some(KeyCode::Char('b')) if running => return background(ctx, i),
            Some(KeyCode::Esc | KeyCode::Char('q')) if running && self.test.is_none() && s.host().is_some() => {
                return background(ctx, i);
            }
            Some(KeyCode::Char('t')) if running => {
                ctx.ask("prompt", Modal::prompt("prompt di prova", "risposta in streaming accanto al log", ""))
            }
            Some(KeyCode::Char('o')) if running => {
                let url = s.base.clone();
                open(ctx, &url);
            }
            Some(KeyCode::Char('c')) if running => {
                let url = format!("{}/v1", s.base);
                copy(ctx, &url);
            }
            Some(KeyCode::Esc) if self.test.is_some() => self.test = None,
            Some(KeyCode::Esc | KeyCode::Char('q')) if running => {
                ctx.ask("leave", Modal::ask("fermo il server e torno indietro?", true));
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
        let Some(i) = self.idx(ctx) else { return Act::Pop };
        if self.leave && !ctx.servers[i].running() {
            self.forget(ctx);
            return Act::Pop;
        }
        Act::None
    }
}

// -- running servers ----------------------------------------------------------------------------

/// The dashboard of server `s` (already in `ctx.servers`): a router's, or a single model's.
pub(crate) fn screen_of(s: &Server) -> Box<dyn Screen> {
    if s.dash.is_router() {
        Box::new(crate::tui::router::RouterLive::attach(s.id))
    } else {
        Box::new(Live::attach(s.id))
    }
}

/// The servers Corral runs (or re-adopted): open the dashboard, stop, keep after exit.
pub struct Servers {
    list: List,
}

impl Servers {
    pub fn new() -> Self {
        Self { list: List::new(0) }
    }
}

impl Screen for Servers {
    fn title(&self) -> String {
        "server attivi".into()
    }

    fn keys(&self, _ctx: &Ctx) -> Vec<Key> {
        vec![
            (KeyCode::Enter, "invio", "dashboard"),
            (KeyCode::Char('s'), "s", "ferma"),
            (KeyCode::Char('l'), "l", "lascia attivo all'uscita"),
            (KeyCode::Char('x'), "x", "togli i terminati"),
            (KeyCode::Esc, "esc", "indietro"),
        ]
    }

    fn draw(&mut self, f: &mut Frame, area: Rect, ctx: &Ctx) {
        if ctx.servers.is_empty() {
            let t = vec![
                Line::from(sp("nessun server in esecuzione", st(WARN))),
                Line::from(sp("avviane uno da «avvia server»; nella dashboard b lo lascia in background", st(DIM))),
                Line::from(sp("quelli di altri computer si collegano da «host remoti»", st(DIM))),
            ];
            f.render_widget(Paragraph::new(t), area);
            return;
        }
        let rows = ctx
            .servers
            .iter()
            .map(|s| {
                let d = &s.dash;
                let (state, color) = d.state();
                let up = d.uptime();
                let on_exit = match s.host() {
                    Some(_) => sp("resta (è sull'host)", st(DIM)),
                    None if s.keep => sp("resta attivo", bold(WARN)),
                    None if s.uses_rpc() => sp("si ferma (usa GPU di altri host)", st(DIM)),
                    None => sp("si ferma", st(DIM)),
                };
                vec![
                    sp(d.alias().to_string(), bold(FG)),
                    sp(s.host().unwrap_or("locale").to_string(), st(if s.host().is_some() { CYAN } else { DIM })),
                    cell(d.endpoint().to_string()),
                    sp(state, bold(color)),
                    cell(format!("{:02}:{:02}:{:02}", up / 3600, up % 3600 / 60, up % 60)),
                    cell(d.requests.to_string()),
                    cell(format!("{:.1}", d.live.back().copied().unwrap_or(0.0))),
                    cell(s.pid.to_string()),
                    on_exit,
                ]
            })
            .collect();
        let head =
            ["alias", "host", "endpoint", "stato", "uptime", "richieste", "gen tok/s", "pid", "all'uscita di Corral"];
        self.list.draw(f, area, &head, rows, ctx);
    }

    fn on(&mut self, ev: &Ev, ctx: &mut Ctx) -> Act {
        if let Some(Hit::Open(i)) = self.list.on(ev)
            && let Some(s) = ctx.servers.get(i)
        {
            return Act::Push(screen_of(s));
        }
        let sel = self.list.sel;
        match key(ev) {
            Some(KeyCode::Char('s')) if sel < ctx.servers.len() => ctx.servers[sel].stop(),
            Some(KeyCode::Char('l')) if sel < ctx.servers.len() && ctx.servers[sel].host().is_some() => {
                ctx.toast("è su un altro computer: uscire da Corral chiude solo il tunnel", OK);
            }
            Some(KeyCode::Char('l')) if sel < ctx.servers.len() && ctx.servers[sel].uses_rpc() => {
                ctx.toast("usa GPU di altri host: i collegamenti finiscono con Corral, e il server con loro", WARN);
            }
            Some(KeyCode::Char('l')) if sel < ctx.servers.len() => {
                let s = &mut ctx.servers[sel];
                s.set_keep(!s.keep);
            }
            Some(KeyCode::Char('x')) => ctx.servers.retain(|s| s.running()),
            _ => return Act::Pass,
        }
        Act::None
    }
}
