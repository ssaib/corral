//! llama-bench with a preset's memory settings (pp512 / tg128), and the results history in
//! <data>/bench.jsonl: one line per run, shown for every quantization of the same model so
//! quantizations and settings can be compared.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ratatui::Frame;
use ratatui::crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::Line;
use ratatui::widgets::{Paragraph, Wrap};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::installs::{self, Install};
use crate::preset::base_alias;
use crate::tui::widgets::{List, key, panel, spinner};
use crate::tui::{Act, Ctx, Ev, Key, Screen, Task};
use crate::ui::*;

#[derive(Clone, Serialize, Deserialize)]
pub struct Record {
    pub date: String,
    pub model: String,
    pub install: String,
    pub args: String, // the preset-derived llama-bench arguments
    pub pp: f64,      // prompt processing, tokens/s
    pub tg: f64,      // generation, tokens/s
}

pub fn history_path() -> PathBuf {
    BASE.join("bench.jsonl")
}

/// llama-bench arguments from the preset; only what differs from llama-bench's defaults. Its
/// lists use other separators than llama-server: '/' between devices and split ratios, ';'
/// between tensor overrides (a ',' would mean another test to run).
pub fn params(p: &Preset) -> Vec<String> {
    let v = |k: &str| p.get(k).map_or("", String::as_str);
    let mut out: Vec<String> = vec![];
    match v("ngl") {
        "" | "auto" => {}
        "all" => out.extend(["-ngl".into(), "999".into()]),
        n => out.extend(["-ngl".into(), n.into()]),
    }
    let plain = [
        ("fa", "-fa", "auto"),
        ("ctk", "-ctk", "f16"),
        ("ctv", "-ctv", "f16"),
        ("threads", "-t", "-1"),
        ("batch", "-b", ""),
        ("ubatch", "-ub", ""),
        ("n_cpu_moe", "-ncmoe", ""),
    ];
    for (k, flag, default) in plain {
        if !v(k).is_empty() && v(k) != default {
            out.extend([flag.into(), v(k).into()]);
        }
    }
    if !v("device").is_empty() {
        out.extend(["-dev".into(), v("device").replace(',', "/")]);
    }
    if !v("ts").is_empty() {
        out.extend(["-ts".into(), v("ts").replace(',', "/")]);
    }
    let mut overrides: Vec<String> = vec![];
    if v("cpu_moe") == "on" {
        overrides.push(crate::preset::OT_PRESETS[0].0.into()); // what --cpu-moe does
    }
    if !v("ot").is_empty() {
        overrides.push(v("ot").replace(',', ";"));
    }
    if !overrides.is_empty() {
        out.extend(["-ot".into(), overrides.join(";")]);
    }
    if v("kv_offload") == "off" {
        out.extend(["-nkvo".into(), "1".into()]);
    }
    out
}

/// pp / tg tokens/s from llama-bench's `-o jsonl` lines.
pub fn parse(lines: &[String]) -> Option<(f64, f64)> {
    let (mut pp, mut tg) = (None, None);
    for v in lines.iter().filter_map(|l| serde_json::from_str::<Value>(l).ok()) {
        let ts = v["avg_ts"].as_f64();
        match (v["n_prompt"].as_u64(), v["n_gen"].as_u64()) {
            (Some(p), Some(0)) if p > 0 => pp = ts,
            (Some(0), Some(g)) if g > 0 => tg = ts,
            _ => {}
        }
    }
    Some((pp?, tg?))
}

pub fn history() -> Vec<Record> {
    let text = fs::read_to_string(history_path()).unwrap_or_default();
    text.lines().filter_map(|l| serde_json::from_str(l).ok()).collect()
}

fn append(r: &Record) -> Result<(), String> {
    let mut f = OpenOptions::new().create(true).append(true).open(history_path()).map_err(|e| e.to_string())?;
    writeln!(f, "{}", serde_json::to_string(r).unwrap_or_default()).map_err(|e| e.to_string())
}

/// llama-bench for `model` with `args`: pp512 and tg128, 3 runs each.
pub fn command(inst: &Install, model: &Path, args: &[String]) -> Vec<String> {
    let mut c = vec![ps(&installs::binary(inst, "llama-bench")), "-m".into(), ps(model)];
    c.extend(["-p", "512", "-n", "128", "-r", "3", "-o", "jsonl"].map(String::from));
    c.extend(args.iter().cloned());
    c
}

/// Run llama-bench (see `command`) with its output in logs/, and add the result to the history.
/// Blocking: run it in a thread. `on_line` gets its progress lines; `cancel` stops it.
pub fn measure(
    inst: &Install,
    model: &Path,
    args: &[String],
    cancel: &AtomicBool,
    on_line: &mut dyn FnMut(String),
) -> Result<Record, String> {
    let cmd = command(inst, model, args);
    if !Path::new(&cmd[0]).is_file() {
        return Err(format!("questa installazione non ha llama-bench ({})", cmd[0]));
    }
    let log = logs().join(format!("bench-{}.log", file_stamp()));
    let mut json = vec![];
    run_logged(&cmd, &BASE, &log, cancel, &mut |l| {
        if l.starts_with('{') {
            json.push(l);
        } else {
            on_line(l);
        }
    })?;
    let Some((pp, tg)) = parse(&json) else {
        return Err(format!("nessun risultato · log: {}", log.display()));
    };
    let rec = Record { date: stamp(), model: ps(model), install: inst.name.clone(), args: args.join(" "), pp, tg };
    append(&rec)?;
    Ok(rec)
}

enum Msg {
    Line(String),
    Done(Result<Record, String>),
}

/// Runs llama-bench for one model + preset and shows the history of that model's quantizations.
pub struct Bench {
    model: PathBuf,
    inst: Option<Install>,
    args: Vec<String>,
    task: Option<Task<Msg>>,
    cancel: Arc<AtomicBool>,
    last: String,
    error: String,
    rows: Vec<Record>,
    list: List,
}

impl Bench {
    pub fn new(ctx: &Ctx, model: &Path, p: &Preset) -> Self {
        let mut b = Self {
            model: model.into(),
            inst: ctx.hw.inst.clone(),
            args: params(p),
            task: None,
            cancel: Arc::new(AtomicBool::new(false)),
            last: String::new(),
            error: String::new(),
            rows: vec![],
            list: List::new(0),
        };
        b.reload();
        b.run();
        b
    }

    fn reload(&mut self) {
        let base = base_alias(&self.model);
        self.rows = history().into_iter().rev().filter(|r| base_alias(Path::new(&r.model)) == base).collect();
    }

    fn run(&mut self) {
        let Some(inst) = self.inst.clone() else {
            self.error = "nessuna installazione di llama.cpp attiva".into();
            return;
        };
        self.error.clear();
        self.cancel = Arc::new(AtomicBool::new(false));
        let cancel = self.cancel.clone();
        let model = self.model.clone();
        let args = self.args.clone();
        self.task = Some(Task::spawn(move |tx| {
            let mut progress = |l: String| {
                let _ = tx.send(Msg::Line(l));
            };
            let rec = measure(&inst, &model, &args, &cancel, &mut progress);
            let _ = tx.send(Msg::Done(rec));
        }));
    }
}

impl Drop for Bench {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed); // leaving the screen stops llama-bench
    }
}

impl Screen for Bench {
    fn title(&self) -> String {
        "benchmark".into()
    }

    fn keys(&self, _ctx: &Ctx) -> Vec<Key> {
        if self.task.is_some() {
            vec![(KeyCode::Esc, "esc", "annulla")]
        } else {
            vec![(KeyCode::Char('r'), "r", "ripeti"), (KeyCode::Esc, "esc", "indietro")]
        }
    }

    fn draw(&mut self, f: &mut Frame, area: Rect, ctx: &Ctx) {
        let [top, hist] = Layout::vertical([Constraint::Length(7), Constraint::Fill(1)]).areas(area);
        let shown = self.inst.as_ref().map_or(String::new(), |i| shell_join(&command(i, &self.model, &self.args)));
        let mut t = vec![Line::from(sp(short(&shown), st(DIM))), Line::default()];
        t.push(if !self.error.is_empty() {
            Line::from(sp(format!("✖ {}", self.error), bold(ERR)))
        } else if self.task.is_some() {
            Line::from(vec![
                sp(format!("{} llama-bench in corso… ", spinner()), bold(WARN)),
                sp(clip(&self.last, 100), st(DIM)),
            ])
        } else {
            match self.rows.first() {
                Some(r) => Line::from(sp(format!("✓ pp512 {:.1} tok/s · tg128 {:.1} tok/s", r.pp, r.tg), bold(OK))),
                None => Line::default(),
            }
        });
        let blk = panel(&format!("benchmark · {}", fname(&ps(&self.model))), LOAD);
        let inner = blk.inner(top);
        f.render_widget(blk, top);
        f.render_widget(Paragraph::new(t).wrap(Wrap { trim: false }), inner);
        let blk = panel("storico · stesse quantizzazioni e impostazioni a confronto", SLOT);
        let inner = blk.inner(hist);
        f.render_widget(blk, hist);
        let rows = self
            .rows
            .iter()
            .map(|r| {
                let mine = r.model == ps(&self.model);
                vec![
                    cell(r.date.clone()),
                    sp(fname(&r.model).to_string(), if mine { bold(FG) } else { st(FG) }),
                    cell(r.install.clone()),
                    sp(if r.args.is_empty() { "(default)".into() } else { r.args.clone() }, st(DIM)),
                    sp(format!("{:.1}", r.pp), bold(SRV)),
                    sp(format!("{:.1}", r.tg), bold(SLOT)),
                ]
            })
            .collect();
        let head = ["data", "modello", "installazione", "parametri", "pp512 t/s", "tg128 t/s"];
        self.list.draw(f, inner, &head, rows, ctx);
    }

    fn on(&mut self, ev: &Ev, _ctx: &mut Ctx) -> Act {
        if self.list.on(ev).is_some() {
            return Act::None;
        }
        match key(ev) {
            Some(KeyCode::Char('r')) if self.task.is_none() => self.run(),
            _ => return Act::Pass,
        }
        Act::None
    }

    fn tick(&mut self, ctx: &mut Ctx) -> Act {
        let Some(t) = self.task.as_mut() else { return Act::None };
        let (msgs, done) = (t.poll(), t.done);
        for m in msgs {
            match m {
                Msg::Line(l) => self.last = l,
                Msg::Done(Ok(_)) => {
                    self.reload();
                    ctx.toast("✓ benchmark salvato nello storico", OK);
                    crate::tui::notify("benchmark finito");
                }
                Msg::Done(Err(e)) => {
                    crate::tui::notify("benchmark non riuscito");
                    self.error = e;
                }
            }
        }
        if done {
            self.task = None;
        }
        Act::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_args_and_results() {
        let p: Preset = [
            ("ngl", "all"),
            ("fa", "on"),
            ("ctk", "q8_0"),
            ("ctv", "f16"),
            ("threads", "-1"),
            ("device", "Vulkan0,Vulkan1"),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        assert_eq!(params(&p), ["-ngl", "999", "-fa", "on", "-ctk", "q8_0", "-dev", "Vulkan0/Vulkan1"]);
        let lines = [
            r#"{"n_prompt": 512, "n_gen": 0, "avg_ts": 22798.6}"#.to_string(),
            r#"{"n_prompt": 0, "n_gen": 128, "avg_ts": 775.6}"#.to_string(),
        ];
        assert_eq!(parse(&lines), Some((22798.6, 775.6)));
        assert_eq!(parse(&lines[..1]), None);
    }
}
