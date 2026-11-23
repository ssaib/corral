//! Preset editor for one model (inline edits, choice lists, memory estimate) and the launch.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;

use ratatui::Frame;
use ratatui::crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::Line;
use ratatui::widgets::{Paragraph, Wrap};

use super::widgets::*;
use super::{Act, Answer, Ctx, Ev, Key, Screen, Task};
use crate::preset::{self, Choices, FIELDS, Fit, Flags, Gguf, OT_PRESETS};
use crate::remote::{self, RpcLink};
use crate::server::{self, Launch};
use crate::ui::*;
use crate::{dashboard, help, installs};

fn sec_color(sec: &str) -> u32 {
    match sec {
        "base" => SIDE,
        "memoria" => LOAD,
        "server" => SLOT,
        "sampling" => SRV,
        "template" => WARN,
        "moe" => CYAN,
        "draft" => SIDE,
        _ => DIM,
    }
}

pub struct Config {
    model: PathBuf,
    pname: String, // the preset being edited (the model's selected one)
    preset: Preset,
    defaults: Preset,
    size: u64,
    list: List,
    editing: Option<TextInput>,
    near: Vec<PathBuf>, // mmproj files next to the model
    est: Option<Task<Result<Fit, String>>>,
    fitted: String,
    gguf: Option<Gguf>,
    adv: bool,           // advanced rows shown
    picks: Vec<PathBuf>, // files offered by the open file chooser
    hosts: Vec<String>,  // hosts offered by the open RPC chooser
}

fn is_lora(p: &Path) -> bool {
    fname(&ps(p)).to_lowercase().contains("lora")
}

impl Config {
    pub fn new(ctx: &Ctx, model: &Path, mmproj: Option<PathBuf>) -> Self {
        let mut preset = preset::get_preset(&ctx.state, model);
        if let Some(m) = mmproj {
            preset.insert("mmproj".into(), ps(&m));
        }
        if let Some(i) = &ctx.hw.inst {
            let srv = installs::server(i);
            thread::spawn(move || preset::supported_flags(&srv)); // warm the cache for draw()
        }
        let mut c = Self {
            model: model.into(),
            pname: preset::preset_name(&ctx.state, model),
            preset,
            defaults: preset::defaults(model),
            size: preset::model_size(model),
            editing: None,
            near: preset::mmprojs_near(model),
            est: None,
            fitted: String::new(),
            gguf: preset::gguf_info(model),
            adv: false,
            picks: vec![],
            hosts: vec![],
            list: List::new(0),
        };
        c.list.set_len(c.rows().len());
        c
    }

    /// Table rows: field indices, None = the show/hide row for the advanced ones.
    fn rows(&self) -> Vec<Option<usize>> {
        let mut v: Vec<Option<usize>> = (0..FIELDS.len()).filter(|&i| !self.hidden(i)).map(Some).collect();
        v.insert(v.len() - 1, None); // before "extra"
        v
    }

    /// Collapsed: "avanzate", and "moe" for a model the GGUF header says is dense.
    fn hidden(&self, i: usize) -> bool {
        let dense = self.gguf.as_ref().is_some_and(|g| g.experts == 0);
        !self.adv && (FIELDS[i].sec == "avanzate" || (FIELDS[i].sec == "moe" && dense))
    }

    fn field(&self) -> Option<usize> {
        self.rows().get(self.list.sel).copied().flatten()
    }

    fn toggle(&mut self) {
        let cur = self.rows().get(self.list.sel).copied().flatten();
        self.adv = !self.adv;
        let rows = self.rows();
        self.list.set_len(rows.len());
        self.list.sel =
            rows.iter().position(|r| r.is_some() && *r == cur).or(rows.iter().position(Option::is_none)).unwrap_or(0);
    }

    fn flags(ctx: &Ctx) -> Option<Arc<Flags>> {
        ctx.hw.inst.as_ref().and_then(|i| preset::cached_flags(&installs::server(i)))
    }

    /// File chooser for a path field: `files`, "nessuno", "percorso…".
    fn choose_file(&mut self, ctx: &mut Ctx, title: &str, files: Vec<PathBuf>, cur: &str) {
        let mut rows: Vec<_> = files
            .iter()
            .map(|p| vec![cell(fname(&ps(p)).to_string()), sp(size(preset::model_size(p)), st(DIM))])
            .collect();
        if files.is_empty() {
            rows.push(vec![sp("(nessun file trovato nelle cartelle modelli)", st(DIM)), cell("")]);
        }
        rows.push(vec![sp("nessuno", st(DIM)), cell("")]);
        rows.push(vec![sp("percorso…", st(DIM)), cell("")]);
        let off = usize::from(files.is_empty());
        let sel = files.iter().position(|p| ps(p) == cur).unwrap_or(files.len() + off);
        self.picks = files;
        ctx.ask("file", Modal::choose(title, &["file", "dimensione"], rows, sel));
    }

    fn set(&mut self, ctx: &mut Ctx, key: &str, v: String) {
        self.preset.insert(key.into(), v);
        self.store(ctx);
    }

    fn store(&mut self, ctx: &mut Ctx) {
        if let Err(e) = preset::put_preset(&mut ctx.state, &self.model, &self.preset) {
            ctx.error(e);
        }
    }

    fn edit(&mut self, row: usize, ctx: &mut Ctx) {
        let Some(i) = self.rows()[row] else { return self.toggle() };
        let f = &FIELDS[i];
        if !preset::supported(f, Self::flags(ctx).as_deref()) {
            return ctx
                .toast(format!("{}: non supportato da questa build di llama.cpp ({})", f.label, ctx.hw.version), WARN);
        }
        let cur = self.preset[f.key].clone();
        match f.choices {
            Choices::Free => self.editing = Some(TextInput::new(&cur)),
            Choices::List(choices) => {
                let rows =
                    choices.iter().map(|c| vec![cell(if c.is_empty() { "(default server)" } else { c })]).collect();
                let sel = choices.iter().position(|c| *c == cur).unwrap_or(0);
                ctx.ask("list", Modal::choose(f.label, &["valore"], rows, sel));
            }
            Choices::Mmproj => self.choose_file(ctx, f.label, self.near.clone(), &cur),
            Choices::Model => {
                let models = preset::local_models().into_iter().filter(|p| *p != self.model && !is_lora(p)).collect();
                self.choose_file(ctx, "modello draft (stessa famiglia/vocabolario, molto più piccolo)", models, &cur)
            }
            Choices::Lora => {
                let loras = preset::local_models().into_iter().filter(|p| is_lora(p)).collect();
                self.choose_file(ctx, f.label, loras, &cur)
            }
            Choices::Ot => {
                let mut rows: Vec<_> = OT_PRESETS.iter().map(|(v, d)| vec![cell(*d), sp(*v, st(DIM))]).collect();
                rows.push(vec![sp("nessuno", st(DIM)), cell("")]);
                rows.push(vec![sp("personalizzato…", st(DIM)), cell("pattern=buffer, separati da virgola")]);
                let sel = OT_PRESETS.iter().position(|(v, _)| *v == cur).unwrap_or(if cur.is_empty() {
                    OT_PRESETS.len()
                } else {
                    OT_PRESETS.len() + 1
                });
                ctx.ask("ot", Modal::choose(f.label, &["preset", "valore"], rows, sel));
            }
            Choices::Device => {
                let devs = &ctx.hw.devs;
                let mut rows: Vec<_> = devs
                    .iter()
                    .map(|d| {
                        vec![
                            cell(d.name.clone()),
                            sp(format!("{} · {}/{} MiB liberi", d.desc, d.free, d.total), st(DIM)),
                        ]
                    })
                    .collect();
                rows.push(vec![cell("auto"), sp("tutti i dispositivi", st(DIM))]);
                rows.push(vec![cell("none"), sp("solo CPU", st(DIM))]);
                rows.push(vec![cell("…"), sp("più dispositivi (es. Vulkan0,Vulkan1)", st(DIM))]);
                let sel = match cur.as_str() {
                    "" => devs.len(),
                    "none" => devs.len() + 1,
                    c => devs.iter().position(|d| d.name == c).unwrap_or(devs.len() + 2),
                };
                ctx.ask("device", Modal::choose("dispositivi", &["dispositivo", "memoria"], rows, sel));
            }
            Choices::Hosts => {
                let hosts = remote::config_hosts();
                let lend = || sp("presta le sue GPU (serve Corral lì: host remoti)", st(DIM));
                let mut rows: Vec<_> = hosts.iter().map(|h| vec![cell(h.clone()), lend()]).collect();
                rows.push(vec![cell("nessuno"), sp("solo le GPU di questo computer", st(DIM))]);
                rows.push(vec![cell("…"), sp("più host, separati da virgola", st(DIM))]);
                let sel = match cur.as_str() {
                    "" => hosts.len(),
                    c => hosts.iter().position(|h| h == c).unwrap_or(hosts.len() + 1),
                };
                self.hosts = hosts;
                ctx.ask("rpc", Modal::choose("GPU di altri host (RPC)", &["host", ""], rows, sel));
            }
        }
    }

    fn answer(&mut self, tag: &str, a: &Answer, ctx: &mut Ctx) {
        let f = &FIELDS[self.field().unwrap_or(0)];
        let n_dev = ctx.hw.devs.len();
        let n_files = self.picks.len().max(1); // the "(none found)" row stands in for an empty list
        match (tag, a) {
            ("list", Answer::Pick(i)) => {
                if let Choices::List(c) = f.choices {
                    self.set(ctx, f.key, c[*i].to_string());
                }
            }
            ("file", Answer::Pick(i)) if *i < self.picks.len() => self.set(ctx, f.key, ps(&self.picks[*i].clone())),
            ("file", Answer::Pick(i)) if *i < n_files => {}
            ("file", Answer::Pick(i)) if *i == n_files => self.set(ctx, f.key, String::new()),
            ("file", Answer::Pick(_)) => ctx.ask("path", Modal::prompt(f.label, "percorso del file .gguf", "")),
            ("path", Answer::Text(t)) if !t.is_empty() => {
                let p = expand(t);
                if p.is_file() {
                    self.set(ctx, f.key, ps(&p))
                } else {
                    ctx.error(format!("{} non esiste", p.display()))
                }
            }
            ("ot", Answer::Pick(i)) if *i < OT_PRESETS.len() => self.set(ctx, "ot", OT_PRESETS[*i].0.into()),
            ("ot", Answer::Pick(i)) if *i == OT_PRESETS.len() => self.set(ctx, "ot", String::new()),
            ("ot", Answer::Pick(_)) => {
                let cur = self.preset["ot"].clone();
                ctx.ask("otfree", Modal::prompt(f.label, r"es. blk\.(1[0-9])\.ffn_.*_exps=CPU", &cur))
            }
            ("otfree", Answer::Text(t)) => self.submit(t.clone(), ctx),
            ("rpc", Answer::Pick(i)) if *i < self.hosts.len() => {
                let host = self.hosts[*i].clone();
                self.set(ctx, "rpc", host);
            }
            ("rpc", Answer::Pick(i)) if *i == self.hosts.len() => self.set(ctx, "rpc", String::new()),
            ("rpc", Answer::Pick(_)) => {
                let cur = self.preset["rpc"].clone();
                let hint = "host di ~/.ssh/config separati da virgola, ognuno con Corral";
                ctx.ask("rpc_list", Modal::prompt("GPU di altri host (RPC)", hint, &cur));
            }
            ("rpc_list", Answer::Text(t)) => self.set(ctx, "rpc", t.clone()),
            ("device", Answer::Pick(i)) if *i < n_dev => {
                let name = ctx.hw.devs[*i].name.clone();
                self.set(ctx, "device", name);
            }
            ("device", Answer::Pick(i)) if *i == n_dev => self.set(ctx, "device", String::new()),
            ("device", Answer::Pick(i)) if *i == n_dev + 1 => self.set(ctx, "device", "none".into()),
            ("device", Answer::Pick(_)) => {
                let cur = self.preset["device"].clone();
                ctx.ask("devices", Modal::prompt("dispositivi", "nomi separati da virgola, es. Vulkan0,Vulkan1", &cur));
            }
            ("devices", Answer::Text(t)) => {
                let v = if t == "auto" { String::new() } else { t.replace(' ', "") };
                self.set(ctx, "device", v);
            }
            ("defaults", Answer::Yes) => {
                self.preset = self.defaults.clone();
                self.store(ctx);
                ctx.toast("preset riportato ai default", OK);
            }
            ("fit", Answer::Yes) => {
                let fitted = self.fitted.clone();
                preset::apply_fit(&mut self.preset, &fitted);
                self.store(ctx);
                ctx.toast("valori di llama-fit-params applicati", OK);
            }
            _ => {}
        }
    }

    fn submit(&mut self, v: String, ctx: &mut Ctx) {
        let Some(i) = self.field() else { return };
        let f = &FIELDS[i];
        let v = if v == "-" { f.default.to_string() } else { v.trim().to_string() }; // "-" = back to the default
        match preset::check(f.key, &v) {
            Ok(()) => {
                self.editing = None;
                self.set(ctx, f.key, v);
            }
            Err(e) => ctx.error(e),
        }
    }

    fn estimate(&mut self, ctx: &mut Ctx) {
        let Some(inst) = ctx.hw.inst.clone() else { return ctx.error("nessuna installazione attiva") };
        let (model, p) = (self.model.clone(), self.preset.clone());
        self.est = Some(Task::once(move || preset::fit_params(&inst, &model, &p)));
    }

    fn show_fit(&mut self, fit: Fit, ctx: &mut Ctx) {
        let mut body: Vec<Line<'static>> = fit.breakdown.iter().map(|l| Line::from(cell(l.clone()))).collect();
        body.push(Line::default());
        body.extend(fit.verdict.iter().map(|l| Line::from(sp(l.clone(), st(LOAD)))));
        for k in ["mmproj", "draft"] {
            if let Ok(m) = fs::metadata(&self.preset[k]) {
                body.push(Line::from(sp(format!("+ {k} {} non incluso nella stima", size(m.len())), st(WARN))));
            }
        }
        let hosts = preset::rpc_hosts(&self.preset);
        if !hosts.is_empty() {
            let msg = format!("stima di questo computer: all'avvio si aggiungono le GPU di {}", hosts.join(", "));
            body.push(Line::from(sp(msg, st(WARN))));
        }
        body.extend(fit.tail.iter().map(|l| Line::from(sp(l.clone(), st(ERR)))));
        let shown = if fit.fitted.is_empty() { "—".to_string() } else { fit.fitted.clone() };
        body.push(Line::from(vec![sp("parametri adattati: ", st(DIM)), sp(shown, bold(FG))]));
        let title = "stima memoria · llama-fit-params";
        if preset::apply_fit(&mut self.preset.clone(), &fit.fitted) {
            body.push(Line::default());
            body.push(Line::from(sp("applico questi valori al preset?", st(FG))));
            self.fitted = fit.fitted;
            ctx.ask("fit", Modal::confirm(title, body, true));
        } else {
            ctx.ask("info", Modal::info(title, if fit.ok { LOAD } else { ERR }, body));
        }
    }
}

impl Config {
    /// Under the table: what the selected parameter does, then what the keys do right now.
    fn help_panel(&self) -> (String, Vec<Line<'static>>) {
        let Some(i) = self.field() else {
            let what = "caricamento, batch, cache, LoRA, endpoint: invio o t per mostrarli o nasconderli";
            return (" parametri avanzati ".into(), vec![Line::from(sp(what, st(FG)))]);
        };
        let fl = &FIELDS[i];
        let short = help::of(fl.key).map_or("", |h| h.0);
        let def = if fl.default.is_empty() { "default del server" } else { fl.default };
        let dense = self.gguf.as_ref().is_some_and(|g| g.experts == 0);
        let status = if self.editing.is_some() {
            sp(format!("✎ invio conferma · esc annulla · ctrl+u svuota · - = {def}"), st(CYAN))
        } else if self.est.is_some() {
            sp(format!("{} llama-fit-params: calcolo memoria…", spinner()), st(LOAD))
        } else if fl.sec == "moe" && dense {
            sp("questo modello è denso (non MoE): qui non cambia nulla", st(WARN))
        } else {
            sp(format!("default: {def} · ? spiegazione · grassetto = cambiato, grigio = non in questa build"), st(DIM))
        };
        let title = format!(" {} · {} ", fl.label, help::flag(fl));
        (title, vec![Line::from(sp(short, st(FG))), Line::from(status)])
    }
}

impl Screen for Config {
    fn title(&self) -> String {
        format!("configura · {} · {}", fname(&ps(&self.model)), self.pname)
    }

    fn keys(&self, _ctx: &Ctx) -> Vec<Key> {
        if self.editing.is_some() {
            return vec![(KeyCode::Enter, "invio", "conferma"), (KeyCode::Esc, "esc", "annulla")];
        }
        vec![
            (KeyCode::Enter, "invio", "modifica"),
            (KeyCode::Char('a'), "a", "avvia server"),
            (KeyCode::Char('v'), "v", "stima VRAM"),
            (KeyCode::Char('b'), "b", "benchmark"),
            (KeyCode::Char('o'), "o", "ottimizza"),
            (KeyCode::Char('?'), "?", "aiuto"),
            (KeyCode::Char('t'), "t", if self.adv { "nascondi avanzate" } else { "avanzate" }),
            (KeyCode::Char('d'), "d", "ripristina default"),
            (KeyCode::Esc, "esc", "indietro"),
        ]
    }

    fn draw(&mut self, f: &mut Frame, area: Rect, ctx: &Ctx) {
        let mut shown = match &ctx.hw.inst {
            Some(i) => shell_join(&preset::build_cmd(&installs::server(i), &self.model, &self.preset, None)),
            None => "—".into(),
        };
        let hosts = preset::rpc_hosts(&self.preset);
        if !hosts.is_empty() {
            shown += &format!(" --rpc <tunnel verso {}>", hosts.join(", ")); // added at launch
        }
        if !self.preset["api_key"].is_empty() {
            shown = shown.replace(&self.preset["api_key"], "•••");
        }
        let shown = short(&shown);
        let n = wrapped(&[Line::from(shown.as_str())], area.width.saturating_sub(4));
        let cmd_h = n.min(6) + 2 + u16::from(n > 1); // word wrap can take one line more than the estimate
        let (help_title, help_lines) = self.help_panel();
        let help_h = wrapped(&help_lines, area.width.saturating_sub(4)).clamp(2, 3) + 2;
        let [info, table, help_area, cmd] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Fill(1),
            Constraint::Length(help_h),
            Constraint::Length(cmd_h),
        ])
        .areas(area);
        let gpus: Vec<String> =
            ctx.hw.devs.iter().map(|x| format!("{} {}/{} MiB liberi", x.name, x.free, x.total)).collect();
        let gpus = if gpus.is_empty() { "nessuna GPU".into() } else { gpus.join(" · ") };
        let inst = ctx
            .hw
            .inst
            .as_ref()
            .map_or("nessuna installazione".into(), |i| format!("llama.cpp {} ({})", i.name, i.backend));
        let shape = match &self.gguf {
            Some(g) if g.experts > 0 => format!("  {} · {} layer · MoE {} esperti", g.arch, g.layers, g.experts),
            Some(g) => format!("  {} · {} layer", g.arch, g.layers),
            None => String::new(),
        };
        f.render_widget(
            Paragraph::new(vec![
                Line::from(vec![
                    sp(short(&ps(&self.model)), bold(FG)),
                    sp(format!("  {}", size(self.size)), st(DIM)),
                    sp(shape, st(CYAN)),
                ]),
                Line::from(sp(format!("{inst} · {gpus}"), st(DIM))),
            ]),
            info,
        );
        let flags = Self::flags(ctx);
        let changed = |i: usize| self.defaults.get(FIELDS[i].key) != Some(&self.preset[FIELDS[i].key]);
        let hidden: Vec<usize> = (0..FIELDS.len()).filter(|&i| self.hidden(i)).collect();
        let n_hidden = hidden.len();
        let n_changed = hidden.iter().filter(|&&i| changed(i)).count();
        let rows = self
            .rows()
            .into_iter()
            .map(|r| {
                let Some(i) = r else {
                    let what = if self.adv {
                        "▾ nascondi avanzate".into()
                    } else {
                        format!("▸ altri {n_hidden} parametri ({n_changed} cambiati)")
                    };
                    return vec![sp("…", st(DIM)), sp(what, st(CYAN)), sp("invio o t", st(DIM))];
                };
                let fl = &FIELDS[i];
                let v = &self.preset[fl.key];
                let mut shown = match (fl.key, v.is_empty()) {
                    ("api_key", true) => "—".into(),
                    ("api_key", false) => "••••••".into(),
                    (_, true) => "(default server)".into(),
                    ("mmproj" | "draft" | "lora", false) => short(v),
                    _ => v.clone(),
                };
                let ok = preset::supported(fl, flags.as_deref());
                let style = if matches!(fl.key, "mmproj" | "draft" | "lora") && !v.is_empty() && !Path::new(v).exists()
                {
                    st(ERR)
                } else if ok && changed(i) {
                    bold(FG)
                } else {
                    st(DIM)
                };
                if !ok {
                    shown = format!("{shown} · non in questa build");
                }
                let label = if ok { cell(fl.label) } else { sp(fl.label, st(DIM)) };
                vec![sp(fl.sec, st(sec_color(fl.sec))), label, sp(shown, style)]
            })
            .collect();
        self.list.draw(f, table, &["sezione", "parametro", "valore"], rows, ctx);
        if let Some(input) = self.editing.as_mut()
            && let Some(r) = self.list.cell_rect(self.list.sel, 2)
        {
            input.draw(f, r, true);
        }
        let help_block = block(help_title, bold(CYAN), DIM);
        f.render_widget(Paragraph::new(help_lines).wrap(Wrap { trim: true }).block(help_block), help_area);
        f.render_widget(
            Paragraph::new(sp(shown, st(LOAD))).wrap(Wrap { trim: false }).block(panel("comando", DIM)),
            cmd,
        );
    }

    fn on(&mut self, ev: &Ev, ctx: &mut Ctx) -> Act {
        if let Ev::Answer(tag, a) = ev {
            self.answer(tag, a, ctx);
            return Act::None;
        }
        if let Some(input) = self.editing.as_mut() {
            return match input.on(ev) {
                Some(Edit::Submit) => {
                    let v = input.value.clone();
                    self.submit(v, ctx);
                    Act::None
                }
                Some(Edit::Cancel) => {
                    self.editing = None;
                    Act::None
                }
                Some(Edit::Changed) => Act::None,
                // a click elsewhere ends the edit; other keys (Ctrl+C) go to the defaults
                None if click(ev).is_some() => {
                    self.editing = None;
                    self.on(ev, ctx)
                }
                None if matches!(ev, Ev::Key(_)) => Act::Pass,
                None => Act::None,
            };
        }
        if let Some(Hit::Open(i)) = self.list.on(ev) {
            self.edit(i, ctx);
            return Act::None;
        }
        match key(ev) {
            Some(KeyCode::Char('a')) => match launch(ctx, &self.model, &self.preset) {
                Ok(s) => Act::Push(s),
                Err(e) => {
                    ctx.error(e);
                    Act::None
                }
            },
            Some(KeyCode::Char('v')) if self.est.is_none() => {
                self.estimate(ctx);
                Act::None
            }
            Some(KeyCode::Char('t')) => {
                self.toggle();
                Act::None
            }
            Some(KeyCode::Char('b')) => Act::Push(Box::new(crate::bench::Bench::new(ctx, &self.model, &self.preset))),
            Some(KeyCode::Char('o')) => Act::Push(Box::new(crate::tune::Tune::new(ctx, &self.model))),
            Some(KeyCode::Char('?')) => {
                if let Some(i) = self.field() {
                    ctx.ask("info", Modal::info(FIELDS[i].label, CYAN, help::explain(&FIELDS[i])));
                }
                Act::None
            }
            Some(KeyCode::Char('d')) => {
                ctx.ask("defaults", Modal::ask("riporto tutti i parametri ai valori di default?", false));
                Act::None
            }
            _ => Act::Pass,
        }
    }

    /// Back from «ottimizza», which may have saved presets and selected one of them.
    fn resume(&mut self, ctx: &mut Ctx) {
        let mmproj = self.preset["mmproj"].clone(); // may come from the caller, not saved yet
        self.pname = preset::preset_name(&ctx.state, &self.model);
        self.preset = preset::get_preset(&ctx.state, &self.model);
        self.preset.insert("mmproj".into(), mmproj);
    }

    fn tick(&mut self, ctx: &mut Ctx) -> Act {
        if let Some(t) = self.est.as_mut()
            && let Some(r) = t.poll().pop()
        {
            self.est = None;
            match r {
                Ok(fit) => self.show_fit(fit, ctx),
                Err(e) => ctx.error(format!("llama-fit-params: {e}")),
            }
        }
        Act::None
    }
}

/// Start llama-server for `model` under the dashboard (the port moves if it's taken). A preset
/// that uses other hosts' GPUs first links them (`Lending`).
pub fn launch(ctx: &mut Ctx, model: &Path, p: &Preset) -> Result<Box<dyn Screen>, String> {
    let inst = ctx.hw.inst.clone().ok_or("nessuna installazione di llama.cpp attiva")?;
    preset::mark_used(&mut ctx.state, model);
    ctx.save();
    let launch = server::model_launch(&inst, &ctx.hw.version, model, p);
    let hosts = preset::rpc_hosts(p);
    if !hosts.is_empty() {
        return Ok(Box::new(Lending::new(launch, hosts)));
    }
    for note in &launch.notes {
        ctx.toast(format!("⚠ {note}"), WARN);
    }
    Ok(Box::new(dashboard::Live::start(ctx, &launch, vec![])?))
}

/// Before a server that uses other hosts' GPUs: their RPC servers, each through ssh (see
/// `remote::rpc_links`), then the dashboard. Esc gives up (the links close again).
pub struct Lending {
    launch: Launch,
    hosts: Vec<String>,
    task: Task<Result<Vec<RpcLink>, String>>,
}

impl Lending {
    fn new(launch: Launch, hosts: Vec<String>) -> Self {
        let wanted = hosts.clone();
        let task = Task::once(move || remote::rpc_links(&wanted));
        Self { launch, hosts, task }
    }
}

impl Screen for Lending {
    fn title(&self) -> String {
        "GPU degli host".into()
    }

    fn keys(&self, _ctx: &Ctx) -> Vec<Key> {
        vec![(KeyCode::Esc, "esc", "annulla")]
    }

    fn draw(&mut self, f: &mut Frame, area: Rect, _ctx: &Ctx) {
        let t = vec![
            Line::from(sp(format!("{} collego le GPU di {}…", spinner(), self.hosts.join(", ")), bold(WARN))),
            Line::default(),
            Line::from(sp(
                "su ogni host Corral avvia il server RPC di llama.cpp, raggiungibile solo dal tunnel ssh;",
                st(DIM),
            )),
            Line::from(sp("poi llama-server parte qui e ci carica una parte del modello", st(DIM))),
        ];
        f.render_widget(Paragraph::new(t).wrap(Wrap { trim: true }), area);
    }

    fn on(&mut self, _ev: &Ev, _ctx: &mut Ctx) -> Act {
        Act::Pass // Esc goes back; the links that open later close at once
    }

    fn tick(&mut self, ctx: &mut Ctx) -> Act {
        let Some(result) = self.task.poll().pop() else {
            return if self.task.done { Act::Pop } else { Act::None };
        };
        let started = result.and_then(|links| {
            server::add_rpc(&mut self.launch, &links, &ctx.hw.version)?;
            for note in &self.launch.notes {
                ctx.toast(format!("⚠ {note}"), WARN);
            }
            dashboard::Live::start(ctx, &self.launch, links)
        });
        match started {
            Ok(live) => Act::Replace(Box::new(live)),
            Err(e) => {
                ctx.error(e);
                Act::Pop
            }
        }
    }
}

/// `corral -y`: the last model with its preset.
pub fn launch_last(ctx: &mut Ctx) -> Result<Box<dyn Screen>, String> {
    let last = ctx.state.last_model.clone().map(PathBuf::from).filter(|p| p.exists());
    let last = last.ok_or("niente da avviare: nessun ultimo modello usato")?;
    let p = preset::get_preset(&ctx.state, &last);
    launch(ctx, &last, &p)
}
