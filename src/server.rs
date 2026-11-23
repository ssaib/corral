//! llama-server processes run by Corral (logic only; the screens are in `dashboard`).
//!
//! The server writes straight into its log file and a thread tails that file, so the dashboard
//! is the same for a server started now and for one re-adopted after Corral restarted. Each
//! server has its own process group (Windows: its own hidden console): the terminal's Ctrl+C or
//! hang-up never reaches it, and it can outlive Corral when asked to ("lascia attivo
//! all'uscita"). A state file per server in <data>/run/ lets the next Corral find it again.
//! Limit: an adopted server is not our child, so its end is noticed when /health stops
//! answering and its exit code is unknown.
//!
//! A server on another computer (`Server::remote`) is the same to the screens: its HTTP API
//! comes through an ssh tunnel, its log and GPUs through the same ssh connection (see `remote`).

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent::Served;
use crate::dashboard::{Dashboard, Info};
use crate::installs::{self, Install};
use crate::remote::{self, Link, RpcLink, Running};
use crate::router;
use crate::ui::*;

// ponytail: one fixed grace per server on a quit signal, sized for Windows' ~5 s console-close
// budget (there it is a kill anyway); servers stop one after another, so N of them can take N x 3 s
const QUIT_GRACE: Duration = Duration::from_secs(3);

pub fn run_dir() -> PathBuf {
    BASE.join("run")
}

/// <data>/run/<pid>.json
#[derive(Serialize, Deserialize)]
pub struct RunFile {
    pub pid: u32,
    pub base: String,
    pub api_key: String,
    pub log: PathBuf,
    pub started: u64, // unix time
    pub keep: bool,
    pub info: Info,
}

// -- live metrics ------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct Slot {
    pub id: usize,
    pub n_ctx: u64,
    pub used: u64,    // tokens in the slot's context (prompt + generated while busy)
    pub decoded: u64, // tokens generated so far by the running task
    pub busy: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Gpu {
    /// the computer it's in, for one lent over RPC ("" = this one, or the one asked)
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub host: String,
    pub name: String,
    pub used: u64, // MiB
    pub total: u64,
    // what the driver reports besides the memory (None = not reported)
    pub busy: Option<u32>, // utilization, %
    pub temp: Option<u32>, // °C
    pub watts: Option<u32>,
}

impl Gpu {
    /// "87% · 65°C · 310 W": the parts the driver reports ("" = none).
    pub fn load(&self) -> String {
        let mut parts = vec![];
        if let Some(b) = self.busy {
            parts.push(format!("{b}%"));
        }
        if let Some(t) = self.temp {
            parts.push(format!("{t}°C"));
        }
        if let Some(w) = self.watts {
            parts.push(format!("{w} W"));
        }
        parts.join(" · ")
    }
}

/// One poll of the server's HTTP API (and of the GPUs).
#[derive(Clone, Default)]
pub struct Metrics {
    pub health: String, // "ok", "caricamento", "" = no answer
    pub slots: Vec<Slot>,
    pub prom: BTreeMap<String, f64>, // /metrics without the "llamacpp:" prefix (empty = not enabled)
    pub gpus: Vec<Gpu>,
    pub vram: Option<u64>, // MiB used by this server's process (nvidia-smi)
    /// router mode: its models (GET /models) and the /metrics of the loaded ones
    pub models: Vec<crate::router::Status>,
}

fn get(base: &str, key: &str, path: &str) -> ureq::Request {
    let r = HTTP.get(&format!("{base}{path}")).timeout(Duration::from_secs(2));
    if key.is_empty() { r } else { r.set("Authorization", &format!("Bearer {key}")) }
}

/// POST a JSON body to the local server; the error carries the server's message.
pub fn post_json(base: &str, key: &str, path: &str, body: Value, timeout: Duration) -> Result<Value, String> {
    let mut r = HTTP.post(&format!("{base}{path}")).timeout(timeout);
    if !key.is_empty() {
        r = r.set("Authorization", &format!("Bearer {key}"));
    }
    match r.send_json(body) {
        Ok(resp) => resp.into_json().map_err(|e| e.to_string()),
        Err(ureq::Error::Status(code, resp)) => {
            let text = resp.into_string().unwrap_or_default();
            let v: Option<Value> = serde_json::from_str(&text).ok();
            let msg = v.as_ref().and_then(|v| v["error"]["message"].as_str()).map_or(text.clone(), String::from);
            Err(format!("HTTP {code}: {}", clip(msg.trim(), 200)))
        }
        Err(e) => Err(e.to_string()),
    }
}

/// Prometheus text: `llamacpp:name value` lines.
pub fn parse_prom(text: &str) -> BTreeMap<String, f64> {
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| {
            let (k, v) = l.split_once(' ')?;
            Some((k.trim_start_matches("llamacpp:").to_string(), v.trim().parse().ok()?))
        })
        .collect()
}

/// /slots: `n_past` where the build reports it, else prompt + decoded tokens of the running task.
pub fn parse_slots(v: &Value) -> Vec<Slot> {
    let n = |v: &Value, k: &str| v[k].as_u64().unwrap_or(0);
    v.as_array()
        .into_iter()
        .flatten()
        .map(|s| {
            let busy = s["is_processing"].as_bool().unwrap_or(false);
            let decoded = if busy { n(&s["next_token"][0], "n_decoded") } else { 0 };
            let running = if busy { n(s, "n_prompt_tokens") + decoded } else { 0 };
            Slot {
                id: n(s, "id") as usize,
                n_ctx: n(s, "n_ctx"),
                used: s["n_past"].as_u64().unwrap_or(running),
                decoded,
                busy,
            }
        })
        .collect()
}

/// nvidia-smi's CSV for --query-gpu=name,memory.used,memory.total,utilization.gpu,
/// temperature.gpu,power.draw (a field it can't read is "[N/A]").
pub fn parse_nvidia(csv: &str) -> Vec<Gpu> {
    let mut out = vec![];
    for line in csv.lines() {
        // from the right: the name is the only field that could hold a comma
        let fields: Vec<&str> = line.rsplitn(6, ',').map(str::trim).collect();
        if let [watts, temp, busy, total, used, name] = fields[..] {
            out.push(Gpu {
                host: String::new(),
                name: name.into(),
                used: used.parse().unwrap_or(0),
                total: total.parse().unwrap_or(0),
                busy: busy.parse().ok(),
                temp: temp.parse().ok(),
                watts: watts.parse::<f64>().ok().map(|w| w.round() as u32),
            });
        }
    }
    out
}

/// An amdgpu card from its sysfs folder (/sys/class/drm/cardN/device); None = not amdgpu.
#[cfg(target_os = "linux")]
fn amd_gpu(name: &str, device: &Path) -> Option<Gpu> {
    let number = |p: PathBuf| fs::read_to_string(p).ok()?.trim().parse::<u64>().ok();
    let used = number(device.join("mem_info_vram_used"))?;
    let total = number(device.join("mem_info_vram_total"))?;
    // the sensors: millidegrees and microwatts (power1_average on older kernels)
    let hwmon = fs::read_dir(device.join("hwmon")).into_iter().flatten().flatten().next().map(|e| e.path());
    let sensor = |file: &str| hwmon.as_ref().and_then(|h| number(h.join(file)));
    let watts = sensor("power1_input").or_else(|| sensor("power1_average"));
    Some(Gpu {
        host: String::new(),
        name: format!("amdgpu {name}"),
        used: used >> 20,
        total: total >> 20,
        busy: number(device.join("gpu_busy_percent")).map(|b| b as u32),
        temp: sensor("temp1_input").map(|t| (t / 1000) as u32),
        watts: watts.map(|w| (w / 1_000_000) as u32),
    })
}

/// GPUs: nvidia-smi (Linux/Windows) and amdgpu's sysfs counters (Linux); empty elsewhere. Also
/// the MiB that process `pid` holds (NVIDIA only).
pub(crate) fn gpus(pid: u32) -> (Vec<Gpu>, Option<u64>) {
    let mut out = vec![];
    let mut mine = None;
    if which("nvidia-smi").is_some() {
        let q = |a: &str| {
            capture(Command::new("nvidia-smi").args([a, "--format=csv,noheader,nounits"]), Duration::from_secs(5))
                .map_or(String::new(), |o| o.stdout)
        };
        out = parse_nvidia(&q("--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw"));
        let apps = q("--query-compute-apps=pid,used_memory");
        let used = apps.lines().filter_map(|l| l.split_once(',')).filter(|(p, _)| p.trim() == pid.to_string());
        mine = used.map(|(_, m)| m.trim().parse::<u64>().unwrap_or(0)).reduce(|a, b| a + b);
    }
    #[cfg(target_os = "linux")]
    for e in fs::read_dir("/sys/class/drm").into_iter().flatten().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let card = name.starts_with("card") && !name.contains('-'); // card1, not its card1-DP-1 outputs
        if card && let Some(g) = amd_gpu(&name, &e.path().join("device")) {
            out.push(g);
        }
    }
    (out, mine)
}

/// GPUs a host lends over RPC: its name, and what its Corral last sent (see `remote::RpcLink`).
type Lent = (String, Arc<Mutex<(Vec<Gpu>, Option<u64>)>>);

/// This computer's GPUs (`gpus`) and those the hosts in `lent` lend, named after their host; the
/// MiB the server holds count the RPC servers' too.
fn all_gpus(pid: u32, lent: &[Lent]) -> (Vec<Gpu>, Option<u64>) {
    let (mut all, mut mine) = gpus(pid);
    for (host, shared) in lent {
        let (theirs, held) = lock(shared).clone();
        for g in theirs {
            all.push(Gpu { host: host.clone(), ..g });
        }
        if let Some(h) = held {
            mine = Some(mine.unwrap_or(0) + h);
        }
    }
    (all, mine)
}

/// What /health says: "ok", "caricamento" (the model is loading), "errore", "" (no answer).
pub fn health_of(base: &str, key: &str) -> &'static str {
    match get(base, key, "/health").call() {
        Ok(_) => "ok",
        Err(ureq::Error::Status(503, _)) => "caricamento",
        Err(ureq::Error::Status(..)) => "errore",
        Err(_) => "",
    }
}

/// Every second: /health, /slots, /metrics (a router: /models and each loaded model's /metrics);
/// every 2 s the GPUs from `gpus_now` (this computer's, or the last a host sent). Stops when `on`
/// turns false.
fn poller(
    base: String,
    key: String,
    router: bool,
    gpus_now: impl Fn() -> (Vec<Gpu>, Option<u64>),
    on: Arc<AtomicBool>,
    tx: Sender<Metrics>,
) {
    let mut gpu = vec![];
    let mut vram = None;
    let mut last_gpu: Option<Instant> = None;
    // an endpoint this server doesn't offer (404/501: disabled or not in the build): stop asking
    let mut has_slots = true;
    let mut has_metrics = true;
    while on.load(Ordering::Relaxed) {
        let t = Instant::now();
        let health = health_of(&base, &key);
        let mut m = Metrics { health: health.into(), ..Metrics::default() };
        if health == "ok" && router {
            if let Ok(r) = get(&base, &key, "/models").call() {
                m.models = r.into_json().map(|v| crate::router::parse_models(&v)).unwrap_or_default();
            }
            // autoload=false: asking an unloaded model's metrics would load it
            for s in m.models.iter_mut().filter(|s| s.state == "loaded") {
                if let Ok(r) = get(&base, &key, &format!("/metrics?model={}&autoload=false", s.id)).call() {
                    s.prom = parse_prom(&r.into_string().unwrap_or_default());
                }
            }
        } else if health == "ok" {
            if has_slots {
                match get(&base, &key, "/slots").call() {
                    Ok(r) => m.slots = r.into_json().map(|v| parse_slots(&v)).unwrap_or_default(),
                    Err(ureq::Error::Status(404 | 501, _)) => has_slots = false,
                    Err(_) => {}
                }
            }
            if has_metrics {
                match get(&base, &key, "/metrics").call() {
                    Ok(r) => m.prom = parse_prom(&r.into_string().unwrap_or_default()),
                    Err(ureq::Error::Status(404 | 501, _)) => has_metrics = false,
                    Err(_) => {}
                }
            }
        }
        if last_gpu.is_none_or(|l| l.elapsed() >= Duration::from_secs(2)) {
            (gpu, vram) = gpus_now();
            last_gpu = Some(Instant::now());
        }
        (m.gpus, m.vram) = (gpu.clone(), vram);
        if tx.send(m).is_err() {
            return;
        }
        thread::sleep(Duration::from_secs(1).saturating_sub(t.elapsed()));
    }
}

/// Lines appended to `path` (a partial last line waits for its newline). Skips the "# …" header.
pub(crate) fn tail(path: PathBuf, on: Arc<AtomicBool>, tx: Sender<String>) {
    let Ok(f) = File::open(&path) else { return };
    let mut reader = BufReader::new(f);
    let mut buf = vec![];
    let mut first = true;
    while on.load(Ordering::Relaxed) {
        match reader.read_until(b'\n', &mut buf) {
            Ok(_) if buf.ends_with(b"\n") => {
                let line = String::from_utf8_lossy(&buf).trim_end().to_string();
                buf.clear();
                let header = first && line.starts_with("# "); // Corral's "# date  command" line
                first = false;
                if !header && tx.send(line).is_err() {
                    return;
                }
            }
            // end of the file, or half a line: wait for the server to write more
            Ok(_) => thread::sleep(Duration::from_millis(100)),
            Err(_) => return,
        }
    }
}

// -- command lines --------------------------------------------------------------------------------

/// The value given to `flag` (`--flag v` or `--flag=v`); the last one wins, as in llama-server.
pub fn flag_value(cmd: &[String], flag: &str) -> Option<String> {
    let mut value = None;
    for (i, arg) in cmd.iter().enumerate() {
        if arg == flag {
            value = cmd.get(i + 1).cloned();
        } else if let Some(v) = arg.strip_prefix(flag).and_then(|rest| rest.strip_prefix('=')) {
            value = Some(v.to_string());
        }
    }
    value
}

/// Move `--api-key KEY` from the command line to LLAMA_API_KEY: `ps` shows every user the
/// command line of every process, the environment only to its owner.
pub fn key_to_env(cmd: &mut Vec<String>, env: &mut Vec<(String, String)>) {
    let Some(key) = flag_value(cmd, "--api-key") else { return };
    let old = std::mem::take(cmd); // rebuilt below without the key
    let mut args = old.into_iter();
    while let Some(arg) = args.next() {
        if arg == "--api-key" {
            args.next(); // its value goes too
        } else if !arg.starts_with("--api-key=") {
            cmd.push(arg);
        }
    }
    env.push(("LLAMA_API_KEY".into(), key));
}

/// Other computers can reach it (it doesn't listen on a loopback address) and it asks no API key.
pub fn exposed_without_key(cmd: &[String]) -> bool {
    let host = flag_value(cmd, "--host").unwrap_or_else(|| "127.0.0.1".into());
    let local = host == "localhost" || host == "::1" || host.starts_with("127.") || host.ends_with(".sock");
    let has_key = flag_value(cmd, "--api-key").is_some() || flag_value(cmd, "--api-key-file").is_some();
    !local && !has_key
}

// -- what to run ----------------------------------------------------------------------------------

/// A server ready to start, the same for the screens and for `corral start`.
#[derive(Clone)]
pub struct Launch {
    pub cmd: Vec<String>,
    pub env: Vec<(String, String)>,
    pub info: Info,
    /// warnings for the user: the port moved, other computers can reach it without a key
    pub notes: Vec<String>,
}

/// The first free port from `port` on `host`, and a note when it isn't `port` itself.
fn free_port_note(host: &str, port: &str) -> Option<(u16, String)> {
    let port: u16 = port.parse().ok()?;
    let free = crate::preset::free_port(host, port)?;
    let note = format!("porta {port} occupata su {host}: uso la {free} (solo per questo avvio)");
    (free != port).then_some((free, note))
}

/// What the dashboard calls the endpoint: 0.0.0.0 is reached as localhost.
fn endpoint(host: &str, port: &str) -> String {
    let host = if host == "0.0.0.0" { "localhost" } else { host };
    format!("{host}:{port}")
}

/// llama-server for `model` with the preset values `p` (the port moves if it's taken).
/// `version`: the install's llama.cpp version, for the dashboard ("" = not known yet).
pub fn model_launch(inst: &Install, version: &str, model: &Path, p: &Preset) -> Launch {
    let mut p = p.clone();
    let mut notes = vec![];
    if let Some((free, note)) = free_port_note(&p["host"], &p["port"]) {
        p.insert("port".into(), free.to_string());
        notes.push(note);
    }
    let or = |v: &str, alt: &str| if v.is_empty() { alt.to_string() } else { v.to_string() };
    let ctx_size = if p["ctx"] == "0" { "modello".into() } else { p["ctx"].clone() };
    let slots = if p["np"] == "-1" { "auto".into() } else { p["np"].clone() };
    let info = Info {
        alias: p["alias"].clone(),
        model: ps(model),
        mmproj: p["mmproj"].clone(),
        endpoint: endpoint(&p["host"], &p["port"]),
        install: inst.name.clone(),
        backend: inst.backend.clone(),
        version: or(version, "…"),
        settings: vec![
            ("ngl".into(), p["ngl"].clone()),
            ("ctx".into(), ctx_size),
            ("fit".into(), p["fit"].clone()),
            ("flash".into(), p["fa"].clone()),
            ("kv".into(), format!("{}/{}", p["ctk"], p["ctv"])),
            ("slot".into(), slots),
            ("device".into(), or(&p["device"], "auto")),
        ],
        router: false,
        host: String::new(),
    };
    let srv = installs::server(inst);
    crate::preset::supported_flags(&srv); // blocking once per build: build_cmd then drops what it doesn't know
    let cmd = crate::preset::build_cmd(&srv, model, &p, None);
    if exposed_without_key(&cmd) {
        notes.push("raggiungibile da altri computer senza API key: impostane una nel preset".into());
    }
    Launch { cmd, env: vec![], info, notes }
}

/// The router of multi-model profile `name` (the port moves if it's taken); writes its INI.
pub fn profile_launch(state: &State, inst: &Install, version: &str, name: &str) -> Result<Launch, String> {
    let mut prof = state.routers.get(name).cloned().ok_or(format!("profilo «{name}» non trovato"))?;
    let mut notes = vec![];
    let o = router::opts(&prof);
    if let Some((free, note)) = free_port_note(&o["host"], &o["port"]) {
        prof.opts.insert("port".into(), free.to_string());
        notes.push(note);
    }
    let cmd = router::prepare(state, name, &prof, inst)?;
    let o = router::opts(&prof);
    let info = Info {
        alias: name.into(),
        model: router::count(prof.models.len()),
        mmproj: String::new(),
        endpoint: endpoint(&o["host"], &o["port"]),
        install: inst.name.clone(),
        backend: inst.backend.clone(),
        version: if version.is_empty() { "…".into() } else { version.into() },
        settings: vec![("modelli".into(), prof.models.len().to_string()), ("insieme".into(), o["models_max"].clone())],
        router: true,
        host: String::new(),
    };
    if exposed_without_key(&cmd) {
        notes.push("raggiungibile da altri computer senza API key: impostane una nelle opzioni del profilo".into());
    }
    Ok(Launch { cmd, env: router::env(), info, notes })
}

/// Point `launch` at the GPUs that `links` lend: --rpc with their tunnels here, an "rpc" line in
/// the dashboard, and a note for a host with another llama.cpp build than this one's `version`.
/// Err: this build of llama-server has no RPC.
pub fn add_rpc(launch: &mut Launch, links: &[RpcLink], version: &str) -> Result<(), String> {
    if links.is_empty() {
        return Ok(());
    }
    let flags = crate::preset::cached_flags(Path::new(&launch.cmd[0]));
    if flags.as_ref().is_some_and(|f| !f.contains("--rpc")) {
        return Err("questa build di llama.cpp non ha l'RPC: usa una release, o ricompilala (installazioni)".into());
    }
    // an explicit list of devices leaves out every device it doesn't name, the lent ones too:
    // they are RPC0, RPC1, … in the order of --rpc
    let lent: usize = links.iter().map(|l| l.devices.len()).sum();
    if let Some(i) = launch.cmd.iter().position(|a| a == "-dev")
        && let Some(list) = launch.cmd.get_mut(i + 1)
        && list != "none"
    {
        for n in 0..lent {
            list.push_str(&format!(",RPC{n}"));
        }
    }
    let endpoints: Vec<String> = links.iter().map(|l| format!("127.0.0.1:{}", l.port)).collect();
    launch.cmd.push("--rpc".into());
    launch.cmd.push(endpoints.join(","));
    let hosts: Vec<&str> = links.iter().map(|l| l.host.as_str()).collect();
    launch.info.settings.push(("rpc".into(), hosts.join(", ")));
    // "b12347 · 8ca5a80" -> "b12347"
    let build = |v: &str| v.split(" ·").next().unwrap_or(v).trim().to_string();
    let known = |v: &str| !matches!(build(v).as_str(), "" | "…" | "?");
    for l in links {
        if known(version) && known(&l.version) && build(&l.version) != build(version) {
            let (there, here) = (build(&l.version), build(version));
            launch
                .notes
                .push(format!("{}: llama.cpp {there} sull'host, {here} qui: l'RPC vuole la stessa versione", l.host));
        }
    }
    Ok(())
}

// -- processes ------------------------------------------------------------------------------------

/// Own process group / hidden console: independent of the terminal and of the other servers.
pub(crate) fn detach(c: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        c.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x200;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        c.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
    }
}

#[cfg(unix)]
pub(crate) fn pid_alive(pid: u32) -> bool {
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

/// Screens find a server by this id: unlike the pid it stays the same across an automatic restart.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Automatic restarts after a crash: at most this many within the window, then it stays down.
pub const RESTARTS: usize = 3;
pub const RESTART_WINDOW: Duration = Duration::from_secs(300);

/// What `Server::restart` did.
pub enum Restart {
    /// the same launch is running again, with the same id (the screens follow it)
    Started(Box<Server>),
    /// it crashed RESTARTS times within RESTART_WINDOW: it stays down
    GaveUp,
    Failed(String),
}

/// What a server on another computer has instead of a process here.
struct Remote {
    host: String,
    ssh: Arc<Mutex<Child>>,     // the tunnel; `corral watch` on the host is at its other end
    ended: Arc<AtomicBool>,     // the host said the server ended
    closed: Arc<AtomicBool>,    // the link is gone (its last lines read)
    errors: Arc<Mutex<String>>, // what ssh said: why the link dropped
    served: Vec<Served>,        // a router's models as clients see them
}

pub struct Server {
    pub id: u64,
    pub dash: Dashboard,
    pub pid: u32,
    pub base: String, // http://host:port
    pub api_key: String,
    pub keep: bool,                   // leave it running when Corral exits
    pub stops: u32,                   // stop requests so far (the second one kills)
    child: Option<Arc<Mutex<Child>>>, // shared with the quit stop
    _quit: Option<QuitGuard>,         // a quit signal stops it (None while kept)
    lost: u32,                        // polls in a row with no answer (how an adopted server's end shows)
    lines: Receiver<String>,
    polls: Receiver<Metrics>,
    tailing: Arc<AtomicBool>,
    polling: Arc<AtomicBool>,
    run: RunFile,
    launch: Option<Launch>, // what started it, to start it again (None = adopted)
    was_up: bool,           // /health answered ok at least once
    restarts: Vec<Instant>, // the automatic restarts so far
    remote: Option<Remote>, // None = it runs on this computer
    rpc: Vec<RpcLink>,      // the hosts whose GPUs it uses (their links close after it stops)
}

impl Server {
    /// Start the launch's llama-server with its output in a new file under logs/; adds --metrics
    /// when the build knows it (the dashboard's LIVE panel reads it). `rpc`: the links to the
    /// hosts whose GPUs it uses (see `add_rpc`), kept as long as the server.
    pub fn start(launch: &Launch, rpc: Vec<RpcLink>) -> Result<Self, String> {
        let mut cmd = launch.cmd.clone();
        let mut env = launch.env.clone();
        let flags = crate::preset::cached_flags(Path::new(&cmd[0]));
        let has_metrics = flags.as_ref().is_none_or(|f| f.contains("--metrics"));
        if has_metrics && !cmd.iter().any(|a| a == "--metrics") {
            cmd.push("--metrics".into());
        }
        let api_key = flag_value(&cmd, "--api-key").unwrap_or_default();
        // unknown build (no --help read): the key stays on the command line rather than risk a
        // server that ignores the variable and asks no key at all
        let reads_key_from_env = flags.as_ref().is_some_and(|f| f.contains("LLAMA_API_KEY"));
        if reads_key_from_env {
            key_to_env(&mut cmd, &mut env);
        }
        let port = flag_value(&cmd, "--port").unwrap_or_default();
        let _ = fs::create_dir_all(logs());
        let log_path = logs().join(format!("llama-{}-{port}.log", file_stamp()));
        let err = |e: io::Error| format!("{}: {e}", log_path.display());
        let mut log = File::create(&log_path).map_err(err)?;
        writeln!(log, "# {}  {}", stamp(), shell_join(&cmd)).map_err(err)?;
        let mut c = Command::new(&cmd[0]);
        c.args(&cmd[1..]).envs(env.iter().map(|(k, v)| (k, v))).current_dir(&*BASE).stdin(Stdio::null());
        c.stdout(log.try_clone().map_err(err)?).stderr(log);
        detach(&mut c);
        let child = c.spawn().map_err(|e| format!("impossibile avviare {}: {e}", cmd[0]))?;
        let run = RunFile {
            pid: child.id(),
            base: format!("http://{}", launch.info.endpoint),
            api_key,
            log: log_path,
            started: unix_now(),
            keep: false,
            info: launch.info.clone(),
        };
        let lent: Vec<Lent> = rpc.iter().map(|l| (l.host.clone(), l.gpus())).collect();
        let mut s = Self::new(run, Some(child), lent);
        s.launch = Some(launch.clone());
        s.rpc = rpc;
        s.save();
        Ok(s)
    }

    fn new(run: RunFile, child: Option<Child>, lent: Vec<Lent>) -> Self {
        let tailing = Arc::new(AtomicBool::new(true));
        let (tx, lines) = mpsc::channel();
        let (path, on) = (run.log.clone(), tailing.clone());
        thread::spawn(move || tail(path, on, tx));
        let pid = run.pid;
        Self::assemble(run, child, None, lines, tailing, move || all_gpus(pid, &lent))
    }

    /// A server on another computer, reached through `link` (see `remote::link`); `r` is what its
    /// host says about it. Its run file stays on the host, and quitting Corral here only closes
    /// the tunnel: `stop` asks the host to stop it.
    pub fn remote(host: &str, r: &Running, link: Link) -> Self {
        let mut info = r.info.clone();
        info.endpoint = format!("127.0.0.1:{}", link.port);
        info.host = host.into();
        let run = RunFile {
            pid: r.pid,
            base: format!("http://127.0.0.1:{}", link.port),
            api_key: link.hello["key"].as_str().unwrap_or_default().to_string(),
            log: PathBuf::from(format!("{host}:{}", r.log)), // shown only
            started: r.started,
            keep: true,
            info,
        };
        let remote = Remote {
            host: host.into(),
            ssh: Arc::new(Mutex::new(link.ssh)),
            ended: link.ended,
            closed: link.closed,
            errors: link.errors,
            served: r.served.clone(),
        };
        let gpus = link.gpus;
        // the log comes through the link: no file here to tail
        let tailing = Arc::new(AtomicBool::new(true));
        Self::assemble(run, None, Some(remote), link.lines, tailing, move || lock(&gpus).clone())
    }

    /// The parts every server has: the poller of its HTTP API, its dashboard, its quit guard.
    fn assemble(
        run: RunFile,
        child: Option<Child>,
        remote: Option<Remote>,
        lines: Receiver<String>,
        tailing: Arc<AtomicBool>,
        gpus_now: impl Fn() -> (Vec<Gpu>, Option<u64>) + Send + 'static,
    ) -> Self {
        let child = child.map(|c| Arc::new(Mutex::new(c)));
        let polling = Arc::new(AtomicBool::new(true));
        let (tx, polls) = mpsc::channel();
        let (base, key, router, on) = (run.base.clone(), run.api_key.clone(), run.info.router, polling.clone());
        thread::spawn(move || poller(base, key, router, gpus_now, on, tx));
        let mut dash = Dashboard::new(run.info.clone(), run.log.clone());
        // an adopted server started before now; checked: the wall clock may be ahead of the
        // monotonic one (clock set after boot), and Instant can't go back past the boot
        let age = Duration::from_secs(unix_now().saturating_sub(run.started));
        dash.t0 = Instant::now().checked_sub(age).unwrap_or_else(Instant::now);
        let mut s = Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            dash,
            pid: run.pid,
            base: run.base.clone(),
            api_key: run.api_key.clone(),
            keep: run.keep,
            stops: 0,
            child,
            _quit: None,
            lost: 0,
            lines,
            polls,
            tailing,
            polling,
            run,
            launch: None,
            was_up: false,
            restarts: vec![],
            remote,
            rpc: vec![],
        };
        s._quit = s.quit_guard();
        s
    }

    /// What a quit signal (SIGTERM, SIGHUP, console close: `ui::on_quit`) does to it unless it is
    /// kept: our child gets the graceful stop, an adopted one (not our child) just the signal.
    fn quit_guard(&self) -> Option<QuitGuard> {
        if let Some(r) = &self.remote {
            // the server stays on its host: only the tunnel closes
            let ssh = r.ssh.clone();
            return Some(on_quit(move || {
                let _ = lock(&ssh).kill();
            }));
        }
        if self.keep {
            return None;
        }
        let (child, pid, file) = (self.child.clone(), self.pid, self.run_path());
        Some(on_quit(move || {
            if hung_up() && over_ssh() {
                // the ssh connection dropped, not a decision to stop: the server keeps running in
                // its own process group, and its run file stays for the next Corral to adopt it
                return;
            }
            match child {
                Some(c) => stop(&c, QUIT_GRACE),
                #[cfg(unix)]
                None if pid_alive(pid) => unsafe {
                    libc::kill(pid as libc::pid_t, libc::SIGINT);
                },
                #[cfg(windows)]
                None => {
                    // not capture(): it kills its child once a quit is under way
                    let mut c = Command::new("taskkill");
                    c.args(["/F", "/PID", &pid.to_string()]).stdin(Stdio::null()).stdout(Stdio::null());
                    let _ = c.stderr(Stdio::null()).status();
                }
                #[cfg(unix)]
                None => {}
            }
            let _ = fs::remove_file(file);
        }))
    }

    fn run_path(&self) -> PathBuf {
        run_dir().join(format!("{}.json", self.pid))
    }

    fn save(&self) {
        if self.remote.is_some() {
            return; // its run file is on its host
        }
        let _ = fs::create_dir_all(run_dir());
        let json = serde_json::to_string_pretty(&self.run).unwrap_or_default();
        let _ = write_private(&self.run_path(), &json); // it holds the API key
    }

    pub fn set_keep(&mut self, keep: bool) {
        if self.remote.is_some() {
            return; // Corral here never stops it on exit anyway
        }
        if self.uses_rpc() {
            return; // the links to the hosts' GPUs end with this Corral, and it with them
        }
        (self.keep, self.run.keep) = (keep, keep);
        self._quit = None; // the old guard goes first, so at most one per server
        self._quit = self.quit_guard();
        if self.running() {
            self.save();
        }
    }

    pub fn running(&self) -> bool {
        self.dash.exit_code.is_none()
    }

    /// It uses GPUs that other computers lend over RPC.
    pub fn uses_rpc(&self) -> bool {
        !self.rpc.is_empty()
    }

    /// The hosts whose GPUs it uses.
    pub fn lenders(&self) -> Vec<&str> {
        self.rpc.iter().map(|l| l.host.as_str()).collect()
    }

    /// The computer it runs on (None = this one).
    pub fn host(&self) -> Option<&str> {
        self.remote.as_ref().map(|r| r.host.as_str())
    }

    /// A router on another computer: its models as clients see them (from its host).
    pub fn served(&self) -> &[Served] {
        self.remote.as_ref().map_or(&[], |r| r.served.as_slice())
    }

    /// Feed new log lines and metrics to the dashboard; notice the end. Call every frame.
    pub fn pump(&mut self) {
        while let Ok(l) = self.lines.try_recv() {
            self.dash.feed(&l);
        }
        while let Ok(m) = self.polls.try_recv() {
            self.lost = if m.health.is_empty() { self.lost + 1 } else { 0 };
            self.was_up |= m.health == "ok";
            self.dash.metrics(m);
        }
        if !self.running() {
            return;
        }
        let code = match self.child.as_ref().map(|c| lock(c).try_wait()) {
            Some(Ok(Some(s))) => Some(exit_code(&s)),
            Some(Err(_)) => Some(-1),
            Some(Ok(None)) => return,
            None if self.gone() => None,
            None => return,
        };
        self.polling.store(false, Ordering::Relaxed);
        match &self.remote {
            // the link dropped before the host said the server ended: it may still run there
            Some(r) if !r.ended.load(Ordering::Relaxed) => {
                let errors = lock(&r.errors).clone();
                let why = errors.lines().map(str::trim).rfind(|l| !l.is_empty()).unwrap_or("");
                self.dash.lost(&clip(why, 120));
            }
            Some(_) => self.dash.finished(None),
            None => {
                self.dash.finished(code);
                let _ = fs::remove_file(self.run_path());
            }
        }
    }

    /// First call: graceful (SIGINT, what Ctrl+C does); then kill. Windows: always a kill,
    /// console events can't target one server (llama-server keeps nothing to flush).
    pub fn stop(&mut self) {
        if !self.running() {
            return;
        }
        self.stops += 1;
        self.dash.stopping(self.stops);
        if let Some(r) = &self.remote {
            // its host stops it; the end comes back through the link
            let (host, pid, errors) = (r.host.clone(), self.pid, r.errors.clone());
            thread::spawn(move || {
                if let Err(e) = remote::stop(&host, pid) {
                    lock(&errors).push_str(&format!("\n{e}"));
                }
            });
            return;
        }
        #[cfg(windows)]
        if let Some(c) = self.child.as_ref() {
            let _ = lock(c).kill();
            return;
        }
        signal_stop(self.pid, self.stops > 1);
    }

    pub fn log_path(&self) -> &Path {
        &self.run.log
    }

    /// After a crash, start the same launch again. A crash: an exit code other than 0 that nobody
    /// asked for, from a server that had been up; one that never came up (a wrong flag, too
    /// little memory) would only fail again. Answers once per crash; None = nothing to do.
    pub fn restart(&mut self) -> Option<Restart> {
        let crashed = self.dash.exit_code.is_some_and(|c| c != 0) && self.was_up && self.stops == 0;
        if !crashed || quit_requested() {
            return None;
        }
        let launch = self.launch.take()?; // adopted servers have none; taken: one answer per crash
        let recent: Vec<Instant> = self.restarts.iter().copied().filter(|t| t.elapsed() < RESTART_WINDOW).collect();
        if recent.len() >= RESTARTS {
            return Some(Restart::GaveUp);
        }
        if let Some(l) = self.rpc.iter().find(|l| l.closed()) {
            // likely why it crashed; without that host's GPUs it would only fail again
            return Some(Restart::Failed(format!("il collegamento alle GPU di {} è caduto", l.host)));
        }
        match Server::start(&launch, std::mem::take(&mut self.rpc)) {
            Ok(mut fresh) => {
                fresh.id = self.id;
                fresh.restarts = recent;
                fresh.restarts.push(Instant::now());
                if self.keep {
                    fresh.set_keep(true);
                }
                Some(Restart::Started(Box::new(fresh)))
            }
            Err(e) => Some(Restart::Failed(e)),
        }
    }

    fn gone(&mut self) -> bool {
        if let Some(r) = &self.remote {
            return r.closed.load(Ordering::Relaxed);
        }
        match self.child.as_ref() {
            Some(c) => !lock(c).try_wait().is_ok_and(|s| s.is_none()),
            #[cfg(unix)]
            None => !pid_alive(self.pid),
            // adopted on Windows: no handle; gone after our taskkill /F or when it stops answering
            #[cfg(not(unix))]
            None => self.stops > 0 || self.lost >= 3,
        }
    }
}

impl Drop for Server {
    /// Corral closing (or a panic): stop the server unless it was asked to stay up.
    fn drop(&mut self) {
        self.tailing.store(false, Ordering::Relaxed);
        self.polling.store(false, Ordering::Relaxed);
        if let Some(r) = &self.remote {
            // the server stays on its host: only the tunnel closes
            let mut ssh = lock(&r.ssh);
            let _ = ssh.kill();
            let _ = ssh.wait();
            return;
        }
        if self.keep && self.running() {
            return; // its run file stays: the next Corral adopts it
        }
        if self.running() && !self.gone() {
            self.stop();
            let until = Instant::now() + Duration::from_secs(5);
            while Instant::now() < until && !self.gone() {
                thread::sleep(Duration::from_millis(50));
            }
            if !self.gone() {
                self.stop();
            }
        }
        if let Some(c) = self.child.as_ref() {
            let _ = lock(c).wait();
        }
        let _ = fs::remove_file(self.run_path());
    }
}

fn read_run(path: &Path) -> Option<RunFile> {
    serde_json::from_str(&fs::read_to_string(path).ok()?).ok()
}

/// Its process is there.
#[cfg(unix)]
pub(crate) fn process_alive(r: &RunFile) -> bool {
    pid_alive(r.pid)
}

/// Its process is there: Windows can't check the pid here, so its port answers.
#[cfg(not(unix))]
pub(crate) fn process_alive(r: &RunFile) -> bool {
    !health_of(&r.base, &r.api_key).is_empty()
}

/// Servers a previous Corral left running (still answering on their port); stale files go.
pub fn adopt() -> Vec<Server> {
    let mut out = vec![];
    for e in fs::read_dir(run_dir()).into_iter().flatten().flatten() {
        let run = read_run(&e.path());
        // the port check too: after a reboot the pid may belong to another program
        let alive = run.as_ref().is_some_and(|r| {
            process_alive(r) && !matches!(get(&r.base, &r.api_key, "/health").call(), Err(ureq::Error::Transport(_)))
        });
        match run {
            Some(r) if alive => out.push(Server::new(r, None, vec![])),
            _ => {
                let _ = fs::remove_file(e.path());
            }
        }
    }
    out
}

/// The servers running now, for `corral status` and `corral stop`. Read only: a server still
/// binding its port counts, and no file is removed (a running Corral may own them).
pub fn running_now() -> Vec<RunFile> {
    let mut out = vec![];
    for e in fs::read_dir(run_dir()).into_iter().flatten().flatten() {
        if let Some(r) = read_run(&e.path())
            && process_alive(&r)
        {
            out.push(r);
        }
    }
    out.sort_by_key(|r| r.started);
    out
}

/// Ask the process `pid` to stop: SIGINT, the graceful Ctrl+C (`force`: SIGKILL). Windows:
/// always a kill, console events can't target one process.
pub fn signal_stop(pid: u32, force: bool) {
    #[cfg(unix)]
    unsafe {
        libc::kill(pid as libc::pid_t, if force { libc::SIGKILL } else { libc::SIGINT });
    }
    #[cfg(windows)]
    {
        let _ = force;
        let _ = capture(Command::new("taskkill").args(["/F", "/PID", &pid.to_string()]), Duration::from_secs(10));
    }
}

/// Stop a server that may not be our child and wait for it: graceful first, a kill after `grace`.
pub fn stop_pid(pid: u32, grace: Duration) {
    signal_stop(pid, false);
    #[cfg(unix)]
    {
        let t0 = Instant::now();
        while pid_alive(pid) && t0.elapsed() < grace {
            thread::sleep(Duration::from_millis(100));
        }
        if pid_alive(pid) {
            signal_stop(pid, true);
        }
    }
    #[cfg(not(unix))]
    let _ = grace;
    let _ = fs::remove_file(run_dir().join(format!("{pid}.json")));
}

// -- the end of a log -------------------------------------------------------------------------------

/// The last `max` bytes of a file as text, from the first whole line.
pub fn read_tail(path: &Path, max: u64) -> String {
    let Ok(mut f) = File::open(path) else { return String::new() };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let start = len.saturating_sub(max);
    let mut buf = vec![];
    if f.seek(SeekFrom::Start(start)).is_err() || f.read_to_end(&mut buf).is_err() {
        return String::new();
    }
    let text = String::from_utf8_lossy(&buf).into_owned();
    if start == 0 {
        return text;
    }
    // the read began halfway through a line: drop that piece
    text.split_once('\n').map(|(_, rest)| rest.to_string()).unwrap_or_default()
}

/// A log line that reports a problem: llama.cpp's "E" level, or the words of an abort.
fn says_error(line: &str) -> bool {
    let level_e = line.split_whitespace().nth(1) == Some("E"); // "0.01.234.567 E srv  …"
    let lower = line.to_lowercase();
    level_e || lower.contains("error") || lower.contains("failed") || lower.contains("assert")
}

/// The lines llama-server prints after the one that says why the model did not load.
fn only_repeats_failure(line: &str) -> bool {
    line.contains("exiting due to") || line.contains("failed to load model")
}

/// Why a server ended, from its log: the last of its last lines that reports a problem,
/// passing over the ones that only repeat that the load failed. "" when none does (a
/// segfault writes nothing): any other line would look like the cause.
pub fn last_error(log: &Path) -> String {
    let tail = read_tail(log, 16 << 10);
    let lines: Vec<&str> = tail.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let recent = &lines[lines.len().saturating_sub(20)..];
    let errors: Vec<&str> = recent.iter().rev().copied().filter(|l| says_error(l)).collect();
    let Some(cause) = errors.iter().find(|l| !only_repeats_failure(l)).or(errors.first()) else {
        return String::new();
    };
    if cause.contains("unknown model architecture") {
        return format!("{} · llama.cpp troppo vecchio per questo modello: aggiornalo", clip(cause, 120));
    }
    clip(cause, 160)
}

// -- helpers for the dashboard ------------------------------------------------------------------------

/// Only web pages may be opened: links come from feeds and APIs, and another scheme (file:,
/// a custom handler) could start whatever program is registered for it.
pub fn browsable(url: &str) -> bool {
    url.starts_with("http://") || url.starts_with("https://")
}

/// Open `url` in the default browser, without waiting for it.
pub fn open_url(url: &str) -> io::Result<()> {
    if !browsable(url) {
        return Err(io::Error::other("non è un indirizzo http(s)"));
    }
    if cfg!(test) {
        return Ok(()); // no browser opening during the tests
    }
    let mut c = if cfg!(windows) {
        // not `cmd /C start`: cmd would run whatever follows a & or | in the URL
        let mut c = Command::new("rundll32");
        c.args(["url.dll,FileProtocolHandler", url]);
        c
    } else {
        let mut c = Command::new(if cfg!(target_os = "macos") { "open" } else { "xdg-open" });
        c.arg(url);
        c
    };
    let mut child = c.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn()?;
    thread::spawn(move || child.wait()); // reap it
    Ok(())
}

/// The address of a server on this machine in `url` ("http://127.0.0.1:8080/…" -> host, port).
pub fn local_address(url: &str) -> Option<(&str, &str)> {
    let rest = url.strip_prefix("http://").or_else(|| url.strip_prefix("https://"))?;
    let authority = rest.split('/').next()?;
    let (host, port) = authority.rsplit_once(':')?;
    let local = host == "localhost" || host == "0.0.0.0" || host.starts_with("127.");
    local.then_some((host, port))
}

/// The ssh command, run on the user's computer, that makes `host:port` of this machine reachable
/// there as localhost:port. `ssh_connection` is $SSH_CONNECTION ("client_ip client_port
/// server_ip server_port"), `user` the user name here.
pub fn tunnel_command(host: &str, port: &str, ssh_connection: &str, user: &str) -> String {
    let fields: Vec<&str> = ssh_connection.split_whitespace().collect();
    let server = fields.get(2).copied().unwrap_or("<server>");
    let ssh_port = fields.get(3).copied().unwrap_or("22");
    let target = if host == "localhost" || host == "0.0.0.0" { "127.0.0.1" } else { host };
    let port_flag = if ssh_port == "22" { String::new() } else { format!("-p {ssh_port} ") };
    let login = if user.is_empty() { server.to_string() } else { format!("{user}@{server}") };
    format!("ssh {port_flag}-L {port}:{target}:{port} {login}")
}

/// Hand `text` to the system clipboard tool; false = none available.
pub fn clipboard(text: &str) -> bool {
    if cfg!(test) {
        return !text.is_empty(); // the tests leave the real clipboard alone
    }
    let tools: &[&[&str]] = if cfg!(windows) {
        &[&["clip"]]
    } else if cfg!(target_os = "macos") {
        &[&["pbcopy"]]
    } else {
        &[&["wl-copy"], &["xclip", "-selection", "clipboard"], &["xsel", "-ib"]]
    };
    tools.iter().any(|t| {
        let Ok(mut c) =
            Command::new(t[0]).args(&t[1..]).stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::null()).spawn()
        else {
            return false;
        };
        let wrote = c.stdin.take().is_some_and(|mut i| i.write_all(text.as_bytes()).is_ok());
        c.wait().is_ok_and(|s| s.success()) && wrote
    })
}

/// Standard base64 with '=' padding (the terminal's OSC 52 clipboard wants it).
pub fn base64(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in data.chunks(3) {
        // 3 bytes = 24 bits = 4 letters of 6 bits; a short last chunk ends with '='
        let byte = |i: usize| u32::from(chunk.get(i).copied().unwrap_or(0));
        let bits = (byte(0) << 16) | (byte(1) << 8) | byte(2);
        for i in 0..4 {
            let letter = ALPHABET[((bits >> (18 - 6 * i)) & 63) as usize] as char;
            out.push(if i <= chunk.len() { letter } else { '=' });
        }
    }
    out
}

/// Streamed answer to the dashboard's test prompt.
pub enum Chunk {
    Text(String),
    /// generated tokens, tokens/s (from the server's timings, else measured)
    Done(u64, f64),
    Err(String),
}

/// POST /v1/chat/completions with stream=true; blocking (run it in a Task). Returns when the
/// answer ends or the receiver is dropped (the connection closes and the server stops).
pub fn chat(base: &str, key: &str, model: &str, prompt: &str, tx: &Sender<Chunk>) {
    let mut body = serde_json::json!({
        "messages": [{"role": "user", "content": prompt}],
        "stream": true,
        "max_tokens": 1024,
    });
    if !model.is_empty() {
        body["model"] = model.into(); // a router needs it to pick the model
    }
    let mut req = HTTP.post(&format!("{base}/v1/chat/completions"));
    if !key.is_empty() {
        req = req.set("Authorization", &format!("Bearer {key}"));
    }
    let resp = match req.send_json(body) {
        Ok(r) => r,
        Err(ureq::Error::Status(code, r)) => {
            let msg = r.into_string().unwrap_or_default();
            let _ = tx.send(Chunk::Err(format!("HTTP {code}: {}", clip(msg.trim(), 200))));
            return;
        }
        Err(e) => {
            let _ = tx.send(Chunk::Err(e.to_string()));
            return;
        }
    };
    let (t0, mut n, mut done) = (Instant::now(), 0u64, None);
    let mut r = BufReader::new(resp.into_reader().take(64 << 20));
    let mut line = String::new();
    while r.read_line(&mut line).is_ok_and(|n| n > 0) {
        let data = line.trim().strip_prefix("data:").map(str::trim).unwrap_or("").to_string();
        line.clear();
        let Ok(v) = serde_json::from_str::<Value>(&data) else { continue };
        let delta = &v["choices"][0]["delta"];
        for k in ["reasoning_content", "content"] {
            if let Some(s) = delta[k].as_str().filter(|s| !s.is_empty()) {
                n += 1;
                if tx.send(Chunk::Text(s.into())).is_err() {
                    return;
                }
            }
        }
        if let Some(t) = v.get("timings") {
            done = Some((t["predicted_n"].as_u64().unwrap_or(n), t["predicted_per_second"].as_f64().unwrap_or(0.0)));
        }
    }
    let (n, tps) = done.unwrap_or((n, n as f64 / t0.elapsed().as_secs_f64().max(1e-3)));
    let _ = tx.send(Chunk::Done(n, tps));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_parsing() {
        let prom =
            parse_prom("# HELP x\nllamacpp:tokens_predicted_total 400\nllamacpp:requests_deferred 2\nbad line\n");
        assert_eq!(prom.get("tokens_predicted_total"), Some(&400.0));
        assert_eq!(prom.get("requests_deferred"), Some(&2.0));
        assert_eq!(prom.len(), 2);
        // b12283: busy slot without n_past; older builds report n_past
        let v: Value = serde_json::from_str(
            r#"[{"id":0,"n_ctx":4096,"is_processing":true,"n_prompt_tokens":155,"next_token":[{"n_decoded":139}]},
                {"id":1,"n_ctx":4096,"is_processing":false},
                {"id":2,"n_ctx":4096,"is_processing":false,"n_past":77}]"#,
        )
        .unwrap();
        let s = parse_slots(&v);
        assert_eq!(s[0], Slot { id: 0, n_ctx: 4096, used: 294, decoded: 139, busy: true });
        assert_eq!((s[1].used, s[1].busy, s[2].used), (0, false, 77));
    }

    #[test]
    fn base64_vectors() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"http://127.0.0.1:8080/v1"), "aHR0cDovLzEyNy4wLjAuMTo4MDgwL3Yx");
    }

    #[test]
    fn log_filter_pause_and_live_rate() {
        let info = Info {
            alias: "a".into(),
            model: "m.gguf".into(),
            mmproj: String::new(),
            endpoint: "h:1".into(),
            install: "i".into(),
            backend: "b".into(),
            version: "v".into(),
            settings: vec![("ngl".into(), "auto".into())],
            router: false,
            host: String::new(),
        };
        // the run file round-trips
        let back: Info = serde_json::from_str(&serde_json::to_string(&info).unwrap()).unwrap();
        assert_eq!(back.settings, info.settings);
        let mut d = Dashboard::new(info, PathBuf::from("x.log"));
        d.feed("0.00.1 I srv    load_model: loading model 'm.gguf'");
        d.feed("0.00.2 W srv    something odd");
        d.paused = true;
        d.feed("0.00.3 I slot   launch_slot_: id  0 | task 1 | processing task");
        assert_eq!(d.scroll, 1); // paused: the view stays where it was
        d.only_warn = true;
        d.feed("0.00.4 I srv    not a warning");
        assert_eq!(d.scroll, 1); // hidden by the filter: nothing moves
        d.follow();
        assert_eq!((d.scroll, d.paused), (0, false));
        let live = |tot: &str, busy: u64| Metrics {
            health: "ok".into(),
            prom: parse_prom(&format!("llamacpp:tokens_predicted_total {tot}")),
            slots: vec![Slot { id: 0, n_ctx: 10, used: 0, decoded: busy, busy: busy > 0 }],
            ..Metrics::default()
        };
        d.metrics(live("100", 0));
        d.metrics(live("100", 50)); // a running task counts before it ends
        assert!(d.live.back().is_some_and(|&r| r > 0.0));
    }
}
