//! The commands that work without the full-screen interface: start, list and stop servers, and
//! turn one into a service that starts with the computer (systemd on Linux, launchd on macOS).
//! A server started here leaves a run file like the interface's, so each one sees the other's.
//! They are also how the Corral of another computer works this one over ssh (see `remote`):
//! `status --json`, `models --json`, `start`, `stop`, and `watch` and `rpc`, which only that one
//! uses.

use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, IsTerminal, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode};
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::server::{self, Launch, RunFile, Server};
use crate::ui::*;
use crate::{agent, installs, models, preset, remote, router};

/// The commands' part of `corral --help`.
pub const HELP: &str = "\
comandi, senza interfaccia:
  corral start <modello> [--preset NOME]   avvia un modello e aspetta che sia pronto
  corral start --profile NOME              avvia un profilo multi-modello
                 --foreground              resta con il server fino alla fine (per i servizi, e
                                           per un preset che usa le GPU di altri host)
  corral status [--json]                   i server attivi
  corral models [--json]                   i modelli (con i loro preset) e i profili da avviare
  corral stop <nome|porta|pid> | --all     ferma un server (anche una parte del nome), o tutti
  corral service install <modello> [--preset NOME] | --profile NOME
                                           lo avvia con il computer e lo riavvia se cade
  corral service list                      i servizi installati
  corral service remove <nome>             toglie un servizio (e ferma il suo server)
  <modello>: il percorso del file, o il nome o una sua parte (es. qwen3-8b)";

/// How a command failed.
enum Fail {
    /// wrong words on the command line: the help, exit code 2 (as for an unknown option)
    Usage(String),
    /// exit code 1
    Error(String),
}

impl From<String> for Fail {
    fn from(e: String) -> Self {
        Fail::Error(e)
    }
}

impl From<&str> for Fail {
    fn from(e: &str) -> Self {
        Fail::Error(e.into())
    }
}

/// Runs `corral <command> …`; None when the first word isn't a command (the interface starts).
pub fn run(args: &[String]) -> Option<ExitCode> {
    let (command, rest) = args.split_first()?;
    let result = match command.as_str() {
        "start" => start(rest),
        "status" => status(rest),
        "models" => list_models(rest),
        "stop" => stop(rest),
        "watch" => watch(rest),
        "rpc" => rpc(rest),
        "service" => service(rest),
        _ => return None,
    };
    Some(match result {
        Ok(code) => code,
        Err(Fail::Usage(msg)) => {
            eprintln!("corral {command}: {msg}\n\n{HELP}");
            ExitCode::from(2)
        }
        Err(Fail::Error(msg)) => {
            eprintln!("corral {command}: {msg}");
            ExitCode::FAILURE
        }
    })
}

// -- the words of a command ---------------------------------------------------------------------

/// A command's words: the options with a value, the bare flags, and the rest in order.
struct Words {
    plain: Vec<String>,
    values: BTreeMap<String, String>,
    flags: HashSet<String>,
}

/// Sort `rest` into the options a command knows; any other word starting with - is an error.
fn split(rest: &[String], with_value: &[&str], bare: &[&str]) -> Result<Words, Fail> {
    let mut words = Words { plain: vec![], values: BTreeMap::new(), flags: HashSet::new() };
    let mut i = 0;
    while i < rest.len() {
        let word = &rest[i];
        i += 1;
        if with_value.contains(&word.as_str()) {
            let value = rest.get(i).ok_or_else(|| Fail::Usage(format!("{word} vuole un valore")))?;
            words.values.insert(word.clone(), value.clone());
            i += 1;
        } else if bare.contains(&word.as_str()) {
            words.flags.insert(word.clone());
        } else if word.starts_with('-') {
            return Err(Fail::Usage(format!("opzione sconosciuta: {word}")));
        } else {
            words.plain.push(word.clone());
        }
    }
    Ok(words)
}

/// What `start` and `service install` act on.
enum Target {
    Profile(String),
    Model { path: PathBuf, preset: Option<String> },
}

/// The target the words name: `--profile NAME`, or a model with an optional `--preset NAME`.
fn target(state: &State, words: &Words) -> Result<Target, Fail> {
    let preset = words.values.get("--preset").cloned();
    if let Some(name) = words.values.get("--profile") {
        if !words.plain.is_empty() || preset.is_some() {
            return Err(Fail::Usage("--profile non va insieme a un modello o a --preset".into()));
        }
        if !state.routers.contains_key(name) {
            let names: Vec<&str> = state.routers.keys().map(String::as_str).collect();
            return Err(format!("nessun profilo «{name}» (profili: {})", or_none(&names)).into());
        }
        return Ok(Target::Profile(name.clone()));
    }
    let [query] = &words.plain[..] else {
        return Err(Fail::Usage("indica un modello, o --profile NOME".into()));
    };
    let path = find_model(state, query)?;
    if let Some(name) = &preset {
        let names = preset::preset_names(state, &path);
        if !names.contains(name) {
            let model = fname(&ps(&path)).to_string();
            return Err(format!("{model} non ha un preset «{name}» (preset: {})", names.join(", ")).into());
        }
    }
    Ok(Target::Model { path, preset })
}

impl Target {
    /// Short name, for the service: the profile, or the model's alias (and the preset, if given).
    fn name(&self) -> String {
        match self {
            Target::Profile(name) => name.clone(),
            Target::Model { path, preset: None } => preset::base_alias(path),
            Target::Model { path, preset: Some(p) } => format!("{}-{p}", preset::base_alias(path)),
        }
    }

    /// For the service's description.
    fn describe(&self) -> String {
        match self {
            Target::Profile(name) => format!("profilo multi-modello «{name}»"),
            Target::Model { path, preset: None } => fname(&ps(path)).to_string(),
            Target::Model { path, preset: Some(p) } => format!("{}, preset «{p}»", fname(&ps(path))),
        }
    }

    /// The words after `corral start` that start it again.
    fn start_words(&self) -> Vec<String> {
        match self {
            Target::Profile(name) => vec!["--profile".into(), name.clone()],
            Target::Model { path, preset } => {
                let mut words = vec![ps(path)];
                if let Some(p) = preset {
                    words.push("--preset".into());
                    words.push(p.clone());
                }
                words
            }
        }
    }

    /// What to start, and the hosts whose GPUs it uses (a model's preset may name some).
    fn launch(&self, state: &State, inst: &installs::Install, version: &str) -> Result<(Launch, Vec<String>), String> {
        match self {
            Target::Profile(name) => Ok((server::profile_launch(state, inst, version, name)?, vec![])),
            Target::Model { path, preset } => {
                let values = match preset {
                    Some(name) => preset::get_named(state, path, name),
                    None => preset::get_preset(state, path), // the one selected in the interface
                };
                Ok((server::model_launch(inst, version, path, &values), preset::rpc_hosts(&values)))
            }
        }
    }
}

/// The model `query` names: a file path, or among the models Corral sees (ignoring case) the
/// file name, the name without .gguf, or else a part of the name only one of them has.
fn find_model(state: &State, query: &str) -> Result<PathBuf, String> {
    let as_path = Path::new(query);
    if as_path.is_file() {
        return Ok(std::path::absolute(as_path).unwrap_or_else(|_| as_path.to_path_buf()));
    }
    let wanted = query.to_lowercase();
    let all = models::scan(state);
    let name = |m: &models::Local| fname(&ps(&m.path)).to_lowercase();
    let exact: Vec<&models::Local> =
        all.iter().filter(|m| name(m) == wanted || name(m) == format!("{wanted}.gguf")).collect();
    let found = if exact.is_empty() { all.iter().filter(|m| name(m).contains(&wanted)).collect() } else { exact };
    match found[..] {
        [one] => Ok(one.path.clone()),
        [] => {
            let names: Vec<&str> = all.iter().take(12).map(|m| fname_of(&m.path)).collect();
            Err(format!("nessun modello «{query}» (modelli: {})", or_none(&names)))
        }
        _ => {
            let paths: Vec<String> = found.iter().map(|m| format!("  {}", ps(&m.path))).collect();
            Err(format!("«{query}» indica {} modelli: scrivi di più, o il percorso\n{}", found.len(), paths.join("\n")))
        }
    }
}

fn fname_of(p: &Path) -> &str {
    p.file_name().and_then(|n| n.to_str()).unwrap_or("?")
}

/// "a, b, c", or "nessuno".
fn or_none(names: &[&str]) -> String {
    if names.is_empty() { "nessuno".into() } else { names.join(", ") }
}

// -- start / status / stop ---------------------------------------------------------------------

fn start(rest: &[String]) -> Result<ExitCode, Fail> {
    let words = split(rest, &["--preset", "--profile"], &["--foreground"])?;
    let (state, problem) = load_state();
    if let Some(p) = problem {
        eprintln!("attenzione: {p}");
    }
    let target = target(&state, &words)?;
    let inst = installs::active(&state).ok_or("nessuna installazione di llama.cpp: aggiungine una da `corral`")?;
    let version = installs::version(&inst);
    let (mut launch, hosts) = target.launch(&state, &inst, &version)?;
    let in_foreground = words.flags.contains("--foreground");
    let mut links = vec![];
    if !hosts.is_empty() {
        if !in_foreground {
            let msg = format!(
                "usa le GPU di {}: i collegamenti vivono con questo comando, quindi serve --foreground",
                hosts.join(", ")
            );
            return Err(msg.into());
        }
        println!("collego le GPU di {}…", hosts.join(", "));
        links = remote::rpc_links(&hosts)?;
        server::add_rpc(&mut launch, &links, &version)?;
    }
    for note in &launch.notes {
        eprintln!("attenzione: {note}");
    }
    let mut s = Server::start(&launch, links)?;
    println!("avvio {} · pid {} · log {}", launch.info.alias, s.pid, s.log_path().display());
    if in_foreground {
        return Ok(foreground(s));
    }
    s.set_keep(true); // it outlives this command: `corral stop` or the interface stop it
    println!("aspetto che sia pronto (Ctrl+C smette di aspettare, il server resta attivo)…");
    Ok(wait_ready(&mut s))
}

/// Wait until the server answers, or ends: a script can send its first request right after.
fn wait_ready(s: &mut Server) -> ExitCode {
    loop {
        s.pump();
        if !s.running() {
            let code = s.dash.exit_code.unwrap_or(-1);
            eprintln!("llama-server si è fermato prima di essere pronto (codice {code}); fine del log:");
            for line in last_lines(s.log_path(), 15) {
                eprintln!("  {line}");
            }
            return ExitCode::FAILURE;
        }
        if s.dash.m.health == "ok" {
            println!("pronto: {}/v1", s.base);
            return ExitCode::SUCCESS;
        }
        thread::sleep(Duration::from_millis(250));
    }
}

/// Stay with the server until it ends, for a service manager watching this process: the
/// server's exit code becomes ours, so a crash is a failure that gets restarted. SIGTERM (the
/// service stopping) stops the server the graceful way first: the quit guard of `Server`.
fn foreground(mut s: Server) -> ExitCode {
    let mut ready = false;
    while s.running() {
        s.pump();
        if !ready && s.dash.m.health == "ok" {
            println!("pronto: {}/v1", s.base);
            ready = true;
        }
        thread::sleep(Duration::from_millis(250));
    }
    let code = s.dash.exit_code.unwrap_or(-1);
    eprintln!("llama-server terminato, codice {code}");
    if quit_requested() {
        // a signal stopped it (the service stopping): the quit thread ends the process with 128 +
        // the signal, as for any other program; the links to other hosts' GPUs close first
        drop(s);
        loop {
            thread::park();
        }
    }
    if code == 0 { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}

/// The last `n` lines of a log file.
fn last_lines(path: &Path, n: usize) -> Vec<String> {
    let text = server::read_tail(path, 16 << 10);
    let lines: Vec<&str> = text.lines().collect();
    let first = lines.len().saturating_sub(n);
    lines[first..].iter().map(|l| l.to_string()).collect()
}

/// "40s", "7m", "2h05" since the unix time `t`.
fn since(t: u64) -> String {
    let s = unix_now().saturating_sub(t);
    match s {
        0..60 => format!("{s}s"),
        60..3600 => format!("{}m", s / 60),
        _ => format!("{}h{:02}", s / 3600, s % 3600 / 60),
    }
}

fn status(rest: &[String]) -> Result<ExitCode, Fail> {
    let words = split(rest, &[], &["--json"])?;
    if !words.plain.is_empty() {
        return Err(Fail::Usage(format!("parola in più: {}", words.plain[0])));
    }
    let servers = server::running_now();
    if words.flags.contains("--json") {
        let (state, _) = load_state();
        println!("{}", serde_json::to_string_pretty(&status_json(&servers, &state)).unwrap_or_default());
        return Ok(ExitCode::SUCCESS);
    }
    if servers.is_empty() {
        println!("nessun server attivo");
        return Ok(ExitCode::SUCCESS);
    }
    println!("{:<8} {:<24} {:<26} {:<13} {:<6} MODELLO", "PID", "NOME", "INDIRIZZO", "STATO", "DA");
    for r in &servers {
        let health = match server::health_of(&r.base, &r.api_key) {
            "ok" => "pronto",
            "caricamento" => "caricamento",
            "errore" => "errore",
            _ => "non risponde",
        };
        // a router's "model" is already a count ("3 modelli")
        let model = if r.info.router { r.info.model.clone() } else { fname(&r.info.model).to_string() };
        let name = clip(&r.info.alias, 24);
        println!("{:<8} {name:<24} {:<26} {health:<13} {:<6} {model}", r.pid, r.base, since(r.started));
    }
    Ok(ExitCode::SUCCESS)
}

/// `corral status --json`: this Corral, the GPUs and the running servers. The Corral of another
/// computer reads it over ssh (`remote::Status`). No API key in it: that goes only through `watch`.
pub fn status_json(servers: &[RunFile], state: &State) -> Value {
    let (gpus, _) = server::gpus(0);
    let mut rows = vec![];
    for r in servers {
        let health = match server::health_of(&r.base, &r.api_key) {
            "ok" => "ok",
            "caricamento" => "loading",
            "errore" => "error",
            _ => "no-answer",
        };
        // a router's models as clients see them: the other computer makes their configs
        let profile = state.routers.get(&r.info.alias).filter(|_| r.info.router);
        let served = profile.map(|p| agent::target(state, &r.info.alias, p, Some(&r.base)).models).unwrap_or_default();
        rows.push(json!({
            "pid": r.pid,
            "name": r.info.alias,
            "url": r.base,
            "model": r.info.model,
            "router": r.info.router,
            "health": health,
            "started": r.started,
            "keep": r.keep,
            "log": ps(&r.log),
            "info": r.info,
            "served": served,
        }));
    }
    json!({
        "version": env!("CARGO_PKG_VERSION"),
        "os": env::consts::OS,
        "arch": env::consts::ARCH,
        "gpus": gpus,
        "servers": rows,
    })
}

/// `corral models [--json]`: what `start` can start here.
fn list_models(rest: &[String]) -> Result<ExitCode, Fail> {
    let words = split(rest, &[], &["--json"])?;
    if let Some(extra) = words.plain.first() {
        return Err(Fail::Usage(format!("parola in più: {extra}")));
    }
    let (state, _) = load_state();
    if words.flags.contains("--json") {
        println!("{}", serde_json::to_string_pretty(&models_json(&state)).unwrap_or_default());
        return Ok(ExitCode::SUCCESS);
    }
    let found = models::scan(&state);
    if found.is_empty() {
        println!("nessun modello: scaricane uno con `corral` (Hugging Face) o aggiungi una cartella");
    } else {
        println!("{:<50} {:>10}  PRESET (* = in uso)", "MODELLO", "DIMENSIONE");
    }
    for m in &found {
        let current = preset::preset_name(&state, &m.path);
        let mut names = vec![];
        for name in preset::preset_names(&state, &m.path) {
            names.push(if name == current { format!("{name}*") } else { name });
        }
        println!("{:<50} {:>10}  {}", clip(fname_of(&m.path), 50), size(m.size), names.join(", "));
    }
    if !state.routers.is_empty() {
        println!("\nPROFILI (corral start --profile NOME)");
        for (name, p) in &state.routers {
            println!("  {name:<30} {}", router::count(p.models.len()));
        }
    }
    Ok(ExitCode::SUCCESS)
}

/// `corral models --json`: the models with their presets, and the profiles.
pub fn models_json(state: &State) -> Value {
    let mut found = vec![];
    for m in models::scan(state) {
        found.push(json!({
            "path": ps(&m.path),
            "size": m.size,
            "presets": preset::preset_names(state, &m.path),
            "preset": preset::preset_name(state, &m.path),
        }));
    }
    let mut profiles = vec![];
    for (name, p) in &state.routers {
        profiles.push(json!({ "name": name, "models": p.models.len() }));
    }
    json!({ "models": found, "profiles": profiles })
}

/// `corral watch <pid>`, for the Corral of another computer that reached this one over ssh (see
/// `remote::link`): a hello with the server's API key, the server's whole log and then its new
/// lines, the GPUs every 2 s, and a last line when the server ends. Corral's own lines start
/// with \x1e, which no log line does. It ends when the server does, or when the other side goes
/// (a write fails).
fn watch(rest: &[String]) -> Result<ExitCode, Fail> {
    let [pid] = rest else {
        return Err(Fail::Usage("indica il pid del server (corral status)".into()));
    };
    let pid: u32 = pid.parse().map_err(|_| Fail::Usage(format!("pid non valido: {pid}")))?;
    let Some(run) = server::running_now().into_iter().find(|r| r.pid == pid) else {
        return Err(format!("nessun server attivo con pid {pid}").into());
    };
    let (tx, lines) = mpsc::channel();
    let (path, on) = (run.log.clone(), Arc::new(AtomicBool::new(true)));
    thread::spawn(move || server::tail(path, on, tx));
    let mut out = io::stdout().lock();
    // the key travels only through a pipe (the ssh of the Corral that asked), never to a screen
    let key = if out.is_terminal() { String::new() } else { run.api_key.clone() };
    // a failed write: the other side went away, nothing left to do
    let _ = stream(&mut out, &run, &lines, &key);
    Ok(ExitCode::SUCCESS)
}

/// What `watch` sends, until the server ends or a write fails.
pub(crate) fn stream(out: &mut impl Write, run: &RunFile, lines: &mpsc::Receiver<String>, key: &str) -> io::Result<()> {
    writeln!(out, "\u{1e}hello {}", json!({ "key": key }))?;
    let mut last_gpus: Option<Instant> = None;
    loop {
        let alive = server::process_alive(run);
        if !alive {
            thread::sleep(Duration::from_millis(300)); // the tail reads the last lines
        }
        while let Ok(line) = lines.try_recv() {
            writeln!(out, "{line}")?;
        }
        if !alive {
            writeln!(out, "\u{1e}end")?;
            return out.flush();
        }
        if last_gpus.is_none_or(|t| t.elapsed() >= Duration::from_secs(2)) {
            let (gpus, vram) = server::gpus(run.pid);
            writeln!(out, "\u{1e}gpus {}", json!({ "gpus": gpus, "vram": vram }))?;
            last_gpus = Some(Instant::now());
        }
        out.flush()?;
        thread::sleep(Duration::from_millis(200));
    }
}

/// `corral rpc [--port N]`, for the Corral of another computer that reached this one over ssh
/// (see `remote::rpc_link`): the RPC server of the active llama.cpp on 127.0.0.1:N, which lends
/// this computer's GPUs to that one's llama-server. Only on the loopback: it asks no password,
/// the ssh tunnel is the way in. It lasts as long as the connection: a hello once it listens
/// (the devices lent, the llama.cpp build), the GPUs every 2 s, and when a write fails (the
/// other side went) the RPC server stops. Its output goes to logs/rpc-….log.
fn rpc(rest: &[String]) -> Result<ExitCode, Fail> {
    let words = split(rest, &["--port"], &[])?;
    let port: u16 = match words.values.get("--port") {
        Some(p) => p.parse().map_err(|_| Fail::Usage(format!("porta non valida: {p}")))?,
        None => remote::RPC_PORT,
    };
    let (state, _) = load_state();
    let inst = installs::active(&state).ok_or("nessuna installazione di llama.cpp: aggiungine una da `corral`")?;
    let no_rpc = "questa installazione di llama.cpp non ha il server RPC: usa una release, o ricompilala da Corral";
    let bin = installs::rpc_server(&inst).ok_or(no_rpc)?;
    // the other side tries the next port when this one is taken, or claimed by another `corral rpc`
    let taken = || Fail::Error(format!("{}: {port}", remote::PORT_TAKEN));
    let Some(_claim) = PortClaim::take(port) else { return Err(taken()) };
    if preset::free_port("127.0.0.1", port) != Some(port) {
        return Err(taken());
    }
    let mut cmd = Command::new(&bin);
    cmd.args(["-H", "127.0.0.1", "-p", &port.to_string()]);
    // the weights sent once stay in the host's cache (~/.cache/llama.cpp/rpc): the next load is fast
    cmd.arg("-c");
    // its --help loads the backends, which print what kind of GPU each one is
    let help = capture(Command::new(&bin).arg("--help"), Duration::from_secs(30)).map(|o| o.stdout + &o.stderr);
    if let Some(dedicated) = installs::dedicated_gpus(&help.unwrap_or_default()) {
        cmd.args(["-d", &dedicated.join(",")]);
    }
    let _ = fs::create_dir_all(logs());
    let log_path = logs().join(format!("rpc-{}-{port}.log", file_stamp()));
    let mut log = File::create(&log_path).map_err(|e| format!("{}: {e}", log_path.display()))?;
    let (mut child, output) = spawn_merged(cmd).map_err(|e| format!("impossibile avviare {}: {e}", bin.display()))?;
    // its lines come a moment before it listens: ready once it takes a connection
    let address = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let mut devices = vec![];
    let mut last = String::new(); // its last line, for when it stops
    let t0 = Instant::now();
    loop {
        for line in output.try_iter() {
            let _ = writeln!(log, "{line}");
            if installs::device_line(&line).is_some() {
                devices.push(line.trim().to_string());
            }
            if !line.trim().is_empty() {
                last = line.trim().to_string();
            }
        }
        if let Ok(Some(status)) = child.try_wait() {
            // what it said last, once its output has ended
            while let Ok(line) = output.recv_timeout(Duration::from_millis(500)) {
                let _ = writeln!(log, "{line}");
                if !line.trim().is_empty() {
                    last = line.trim().to_string();
                }
            }
            // some other program took the port meanwhile: the next port, then
            if preset::free_port("127.0.0.1", port) != Some(port) {
                return Err(taken());
            }
            return Err(format!("il server RPC si è fermato (codice {}): {last}", exit_code(&status)).into());
        }
        if TcpStream::connect_timeout(&address, Duration::from_millis(200)).is_ok() {
            break;
        }
        if t0.elapsed() > Duration::from_secs(60) {
            crate::ui::stop(&Mutex::new(child), Duration::from_secs(3));
            return Err("il server RPC non risponde".into());
        }
        thread::sleep(Duration::from_millis(100));
    }
    let hello = json!({ "version": installs::version(&inst), "backend": inst.backend, "devices": devices });
    let mut out = io::stdout().lock();
    // a failed write: the other side went away; either way the RPC server doesn't outlive it
    let _ = lend(&mut out, &mut child, &output, &mut log, &hello);
    crate::ui::stop(&Mutex::new(child), Duration::from_secs(3));
    Ok(ExitCode::SUCCESS)
}

/// A port held for one `corral rpc`: two that start at the same moment both find it free, and
/// one RPC server would fail on it while the other's answer made it look ready. The claim is a
/// file only one of them can create (<data>/rpc/<port>.lock, with its pid), gone when it ends.
struct PortClaim(PathBuf);

impl PortClaim {
    fn take(port: u16) -> Option<Self> {
        let dir = BASE.join("rpc");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join(format!("{port}.lock"));
        // the claim of a `corral rpc` that was killed: the port is free again
        #[cfg(unix)]
        if let Ok(text) = fs::read_to_string(&path)
            && let Ok(pid) = text.trim().parse::<u32>()
            && !server::pid_alive(pid)
        {
            let _ = fs::remove_file(&path);
        }
        let mut file = OpenOptions::new().write(true).create_new(true).open(&path).ok()?;
        let _ = write!(file, "{}", std::process::id());
        Some(Self(path))
    }
}

impl Drop for PortClaim {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// What `rpc` sends while the RPC server runs: the hello, then the GPUs every 2 s (with the MiB
/// the RPC server holds). Ends when a write fails or the RPC server ends.
pub(crate) fn lend(
    out: &mut impl Write,
    child: &mut Child,
    output: &Receiver<String>,
    log: &mut File,
    hello: &Value,
) -> io::Result<()> {
    writeln!(out, "\u{1e}hello {hello}")?;
    loop {
        for line in output.try_iter() {
            let _ = writeln!(log, "{line}");
        }
        if let Ok(Some(_)) = child.try_wait() {
            writeln!(out, "\u{1e}end")?;
            return out.flush();
        }
        let (gpus, held) = server::gpus(child.id());
        writeln!(out, "\u{1e}gpus {}", json!({ "gpus": gpus, "vram": held }))?;
        out.flush()?;
        thread::sleep(Duration::from_secs(2));
    }
}

/// `what` is this server's pid or port, or (ignoring case) its name or a part of it, as for
/// `start`; `stop` refuses a `what` that fits more than one server.
fn names_server(r: &RunFile, what: &str) -> bool {
    let port = r.base.rsplit(':').next().unwrap_or_default();
    let name = r.info.alias.to_lowercase();
    what == r.pid.to_string() || what == port || name.contains(&what.to_lowercase())
}

fn stop(rest: &[String]) -> Result<ExitCode, Fail> {
    let words = split(rest, &[], &["--all"])?;
    let servers = server::running_now();
    let chosen: Vec<&RunFile> = if words.flags.contains("--all") {
        if !words.plain.is_empty() {
            return Err(Fail::Usage("--all non va insieme a un nome".into()));
        }
        servers.iter().collect()
    } else {
        let [what] = &words.plain[..] else {
            return Err(Fail::Usage("indica quale server (nome, porta o pid), o --all".into()));
        };
        let found: Vec<&RunFile> = servers.iter().filter(|r| names_server(r, what)).collect();
        if found.is_empty() {
            return Err(format!("nessun server attivo «{what}» (corral status li elenca)").into());
        }
        if found.len() > 1 {
            return Err(format!("«{what}» indica {} server: usa il pid (corral status)", found.len()).into());
        }
        found
    };
    if chosen.is_empty() {
        println!("nessun server attivo");
    }
    for r in chosen {
        print!("fermo {} (pid {})… ", r.info.alias, r.pid);
        let _ = std::io::stdout().flush();
        server::stop_pid(r.pid, Duration::from_secs(15));
        println!("fatto");
    }
    Ok(ExitCode::SUCCESS)
}

// -- services ---------------------------------------------------------------------------------

fn service(rest: &[String]) -> Result<ExitCode, Fail> {
    let Some((action, rest)) = rest.split_first() else {
        return Err(Fail::Usage("indica install, list o remove".into()));
    };
    match action.as_str() {
        "install" => {
            let words = split(rest, &["--preset", "--profile"], &[])?;
            let (state, _) = load_state();
            let target = target(&state, &words)?;
            let exe = env::current_exe().map_err(|e| format!("non trovo il programma corral: {e}"))?;
            let mut cmd = vec![ps(&exe), "start".into()];
            cmd.extend(target.start_words());
            cmd.push("--foreground".into());
            install_service(&service_name(&target.name()), &target.describe(), &cmd)?;
        }
        "list" => list_services()?,
        "remove" => {
            let [name] = rest else {
                return Err(Fail::Usage("indica il servizio da togliere (corral service list)".into()));
            };
            // "corral-x" as listed, or the name it was made from
            let name = name.trim_end_matches(".service").trim_end_matches(".plist");
            let name = if name.starts_with("corral-") { name.to_string() } else { service_name(name) };
            remove_service(&name)?;
        }
        _ => return Err(Fail::Usage(format!("azione sconosciuta: {action}"))),
    }
    Ok(ExitCode::SUCCESS)
}

/// corral-<name>, in the characters every service manager accepts.
fn service_name(name: &str) -> String {
    let clean: String =
        name.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' }).collect();
    format!("corral-{}", clean.trim_matches('-'))
}

/// The names (without `ext`) of Corral's service files in `dir`, sorted.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn service_files(dir: &Path, ext: &str) -> Vec<String> {
    let mut names = vec![];
    for e in fs::read_dir(dir).into_iter().flatten().flatten() {
        let file = e.file_name().to_string_lossy().into_owned();
        if let Some(name) = file.strip_suffix(ext)
            && name.starts_with("corral-")
        {
            names.push(name.to_string());
        }
    }
    names.sort();
    names
}

/// A word for a systemd unit: in double quotes, with \ and " escaped, % doubled (it starts a
/// specifier) and, for ExecStart= (`exec`), $ doubled too (it would expand a variable).
#[cfg(any(target_os = "linux", test))]
fn systemd_word(s: &str, exec: bool) -> String {
    let mut out = String::from('"');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '%' => out.push_str("%%"),
            '$' if exec => out.push_str("$$"),
            '\n' | '\r' => out.push(' '),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The systemd user unit that runs `cmd` (corral start … --foreground) with the data in `home`.
#[cfg(any(target_os = "linux", test))]
pub fn systemd_unit(description: &str, cmd: &[String], home: &Path) -> String {
    let exec: Vec<String> = cmd.iter().map(|w| systemd_word(w, true)).collect();
    let home_var = systemd_word(&format!("CORRAL_HOME={}", ps(home)), false);
    let description = description.replace('%', "%%").replace(['\n', '\r'], " ");
    let lines = [
        "# scritto da `corral service install`; si toglie con `corral service remove`".to_string(),
        "[Unit]".into(),
        format!("Description=Corral: {description}"),
        String::new(),
        "[Service]".into(),
        format!("Environment={home_var}"),
        format!("ExecStart={}", exec.join(" ")),
        "# un'uscita con codice diverso da 0 (un crash) lo riavvia dopo 5 s; `systemctl stop` no".into(),
        "Restart=on-failure".into(),
        "RestartSec=5".into(),
        "# SIGTERM solo a Corral, che ferma llama-server nel modo ordinato".into(),
        "KillMode=mixed".into(),
        "TimeoutStopSec=20".into(),
        String::new(),
        "[Install]".into(),
        "WantedBy=default.target".into(),
    ];
    lines.join("\n") + "\n"
}

/// The launchd agent that runs `cmd` at login with the data in `home` and its messages in `log`.
#[cfg(any(target_os = "macos", test))]
pub fn launchd_plist(label: &str, cmd: &[String], home: &Path, log: &Path) -> String {
    let xml = |s: &str| s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
    let mut lines = vec![
        r#"<?xml version="1.0" encoding="UTF-8"?>"#.to_string(),
        r#"<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">"#
            .into(),
        r#"<plist version="1.0">"#.into(),
        "<dict>".into(),
        format!("  <key>Label</key><string>{}</string>", xml(label)),
        "  <key>ProgramArguments</key>".into(),
        "  <array>".into(),
    ];
    for word in cmd {
        lines.push(format!("    <string>{}</string>", xml(word)));
    }
    lines.extend([
        "  </array>".into(),
        format!(
            "  <key>EnvironmentVariables</key><dict><key>CORRAL_HOME</key><string>{}</string></dict>",
            xml(&ps(home))
        ),
        "  <key>RunAtLoad</key><true/>".into(),
        "  <!-- un'uscita con codice diverso da 0 (un crash) lo riavvia -->".into(),
        "  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>".into(),
        format!("  <key>StandardOutPath</key><string>{}</string>", xml(&ps(log))),
        format!("  <key>StandardErrorPath</key><string>{}</string>", xml(&ps(log))),
        "</dict>".into(),
        "</plist>".into(),
    ]);
    lines.join("\n") + "\n"
}

// systemd (Linux)

#[cfg(target_os = "linux")]
fn unit_dir() -> PathBuf {
    let xdg = env::var_os("XDG_CONFIG_HOME").filter(|_| !cfg!(test)); // tests: their own home
    let config = xdg.map(PathBuf::from).unwrap_or_else(|| home().join(".config"));
    config.join("systemd").join("user")
}

#[cfg(target_os = "linux")]
fn systemctl(args: &[&str]) -> Result<String, String> {
    if cfg!(test) {
        return Ok(String::new()); // the tests never touch the user's services
    }
    let out = capture(Command::new("systemctl").arg("--user").args(args), Duration::from_secs(60))
        .map_err(|e| format!("systemctl: {e}"))?;
    if out.ok { Ok(out.stdout) } else { Err(format!("systemctl --user {}: {}", args.join(" "), out.stderr.trim())) }
}

/// systemd keeps the user's services up without a login session ("linger").
#[cfg(target_os = "linux")]
fn lingering() -> bool {
    if cfg!(test) {
        return false;
    }
    let user = env::var("USER").unwrap_or_default();
    let query = ["show-user", &user, "--property=Linger", "--value"];
    let out = capture(Command::new("loginctl").args(query), Duration::from_secs(10));
    out.is_ok_and(|o| o.stdout.trim() == "yes")
}

#[cfg(target_os = "linux")]
fn install_service(name: &str, description: &str, cmd: &[String]) -> Result<(), String> {
    let dir = unit_dir();
    let file = dir.join(format!("{name}.service"));
    let err = |e: std::io::Error| format!("{}: {e}", file.display());
    fs::create_dir_all(&dir).map_err(err)?;
    fs::write(&file, systemd_unit(description, cmd, &BASE)).map_err(err)?;
    println!("scritto {}", file.display());
    systemctl(&["daemon-reload"])?;
    systemctl(&["enable", "--now", &format!("{name}.service")])?;
    println!("attivo: parte quando accedi e riparte se cade");
    println!("messaggi: journalctl --user -u {name} · log del server in {}", logs().display());
    if !lingering() {
        println!("per farlo partire con il computer, anche senza accesso (es. una macchina usata via SSH):");
        println!("  loginctl enable-linger");
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn list_services() -> Result<(), String> {
    let names = service_files(&unit_dir(), ".service");
    if names.is_empty() {
        println!("nessun servizio di Corral");
    }
    for name in names {
        // `is-active` exits 3 when the service is stopped: the answer is on stdout either way
        let query = ["--user", "is-active", &format!("{name}.service")];
        let state = capture(Command::new("systemctl").args(query), Duration::from_secs(10)).map(|o| o.stdout);
        println!("{name:<32} {}", state.unwrap_or_default().trim());
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn remove_service(name: &str) -> Result<(), String> {
    let file = unit_dir().join(format!("{name}.service"));
    if !file.exists() {
        return Err(format!("nessun servizio {name} ({} non c'è)", file.display()));
    }
    let _ = systemctl(&["disable", "--now", &format!("{name}.service")]); // also stops its server
    fs::remove_file(&file).map_err(|e| format!("{}: {e}", file.display()))?;
    systemctl(&["daemon-reload"])?;
    println!("tolto {name}");
    Ok(())
}

// launchd (macOS)

#[cfg(target_os = "macos")]
fn agents_dir() -> PathBuf {
    home().join("Library").join("LaunchAgents")
}

/// The user's launchd domain, gui/<uid>.
#[cfg(target_os = "macos")]
fn gui_domain() -> String {
    format!("gui/{}", unsafe { libc::getuid() })
}

#[cfg(target_os = "macos")]
fn launchctl(args: &[&str]) -> Result<(), String> {
    if cfg!(test) {
        return Ok(()); // the tests never touch the user's services
    }
    let out = capture(Command::new("launchctl").args(args), Duration::from_secs(60))
        .map_err(|e| format!("launchctl: {e}"))?;
    if out.ok { Ok(()) } else { Err(format!("launchctl {}: {}", args.join(" "), out.stderr.trim())) }
}

#[cfg(target_os = "macos")]
fn install_service(name: &str, _description: &str, cmd: &[String]) -> Result<(), String> {
    let dir = agents_dir();
    let file = dir.join(format!("{name}.plist"));
    let log = logs().join(format!("{name}.log"));
    let err = |e: std::io::Error| format!("{}: {e}", file.display());
    fs::create_dir_all(&dir).map_err(err)?;
    fs::create_dir_all(logs()).map_err(err)?;
    if file.exists() {
        let _ = launchctl(&["bootout", &format!("{}/{name}", gui_domain())]); // the old version goes
    }
    fs::write(&file, launchd_plist(name, cmd, &BASE, &log)).map_err(err)?;
    println!("scritto {}", file.display());
    launchctl(&["bootstrap", &gui_domain(), &ps(&file)])?;
    println!("attivo: parte quando accedi e riparte se cade · messaggi in {}", log.display());
    Ok(())
}

#[cfg(target_os = "macos")]
fn list_services() -> Result<(), String> {
    let names = service_files(&agents_dir(), ".plist");
    if names.is_empty() {
        println!("nessun servizio di Corral");
    }
    for name in names {
        println!("{name}");
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn remove_service(name: &str) -> Result<(), String> {
    let file = agents_dir().join(format!("{name}.plist"));
    if !file.exists() {
        return Err(format!("nessun servizio {name} ({} non c'è)", file.display()));
    }
    let _ = launchctl(&["bootout", &format!("{}/{name}", gui_domain())]); // also stops its server
    fs::remove_file(&file).map_err(|e| format!("{}: {e}", file.display()))?;
    println!("tolto {name}");
    Ok(())
}

// elsewhere (Windows)

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn install_service(_name: &str, _description: &str, cmd: &[String]) -> Result<(), String> {
    let quoted: Vec<String> =
        cmd.iter().map(|w| if w.contains(' ') { format!("\"{w}\"") } else { w.clone() }).collect();
    Err(format!(
        "su questo sistema non c'è ancora: crea un'attività che all'accesso esegua, con CORRAL_HOME={}:\n  {}",
        BASE.display(),
        quoted.join(" ")
    ))
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn list_services() -> Result<(), String> {
    Err("su questo sistema i servizi non ci sono ancora".into())
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn remove_service(_name: &str) -> Result<(), String> {
    Err("su questo sistema i servizi non ci sono ancora".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn command_words() {
        let w = split(&words("qwen --preset lungo --foreground"), &["--preset", "--profile"], &["--foreground"]);
        let w = w.ok().unwrap();
        assert_eq!((w.plain, w.values["--preset"].as_str()), (vec!["qwen".to_string()], "lungo"));
        assert!(w.flags.contains("--foreground"));
        assert!(matches!(split(&words("--bogus"), &[], &[]), Err(Fail::Usage(_))));
        assert!(matches!(split(&words("--preset"), &["--preset"], &[]), Err(Fail::Usage(_))));
        assert!(since(unix_now()).ends_with('s'));
        assert_eq!(since(unix_now() - 7_500), "2h05");
    }

    #[test]
    fn service_files_are_quoted() {
        assert_eq!(service_name("Qwen3-8B (veloce)"), "corral-qwen3-8b--veloce");
        let cmd = words("/opt/corral start /m/qwen.gguf --preset 100% --foreground");
        let mut cmd = cmd;
        cmd[2] = "/m/my model $HOME.gguf".into(); // a space and a $ in a path
        let unit = systemd_unit("qwen \"100%\"", &cmd, Path::new("/dati/corral"));
        assert!(unit.contains("Description=Corral: qwen \"100%%\"\n"));
        assert!(unit.contains("Environment=\"CORRAL_HOME=/dati/corral\"\n"));
        assert!(unit.contains(
            "ExecStart=\"/opt/corral\" \"start\" \"/m/my model $$HOME.gguf\" \"--preset\" \"100%%\" \"--foreground\"\n"
        ));
        assert!(unit.contains("Restart=on-failure") && unit.contains("KillMode=mixed"));
        let plist =
            launchd_plist("corral-a&b", &words("/opt/corral start <x>"), Path::new("/d"), Path::new("/d/a.log"));
        assert!(plist.contains("<string>corral-a&amp;b</string>") && plist.contains("<string>&lt;x&gt;</string>"));
        assert!(plist.contains("<key>SuccessfulExit</key><false/>"));
    }
}
