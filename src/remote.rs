//! Corral on other computers, through the system's `ssh`: keys, the agent, ProxyJump and the rest
//! of ~/.ssh/config work as in a terminal, with no SSH library here. The host's own Corral does
//! the work (`corral status --json`, `models --json`, `start`, `stop`, `watch`); this side lists
//! the hosts, asks them, and reaches each server through a tunnel (`ssh -L`), so the dashboard,
//! the test prompt and the clients' configs use it like a server on this computer.
//!
//! The same connection can lend a host's GPUs instead (`corral rpc` there, `rpc_links` here):
//! llama.cpp's RPC server listens only on the host's loopback, and this computer's llama-server
//! reaches it through the tunnel, so the RPC server, which asks no password, is never on the
//! network.
//!
//! Limits: the host needs a POSIX shell (Linux, macOS; not Windows), and the login must work with
//! a key: ssh runs in the background with BatchMode, where a password or a new host key can't be
//! asked.

use std::env;
use std::fmt;
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;

use crate::agent::Served;
use crate::dashboard::Info;
use crate::preset::free_port;
use crate::server::{Gpu, detach};
use crate::ui::*;

/// The oldest Corral on a host that speaks this protocol (`status --json` as an object, `watch`).
pub const MIN_VERSION: &str = "0.4.0";

/// This Corral's version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// -- the hosts ------------------------------------------------------------------------------------

/// The hosts named in ~/.ssh/config and in the files it includes, in order. A pattern (`*`, `?`,
/// `!`) is not a host.
pub fn config_hosts() -> Vec<String> {
    let dir = home().join(".ssh");
    let mut hosts = vec![];
    read_config(&dir.join("config"), &dir, 0, &mut hosts);
    hosts
}

/// The `Host` names of one ssh config file, added to `hosts`; its `Include`s are read in place.
pub fn read_config(path: &Path, ssh_dir: &Path, depth: usize, hosts: &mut Vec<String>) {
    if depth > 8 {
        return; // files that include each other
    }
    let Ok(text) = fs::read_to_string(path) else { return };
    for line in text.lines() {
        let (keyword, args) = config_line(line);
        if keyword == "host" {
            for name in args {
                // a name starting with - would reach ssh as an option
                let pattern = name.contains(['*', '?', '!']) || name.starts_with('-');
                if !pattern && !hosts.contains(&name) {
                    hosts.push(name);
                }
            }
        } else if keyword == "include" {
            for pattern in args {
                for file in included(&pattern, ssh_dir) {
                    read_config(&file, ssh_dir, depth + 1, hosts);
                }
            }
        }
    }
}

/// An ssh config line: its keyword in lower case and its arguments ("Host a b", "Host=a").
fn config_line(line: &str) -> (String, Vec<String>) {
    let line = line.trim();
    if line.starts_with('#') {
        return (String::new(), vec![]);
    }
    let end = line.find(|c: char| c.is_whitespace() || c == '=').unwrap_or(line.len());
    let (keyword, rest) = line.split_at(end);
    let rest = rest.trim_start();
    let rest = rest.strip_prefix('=').unwrap_or(rest);
    (keyword.to_lowercase(), shell_split(rest))
}

/// The files an `Include` names: relative to ~/.ssh, `~/` for the home, `*` and `?` in the file
/// name (sorted, as ssh reads them).
fn included(pattern: &str, ssh_dir: &Path) -> Vec<PathBuf> {
    let path = match pattern.strip_prefix("~/") {
        Some(rest) => home().join(rest),
        None => ssh_dir.join(pattern), // an absolute pattern replaces ssh_dir
    };
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if !name.contains(['*', '?']) {
        return vec![path];
    }
    let dir = path.parent().unwrap_or(ssh_dir);
    let mut files = vec![];
    for e in fs::read_dir(dir).into_iter().flatten().flatten() {
        if wildcard(&name, &e.file_name().to_string_lossy()) {
            files.push(e.path());
        }
    }
    files.sort();
    files
}

/// `name` fits `pattern`, where `*` is any run of characters and `?` any one character.
fn wildcard(pattern: &str, name: &str) -> bool {
    fn fits(pattern: &[char], name: &[char]) -> bool {
        match pattern.split_first() {
            None => name.is_empty(),
            Some(('*', rest)) => (0..=name.len()).any(|skip| fits(rest, &name[skip..])),
            Some(('?', rest)) => !name.is_empty() && fits(rest, &name[1..]),
            Some((c, rest)) => name.first() == Some(c) && fits(rest, &name[1..]),
        }
    }
    let pattern: Vec<char> = pattern.chars().collect();
    let name: Vec<char> = name.chars().collect();
    fits(&pattern, &name)
}

// -- asking a host ----------------------------------------------------------------------------------

/// Why a host's Corral didn't answer.
#[derive(Debug)]
pub enum Problem {
    /// ssh couldn't get there or in: its message
    Ssh(String),
    /// no Corral on the host
    Missing,
    /// a Corral older than MIN_VERSION: it doesn't know what was asked
    Old,
    /// the host's Corral answered with an error: its message
    Corral(String),
}

impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Problem::Ssh(msg) | Problem::Corral(msg) => write!(f, "{msg}"),
            Problem::Missing => write!(f, "Corral non c'è"),
            Problem::Old => write!(f, "c'è un Corral precedente alla {MIN_VERSION}"),
        }
    }
}

impl Problem {
    /// What the user can do about it ("" = nothing to add).
    pub fn hint(&self, host: &str) -> String {
        let same = format!("{} {}", env::consts::OS, env::consts::ARCH);
        match self {
            Problem::Ssh(msg) if msg.contains("Permission denied") => {
                format!("serve l'accesso con una chiave, senza password: ssh-copy-id {host}")
            }
            Problem::Ssh(msg) if msg.contains("Host key verification failed") => {
                format!("collegati una volta con «ssh {host}» per accettare la chiave dell'host")
            }
            Problem::Missing | Problem::Old => {
                format!("u: ci copio questo Corral {VERSION} (se l'host è {same}); o installalo lì in ~/.local/bin")
            }
            _ => String::new(),
        }
    }
}

/// ssh's options for work in the background: no prompts (nobody would see them: see the module's
/// limits), a bounded wait for the connection, no terminal on the other end.
const BACKGROUND: [&str; 5] = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-T"];

/// The user's ssh (in the tests a fake one that plays the hosts on this computer, see `fake`).
fn ssh() -> Command {
    #[cfg(all(test, unix))]
    return Command::new(crate::fake::llama_cpp().join("ssh"));
    #[cfg(not(all(test, unix)))]
    Command::new("ssh")
}

fn scp() -> Command {
    #[cfg(all(test, unix))]
    return Command::new(crate::fake::llama_cpp().join("scp"));
    #[cfg(not(all(test, unix)))]
    Command::new("scp")
}

/// The command line, for the host's shell, that runs its Corral with `args`. A command over ssh
/// gets a shell that may not have read the login profile, so ~/.local/bin (where `copy_corral`
/// puts it) can be missing from PATH: it comes first, then PATH, then cargo's folder.
pub fn corral_command(args: &[&str]) -> String {
    const FIND: &str = r#"for c in "$HOME/.local/bin/corral" corral "$HOME/.cargo/bin/corral"; do command -v "$c" > /dev/null && exec "$c" "$@"; done; echo "corral non trovato" >&2; exit 127"#;
    let mut words = vec!["sh".to_string(), "-c".into(), FIND.into(), "corral".into()];
    words.extend(args.iter().map(|a| a.to_string()));
    shell_join(&words)
}

/// `script` for the host's sh, whatever the login shell is.
fn sh_command(script: &str) -> String {
    shell_join(&["sh".to_string(), "-c".into(), script.into()])
}

/// The last line with something on it ("" = none).
fn last_line(text: &str) -> String {
    text.lines().map(str::trim).rfind(|l| !l.is_empty()).unwrap_or("").to_string()
}

/// Run `command` (a line for the host's shell) on `host`: its output, or ssh's problem.
fn run(host: &str, command: &str, timeout: Duration) -> Result<Output, Problem> {
    let mut c = ssh();
    c.args(BACKGROUND).arg(host).arg(command);
    let out = capture(&mut c, timeout).map_err(|e| Problem::Ssh(format!("ssh: {e}")))?;
    if out.code == 255 {
        let msg = last_line(&out.stderr);
        return Err(Problem::Ssh(if msg.is_empty() { "ssh non riesce a collegarsi".into() } else { msg }));
    }
    Ok(out)
}

/// Run the host's Corral with `args`: what it printed, or why not.
fn call(host: &str, args: &[&str], timeout: Duration) -> Result<String, Problem> {
    let out = run(host, &corral_command(args), timeout)?;
    match out.code {
        0 => Ok(out.stdout),
        127 => Err(Problem::Missing),
        // a wrong command or option: one this Corral uses and the host's doesn't know yet
        2 => Err(Problem::Old),
        _ => Err(Problem::Corral(out.stderr.trim().to_string())),
    }
}

/// Version `a` comes before `b` ("0.3.0" before "0.4.0"; a part that isn't a number counts as 0).
pub fn older(a: &str, b: &str) -> bool {
    let parts = |v: &str| -> Vec<u32> { v.split('.').map(|p| p.parse().unwrap_or(0)).collect() };
    parts(a) < parts(b)
}

/// A server running on a host, as its `corral status --json` describes it.
#[derive(Clone, Deserialize)]
pub struct Running {
    pub pid: u32,
    /// where the host reaches it
    pub url: String,
    /// ok, loading, error, no-answer
    pub health: String,
    pub started: u64, // unix time
    pub keep: bool,
    pub log: String,
    pub info: Info,
    /// a router: its models as clients see them
    #[serde(default)]
    pub served: Vec<Served>,
}

/// What a host's Corral says about itself.
#[derive(Deserialize)]
pub struct Status {
    pub version: String,
    pub os: String,
    pub arch: String,
    pub gpus: Vec<Gpu>,
    pub servers: Vec<Running>,
}

pub fn status(host: &str) -> Result<Status, Problem> {
    let out = call(host, &["status", "--json"], Duration::from_secs(30))?;
    // 0.3.0 printed a list: not an object
    let status: Status = serde_json::from_str(&out).map_err(|_| Problem::Old)?;
    if older(&status.version, MIN_VERSION) {
        return Err(Problem::Old);
    }
    Ok(status)
}

/// A model on a host, as its `corral models --json` lists it.
#[derive(Clone, Debug, Deserialize)]
pub struct Model {
    pub path: String,
    pub size: u64,
    pub presets: Vec<String>,
    /// the one in use there
    pub preset: String,
}

/// A multi-model profile on a host.
#[derive(Clone, Debug, Deserialize)]
pub struct Profile {
    pub name: String,
    pub models: usize,
}

/// What a host can start.
#[derive(Debug, Default, Deserialize)]
pub struct Models {
    pub models: Vec<Model>,
    pub profiles: Vec<Profile>,
}

pub fn models(host: &str) -> Result<Models, Problem> {
    let out = call(host, &["models", "--json"], Duration::from_secs(60))?;
    serde_json::from_str(&out).map_err(|_| Problem::Old)
}

/// `corral start <words>` on the host. It answers once the server is ready or has failed, which
/// a large model can make take minutes. Ok: the line with its address.
pub fn start(host: &str, words: &[String]) -> Result<String, Problem> {
    let mut args = vec!["start"];
    args.extend(words.iter().map(String::as_str));
    let out = call(host, &args, Duration::from_secs(900))?;
    Ok(last_line(&out))
}

/// `corral stop <pid>` on the host (graceful, a kill after 15 s).
pub fn stop(host: &str, pid: u32) -> Result<(), Problem> {
    call(host, &["stop", &pid.to_string()], Duration::from_secs(60)).map(|_| ())
}

/// The host's system and processor, as Rust names them ("linux", "x86_64").
pub fn platform(host: &str) -> Result<(String, String), Problem> {
    let out = run(host, "uname -sm", Duration::from_secs(30))?;
    let text = out.stdout.trim();
    let (os, arch) = text.split_once(' ').unwrap_or((text, ""));
    let os = match os {
        "Linux" => "linux".to_string(),
        "Darwin" => "macos".to_string(),
        other => other.to_lowercase(),
    };
    let arch = match arch {
        "arm64" => "aarch64",
        "amd64" => "x86_64",
        other => other,
    };
    Ok((os, arch.to_string()))
}

/// Copy this program to the host's ~/.local/bin/corral, when the host has the same system and
/// processor. The copy must start there before it replaces anything: a different C library can
/// stop it, and then the Corral already there (if any) stays.
pub fn copy_corral(host: &str) -> Result<(), Problem> {
    let (os, arch) = platform(host)?;
    if os != env::consts::OS || arch != env::consts::ARCH {
        let here = format!("{} {}", env::consts::OS, env::consts::ARCH);
        return Err(Problem::Corral(format!("l'host è {os} {arch}, questo Corral è per {here}: installalo lì")));
    }
    let exe = env::current_exe().map_err(|e| Problem::Corral(format!("non trovo questo programma: {e}")))?;
    // the folder as the host's shell names it: scp's relative paths would follow the login's home
    let out = run(
        host,
        &sh_command(r#"mkdir -p "$HOME/.local/bin" && cd "$HOME/.local/bin" && pwd"#),
        Duration::from_secs(30),
    )?;
    if out.code != 0 {
        return Err(Problem::Corral(out.stderr.trim().to_string()));
    }
    let target = format!("{host}:{}/corral.new", out.stdout.trim());
    let mut scp = scp();
    scp.args(["-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]).arg(&exe).arg(&target);
    let out = capture(&mut scp, Duration::from_secs(600)).map_err(|e| Problem::Ssh(format!("scp: {e}")))?;
    if !out.ok {
        return Err(Problem::Ssh(format!("scp: {}", last_line(&out.stderr))));
    }
    let check = r#"cd "$HOME/.local/bin" && chmod 755 corral.new && ./corral.new status --json > /dev/null && mv -f corral.new corral || { rm -f corral.new; exit 1; }"#;
    let out = run(host, &sh_command(check), Duration::from_secs(60))?;
    if out.code != 0 {
        return Err(Problem::Corral(format!("la copia non parte sull'host: {}", last_line(&out.stderr))));
    }
    Ok(())
}

/// The interactive ssh that opens the host's Corral in this terminal (it may ask a password).
pub fn session(host: &str) -> Command {
    let mut c = ssh();
    c.args(["-t", "-o", "ConnectTimeout=10"]).arg(host).arg(corral_command(&[]));
    c
}

// -- the link to a server ---------------------------------------------------------------------------

/// The live connection to a server on a host: a tunnel from 127.0.0.1:`port` here to the server,
/// and on the same ssh connection the host's `corral watch`, which sends the server's log, the
/// host's GPUs and, at the end, that the server ended.
pub struct Link {
    pub ssh: Child,
    pub port: u16,
    /// what the host's Corral said first (`watch`: the server's API key; `rpc`: its devices)
    pub hello: Value,
    pub lines: Receiver<String>,
    pub gpus: Arc<Mutex<(Vec<Gpu>, Option<u64>)>>,
    /// the host said the server ended
    pub ended: Arc<AtomicBool>,
    /// the connection is gone, its last lines read
    pub closed: Arc<AtomicBool>,
    /// what ssh said: why the connection dropped
    pub errors: Arc<Mutex<String>>,
}

/// Where the host's ssh server connects for the server at `url` (the address the host uses for
/// it): the loopback for one that listens on every address.
pub fn tunnel_target(url: &str) -> Option<(String, u16)> {
    let rest = url.strip_prefix("http://").or_else(|| url.strip_prefix("https://"))?;
    let authority = rest.split('/').next()?;
    let (host, port) = authority.rsplit_once(':')?;
    let host = match host {
        "localhost" | "0.0.0.0" | "[::]" => "127.0.0.1",
        other => other,
    };
    Some((host.to_string(), port.parse().ok()?))
}

/// Open the link to server `r` on `host` (blocking: run it in a task). The port here is the
/// server's own when it's free.
pub fn link(host: &str, r: &Running) -> Result<Link, String> {
    let (target, target_port) = tunnel_target(&r.url).ok_or(format!("indirizzo che non capisco: {}", r.url))?;
    let port = free_port("127.0.0.1", target_port).ok_or("nessuna porta libera su questo computer")?;
    let forward = format!("127.0.0.1:{port}:{target}:{target_port}");
    let mut c = ssh();
    c.args(BACKGROUND);
    // the tunnel or nothing; a dead connection shows within a minute
    c.args(["-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]);
    c.args(["-L", &forward]).arg(host).arg(corral_command(&["watch", &r.pid.to_string()]));
    // out of the terminal's reach: a Ctrl+C meant for another program (the host's Corral opened
    // here) must not close the tunnels. If Corral dies without closing it, ssh follows in a few
    // seconds: the host's next GPU line (every 2 s) finds no reader here.
    detach(&mut c);
    connect(c, port)
}

/// A line from the host's Corral rather than from the server's log: \x1e, a kind, its data.
fn control(line: &str) -> Option<(&str, &str)> {
    let rest = line.strip_prefix('\u{1e}')?;
    Some(rest.split_once(' ').unwrap_or((rest, "")))
}

/// Hand each line of `r` to `on_line` until the end, or until it returns false.
fn each_line(r: impl Read, mut on_line: impl FnMut(String) -> bool) {
    let mut reader = BufReader::new(r);
    let mut buf = vec![];
    while reader.read_until(b'\n', &mut buf).is_ok_and(|n| n > 0) {
        let line = String::from_utf8_lossy(&buf).trim_end().to_string();
        buf.clear();
        if !on_line(line) {
            return;
        }
    }
}

#[derive(Deserialize)]
struct GpuLine {
    gpus: Vec<Gpu>,
    vram: Option<u64>,
}

/// Start `cmd` (the ssh of `link`, whose tunnel ends here at `port`) and wait for the hello of
/// the host's Corral: by then the tunnel is up.
pub fn connect(mut cmd: Command, port: u16) -> Result<Link, String> {
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut ssh = cmd.spawn().map_err(|e| format!("ssh: {e}"))?;
    let (Some(stdout), Some(stderr)) = (ssh.stdout.take(), ssh.stderr.take()) else {
        return Err("ssh: niente da leggere".into());
    };
    let errors = Arc::new(Mutex::new(String::new()));
    let errors_in = errors.clone();
    let stderr_read = thread::spawn(move || {
        each_line(stderr, |line| {
            let mut e = lock(&errors_in);
            e.push_str(&line);
            e.push('\n');
            // the last lines are the ones that tell why
            let lines: Vec<&str> = e.lines().collect();
            if lines.len() > 20 {
                *e = lines[lines.len() - 20..].join("\n") + "\n";
            }
            true
        })
    });
    let gpus = Arc::new(Mutex::new((vec![], None)));
    let ended = Arc::new(AtomicBool::new(false));
    let closed = Arc::new(AtomicBool::new(false));
    let (hello_tx, hello) = mpsc::channel::<String>();
    let (lines_tx, lines) = mpsc::channel::<String>();
    let (gpus_in, ended_in, closed_in) = (gpus.clone(), ended.clone(), closed.clone());
    thread::spawn(move || {
        let mut hello_tx = Some(hello_tx);
        each_line(stdout, |line| {
            match control(&line) {
                Some(("hello", data)) => {
                    if let Some(tx) = hello_tx.take() {
                        let _ = tx.send(data.to_string());
                    }
                }
                Some(("gpus", data)) => {
                    if let Ok(g) = serde_json::from_str::<GpuLine>(data) {
                        *lock(&gpus_in) = (g.gpus, g.vram);
                    }
                }
                Some(("end", _)) => ended_in.store(true, Ordering::Relaxed),
                Some(_) => {} // a kind from a newer Corral: not for this one
                // before the hello: the host's shell talking (a greeting), not the log
                None if hello_tx.is_some() => {}
                None => return lines_tx.send(line).is_ok(),
            }
            true
        });
        closed_in.store(true, Ordering::Relaxed);
    });
    // 60 s: an RPC server first loads its GPU backends
    match hello.recv_timeout(Duration::from_secs(60)) {
        Ok(data) => {
            let hello = serde_json::from_str(&data).unwrap_or(Value::Null);
            Ok(Link { ssh, port, hello, lines, gpus, ended, closed, errors })
        }
        Err(_) => {
            // ssh ended, or nothing came: its last words say why
            let _ = ssh.kill();
            let _ = ssh.wait();
            let _ = stderr_read.join();
            let why = last_line(&lock(&errors));
            Err(if why.is_empty() { "nessuna risposta dall'host".into() } else { why })
        }
    }
}

// -- lending GPUs (llama.cpp RPC) --------------------------------------------------------------------

/// The RPC server's own default port; the next ones are tried when it's taken on the host.
pub const RPC_PORT: u16 = 50052;

/// Where the tunnels to RPC servers start here: a range of their own, so that a host that is
/// this computer as well doesn't find its RPC port taken by the tunnel's end.
const RPC_TUNNEL_PORT: u16 = RPC_PORT + 100;

/// How the host's `corral rpc` says its port is taken (the next one is tried).
pub const PORT_TAKEN: &str = "porta occupata";

/// A host's GPUs lent to this computer: its RPC server, reached at 127.0.0.1:`port` here. It
/// stays up as long as this value: dropping it closes the connection, and the host's Corral then
/// stops the RPC server.
pub struct RpcLink {
    pub host: String,
    pub port: u16,
    /// the host's llama.cpp build: RPC wants compatible versions on both sides
    pub version: String,
    /// the GPUs lent, as llama.cpp lists them ("Vulkan0: NVIDIA … (24210 MiB, 23700 MiB free)")
    pub devices: Vec<String>,
    link: Link,
}

impl RpcLink {
    /// The link to `host`'s `corral rpc`, with what its hello says.
    pub fn new(host: &str, link: Link) -> Self {
        let version = link.hello["version"].as_str().unwrap_or("?").to_string();
        let list = link.hello["devices"].as_array().cloned().unwrap_or_default();
        let devices = list.iter().filter_map(|d| d.as_str()).map(String::from).collect();
        Self { host: host.into(), port: link.port, version, devices, link }
    }

    /// The host's GPUs, as its Corral sends them every 2 s.
    pub fn gpus(&self) -> Arc<Mutex<(Vec<Gpu>, Option<u64>)>> {
        self.link.gpus.clone()
    }

    /// The connection dropped, or the RPC server ended.
    pub fn closed(&self) -> bool {
        self.link.closed.load(Ordering::Relaxed)
    }
}

impl Drop for RpcLink {
    fn drop(&mut self) {
        let _ = self.link.ssh.kill();
        let _ = self.link.ssh.wait();
    }
}

/// Lend the GPUs of `host`: its RPC server through a tunnel whose end here is the first free port
/// from `from` (blocking: run it in a task).
pub fn rpc_link(host: &str, from: u16) -> Result<RpcLink, String> {
    let mut why = String::new();
    for host_port in RPC_PORT..RPC_PORT + 5 {
        let port = free_port("127.0.0.1", from).ok_or("nessuna porta libera su questo computer")?;
        let forward = format!("127.0.0.1:{port}:127.0.0.1:{host_port}");
        let mut c = ssh();
        c.args(BACKGROUND);
        c.args(["-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]);
        c.args(["-L", &forward]).arg(host).arg(corral_command(&["rpc", "--port", &host_port.to_string()]));
        detach(&mut c); // as in `link`
        match connect(c, port) {
            Ok(link) => return Ok(RpcLink::new(host, link)),
            Err(e) if e.contains(PORT_TAKEN) => why = e, // another RPC server there: the next port
            Err(e) => return Err(format!("{host}: {e}")),
        }
    }
    Err(format!("{host}: {why}"))
}

/// The GPUs of every host in `hosts`, all at once; one that fails fails them all (the others
/// close again). Each looks for its port here from a start of its own: at the same moment two
/// of them would find the same one free.
pub fn rpc_links(hosts: &[String]) -> Result<Vec<RpcLink>, String> {
    let mut tries = vec![];
    for (i, h) in hosts.iter().enumerate() {
        let host = h.clone();
        let from = RPC_TUNNEL_PORT + 10 * i as u16;
        tries.push(thread::spawn(move || rpc_link(&host, from)));
    }
    let mut links = vec![];
    for t in tries {
        let result = t.join().unwrap_or_else(|_| Err("collegamento RPC interrotto".into()));
        links.push(result?);
    }
    Ok(links)
}
