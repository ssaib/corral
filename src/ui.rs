//! Shared palette, paths, state, HTTP, child processes and the few host facts std doesn't
//! expose (RAM, disk, local time). No terminal output here: screens live in `tui`.

use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, Read, Write};
use std::path::{MAIN_SEPARATOR, Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{LazyLock, Mutex, MutexGuard, Once, PoisonError};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Data root (state.json, installs/, models/, logs/): $CORRAL_HOME, else the pre-rename
/// $LLAMA_MANAGER_HOME, else ~/.corral (or ~/.llama-manager when only that one exists).
pub static BASE: LazyLock<PathBuf> = LazyLock::new(|| {
    if cfg!(test) {
        // the fake ssh's other computer (see `fake::host`)
        if let Some(dir) = env::var_os("CORRAL_FAKE_BASE") {
            return PathBuf::from(dir);
        }
        // tests never touch the real data: an empty folder of their own, one per test run
        let dir = env::temp_dir().join(format!("corral-test-{}", std::process::id())).join("data");
        let _ = fs::create_dir_all(&dir);
        return dir;
    }
    let env_dir = env::var_os("CORRAL_HOME").or_else(|| env::var_os("LLAMA_MANAGER_HOME"));
    let p = env_dir.map_or_else(|| data_dir(&home()), PathBuf::from);
    std::path::absolute(&p).unwrap_or(p)
});

pub fn data_dir(home: &Path) -> PathBuf {
    let (new, old) = (home.join(".corral"), home.join(".llama-manager"));
    if !new.exists() && old.is_dir() { old } else { new }
}

pub fn home() -> PathBuf {
    if cfg!(test) {
        // tests see an empty home of their own: no ~/.ssh/config hosts, no model caches
        let dir = BASE.parent().map_or_else(env::temp_dir, Path::to_path_buf).join("home");
        let _ = fs::create_dir_all(&dir);
        return dir;
    }
    env::home_dir().unwrap_or_default()
}

pub fn logs() -> PathBuf {
    BASE.join("logs")
}

/// logs/*.log, newest first: (path, size, modified).
pub fn log_files() -> Vec<(PathBuf, u64, std::time::SystemTime)> {
    let mut v: Vec<_> = fs::read_dir(logs())
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().ends_with(".log"))
        .filter_map(|e| {
            let m = e.metadata().ok()?;
            Some((e.path(), m.len(), m.modified().ok()?))
        })
        .collect();
    v.sort_by_key(|f| std::cmp::Reverse(f.2));
    v
}

/// Keep the newest `keep` logs of each kind (llama-*.log, build-*.log, …: the part before the first '-').
pub fn rotate_logs(keep: usize) {
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for (p, ..) in log_files() {
        let name = fname(&ps(&p)).to_string();
        let n = seen.entry(name.split('-').next().unwrap_or("").to_string()).or_default();
        *n += 1;
        if *n > keep {
            let _ = fs::remove_file(&p);
        }
    }
}

// The palette. Neutrals in three depths (canvas, bars and dialogs, selection), text in three
// weights (FG, DIM, FAINT), one brand accent (SIDE, lavender) and colours that mean a state or a
// data series. Borders stay neutral (LINE) unless their colour says something.
pub const BG: u32 = 0x16171d; // canvas
pub const SURFACE: u32 = 0x1f2128; // header and footer bars, dialogs, toasts
pub const LINE: u32 = 0x363a46; // borders and rules
pub const FG: u32 = 0xe2e4e9;
pub const DIM: u32 = 0x8e93a3; // secondary text
pub const FAINT: u32 = 0x5f6474; // row numbers, column heads, separators
pub const SIDE: u32 = 0xb49cf5; // brand lavender: selection, focus, the wordmark
pub const WOOL: u32 = 0xe2c79c; // the llama's wool: brand second colour
pub const SRV: u32 = 0xf29d6c; // server lines, prompt processing
pub const SLOT: u32 = 0x86a9f7; // requests and slots, generation
pub const LOAD: u32 = 0x6fd39c; // model loading, memory
pub const OK: u32 = 0x6fd39c;
pub const WARN: u32 = 0xf0c46e;
pub const ERR: u32 = 0xf07c7c;
pub const CYAN: u32 = 0x6cc5d2;

// ponytail: only Apple Terminal is treated as 256-colour; add a COLORTERM check if another terminal shows wrong colours.
static TRUECOLOR: LazyLock<bool> = LazyLock::new(|| env::var("TERM_PROGRAM").map_or(true, |t| t != "Apple_Terminal"));

/// Palette entry as a terminal colour (nearest xterm-256 entry where truecolor is missing).
pub fn rgb(hex: u32) -> Color {
    let (r, g, b) = ((hex >> 16) as u8, (hex >> 8) as u8, hex as u8);
    if *TRUECOLOR {
        return Color::Rgb(r, g, b);
    }
    if r.max(g).max(b) - r.min(g).min(b) < 24 {
        let avg = (u16::from(r) + u16::from(g) + u16::from(b)) / 3;
        return Color::Indexed(match avg {
            0..8 => 16,
            239.. => 231,
            _ => 232 + ((avg - 8) / 10) as u8,
        });
    }
    let q = |v: u8| {
        if v < 48 {
            0
        } else if v < 115 {
            1
        } else {
            (v - 35) / 40
        }
    };
    Color::Indexed(16 + 36 * q(r) + 6 * q(g) + q(b))
}

pub fn st(hex: u32) -> Style {
    Style::new().fg(rgb(hex))
}

pub fn bold(hex: u32) -> Style {
    st(hex).add_modifier(Modifier::BOLD)
}

pub fn sp(text: impl Into<String>, style: Style) -> Span<'static> {
    Span::styled(text.into(), style)
}

/// Plain table cell / text in the default foreground.
pub fn cell(text: impl Into<String>) -> Span<'static> {
    sp(text, st(FG))
}

pub fn len(s: &str) -> usize {
    s.chars().count() // ponytail: one column per char; fine for model names and the glyphs used here
}

pub fn clip(s: &str, n: usize) -> String {
    if len(s) <= n { s.to_string() } else { s.chars().take(n.saturating_sub(1)).chain(['…']).collect() }
}

pub fn size(n: u64) -> String {
    let mut n = n as f64;
    for unit in ["B", "KiB", "MiB"] {
        if n < 1024.0 {
            return if unit == "MiB" { format!("{n:.1} {unit}") } else { format!("{n:.0} {unit}") };
        }
        n /= 1024.0;
    }
    format!("{n:.1} GiB")
}

// -- paths & strings ----------------------------------------------------------------------------

pub fn ps(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// Last component of a local or repo path.
pub fn fname(s: &str) -> &str {
    s.rsplit(['/', '\\']).next().unwrap_or(s)
}

/// Paths relative to the data dir (and its parent) for display.
pub fn short(s: &str) -> String {
    let mut s = s.to_string();
    for root in [Some(BASE.as_path()), BASE.parent()].into_iter().flatten() {
        s = s.replace(&format!("{}{MAIN_SEPARATOR}", root.display()), "");
    }
    s
}

pub fn expand(p: &str) -> PathBuf {
    let p = match p.strip_prefix('~') {
        Some(rest) if rest.is_empty() || rest.starts_with(['/', '\\']) => {
            home().join(rest.trim_start_matches(['/', '\\']))
        }
        _ => PathBuf::from(p),
    };
    std::path::absolute(&p).unwrap_or(p)
}

/// shlex.split: blanks separate words, '…' is literal, "…" honours \" and \\; outside quotes a
/// backslash escapes the next char. On Windows, where it is a path separator, only \" is special
/// (as for CommandLineToArgvW), so "\\nas\share" stays a UNC path.
pub fn shell_split(s: &str) -> Vec<String> {
    let mut out = vec![];
    let mut cur = String::new(); // the word being read
    let mut word = false; // a word has started (also an empty one: '')
    let mut quote = None; // the quote we are inside of
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        match (quote, c) {
            (None, c) if c.is_whitespace() => {
                if word {
                    out.push(std::mem::take(&mut cur));
                    word = false;
                }
            }
            (None, '\'' | '"') => {
                quote = Some(c);
                word = true;
            }
            (Some(q), c) if c == q => quote = None,
            (Some('"'), '\\') if it.peek() == Some(&'"') || (it.peek() == Some(&'\\') && !cfg!(windows)) => {
                cur.extend(it.next())
            }
            (None, '\\') if !cfg!(windows) => {
                cur.extend(it.next());
                word = true;
            }
            (_, c) => {
                cur.push(c);
                word = true;
            }
        }
    }
    if word {
        out.push(cur);
    }
    out
}

/// shlex.join, for display and logs.
pub fn shell_join(args: &[String]) -> String {
    let safe = |a: &String| !a.is_empty() && a.chars().all(|c| c.is_alphanumeric() || "@%+=:,./-_".contains(c));
    args.iter()
        .map(|a| if safe(a) { a.clone() } else { format!("'{}'", a.replace('\'', r#"'"'"'"#)) })
        .collect::<Vec<_>>()
        .join(" ")
}

// -- state ------------------------------------------------------------------------------------

pub type Preset = BTreeMap<String, String>;
/// A model's presets by name.
pub type Named = BTreeMap<String, Preset>;

pub const STATE_VERSION: u32 = 2;

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct State {
    /// 2 = named presets; missing (1) = one flat preset per model, read as "default"
    #[serde(default)]
    pub version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub install: Option<String>,
    /// model path -> preset name -> values that differ from the defaults
    #[serde(default, deserialize_with = "named_presets")]
    pub presets: BTreeMap<String, Named>,
    /// model path -> name of the preset in use (missing = the first one)
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub preset_sel: BTreeMap<String, String>,
    /// model path -> stamp() of the last server start
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub last_used: BTreeMap<String, String>,
    /// extra folders scanned for GGUF files
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub model_dirs: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_model: Option<String>,
    /// multi-model profiles (llama-server router mode) by name
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub routers: BTreeMap<String, crate::router::Profile>,
    /// news screen: unix time of the last visit (newer items are marked), the RSS/Atom feeds
    /// (None = the defaults)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub news_seen: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feeds: Option<Vec<String>>,
}

pub const DEFAULT_PRESET: &str = "default";

/// Reads both layouts: {model: {name: {k: v}}} and the old flat {model: {k: v}}.
fn named_presets<'de, D: serde::Deserializer<'de>>(d: D) -> Result<BTreeMap<String, Named>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Entry {
        Named(Named),
        Flat(Preset),
    }
    let raw: BTreeMap<String, Entry> = Deserialize::deserialize(d)?;
    Ok(raw
        .into_iter()
        .map(|(model, e)| match e {
            Entry::Named(n) => (model, n),
            Entry::Flat(p) => (model, Named::from([(DEFAULT_PRESET.to_string(), p)])),
        })
        .collect())
}

pub fn load_state() -> (State, Option<String>) {
    load_state_at(&BASE.join("state.json"))
}

pub fn save_state(state: &State) -> Result<(), String> {
    #[cfg(test)]
    if !crate::fake::persist() {
        return Ok(()); // the screens' tests keep the state in memory: see `fake::disk`
    }
    save_state_at(state, &BASE.join("state.json"))
}

/// The state in `path`, or the defaults on the first run. A file that can't be read is moved
/// aside to <path>.bad (the next save would overwrite it) and the message says so.
pub fn load_state_at(path: &Path) -> (State, Option<String>) {
    if !path.exists() {
        return (State::default(), None);
    }
    let read = fs::read_to_string(path).map_err(|e| e.to_string());
    match read.and_then(|text| serde_json::from_str(&text).map_err(|e| e.to_string())) {
        Ok(state) => (state, None),
        Err(e) => {
            let bad = path.with_extension("json.bad");
            let _ = fs::rename(path, &bad);
            let msg = format!("state.json illeggibile ({e}): spostato in {}, riparto dai default", bad.display());
            (State::default(), Some(msg))
        }
    }
}

/// Write the state through a temporary file renamed over `path`: a crash or a full disk halfway
/// leaves the previous file whole instead of a truncated one. Private: presets hold API keys.
pub fn save_state_at(state: &State, path: &Path) -> Result<(), String> {
    let tmp = path.with_extension("json.tmp");
    let err = |e: String| format!("impossibile salvare {}: {e}", path.display());
    let json = serde_json::to_string_pretty(&State { version: STATE_VERSION, ..state.clone() })
        .map_err(|e| err(e.to_string()))?;
    write_private(&tmp, &json).and_then(|()| fs::rename(&tmp, path)).map_err(|e| err(e.to_string()))
}

/// Write a file only its owner can read (unix: mode 600), for anything holding an API key.
/// Windows: left to the folder's permissions (the user profile is private by default).
pub fn write_private(path: &Path, text: &str) -> io::Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600); // a new file is private from the start
    let mut file = options.open(path)?;
    #[cfg(unix)]
    {
        // an existing file keeps its old mode on open: set it before writing the secret
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    file.write_all(text.as_bytes())
}

// -- HTTP -----------------------------------------------------------------------------------------

fn agent(proxy: Option<ureq::Proxy>) -> ureq::Agent {
    let b = ureq::AgentBuilder::new()
        .user_agent("corral")
        .timeout_connect(Duration::from_secs(30))
        .timeout_read(Duration::from_secs(60))
        .redirect_auth_headers(ureq::RedirectAuthHeaders::SameHost)
        .try_proxy_from_env(false);
    match proxy {
        Some(p) => b.proxy(p),
        None => b,
    }
    .build()
}

/// Direct agent, never proxied: for the local server (127.0.0.1 / localhost).
pub static HTTP: LazyLock<ureq::Agent> = LazyLock::new(|| agent(None));

/// Internet agents for http:// and https:// URLs, each through its proxy from the environment
/// when there is one.
// ponytail: ureq 2 speaks http:// and socks:// proxies only; an https:// proxy URL is ignored (direct).
static VIA_HTTP: LazyLock<ureq::Agent> = LazyLock::new(|| proxied(false));
static VIA_HTTPS: LazyLock<ureq::Agent> = LazyLock::new(|| proxied(true));

fn proxied(https: bool) -> ureq::Agent {
    let env = |k: &str| env::var(k).ok();
    agent(proxy_env(https, &env).and_then(|p| ureq::Proxy::new(p).ok()))
}

/// curl's rules: https URLs use HTTPS_PROXY then ALL_PROXY, http URLs HTTP_PROXY then ALL_PROXY;
/// upper or lower case (lower first, like curl).
pub fn proxy_env(https: bool, env: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    let names: &[&str] = if https { &["https_proxy", "HTTPS_PROXY"] } else { &["http_proxy", "HTTP_PROXY"] };
    names.iter().chain(&["all_proxy", "ALL_PROXY"]).find_map(|k| env(k).filter(|v| !v.trim().is_empty()))
}

/// NO_PROXY: comma-separated hosts or domain suffixes ("example.com", ".example.com", "*" = all);
/// loopback is always direct.
pub fn no_proxy(host: &str, list: &str) -> bool {
    let host = host.trim_start_matches('[').trim_end_matches(']').to_ascii_lowercase();
    if host == "localhost" || host.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback()) {
        return true;
    }
    list.split(',').map(|e| e.trim().trim_start_matches("*.").trim_start_matches('.').to_ascii_lowercase()).any(|e| {
        let e = e.split(':').next().unwrap_or("");
        e == "*" || (!e.is_empty() && (host == e || host.ends_with(&format!(".{e}"))))
    })
}

/// GET to the Internet, honouring HTTPS_PROXY / HTTP_PROXY / ALL_PROXY / NO_PROXY.
pub fn get(url: &str) -> ureq::Request {
    #[cfg(test)]
    let url = crate::fake::local(url); // tests never reach the Internet
    #[cfg(test)]
    let url = url.as_str();
    let https = url.starts_with("https:");
    let host = url.split("://").nth(1).unwrap_or("").split(['/', '?']).next().unwrap_or("");
    let host = host.rsplit('@').next().unwrap_or(host);
    let host = if host.starts_with('[') {
        host.split(']').next().unwrap_or(host)
    } else {
        host.split(':').next().unwrap_or(host)
    };
    let list = env::var("no_proxy").or_else(|_| env::var("NO_PROXY")).unwrap_or_default();
    let agent: &ureq::Agent = if no_proxy(host, &list) {
        &HTTP
    } else if https {
        &VIA_HTTPS
    } else {
        &VIA_HTTP
    };
    agent.get(url)
}

pub fn get_json<T: DeserializeOwned>(req: ureq::Request) -> Result<T, String> {
    req.call().map_err(|e| e.to_string())?.into_json().map_err(|e| e.to_string())
}

/// Download progress, reported about 10 times a second.
pub enum Dl {
    /// resuming: hashing the part already on disk (bytes hashed, part size)
    Verify(u64, u64),
    /// receiving (bytes on disk, total; 0 = unknown)
    Get(u64, u64),
}

/// Download `req` into `dest` through `dest.part` (resumed when present), reporting to `on`;
/// `on` returning false cancels (the .part stays for the next try).
/// `expected` > 0 and an existing file of that size = nothing to do (not re-hashed).
/// The resume sends If-Range with the ETag saved in `dest.part.etag`: a changed remote file
/// comes back whole (200) instead of being appended. `sha256` (hex) is checked at the end;
/// on mismatch the file is deleted.
#[allow(clippy::result_large_err)] // cold path; callers match on the status code
pub fn download(
    req: ureq::Request,
    dest: &Path,
    expected: u64,
    sha256: Option<&str>,
    on: &mut dyn FnMut(Dl) -> bool,
) -> Result<(), ureq::Error> {
    if expected > 0 && fs::metadata(dest).is_ok_and(|m| m.len() == expected) {
        return Ok(());
    }
    let cancelled = || io::Error::new(io::ErrorKind::Interrupted, "annullato");
    let part = dest.with_file_name(format!("{}.part", fname(&ps(dest))));
    let etag_file = dest.with_file_name(format!("{}.part.etag", fname(&ps(dest))));
    let etag = fs::read_to_string(&etag_file).unwrap_or_default();
    // no validator = no way to tell the remote file is the same one: start over
    let have = if etag.is_empty() { 0 } else { fs::metadata(&part).map_or(0, |m| m.len()) };
    let ranged =
        if have > 0 { req.clone().set("Range", &format!("bytes={have}-")).set("If-Range", &etag) } else { req.clone() };
    let resp = match ranged.call() {
        Err(ureq::Error::Status(416, _)) => {
            fs::remove_file(&part)?;
            let _ = fs::remove_file(&etag_file);
            return download(req, dest, expected, sha256, on);
        }
        r => r?,
    };
    let start = if resp.status() == 206 { have } else { 0 };
    if start == 0 {
        match resp.header("ETag") {
            Some(e) => fs::write(&etag_file, e)?,
            None => {
                let _ = fs::remove_file(&etag_file);
            }
        }
    }
    let total = resp.header("Content-Length").and_then(|v| v.parse::<u64>().ok()).map_or(expected, |n| n + start);
    let mut hash = Sha256::new();
    let mut buf = vec![0; 1 << 20];
    let mut last = Instant::now(); // last progress report
    let mut f = if start > 0 {
        // resuming: the part already on disk goes into the hash first
        let mut old = File::open(&part)?.take(start);
        let mut seen = 0;
        on(Dl::Verify(0, start));
        loop {
            let n = old.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hash.update(&buf[..n]);
            seen += n as u64;
            if last.elapsed() >= Duration::from_millis(100) {
                if !on(Dl::Verify(seen, start)) {
                    return Err(cancelled().into());
                }
                last = Instant::now();
            }
        }
        OpenOptions::new().append(true).open(&part)?
    } else {
        File::create(&part)?
    };
    let mut body = resp.into_reader();
    let mut done = start;
    on(Dl::Get(done, total));
    loop {
        let n = body.read(&mut buf)?;
        if n == 0 {
            break;
        }
        f.write_all(&buf[..n])?;
        hash.update(&buf[..n]);
        done += n as u64;
        if last.elapsed() >= Duration::from_millis(100) {
            if !on(Dl::Get(done, total)) {
                return Err(cancelled().into());
            }
            last = Instant::now();
        }
    }
    on(Dl::Get(done, total));
    if total > 0 && done < total {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "download interrotto").into());
    }
    drop(f);
    let _ = fs::remove_file(&etag_file);
    let got = format!("{:x}", hash.finalize());
    if let Some(want) = sha256
        && !want.eq_ignore_ascii_case(&got)
    {
        fs::remove_file(&part)?;
        return Err(io::Error::other("file corrotto (sha256 diverso), cancellato: riprova").into());
    }
    fs::rename(&part, dest)?;
    Ok(())
}

// -- child processes ------------------------------------------------------------------------------

/// Spawn with stdout and stderr merged into one ordered stream of lines (stderr=STDOUT).
pub fn spawn_merged(mut cmd: Command) -> io::Result<(Child, Receiver<String>)> {
    let (r, w) = io::pipe()?;
    let child = cmd.stdin(Stdio::null()).stdout(w.try_clone()?).stderr(w).spawn()?;
    drop(cmd); // it still holds the write ends: without this the reader never sees EOF
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut r = io::BufReader::new(r);
        let mut buf = vec![];
        while r.read_until(b'\n', &mut buf).is_ok_and(|n| n > 0) {
            if tx.send(String::from_utf8_lossy(&buf).trim_end().to_string()).is_err() {
                break;
            }
            buf.clear();
        }
    });
    Ok((child, rx))
}

pub struct Output {
    pub ok: bool,
    pub code: i32, // see exit_code
    pub stdout: String,
    pub stderr: String,
}

/// subprocess.run(capture_output=True, timeout=…): Err on spawn failure or timeout. A quit signal
/// (`on_quit`) kills the child too, so don't use it from a quit stop.
pub fn capture(cmd: &mut Command, timeout: Duration) -> Result<Output, String> {
    let mut child =
        cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().map_err(|e| e.to_string())?;
    let slurp = |r: Option<Box<dyn Read + Send>>| {
        thread::spawn(move || {
            let mut s = vec![];
            if let Some(mut r) = r {
                let _ = r.read_to_end(&mut s);
            }
            String::from_utf8_lossy(&s).into_owned()
        })
    };
    let out = slurp(child.stdout.take().map(|r| Box::new(r) as _));
    let err = slurp(child.stderr.take().map(|r| Box::new(r) as _));
    let deadline = Instant::now() + timeout;
    let status = loop {
        if let Some(s) = child.try_wait().map_err(|e| e.to_string())? {
            break s;
        }
        if Instant::now() > deadline || quit_requested() {
            let _ = child.kill();
            let _ = child.wait();
            return Err("timeout".into());
        }
        thread::sleep(Duration::from_millis(20));
    };
    Ok(Output {
        ok: status.success(),
        code: exit_code(&status),
        stdout: out.join().unwrap_or_default(),
        stderr: err.join().unwrap_or_default(),
    })
}

/// Popen.returncode: the exit code, or -signal when killed by a signal (unix).
pub fn exit_code(s: &ExitStatus) -> i32 {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = s.signal() {
            return -sig;
        }
    }
    s.code().unwrap_or(-1)
}

/// Run a long command (git/cmake) appending its merged output to `log_path` and handing each
/// line to `on_line`; `cancel` turning true (or a quit signal, see `on_quit`) kills it.
/// Err = message for the user.
pub fn run_logged(
    cmd: &[String],
    cwd: &Path,
    log_path: &Path,
    cancel: &AtomicBool,
    on_line: &mut dyn FnMut(String),
) -> Result<(), String> {
    let _ = fs::create_dir_all(log_path.parent().unwrap_or(&BASE));
    let mut log = OpenOptions::new().create(true).append(true).open(log_path).ok();
    if let Some(l) = log.as_mut() {
        let _ = writeln!(l, "\n# {}  {}", stamp(), cmd.join(" "));
    }
    let mut c = Command::new(&cmd[0]);
    c.args(&cmd[1..]).current_dir(cwd);
    let (mut child, rx) = spawn_merged(c).map_err(|e| format!("impossibile avviare {} ({e})", cmd[0]))?;
    loop {
        match rx.recv_timeout(Duration::from_millis(150)) {
            Ok(raw) => {
                if let Some(l) = log.as_mut() {
                    let _ = writeln!(l, "{raw}");
                }
                on_line(raw);
            }
            Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {}
        }
        // ponytail: kills only the direct child; make/ninja/compilers die on SIGPIPE at their next write
        // (or with the terminal's SIGHUP), a process group kill if they ever linger
        if cancel.load(Ordering::Relaxed) || quit_requested() {
            // grandchildren (make, compilers) may keep the pipe open: don't wait for EOF
            let _ = child.kill();
            let _ = child.wait();
            return Err("annullato".into());
        }
    }
    match child.wait().map_or(-1, |s| exit_code(&s)) {
        0 => Ok(()),
        code => Err(format!("codice {code} · log completo: {}", log_path.display())),
    }
}

/// Stop the child the way Ctrl+C would: llama-server shuts down gracefully on SIGINT.
/// No Windows twin: CTRL_C_EVENT can't target one process (it hits the whole console, the user's
/// shell included) and the servers run in their own hidden console anyway, so there `stop` kills.
#[cfg(unix)]
pub fn interrupt(child: &Child) {
    unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGINT) };
}

// -- quit signals ---------------------------------------------------------------------------------

/// 128 + the signal (unix) or console event (Windows) caught under a `QuitGuard`; 0 = none.
static QUIT: AtomicI32 = AtomicI32::new(0);
type Stop = Box<dyn FnOnce() + Send>;
static STOPS: Mutex<Vec<(usize, Stop)>> = Mutex::new(Vec::new());
static NEXT: AtomicUsize = AtomicUsize::new(0);

/// While one lives, SIGINT/SIGTERM/SIGHUP (Windows: Ctrl+Break, console close, logoff, shutdown) don't
/// kill the process on the spot: a background thread runs the `stop` of every live guard, newest
/// first (stop a server, then restore the terminal), and exits with 128 + signal. It doesn't count on
/// the UI loop, which may be stuck: crossterm 0.28 spins inside event::poll once the terminal hangs up.
/// The loop only has to stop drawing (and dropping screens) once `quit_requested()`.
/// The last drop puts back the previous handlers.
pub struct QuitGuard(usize);

pub fn on_quit(stop: impl FnOnce() + Send + 'static) -> QuitGuard {
    static WATCHER: Once = Once::new();
    WATCHER.call_once(|| {
        thread::spawn(|| {
            while !quit_requested() {
                thread::sleep(Duration::from_millis(50));
            }
            // ponytail: a fixed pause lets an in-flight frame finish before the terminal is restored;
            // a lock around drawing if a frame ever lands after it
            thread::sleep(Duration::from_millis(100));
            let stops = std::mem::take(&mut *lock(&STOPS));
            for (_, stop) in stops.into_iter().rev() {
                stop();
            }
            thread::sleep(Duration::from_millis(300)); // room for log writers to catch the last lines
            std::process::exit(QUIT.load(Ordering::SeqCst)) // the shell's 128 + signal
        });
    });
    let id = NEXT.fetch_add(1, Ordering::SeqCst);
    let mut stops = lock(&STOPS);
    if stops.is_empty() {
        quit_handlers(true);
    }
    stops.push((id, Box::new(stop)));
    QuitGuard(id)
}

impl Drop for QuitGuard {
    fn drop(&mut self) {
        let mut stops = lock(&STOPS);
        stops.retain(|(id, _)| *id != self.0);
        if stops.is_empty() {
            quit_handlers(false);
        }
    }
}

pub fn quit_requested() -> bool {
    QUIT.load(Ordering::SeqCst) != 0
}

/// The quit signal was a hang-up: the terminal, or the ssh connection, went away.
#[cfg(unix)]
pub fn hung_up() -> bool {
    QUIT.load(Ordering::SeqCst) == 128 + libc::SIGHUP
}

#[cfg(not(unix))]
pub fn hung_up() -> bool {
    false
}

/// Corral runs inside an ssh session: a browser or a clipboard tool would work on this machine,
/// not on the user's.
pub fn over_ssh() -> bool {
    env::var_os("SSH_CONNECTION").is_some() || env::var_os("SSH_TTY").is_some()
}

/// Run `f` while another program has the terminal (an interactive ssh): Ctrl+C is that program's,
/// so here it doesn't count as a quit signal. Unix only: a Windows console in raw mode, as ssh
/// sets it, sends no Ctrl+C event anyway.
pub fn without_interrupt<T>(f: impl FnOnce() -> T) -> T {
    #[cfg(unix)]
    let previous = unsafe { libc::signal(libc::SIGINT, libc::SIG_IGN) };
    let out = f();
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGINT, previous);
    }
    out
}

/// `interrupt`, then kill if it's still running after `grace` (Windows: kill right away, see
/// `interrupt`); always reaped.
pub fn stop(child: &Mutex<Child>, grace: Duration) {
    #[cfg(unix)]
    {
        interrupt(&lock(child));
        let t0 = Instant::now();
        while matches!(lock(child).try_wait(), Ok(None)) && t0.elapsed() < grace {
            thread::sleep(Duration::from_millis(50));
        }
    }
    #[cfg(not(unix))]
    let _ = grace;
    let mut c = lock(child);
    let _ = c.kill(); // no-op once it has exited
    let _ = c.wait();
}

/// A poisoned lock still holds usable data here (a child handle, a list).
pub fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(unix)]
const QUIT_SIGNALS: [libc::c_int; 3] = [libc::SIGINT, libc::SIGTERM, libc::SIGHUP];
#[cfg(unix)]
static PREV: [AtomicUsize; 3] = [const { AtomicUsize::new(libc::SIG_DFL) }; 3];

#[cfg(unix)]
extern "C" fn on_signal(sig: libc::c_int) {
    QUIT.store(128 + sig, Ordering::SeqCst); // an atomic store is all a signal handler may safely do
}

#[cfg(unix)]
fn quit_handlers(on: bool) {
    for (&sig, prev) in QUIT_SIGNALS.iter().zip(&PREV) {
        unsafe {
            if !on {
                libc::signal(sig, prev.load(Ordering::SeqCst));
                continue;
            }
            let old = libc::signal(sig, on_signal as extern "C" fn(libc::c_int) as libc::sighandler_t);
            prev.store(old, Ordering::SeqCst);
            if old == libc::SIG_IGN {
                libc::signal(sig, libc::SIG_IGN); // e.g. under nohup: keep ignoring the hangup
            }
        }
    }
}

#[cfg(windows)]
unsafe extern "system" fn on_ctrl(event: u32) -> i32 {
    QUIT.store(128 + event as i32, Ordering::SeqCst);
    if event >= 2 {
        // close/logoff/shutdown: Windows ends the process as soon as this returns (or ~5 s later
        // anyway), so hold it while the quit thread stops the children and exits
        thread::sleep(Duration::from_secs(10));
    }
    1
}

#[cfg(windows)]
fn quit_handlers(on: bool) {
    unsafe { SetConsoleCtrlHandler(Some(on_ctrl), i32::from(on)) };
}

pub fn which(name: &str) -> Option<PathBuf> {
    let exts =
        if cfg!(windows) { env::var("PATHEXT").unwrap_or_else(|_| ".EXE;.CMD;.BAT".into()) } else { String::new() };
    env::split_paths(&env::var_os("PATH")?)
        .find_map(|dir| exts.split(';').map(|e| dir.join(format!("{name}{e}"))).find(|p| p.is_file()))
}

// -- host ------------------------------------------------------------------------------------------

#[cfg(windows)]
#[repr(C)]
struct MemoryStatusEx {
    len: u32,
    load: u32,
    total_phys: u64,
    avail_phys: u64,
    total_pagefile: u64,
    avail_pagefile: u64,
    total_virtual: u64,
    avail_virtual: u64,
    avail_extended: u64,
}

#[cfg(windows)]
#[repr(C)]
#[derive(Default)]
struct WinTime {
    year: u16,
    month: u16,
    weekday: u16,
    day: u16,
    hour: u16,
    minute: u16,
    second: u16,
    millis: u16,
}

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GlobalMemoryStatusEx(m: *mut MemoryStatusEx) -> i32;
    fn GetDiskFreeSpaceExW(dir: *const u16, avail: *mut u64, total: *mut u64, free: *mut u64) -> i32;
    fn GetLocalTime(t: *mut WinTime);
    fn SetConsoleCtrlHandler(handler: Option<unsafe extern "system" fn(u32) -> i32>, add: i32) -> i32;
}

/// RAM the OS could hand out now, in bytes (0 = unknown).
#[cfg(target_os = "linux")]
pub fn ram_available() -> u64 {
    fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines().find_map(|l| l.strip_prefix("MemAvailable:")?.split_whitespace().next()?.parse::<u64>().ok())
        })
        .map_or(0, |kib| kib * 1024)
}

#[cfg(target_os = "macos")]
pub fn ram_available() -> u64 {
    // psutil's "available" = free_count + inactive; vm_stat prints free_count minus the speculative pages
    let Ok(out) = capture(&mut Command::new("vm_stat"), Duration::from_secs(5)) else {
        return 0;
    };
    let page = out.stdout.split("page size of ").nth(1).and_then(|s| s.split_whitespace().next()?.parse().ok());
    let pages = |k: &str| {
        out.stdout
            .lines()
            .find_map(|l| l.strip_prefix(k)?.trim().trim_end_matches('.').parse::<u64>().ok())
            .unwrap_or(0)
    };
    (pages("Pages free:") + pages("Pages speculative:") + pages("Pages inactive:")) * page.unwrap_or(4096)
}

#[cfg(windows)]
pub fn ram_available() -> u64 {
    let mut m: MemoryStatusEx = unsafe { std::mem::zeroed() };
    m.len = size_of::<MemoryStatusEx>() as u32;
    if unsafe { GlobalMemoryStatusEx(&mut m) } != 0 { m.avail_phys } else { 0 }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
pub fn ram_available() -> u64 {
    0
}

/// Free space for the current user on the volume holding `p`.
#[cfg(unix)]
#[allow(clippy::useless_conversion)] // field widths differ between Linux and macOS
// ponytail: macOS statvfs block counts are 32-bit, so a volume with > 16 TiB free may read low; use statfs
// there if that ever matters
pub fn disk_free(p: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(p.as_os_str().as_bytes()).ok()?;
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    (unsafe { libc::statvfs(c.as_ptr(), &mut s) } == 0).then(|| u64::from(s.f_bavail) * u64::from(s.f_frsize))
}

#[cfg(windows)]
pub fn disk_free(p: &Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    let wide: Vec<u16> = p.as_os_str().encode_wide().chain([0]).collect();
    let mut avail = 0;
    let ok = unsafe { GetDiskFreeSpaceExW(wide.as_ptr(), &mut avail, std::ptr::null_mut(), std::ptr::null_mut()) };
    (ok != 0).then_some(avail)
}

/// Local wall clock: [year, month, day, hour, minute, second].
#[cfg(unix)]
fn local_now() -> [i32; 6] {
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    unsafe { libc::localtime_r(&libc::time(std::ptr::null_mut()), &mut tm) };
    [tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec]
}

#[cfg(windows)]
fn local_now() -> [i32; 6] {
    let mut t = WinTime::default();
    unsafe { GetLocalTime(&mut t) };
    [t.year, t.month, t.day, t.hour, t.minute, t.second].map(i32::from)
}

/// Seconds since 1970 (run files, caches, news dates).
pub fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// "%F %T"
pub fn stamp() -> String {
    let [y, mo, d, h, mi, s] = local_now();
    format!("{y}-{mo:02}-{d:02} {h:02}:{mi:02}:{s:02}")
}

/// "%H:%M:%S"
pub fn clock() -> String {
    let [.., h, mi, s] = local_now();
    format!("{h:02}:{mi:02}:{s:02}")
}

/// "%Y%m%d-%H%M%S", for file names
pub fn file_stamp() -> String {
    let [y, mo, d, h, mi, s] = local_now();
    format!("{y}{mo:02}{d:02}-{h:02}{mi:02}{s:02}")
}
