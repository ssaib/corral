//! What the tests use instead of the outside world.
//!
//! - The web: under `cfg(test)` every request `ui::get` makes comes here, to a server on
//!   127.0.0.1, with the original host as the first piece of the path
//!   (https://huggingface.co/api/models → http://127.0.0.1:PORT/huggingface.co/api/models).
//!   It answers with small pages of each service (Hugging Face, GitHub, RSS feeds), consistent
//!   with each other: the sizes and sha256 of a repo's files are those of the bytes it serves.
//!   `route` adds or replaces a page.
//! - llama.cpp: `llama_cpp()` is a folder of scripts named like its tools (llama-server,
//!   llama-fit-params, llama-bench, rpc-server) that run this very test binary again as
//!   `fake_tool` (see the end of the file), which plays the tool. Unix only.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{LazyLock, Mutex};
use std::thread;

use sha2::{Digest, Sha256};

pub struct Reply {
    pub status: u16,
    pub kind: &'static str,
    pub body: Vec<u8>,
}

fn reply(status: u16, kind: &'static str, body: impl Into<Vec<u8>>) -> Reply {
    Reply { status, kind, body: body.into() }
}

static ROUTES: LazyLock<Mutex<HashMap<String, Reply>>> = LazyLock::new(Mutex::default);

static SERVER: LazyLock<String> = LazyLock::new(|| {
    let listener = TcpListener::bind("127.0.0.1:0").expect("fake web server");
    let port = listener.local_addr().map(|a| a.port()).unwrap_or(0);
    thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            thread::spawn(move || serve(stream));
        }
    });
    format!("http://127.0.0.1:{port}")
});

/// Where a request for `url` goes during the tests: local addresses stay, the rest comes here.
pub fn local(url: &str) -> String {
    let Some((_, rest)) = url.split_once("://") else { return url.to_string() };
    let host = rest.split(['/', '?']).next().unwrap_or("");
    let loopback = host.starts_with("127.") || host.starts_with("localhost") || host.starts_with("[::1]");
    if loopback { url.to_string() } else { format!("{}/{rest}", *SERVER) }
}

/// Answer `path` (no query; the original host first, e.g. "/huggingface.co/api/models/a/b") with
/// `body` from now on.
pub fn route(path: &str, status: u16, kind: &'static str, body: impl Into<Vec<u8>>) {
    let _ = &*SERVER;
    ROUTES.lock().unwrap().insert(path.to_string(), reply(status, kind, body));
}

fn serve(stream: TcpStream) {
    let mut reader = BufReader::new(&stream);
    let mut first = String::new();
    if reader.read_line(&mut first).is_err() {
        return;
    }
    loop {
        let mut header = String::new();
        let read = reader.read_line(&mut header).unwrap_or(0);
        if read == 0 || header == "\r\n" || header == "\n" {
            break;
        }
    }
    let target = first.split_whitespace().nth(1).unwrap_or("/");
    let path = target.split('?').next().unwrap_or(target);
    let registered = ROUTES.lock().unwrap().get(path).map(|r| reply(r.status, r.kind, r.body.clone()));
    let r = registered.unwrap_or_else(|| canned(path));
    let head = format!(
        "HTTP/1.1 {} X\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        r.status,
        r.kind,
        r.body.len()
    );
    let mut out = &stream;
    let _ = out.write_all(head.as_bytes());
    let _ = out.write_all(&r.body);
}

// -- the data folder -----------------------------------------------------------------------------

thread_local! {
    static PERSIST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
static DISK: Mutex<()> = Mutex::new(());

/// Whether `ui::save_state` writes state.json on this thread (see `disk`).
pub fn persist() -> bool {
    PERSIST.with(|p| p.get())
}

/// The data folder's state.json for this test alone: the tests run in parallel and share the
/// folder, so by default saving the state does nothing; a test that reads it back from disk (the
/// commands) holds this until it's done.
pub struct Disk(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);

pub fn disk() -> Disk {
    let guard = DISK.lock().unwrap_or_else(|e| e.into_inner());
    PERSIST.with(|p| p.set(true));
    Disk(guard)
}

impl Drop for Disk {
    fn drop(&mut self) {
        PERSIST.with(|p| p.set(false));
    }
}

/// The run/ folder is shared too: a test that adopts "every server left running" (what a new
/// Corral does) would take the other tests' servers, and drop the run files of the ones still
/// starting. Tests with servers hold this shared; that one alone, exclusive.
static SERVERS: std::sync::RwLock<()> = std::sync::RwLock::new(());

thread_local! {
    static SHARED_HERE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The shared hold on run/ (once per thread: a second read lock behind a waiting writer would
/// wait forever).
pub struct SharedServers(#[allow(dead_code)] Option<std::sync::RwLockReadGuard<'static, ()>>);

pub fn shared_servers() -> SharedServers {
    let first = SHARED_HERE.with(|n| n.get()) == 0;
    SHARED_HERE.with(|n| n.set(n.get() + 1));
    SharedServers(first.then(|| SERVERS.read().unwrap_or_else(|e| e.into_inner())))
}

impl Drop for SharedServers {
    fn drop(&mut self) {
        SHARED_HERE.with(|n| n.set(n.get() - 1));
    }
}

pub fn all_servers() -> std::sync::RwLockWriteGuard<'static, ()> {
    SERVERS.write().unwrap_or_else(|e| e.into_inner())
}

// -- the pages ---------------------------------------------------------------------------------

const JSON: &str = "application/json";

/// A GGUF file with just the header keys Corral reads, then `padding` bytes.
pub fn gguf(layers: u32, experts: u32, ctx: u32, padding: usize) -> Vec<u8> {
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes()); // version
    b.extend(0u64.to_le_bytes()); // tensors
    let kvs: u64 = if experts > 0 { 4 } else { 3 };
    b.extend(kvs.to_le_bytes());
    let string = |b: &mut Vec<u8>, s: &str| {
        b.extend((s.len() as u64).to_le_bytes());
        b.extend(s.as_bytes());
    };
    string(&mut b, "general.architecture");
    b.extend(8u32.to_le_bytes());
    string(&mut b, "qwen3");
    let int = |b: &mut Vec<u8>, key: &str, v: u32| {
        string(b, key);
        b.extend(4u32.to_le_bytes());
        b.extend(v.to_le_bytes());
    };
    int(&mut b, "qwen3.block_count", layers);
    int(&mut b, "qwen3.context_length", ctx);
    if experts > 0 {
        int(&mut b, "qwen3.expert_count", experts);
    }
    b.extend(vec![0u8; padding]);
    b
}

/// The files of every fake repo: two quantizations, an mmproj, a README.
pub fn repo_files() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("Model-Q4_K_M.gguf", gguf(36, 0, 40960, 3000)),
        ("Model-Q8_0.gguf", gguf(36, 0, 40960, 6000)),
        ("mmproj-Model-F16.gguf", gguf(1, 0, 0, 500)),
        ("README.md", b"# fake model\n".to_vec()),
    ]
}

fn sha256(b: &[u8]) -> String {
    Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect()
}

fn models_list() -> String {
    serde_json::json!([
        {"id": "unsloth/Qwen3-8B-GGUF", "downloads": 123456, "pipeline_tag": "text-generation",
         "createdAt": "2026-11-20T10:00:00.000Z",
         "gguf": {"total": 8_190_000_000u64, "architecture": "qwen3", "context_length": 40960}},
        {"id": "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF", "downloads": 90000, "pipeline_tag": "text-generation",
         "createdAt": "2026-11-18T10:00:00.000Z",
         "gguf": {"total": 7_600_000_000u64, "architecture": "qwen2", "context_length": 32768}},
        {"id": "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF", "downloads": 5000, "pipeline_tag": "image-text-to-text",
         "createdAt": "2026-11-19T10:00:00.000Z",
         "gguf": {"total": 3_750_000_000u64, "architecture": "qwen2vl", "context_length": 128000}},
        {"id": "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF", "downloads": 40000, "pipeline_tag": "text-generation",
         "createdAt": "2026-11-17T10:00:00.000Z",
         "gguf": {"total": 14_800_000_000u64, "architecture": "qwen2", "context_length": 131072}},
        {"id": "someone/odd-GGUF"}
    ])
    .to_string()
}

fn repo_info(repo: &str) -> String {
    let siblings: Vec<serde_json::Value> = repo_files()
        .iter()
        .map(|(name, bytes)| {
            serde_json::json!({"rfilename": name, "size": bytes.len(), "lfs": {"sha256": sha256(bytes)}})
        })
        .collect();
    serde_json::json!({"id": repo, "gated": repo.contains("gated"), "siblings": siblings}).to_string()
}

/// The release assets of the fake builds, served at their browser_download_url (see `route`).
pub const ASSETS: [&str; 5] = [
    "llama-b9999-bin-ubuntu-x64.tar.gz",
    "llama-b9999-bin-ubuntu-vulkan-x64.tar.gz",
    "llama-b9999-bin-macos-arm64.tar.gz",
    "llama-b9999-bin-macos-x64.tar.gz",
    "llama-b9999-bin-win-cpu-x64.zip",
];

fn releases() -> String {
    let assets: Vec<serde_json::Value> = ASSETS
        .iter()
        .map(|n| {
            serde_json::json!({"name": n, "size": 0,
                "browser_download_url": format!("https://github.com/ggml-org/llama.cpp/releases/download/b9999/{n}")})
        })
        .collect();
    serde_json::json!([
        {"tag_name": "b9999", "prerelease": true, "published_at": "2026-11-22T08:00:00Z",
         "html_url": "https://github.com/ggml-org/llama.cpp/releases/tag/b9999",
         "body": "<details open>\n\nserver: faster slots (#12345)\n\n</details>\n\n**Windows:** zip", "assets": assets},
        {"tag_name": "b9998", "prerelease": true, "published_at": "2026-11-21T08:00:00Z",
         "html_url": "https://github.com/ggml-org/llama.cpp/releases/tag/b9998", "body": "", "assets": []},
        {"tag_name": "v1.0.0", "prerelease": false, "published_at": "2026-11-01T08:00:00Z",
         "html_url": "https://github.com/ggml-org/llama.cpp/releases/tag/v1.0.0", "body": "stable", "assets": []}
    ])
    .to_string()
}

const RSS: &str = r#"<?xml version="1.0"?>
<rss version="2.0"><channel><title>Fake AI news</title>
<item><title>A new model &amp; more</title><link>https://example.com/a</link>
<pubDate>Sat, 22 Nov 2026 10:00:00 GMT</pubDate><description>&lt;p&gt;Something happened.&lt;/p&gt;</description></item>
<item><title>Another one</title><link>https://example.com/b</link>
<pubDate>Fri, 21 Nov 2026 09:30:00 +0100</pubDate><description>More text</description></item>
</channel></rss>"#;

fn canned(path: &str) -> Reply {
    let parts: Vec<&str> = path.trim_start_matches('/').split('/').collect();
    match parts.as_slice() {
        ["huggingface.co", "api", "models"] => reply(200, JSON, models_list()),
        ["huggingface.co", "api", "models", org, repo] => {
            if repo.contains("missing") {
                return reply(404, JSON, r#"{"error":"Repository not found"}"#);
            }
            reply(200, JSON, repo_info(&format!("{org}/{repo}")))
        }
        ["huggingface.co", _org, _repo, "resolve", "main", file] => {
            match repo_files().into_iter().find(|(name, _)| name == file) {
                Some((_, bytes)) => reply(200, "application/octet-stream", bytes),
                None => reply(404, "text/plain", "not found"),
            }
        }
        ["api.github.com", "repos", "ggml-org", "llama.cpp", "releases"] => reply(200, JSON, releases()),
        // any other site is a news feed
        [host, ..]
            if path.ends_with(".xml") || !matches!(*host, "huggingface.co" | "api.github.com" | "github.com") =>
        {
            reply(200, "application/rss+xml", RSS)
        }
        _ => reply(404, "text/plain", "not found"),
    }
}

// -- llama.cpp ---------------------------------------------------------------------------------

/// A folder of scripts named like llama.cpp's tools, each running this test binary again as
/// `fake_tool`, which plays that tool. Made once per test run, next to the test data.
#[cfg(unix)]
pub fn llama_cpp() -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt;
    static DIR: LazyLock<std::path::PathBuf> = LazyLock::new(|| {
        let dir = crate::ui::BASE.parent().expect("test folder").join("fake-llama.cpp");
        std::fs::create_dir_all(&dir).unwrap();
        let exe = std::env::current_exe().unwrap();
        let tests = dir.parent().expect("test folder").to_path_buf();
        for tool in ["llama-server", "llama-fit-params", "llama-bench", "rpc-server", "ssh", "scp", "corral"] {
            let script = format!(
                // libtest's own "running 1 test" goes nowhere: the tool's stdout is fd 3 (see fake_tool)
                "#!/bin/sh\nCORRAL_FAKE_TOOL={tool} CORRAL_FAKE_DIR='{}' exec '{}' --exact fake::fake_tool --ignored \
                 --nocapture --test-threads=1 -q -- \"$@\" 3>&1 1>/dev/null\n",
                tests.display(),
                exe.display()
            );
            let path = dir.join(tool);
            std::fs::write(&path, script).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        dir
    });
    DIR.clone()
}

/// An install made of the fake tools.
#[cfg(unix)]
pub fn install() -> crate::installs::Install {
    crate::installs::Install {
        name: "fake-b9999".into(),
        kind: "release".into(),
        backend: "vulkan".into(),
        bin_dir: crate::ui::ps(&llama_cpp()),
        git_ref: None,
        flags: vec![],
        src_dir: None,
    }
}

/// `llama-server --help` as far as Corral reads it: every flag of the presets and the router.
fn help_text() -> String {
    use crate::preset::Flag;
    let mut out = String::from(
        "-h,    --help, --usage                  print usage and exit\n\
         --version                               show version and build info\n\
         --list-devices                          print a list of available devices and exit\n\
         -m,    --model FNAME                    model path\n\
         --rpc SERVERS                           comma separated list of RPC servers\n\
         --embeddings                            restrict to only support embedding use case\n\
         --rerank                                enable reranking endpoint\n\
         --models-preset PATH                    INI file of the router's models\n\
         --spec-type TYPE                        speculative decoding\n\
         -d, --device <dev1,dev2,..>             devices (rpc-server)\n\
         --api-key KEY                           API key to use for authentication\n\
                                                 (env: LLAMA_API_KEY)\n\
         --old-thing                             this option has been removed\n",
    );
    for f in crate::preset::FIELDS.iter().chain(crate::router::OPTS) {
        match &f.flag {
            Flag::Opt(spec) => {
                for s in spec.split('|') {
                    out.push_str(&format!("{s} N    {}\n", f.label));
                }
            }
            Flag::Toggle(on, off) => {
                for w in [on, off].into_iter().filter(|w| !w.is_empty()) {
                    out.push_str(&format!("{w}    {}\n", f.label));
                }
            }
            Flag::No => {}
        }
    }
    out
}

#[test]
#[ignore = "plays a llama.cpp tool for the scripts of llama_cpp()"]
fn fake_tool() {
    let Ok(tool) = std::env::var("CORRAL_FAKE_TOOL") else { return };
    // the script's real stdout, kept on fd 3 while libtest printed its header to /dev/null
    #[cfg(unix)]
    unsafe {
        libc::dup2(3, 1);
        libc::close(3);
    }
    let args: Vec<String> = std::env::args().skip_while(|a| a != "--").skip(1).collect();
    let code = play(&tool, &args);
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    std::process::exit(code);
}

fn value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn play(tool: &str, args: &[String]) -> i32 {
    match tool {
        "corral" => return corral(args),
        "ssh" => return ssh(args),
        "scp" => return scp(args),
        _ => {}
    }
    let has = |f: &str| args.iter().any(|a| a == f);
    if has("--help") {
        print!("{}", help_text());
        return 0;
    }
    if has("--version") {
        eprintln!("version: 9999 (abc1234)\nbuild 9999, commit abc1234");
        return 0;
    }
    if has("--list-devices") {
        println!("Available devices:\n  Vulkan0: Fake GPU 4080 (12282 MiB, 11474 MiB free)");
        return 0;
    }
    match tool {
        "llama-fit-params" => fit(args),
        "llama-bench" => bench(args),
        "llama-server" => server(args),
        "rpc-server" => rpc_server(args),
        _ => 0,
    }
}

/// rpc-server: its device line, then it takes connections until it's stopped.
fn rpc_server(args: &[String]) -> i32 {
    let host = value(args, "-H").unwrap_or("127.0.0.1".into());
    let port = value(args, "-p").unwrap_or("50052".into());
    println!("  Vulkan0: Fake GPU 4080 (12282 MiB, 11474 MiB free)");
    println!("Starting RPC server on {host}:{port}");
    let Ok(listener) = TcpListener::bind(format!("{host}:{port}")) else {
        eprintln!("Failed to create server socket");
        return 1;
    };
    let mut held = vec![];
    for s in listener.incoming().flatten() {
        held.push(s); // an RPC client keeps its connection
    }
    0
}

/// llama-fit-params with a simple memory model: 200 MiB a layer, the KV cache by type, a
/// compute buffer by ubatch, 11390 MiB free. It lowers what wasn't set (the context first, down
/// to -fitc, then the layers) to leave -fitt free, like llama.cpp.
fn fit(args: &[String]) -> i32 {
    let model = value(args, "-m").unwrap_or_default();
    if model.contains("broken") {
        eprintln!("llama_model_load: error loading model: fake broken file");
        return 1;
    }
    let shape = crate::preset::gguf_info(std::path::Path::new(&model));
    let layers = shape.as_ref().map_or(33, |g| u64::from(g.layers) + 1);
    let train = shape.as_ref().map_or(4096, |g| g.ctx.max(1) as u64);
    let num = |flag: &str, default: u64| value(args, flag).and_then(|v| v.parse().ok()).unwrap_or(default);
    let free = 11390u64;
    let target = num("-fitt", 1024);
    let min_ctx = num("-fitc", 4096);
    let kv_kib = match value(args, "-ctk").as_deref() {
        Some("q8_0") => 76,
        Some("q4_0") => 40,
        _ => 144,
    };
    let compute = if num("-ub", 512) >= 1024 { 700 } else { 350 };
    let user_ctx = value(args, "-c").filter(|c| c != "0");
    let user_ngl = value(args, "-ngl").filter(|n| n != "-1" && n != "auto");
    let mut ctx = user_ctx.as_ref().and_then(|c| c.parse().ok()).unwrap_or(train);
    let mut ngl = user_ngl.as_ref().and_then(|n| n.parse().ok()).unwrap_or(layers).min(layers);
    let need = |ctx: u64, ngl: u64| ngl * 200 + ctx * kv_kib / 1024 + compute;
    eprintln!("0.00.376.716 I common_memory_breakdown_print: |   - Vulkan0 (Fake GPU) | 12282 = {free} + ... |");
    eprintln!(
        "0.00.388.669 I common_params_fit_impl: projected to use {} MiB of device memory vs. {free} MiB of free device memory",
        need(ctx, ngl)
    );
    let room = free.saturating_sub(target);
    if user_ctx.is_none() {
        while need(ctx, ngl) > room && ctx > min_ctx {
            ctx = ctx.saturating_sub(256).max(min_ctx);
        }
    }
    if user_ngl.is_none() {
        while need(ctx, ngl) > room && ngl > 0 {
            ngl -= 1;
        }
    }
    if need(ctx, ngl) > room {
        eprintln!("0.00.388.673 I common_params_fit_impl: cannot meet free memory target of {target} MiB");
    }
    let shown = if ngl == layers { "-1".to_string() } else { ngl.to_string() };
    println!("-c {ctx} -ngl {shown}");
    0
}

/// llama-bench: faster when every layer is in VRAM.
fn bench(args: &[String]) -> i32 {
    let model = value(args, "-m").unwrap_or_default();
    if model.contains("nobench") {
        eprintln!("error: failed to load model '{model}'");
        return 1;
    }
    let offloaded = value(args, "-ngl").and_then(|n| n.parse::<u64>().ok()).is_some_and(|n| n < 30);
    let (pp, tg) = if offloaded { (1200.5, 14.2) } else { (3000.5, 45.5) };
    eprintln!("llama-bench: benchmark 1/2: starting");
    println!(r#"{{"n_prompt": 512, "n_gen": 0, "avg_ts": {pp}}}"#);
    println!(r#"{{"n_prompt": 0, "n_gen": 128, "avg_ts": {tg}}}"#);
    0
}

/// llama-server: the endpoints Corral asks (health, slots, metrics, models, chat, embeddings,
/// rerank, the router's load/unload), with the log lines of the real one. An alias with "crash"
/// in it exits with an error right away, one with "flaky" a moment after it's up.
fn server(args: &[String]) -> i32 {
    let host = value(args, "--host").unwrap_or("127.0.0.1".into());
    let port = value(args, "--port").unwrap_or("8080".into());
    let alias = value(args, "--alias").unwrap_or("model".into());
    if alias.contains("crash") {
        eprintln!("0.00.100.000 E main: fake crash before loading");
        return 1;
    }
    let key = std::env::var("LLAMA_API_KEY").ok().or_else(|| value(args, "--api-key")).unwrap_or_default();
    let router = args.iter().any(|a| a == "--models-preset");
    eprintln!("0.00.105.703 I srv    load_model: loading model '{}'", value(args, "-m").unwrap_or_default());
    let Ok(listener) = TcpListener::bind(format!("{host}:{port}")) else {
        eprintln!("0.00.200.000 E main: couldn't bind HTTP server socket, hostname: {host}, port: {port}");
        return 1;
    };
    eprintln!("0.04.482.821 I srv    load_model: initializing, n_slots = 2, n_ctx_slot = 4096, kv_unified = 'true'");
    if router {
        eprintln!("0.04.483.000 I srv  main: spawning server instance with name=alpha on port 40001");
    }
    eprintln!("0.04.486.940 I srv  llama_server: listening on http://{host}:{port}");
    let started = std::time::Instant::now();
    if alias.contains("flaky") {
        // up for a moment, then down: what the automatic restart is for
        thread::spawn(|| {
            thread::sleep(std::time::Duration::from_millis(1500));
            eprintln!("0.09.000.000 E srv  operator(): fake crash after start");
            std::process::exit(3);
        });
    }
    let ctx = std::sync::Arc::new((alias, key, router, started));
    for stream in listener.incoming().flatten() {
        let ctx = ctx.clone();
        thread::spawn(move || answer(stream, &ctx.0, &ctx.1, ctx.2, ctx.3));
    }
    0
}

fn answer(stream: TcpStream, alias: &str, key: &str, router: bool, started: std::time::Instant) {
    use std::io::Read;
    let mut reader = BufReader::new(&stream);
    let mut first = String::new();
    if reader.read_line(&mut first).is_err() {
        return;
    }
    let mut length = 0;
    let mut auth = String::new();
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).unwrap_or(0) == 0 || header.trim().is_empty() {
            break;
        }
        let lower = header.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            length = v.trim().parse().unwrap_or(0);
        }
        if lower.starts_with("authorization:") {
            auth = header.split_once(':').map_or("", |x| x.1).trim().to_string();
        }
    }
    let mut body = vec![0; length];
    let _ = reader.read_exact(&mut body);
    let body = String::from_utf8_lossy(&body).to_string();
    let mut parts = first.split_whitespace();
    let method = parts.next().unwrap_or("GET").to_string();
    let path = parts.next().unwrap_or("/").split('?').next().unwrap_or("/").to_string();
    let send = |status: u16, kind: &str, text: String| {
        let head = format!(
            "HTTP/1.1 {status} X\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            text.len()
        );
        let mut out = &stream;
        let _ = out.write_all(head.as_bytes());
        let _ = out.write_all(text.as_bytes());
    };
    if !key.is_empty() && path != "/health" && auth != format!("Bearer {key}") {
        return send(401, JSON, r#"{"error":{"message":"Invalid API Key"}}"#.into());
    }
    eprintln!("0.05.000.000 I srv  log_server_r: request: {method} {path} 127.0.0.1 200");
    let stream_wanted = body.replace(' ', "").contains("\"stream\":true");
    match (method.as_str(), path.as_str()) {
        ("GET", "/health") if started.elapsed() < std::time::Duration::from_millis(300) => {
            send(503, JSON, r#"{"error":{"message":"Loading model"}}"#.into())
        }
        ("GET", "/health") => send(200, JSON, r#"{"status":"ok"}"#.into()),
        ("GET", "/slots") => send(
            200,
            JSON,
            r#"[{"id":0,"n_ctx":4096,"is_processing":true,"n_prompt_tokens":100,"next_token":[{"n_decoded":20}]},
                {"id":1,"n_ctx":4096,"is_processing":false}]"#
                .into(),
        ),
        ("GET", "/metrics") => send(
            200,
            "text/plain",
            "# HELP llamacpp:prompt_tokens_total x\nllamacpp:prompt_tokens_total 1200\n\
             llamacpp:tokens_predicted_total 340\nllamacpp:prompt_tokens_seconds 1500.5\n\
             llamacpp:predicted_tokens_seconds 42.1\nllamacpp:requests_processing 1\n\
             llamacpp:requests_deferred 0\nllamacpp:kv_cache_usage_ratio 0.25\n"
                .into(),
        ),
        ("GET", "/models") if router => send(
            200,
            JSON,
            r#"{"data":[{"id":"alpha","status":{"value":"loaded"}},{"id":"beta","status":{"value":"unloaded"}},
                {"id":"gamma","status":{"value":"unloaded","failed":true,"exit_code":1}}]}"#
                .into(),
        ),
        ("POST", "/models/load" | "/models/unload") => send(200, JSON, r#"{"success":true}"#.into()),
        ("GET", "/v1/models" | "/models") => send(200, JSON, format!(r#"{{"data":[{{"id":"{alias}"}}]}}"#)),
        ("GET", "/props") => {
            send(200, JSON, r#"{"default_generation_settings":{"n_ctx":4096},"total_slots":2}"#.into())
        }
        ("POST", "/v1/chat/completions") if stream_wanted => {
            eprintln!("0.05.074.646 I slot launch_slot_: id  0 | task 7 | processing task, is_child = 0");
            let chunk = |t: &str| format!("data: {{\"choices\":[{{\"delta\":{{\"content\":\"{t}\"}}}}]}}\n\n");
            let mut text = chunk("Ciao");
            text.push_str(&chunk(", sono un modello finto."));
            text.push_str(
                "data: {\"choices\":[],\"timings\":{\"prompt_per_second\":1500.0,\"predicted_per_second\":42.0}}\n\n",
            );
            text.push_str("data: [DONE]\n\n");
            eprintln!(
                "0.05.118.983 I slot print_timing: id  0 | task 7 | prompt eval time =      21.22 ms /    25 tokens \
                 (    0.85 ms per token,  1178.02 tokens per second)"
            );
            eprintln!(
                "0.05.118.986 I slot print_timing: id  0 | task 7 |        eval time =      22.86 ms /     4 tokens \
                 (    7.62 ms per token,   131.22 tokens per second)"
            );
            eprintln!(
                "0.05.119.017 I slot      release: id  0 | task 7 | stop processing: n_tokens = 28, truncated = 0"
            );
            send(200, "text/event-stream", text)
        }
        ("POST", "/v1/chat/completions") if body.replace(' ', "").contains("\"role\":\"tool\"") => send(
            200,
            JSON,
            r#"{"choices":[{"message":{"role":"assistant","content":"A Roma ci sono 20 gradi."}}]}"#.into(),
        ),
        ("POST", "/v1/chat/completions") if body.contains("\"tools\"") => send(
            200,
            JSON,
            r#"{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"1","type":"function",
                "function":{"name":"get_weather","arguments":"{\"city\":\"Roma\"}"}}]}}]}"#
                .into(),
        ),
        ("POST", "/v1/chat/completions") => send(
            200,
            JSON,
            r#"{"choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"completion_tokens":1}}"#.into(),
        ),
        ("POST", "/v1/embeddings") => send(200, JSON, r#"{"data":[{"index":0,"embedding":[0.1,0.2,0.3]}]}"#.into()),
        ("POST", "/v1/rerank") => send(
            200,
            JSON,
            r#"{"results":[{"index":1,"relevance_score":0.1},{"index":0,"relevance_score":0.9}]}"#.into(),
        ),
        _ => send(404, JSON, r#"{"error":{"message":"File Not Found"}}"#.into()),
    }
}

/// The fake builds' archives at their GitHub download addresses: one folder with the fake tools,
/// as the real releases have it.
#[cfg(unix)]
pub fn serve_release() {
    static ARCHIVE: LazyLock<Vec<u8>> = LazyLock::new(|| {
        let dir = crate::ui::BASE.parent().expect("test folder").join("fake-release");
        let top = dir.join("llama-b9999");
        std::fs::create_dir_all(&top).unwrap();
        for e in std::fs::read_dir(llama_cpp()).unwrap().flatten() {
            std::fs::copy(e.path(), top.join(e.file_name())).unwrap();
        }
        let archive = dir.join("release.tar.gz");
        let ok = std::process::Command::new("tar")
            .arg("-czf")
            .arg(&archive)
            .arg("-C")
            .arg(&dir)
            .arg("llama-b9999")
            .status()
            .is_ok_and(|s| s.success());
        assert!(ok, "tar");
        std::fs::read(archive).unwrap()
    });
    for name in ASSETS {
        let path = format!("/github.com/ggml-org/llama.cpp/releases/download/b9999/{name}");
        route(&path, 200, "application/gzip", ARCHIVE.clone());
    }
}

// -- other computers ---------------------------------------------------------------------------------

/// The Corral of a fake host: the real commands (`cli::run`), on that host's data folder.
fn corral(args: &[String]) -> i32 {
    match crate::cli::run(args) {
        Some(c) if c == std::process::ExitCode::SUCCESS => 0,
        Some(c) if c == std::process::ExitCode::from(2) => 2,
        Some(_) => 1,
        None => 2,
    }
}

/// The folder of fake host `name` in this test run: data/ (its Corral's) and home/.
fn host_dir(tests: &std::path::Path, name: &str) -> std::path::PathBuf {
    tests.join("hosts").join(name)
}

/// A host the fake ssh reaches: its Corral (with the fake llama.cpp, one model, a profile) in its
/// own data folder, `corral` in its ~/.local/bin unless the name says "nocorral". Its data folder.
#[cfg(unix)]
pub fn host(name: &str) -> std::path::PathBuf {
    let dir = host_dir(crate::ui::BASE.parent().expect("test folder"), name);
    let data = dir.join("data");
    let bin = dir.join("home").join(".local").join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    if !name.contains("nocorral") {
        std::fs::copy(llama_cpp().join("corral"), bin.join("corral")).unwrap();
    }
    let inst = install();
    let inst_dir = data.join("installs").join(&inst.name);
    std::fs::create_dir_all(&inst_dir).unwrap();
    std::fs::write(inst_dir.join("install.json"), serde_json::to_string(&inst).unwrap()).unwrap();
    let model = data.join("models").join("test__remote").join("Remote-Q4_K_M.gguf");
    std::fs::create_dir_all(model.parent().unwrap()).unwrap();
    std::fs::write(&model, gguf(36, 0, 40960, 100)).unwrap();
    let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let mut state = crate::ui::State { install: Some(inst.name.clone()), ..Default::default() };
    let preset: crate::ui::Preset = [("port".to_string(), port.to_string())].into();
    state.presets.entry(crate::ui::ps(&model)).or_default().insert("default".into(), preset);
    let entry = crate::router::Entry {
        id: "remote".into(),
        model: crate::ui::ps(&model),
        preset: String::new(),
        startup: true,
    };
    let profile = crate::router::Profile { models: vec![entry], ..Default::default() };
    state.routers.insert("lavoro".into(), profile);
    crate::ui::save_state_at(&state, &data.join("state.json")).unwrap();
    data
}

/// ssh: the command runs here, as the host's (its home and its Corral's data folder), and -L
/// opens the tunnel. A host whose name has "down" in it can't be reached.
fn ssh(args: &[String]) -> i32 {
    let mut forward = None;
    let mut i = 0;
    while i < args.len() && args[i].starts_with('-') {
        match args[i].as_str() {
            "-L" => forward = args.get(i + 1).cloned(),
            "-o" | "-p" | "-F" | "-i" => {}
            _ => {
                i += 1;
                continue;
            }
        }
        i += 2;
    }
    let Some(host) = args.get(i) else { return 255 };
    let command = args[i + 1..].join(" ");
    if host.contains("down") {
        eprintln!("ssh: connect to host {host} port 22: Connection refused");
        return 255;
    }
    if let Some(spec) = forward {
        // 127.0.0.1:PORT:TARGET:TARGET_PORT
        let parts: Vec<&str> = spec.split(':').collect();
        let (Some(port), Some(target), Some(target_port)) = (parts.get(1), parts.get(2), parts.get(3)) else {
            return 255;
        };
        let Ok(listener) = TcpListener::bind(format!("127.0.0.1:{port}")) else {
            eprintln!("bind [127.0.0.1]:{port}: Address already in use");
            eprintln!("channel_setup_fwd_listener_tcpip: cannot listen to port: {port}");
            return 255;
        };
        let to = format!("{target}:{target_port}");
        thread::spawn(move || {
            for local in listener.incoming().flatten() {
                let Ok(remote) = TcpStream::connect(&to) else { continue };
                let (Ok(mut up_from), Ok(mut up_to)) = (local.try_clone(), remote.try_clone()) else { continue };
                thread::spawn(move || std::io::copy(&mut up_from, &mut up_to));
                let (mut down_from, mut down_to) = (remote, local);
                thread::spawn(move || std::io::copy(&mut down_from, &mut down_to));
            }
        });
    }
    let tests = std::path::PathBuf::from(std::env::var_os("CORRAL_FAKE_DIR").unwrap_or_default());
    let dir = host_dir(&tests, host);
    let status = std::process::Command::new("sh")
        .arg("-c")
        .arg(&command)
        .env("HOME", dir.join("home"))
        .env("CORRAL_FAKE_BASE", dir.join("data"))
        .status();
    status.ok().and_then(|s| s.code()).unwrap_or(255)
}

/// scp of this Corral to a host: the host gets the fake Corral instead (this one is a test binary).
fn scp(args: &[String]) -> i32 {
    let Some(target) = args.last() else { return 1 };
    let Some((host, path)) = target.split_once(':') else { return 1 };
    if host.contains("down") {
        eprintln!("ssh: connect to host {host} port 22: Connection refused");
        return 1;
    }
    let tests = std::path::PathBuf::from(std::env::var_os("CORRAL_FAKE_DIR").unwrap_or_default());
    match std::fs::copy(tests.join("fake-llama.cpp").join("corral"), path) {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("scp: {path}: {e}");
            1
        }
    }
}
