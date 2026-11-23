//! llama.cpp installations: prebuilt GitHub releases, source builds, imported dirs.
//!
//! Each one lives in installs/<name>/install.json:
//!   {"name", "kind": release|source|external, "backend", "bin_dir", "ref"?, "flags"?, "src_dir"?}
//! "external" = a directory you already had (e.g. ~/Code/AI/llama.cpp): only registered,
//! never modified or deleted. Pure logic (blocking calls: run them in a thread); screens are in
//! `tui::installs`.

use std::env::{self, consts};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::ui::*;

pub(crate) const GH_API: &str = "https://api.github.com/repos/ggml-org/llama.cpp";
// the tests' source builds fail at once, with no network: nothing is there
const GIT_URL: &str =
    if cfg!(test) { "/nonexistent/corral-test/llama.cpp.git" } else { "https://github.com/ggml-org/llama.cpp.git" };
const BUILD_TARGETS: [&str; 3] = ["llama-server", "llama-fit-params", "llama-bench"];

static RE_DEV: LazyLock<Regex> = re!(r"^\s+(\S+): (.+) \((\d+) MiB, (\d+) MiB free\)");
static RE_VULKAN: LazyLock<Regex> = re!(r"^ggml_vulkan: (\d+) = .*\| uma: (\d) \|");
static RE_VER: LazyLock<Regex> = re!(r"build (\d+), commit (\w+)");
static RE_CUDA: LazyLock<Regex> = re!(r"CUDA Version: (\d+)\.(\d+)");
static RE_ASSET: LazyLock<Regex> = re!(r"^llama-(b\d+)-bin-(ubuntu|macos|win)-(?:(.+)-)?(x64|arm64)\.(?:tar\.gz|zip)$");

fn installs_dir() -> PathBuf {
    BASE.join("installs")
}

pub fn external_candidate() -> PathBuf {
    BASE.parent().unwrap_or(&BASE).join("llama.cpp")
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Install {
    pub name: String,
    pub kind: String,
    pub backend: String,
    pub bin_dir: String,
    #[serde(rename = "ref", default, skip_serializing_if = "Option::is_none")]
    pub git_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub flags: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub src_dir: Option<String>,
}

// -- registry -------------------------------------------------------------------

pub fn all_installs() -> Vec<Install> {
    let mut files: Vec<PathBuf> =
        fs::read_dir(installs_dir()).into_iter().flatten().flatten().map(|e| e.path().join("install.json")).collect();
    files.sort();
    files.iter().filter_map(|f| serde_json::from_str(&fs::read_to_string(f).ok()?).ok()).collect()
}

pub fn active(state: &State) -> Option<Install> {
    let all = all_installs();
    all.iter().find(|i| Some(&i.name) == state.install.as_ref()).or(all.first()).cloned()
}

fn save(inst: &Install) -> Result<(), String> {
    let d = installs_dir().join(&inst.name);
    fs::create_dir_all(&d)
        .and_then(|_| fs::write(d.join("install.json"), serde_json::to_string_pretty(inst).unwrap_or_default()))
        .map_err(|e| format!("impossibile salvare {}: {e}", d.display()))
}

pub fn binary(inst: &Install, tool: &str) -> PathBuf {
    Path::new(&inst.bin_dir).join(format!("{tool}{}", consts::EXE_SUFFIX))
}

pub fn server(inst: &Install) -> PathBuf {
    binary(inst, "llama-server")
}

/// The RPC server of `inst`, which lends this computer's GPUs to another one's llama-server:
/// ggml-rpc-server in current builds, rpc-server before. None = not in this build.
pub fn rpc_server(inst: &Install) -> Option<PathBuf> {
    ["ggml-rpc-server", "rpc-server"].into_iter().map(|tool| binary(inst, tool)).find(|p| p.is_file())
}

pub fn version(inst: &Install) -> String {
    let Ok(out) = capture(Command::new(server(inst)).arg("--version"), Duration::from_secs(20)) else {
        return "non avviabile".into();
    };
    match RE_VER.captures(&(out.stdout + &out.stderr)) {
        None => "?".into(),
        Some(m) if &m[1] == "1" => m[2].to_string(), // shallow clone: build no. is 1
        Some(m) => format!("b{} · {}", &m[1], &m[2]),
    }
}

pub struct Device {
    pub name: String,
    pub desc: String,
    pub total: u64,
    pub free: u64,
}

/// GPUs as seen by this build (`llama-server --list-devices`), sizes in MiB.
pub fn devices(inst: &Install) -> Vec<Device> {
    let Ok(out) = capture(Command::new(server(inst)).arg("--list-devices"), Duration::from_secs(30)) else {
        return vec![];
    };
    (out.stdout + &out.stderr).lines().filter_map(device_line).collect()
}

/// A device as llama.cpp lists it: "  Vulkan0: NVIDIA … (12282 MiB, 11237 MiB free)".
pub fn device_line(line: &str) -> Option<Device> {
    let m = RE_DEV.captures(line)?;
    Some(Device {
        name: m[1].into(),
        desc: m[2].into(),
        total: m[3].parse().unwrap_or(0),
        free: m[4].parse().unwrap_or(0),
    })
}

/// The devices to lend over RPC, as the RPC server's -d wants them, from what llama.cpp prints
/// while it loads its backends. Here llama.cpp would leave an integrated GPU out when there's a
/// dedicated one (Vulkan's "uma: 1"), but over RPC the other computer can't tell them apart, so
/// the choice is made here. None = all of them (no integrated GPU, or nothing else).
pub fn dedicated_gpus(backend_lines: &str) -> Option<Vec<String>> {
    let mut dedicated = vec![];
    let mut integrated = false;
    for line in backend_lines.lines() {
        // "ggml_vulkan: 1 = AMD Radeon 610M (RADV RAPHAEL_MENDOCINO) (radv) | uma: 1 | fp16: 1 | …"
        let Some(m) = RE_VULKAN.captures(line) else { continue };
        if &m[2] == "0" {
            dedicated.push(format!("Vulkan{}", &m[1]));
        } else {
            integrated = true;
        }
    }
    (integrated && !dedicated.is_empty()).then_some(dedicated)
}

// -- host checks ------------------------------------------------------------------

pub fn driver_cuda() -> Option<(u32, u32)> {
    let out = capture(&mut Command::new("nvidia-smi"), Duration::from_secs(10)).ok()?;
    let m = RE_CUDA.captures(&out.stdout)?;
    Some((m[1].parse().ok()?, m[2].parse().ok()?))
}

fn nvcc() -> Option<PathBuf> {
    let exe = format!("nvcc{}", consts::EXE_SUFFIX);
    which("nvcc")
        .or_else(|| Some(PathBuf::from(env::var_os("CUDA_PATH")?).join("bin").join(&exe)).filter(|p| p.is_file()))
        .or_else(|| {
            fs::read_dir("/usr/local")
                .ok()?
                .flatten()
                .filter(|e| e.file_name().to_string_lossy().starts_with("cuda"))
                .map(|e| e.path().join("bin").join(&exe))
                .find(|p| p.is_file())
        })
}

fn vulkan_headers() -> bool {
    let sdk = env::var_os("VULKAN_SDK").map(|p| PathBuf::from(p).join("include"));
    sdk.into_iter()
        .chain(["/usr/include", "/usr/local/include", "/opt/homebrew/include"].map(PathBuf::from))
        .any(|d| d.join("vulkan").join("vulkan.h").is_file())
}

pub fn prereqs(backend: &str) -> (bool, String) {
    let mut missing: Vec<&str> = ["git", "cmake"].into_iter().filter(|t| which(t).is_none()).collect();
    if cfg!(target_os = "macos") {
        // /usr/bin/c++ and git are stubs that only offer to install the Command Line Tools
        let clt = capture(Command::new("xcode-select").arg("-p"), Duration::from_secs(10)).is_ok_and(|o| o.ok);
        if !clt {
            missing.push("Xcode Command Line Tools");
        }
    } else if !cfg!(windows) && which("c++").is_none() {
        missing.push("c++"); // on Windows cmake finds Visual Studio by itself
    }
    let mut hint = match consts::OS {
        "macos" => "xcode-select --install && brew install git cmake",
        "windows" => "winget install Git.Git Kitware.CMake + Visual Studio Build Tools (C++)",
        _ => "sudo apt install git cmake build-essential",
    }
    .to_string();
    match backend {
        "vulkan" => {
            if which("glslc").is_none() {
                missing.push("glslc");
            }
            if !vulkan_headers() {
                missing.push("vulkan headers");
            }
            hint += if cfg!(windows) { " + Vulkan SDK (vulkan.lunarg.com)" } else { " libvulkan-dev glslc" };
        }
        "cuda" => {
            if nvcc().is_none() {
                missing.push("CUDA toolkit (nvcc)");
            }
            hint = if cfg!(windows) {
                "CUDA Toolkit da developer.nvidia.com (dopo Visual Studio). Alternativa senza toolkit: release precompilata CUDA.".into()
            } else {
                "sudo apt install nvidia-cuda-toolkit  (o il toolkit da developer.nvidia.com). Se nvcc rifiuta il gcc di \
                 sistema aggiungi -DCMAKE_CUDA_HOST_COMPILER=g++-14. Alternativa senza toolkit: release precompilata CUDA."
                    .into()
            };
        }
        _ => {}
    }
    if missing.is_empty() {
        (true, "ok".into())
    } else {
        (false, format!("manca: {}  →  {hint}", missing.join(", ")))
    }
}

fn backend_from_cmake(bin_dir: &Path) -> String {
    // build/bin (Ninja/Make) or build/bin/Release (Visual Studio): CMakeCache.txt sits in build/
    let Some(cache) =
        bin_dir.ancestors().skip(1).take(2).find_map(|d| fs::read_to_string(d.join("CMakeCache.txt")).ok())
    else {
        return "?".into();
    };
    ["CUDA", "VULKAN", "HIP", "SYCL", "METAL"]
        .iter()
        .find(|b| cache.contains(&format!("GGML_{b}:BOOL=ON")))
        .map_or("cpu".into(), |b| b.to_lowercase())
}

/// Directory holding llama-server inside a llama.cpp checkout / release folder.
pub fn find_server(path: &Path) -> Option<PathBuf> {
    let exe = format!("llama-server{}", consts::EXE_SUFFIX);
    let build = path.join("build").join("bin");
    [build.clone(), build.join("Release"), path.join("bin"), path.to_path_buf()]
        .into_iter()
        .find(|d| d.join(&exe).is_file())
}

// -- release ------------------------------------------------------------------------

#[derive(Clone, Deserialize)]
pub struct Asset {
    pub name: String,
    pub size: u64,
    #[serde(default)]
    pub browser_download_url: String,
    #[serde(default)]
    pub digest: Option<String>, // "sha256:<hex>"
}

#[derive(Deserialize)]
pub struct Release {
    pub tag_name: String,
    #[serde(default)]
    pub published_at: String,
    pub assets: Vec<Asset>,
}

#[derive(Clone)]
pub struct Variant {
    pub variant: String,
    pub assets: Vec<Asset>, // the build, plus the CUDA runtime archive for cuda-* variants
    pub ok: bool,
    pub status: String,
    pub size: u64,
}

/// Release asset naming: ubuntu / macos / win, x64 / arm64.
pub fn platform() -> Option<(&'static str, &'static str)> {
    let os = match consts::OS {
        "linux" => "ubuntu",
        "macos" => "macos",
        "windows" => "win",
        _ => return None,
    };
    let arch = match consts::ARCH {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        _ => return None,
    };
    Some((os, arch))
}

fn cuda_ver(v: &str) -> (u32, u32) {
    let mut it = v.split('.').map(|x| x.parse().unwrap_or(0));
    (it.next().unwrap_or(0), it.next().unwrap_or(0))
}

pub fn release_variants(rel: &Release, drv: Option<(u32, u32)>, os: &str, arch: &str) -> Vec<Variant> {
    let mut out = vec![];
    for a in &rel.assets {
        let Some(m) = RE_ASSET.captures(&a.name) else {
            continue;
        };
        if &m[2] != os || &m[4] != arch {
            continue;
        }
        let variant = m.get(3).map_or(if os == "macos" && arch == "arm64" { "metal" } else { "cpu" }, |v| v.as_str());
        let mut v =
            Variant { variant: variant.into(), assets: vec![a.clone()], ok: true, status: "ok".into(), size: a.size };
        if let Some(need) = variant.strip_prefix("cuda-") {
            // CUDA runtime libs ship separately: cudart-llama-<tag>-bin-ubuntu-… / cudart-llama-bin-win-…
            let tail = format!("-bin-{os}-{variant}-{arch}.");
            v.assets
                .extend(rel.assets.iter().filter(|x| x.name.starts_with("cudart-") && x.name.contains(&tail)).cloned());
            v.size = v.assets.iter().map(|x| x.size).sum();
            v.ok = drv.is_some_and(|d| cuda_ver(need) <= d);
            if !v.ok {
                v.status = drv.map_or("nessun driver NVIDIA".into(), |(a, b)| format!("driver CUDA {a}.{b} < {need}"));
            }
        } else if let rt @ ("rocm" | "sycl" | "openvino" | "opencl") = variant.split('-').next().unwrap_or("") {
            v.status = format!("richiede il runtime {rt}");
            v.ok = false;
        }
        out.push(v);
    }
    out
}

pub fn recommended(variants: &[Variant]) -> usize {
    let cuda = variants.iter().enumerate().filter(|(_, v)| v.ok && v.variant.starts_with("cuda-"));
    match cuda.max_by_key(|(_, v)| cuda_ver(&v.variant[5..])) {
        Some((i, _)) => i,
        None => variants.iter().position(|v| v.variant == "vulkan").unwrap_or(0),
    }
}

/// Extract with the system tar (bsdtar on Windows 10+ and macOS also reads .zip), dropping the
/// archive's single top-level folder if it has one.
fn extract(archive: &Path, dest: &Path) -> Result<(), String> {
    let tmp = dest.with_extension("tmp");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;
    // System32's bsdtar: a GNU tar from Git Bash first in PATH can't read .zip
    // (output captured: stray tar messages would garble the TUI)
    let tar = match env::var_os("SystemRoot") {
        Some(root) if cfg!(windows) => PathBuf::from(root).join("System32").join("tar.exe"),
        _ => PathBuf::from("tar"),
    };
    if !Command::new(&tar).arg("-xf").arg(archive).arg("-C").arg(&tmp).output().is_ok_and(|o| o.status.success()) {
        return Err(format!("estrazione di {} fallita", archive.display()));
    }
    let top: Vec<PathBuf> = fs::read_dir(&tmp).map_err(|e| e.to_string())?.flatten().map(|e| e.path()).collect();
    let root = if top.len() == 1 && top[0].is_dir() { top[0].clone() } else { tmp.clone() };
    for e in fs::read_dir(&root).map_err(|e| e.to_string())?.flatten() {
        fs::rename(e.path(), dest.join(e.file_name())).map_err(|e| e.to_string())?;
    }
    fs::remove_dir_all(&tmp).map_err(|e| e.to_string())
}

/// GitHub API request, authenticated with $GITHUB_TOKEN / $GH_TOKEN when set (higher rate limit).
pub(crate) fn gh(url: &str) -> ureq::Request {
    let token = ["GITHUB_TOKEN", "GH_TOKEN"].iter().find_map(|k| env::var(k).ok().filter(|t| !t.trim().is_empty()));
    match token {
        Some(t) => get(url).set("Authorization", &format!("Bearer {}", t.trim())),
        None => get(url),
    }
}

pub fn releases() -> Result<Vec<Release>, String> {
    get_json(gh(&format!("{GH_API}/releases?per_page=30"))) // the news screen offers the last 30 builds
}

/// Newest build tag (bNNNN), cached in cache/latest-build.json for 6 hours (None = unknown / offline).
/// Not /releases/latest: the bNNNN builds are marked prerelease and "latest" is a vX.Y.Z tag.
pub fn latest_tag() -> Option<String> {
    #[derive(Serialize, Deserialize)]
    struct Cached {
        tag: String,
        at: u64,
    }
    let file = BASE.join("cache").join("latest-build.json");
    let now = unix_now();
    let cached: Option<Cached> = fs::read_to_string(&file).ok().and_then(|s| serde_json::from_str(&s).ok());
    if let Some(c) = &cached
        && now.saturating_sub(c.at) < 6 * 3600
    {
        return Some(c.tag.clone());
    }
    let newest = releases().ok().and_then(|r| r.into_iter().map(|r| r.tag_name).max_by_key(|t| tag_no(t)));
    match newest.filter(|t| tag_no(t).is_some()) {
        Some(tag) => {
            let _ = fs::create_dir_all(BASE.join("cache"));
            let c = Cached { tag, at: now };
            let _ = fs::write(&file, serde_json::to_string(&c).unwrap_or_default());
            Some(c.tag)
        }
        None => cached.map(|c| c.tag), // stale beats nothing
    }
}

/// "b1234" -> 1234, to compare release tags.
pub fn tag_no(tag: &str) -> Option<u64> {
    tag.strip_prefix('b')?.parse().ok()
}

/// `latest` is a newer tag than this release install's.
pub fn newer_release(inst: &Install, latest: &str) -> bool {
    inst.kind == "release" && tag_no(latest) > inst.git_ref.as_deref().and_then(tag_no)
}

/// The variant of a release install ("rel-b1234-cuda-12.4" -> "cuda-12.4").
pub fn release_variant(inst: &Install) -> Option<String> {
    let tag = inst.git_ref.as_deref()?;
    inst.name.strip_prefix(&format!("rel-{tag}-")).map(String::from)
}

pub fn release_name(tag: &str, variant: &str) -> String {
    format!("rel-{tag}-{variant}")
}

/// A single ordinary folder name. "", "." and ".." (or a path) would point at installs/ itself or
/// above it, and deleting that would take the whole data folder with it.
fn is_plain_name(name: &str) -> bool {
    let mut parts = Path::new(name).components();
    matches!((parts.next(), parts.next()), (Some(Component::Normal(_)), None))
}

pub fn exists(name: &str) -> bool {
    is_plain_name(name) && installs_dir().join(name).exists()
}

/// Delete installs/<name> (a failed or cancelled install).
pub fn discard(name: &str) {
    if is_plain_name(name) {
        let _ = fs::remove_dir_all(installs_dir().join(name));
    }
}

/// Download and unpack a release variant into installs/rel-<tag>-<variant>, replacing it.
/// `on(file, progress)` reports progress; returning false cancels.
pub fn install_release(tag: &str, v: &Variant, on: &mut dyn FnMut(&str, Dl) -> bool) -> Result<Install, String> {
    let name = release_name(tag, &v.variant);
    let dest = installs_dir().join(&name);
    discard(&name);
    let bin = dest.join("bin");
    let mut fetch = || -> Result<(), String> {
        fs::create_dir_all(&bin).map_err(|e| e.to_string())?;
        for a in &v.assets {
            let arc = dest.join(&a.name);
            let sha = a.digest.as_deref().and_then(|d| d.strip_prefix("sha256:"));
            download(get(&a.browser_download_url), &arc, a.size, sha, &mut |p| on(&a.name, p))
                .map_err(|e| e.to_string())?;
            extract(&arc, &bin)?;
            let _ = fs::remove_file(&arc);
        }
        Ok(())
    };
    if let Err(e) = fetch() {
        discard(&name);
        return Err(e);
    }
    let inst = Install {
        name,
        kind: "release".into(),
        backend: v.variant.split('-').next().unwrap_or("cpu").into(),
        bin_dir: ps(&bin),
        git_ref: Some(tag.into()),
        flags: vec![],
        src_dir: None,
    };
    save(&inst)?;
    Ok(inst)
}

// -- source build ---------------------------------------------------------------------

/// Backends offered for a source build: (name, description, cmake flags).
pub fn build_options() -> Vec<(&'static str, &'static str, &'static [&'static str])> {
    if cfg!(target_os = "macos") {
        vec![
            ("metal", "Metal · GPU Apple, il default su Mac", &[]),
            ("cpu", "solo CPU", &["-DGGML_METAL=OFF"]),
            ("custom", "flag cmake personalizzati (Vulkan, BLAS, …)", &[]),
        ]
    } else {
        vec![
            ("vulkan", "Vulkan · GPU NVIDIA/AMD/Intel", &["-DGGML_VULKAN=ON"]),
            ("cuda", "CUDA · NVIDIA, la più veloce sulle GPU NVIDIA", &["-DGGML_CUDA=ON"]),
            ("cpu", "solo CPU", &[]),
            ("custom", "flag cmake personalizzati (HIP, BLAS, …)", &[]),
        ]
    }
}

/// Installation names (and other names used as folder or file names) as a safe folder name:
/// anything but letters, digits and `_ . -` becomes '_'; a name made only of dots ("." / "..")
/// would mean this folder or its parent, so it gets a '_' in front.
pub fn sanitize(name: &str) -> String {
    let clean: String = name.chars().map(|c| if c.is_alphanumeric() || "_.-".contains(c) { c } else { '_' }).collect();
    if clean.chars().all(|c| c == '.') { format!("_{clean}") } else { clean }
}

/// Fresh, empty checkout for a source build (installs/<name>/src with origin set), replacing any old one.
pub fn new_source(name: &str, backend: &str, git_ref: &str, flags: Vec<String>) -> Result<Install, String> {
    if !is_plain_name(name) {
        return Err(format!("nome «{name}» non valido per una cartella"));
    }
    discard(name);
    let src = installs_dir().join(name).join("src");
    let git =
        |args: &[&str]| Command::new("git").args(args).current_dir(&src).output().is_ok_and(|o| o.status.success());
    if fs::create_dir_all(&src).is_err() || !git(&["init", "-q"]) || !git(&["remote", "add", "origin", GIT_URL]) {
        discard(name);
        return Err("git init fallito (git è installato?)".into());
    }
    Ok(Install {
        name: name.into(),
        kind: "source".into(),
        backend: backend.into(),
        bin_dir: String::new(),
        git_ref: Some(git_ref.into()),
        flags,
        src_dir: Some(ps(&src)),
    })
}

pub fn build_log(inst: &Install) -> PathBuf {
    logs().join(format!("build-{}.log", inst.name))
}

pub enum Progress {
    Step(&'static str),
    Line(String),
}

/// fetch + checkout + cmake configure + compile (blocking), then record bin_dir and save.
pub fn build(inst: &mut Install, cancel: &AtomicBool, on: &mut dyn FnMut(Progress)) -> Result<(), String> {
    let src = PathBuf::from(inst.src_dir.clone().unwrap_or_default());
    let log = build_log(inst);
    let jobs = std::thread::available_parallelism().map_or(4, |n| n.get()).to_string();
    let r = inst.git_ref.as_deref().unwrap_or("master");
    let s = |a: &[&str]| a.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    // RPC: llama-server's --rpc, and the RPC server that lends the GPUs (see `corral rpc`)
    let mut configure = s(&["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_RPC=ON"]);
    configure.extend(inst.flags.iter().cloned());
    let mut compile = s(&["cmake", "--build", "build", "--config", "Release", "-j", &jobs, "--target"]);
    compile.extend(s(&BUILD_TARGETS));
    let steps = [
        (s(&["git", "fetch", "--depth", "1", "origin", r]), "git fetch"),
        (s(&["git", "checkout", "--force", "FETCH_HEAD"]), "git checkout"),
        (configure, "cmake configure"),
        (compile, "compilazione"),
    ];
    for (cmd, label) in steps {
        on(Progress::Step(label));
        on(Progress::Line(format!("$ {}", cmd.join(" "))));
        run_logged(&cmd, &src, &log, cancel, &mut |l| on(Progress::Line(l))).map_err(|e| format!("{label}: {e}"))?;
    }
    // the RPC server: its target has another name in older versions, and a checkout from before
    // RPC has none, which is no reason to fail the build
    on(Progress::Step("server RPC"));
    for target in ["ggml-rpc-server", "rpc-server"] {
        let cmd = s(&["cmake", "--build", "build", "--config", "Release", "-j", &jobs, "--target", target]);
        on(Progress::Line(format!("$ {}", cmd.join(" "))));
        if run_logged(&cmd, &src, &log, cancel, &mut |l| on(Progress::Line(l))).is_ok() {
            break;
        }
        if cancel.load(Ordering::Relaxed) {
            return Err("annullato".into());
        }
    }
    inst.bin_dir = ps(&find_server(&src).unwrap_or_else(|| src.join("build").join("bin")));
    save(inst)
}

// -- import / remove ----------------------------------------------------------------------

/// Register an existing llama.cpp folder (checkout with build/bin, or an unpacked release).
pub fn import_dir(path: &Path) -> Result<Install, String> {
    let bin = find_server(path).ok_or_else(|| format!("nessun llama-server in {}", path.display()))?;
    let name = format!("ext-{}", path.file_name().map_or("llama.cpp".into(), |n| n.to_string_lossy()));
    let inst = Install {
        name,
        kind: "external".into(),
        backend: backend_from_cmake(&bin),
        bin_dir: ps(&bin),
        git_ref: None,
        flags: vec![],
        src_dir: None,
    };
    save(&inst)?;
    Ok(inst)
}

/// Delete installs/<name>; for "external" only the registration goes, the folder stays.
pub fn remove(inst: &Install) -> Result<(), String> {
    if !is_plain_name(&inst.name) {
        return Err(format!("nome «{}» non valido: non cancello nulla", inst.name));
    }
    let d = installs_dir().join(&inst.name);
    fs::remove_dir_all(&d).map_err(|e| format!("{}: {e}", d.display()))
}

pub fn install_dir(inst: &Install) -> PathBuf {
    installs_dir().join(&inst.name)
}
