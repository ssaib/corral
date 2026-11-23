//! Hugging Face: GGUF search, hardware-based suggestions, quant choice and download.
//! Pure logic (blocking HTTP: run it in a thread); screens are in `tui::hf`.

use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ui::*;

/// Hugging Face (or the mirror in $HF_ENDPOINT, like huggingface_hub), without the trailing slash.
pub fn endpoint() -> String {
    env::var("HF_ENDPOINT")
        .ok()
        .filter(|e| !e.trim().is_empty())
        .map_or("https://huggingface.co".into(), |e| e.trim().trim_end_matches('/').to_string())
}

fn api() -> String {
    format!("{}/api/models", endpoint())
}
// ponytail: flat allowance for KV cache + compute buffers at ~8k ctx; the exact
// figure comes from llama-fit-params once the file is on disk (config screen, 'v').
pub const OVERHEAD: u64 = 3 << 29; // 1.5 GiB
const Q4_BYTES_PER_PARAM: f64 = 0.6; // ~4.8 bits/weight (Q4_K_M), to size a repo before listing its files
const FULL: [&str; 3] = ["F32", "F16", "BF16"];
const LLM_TAGS: [Option<&str>; 4] = [None, Some("text-generation"), Some("image-text-to-text"), Some("conversational")];
// suggestions only: official orgs + well-known quantizers (free search shows everyone)
const TRUSTED: [&str; 20] = [
    "ggml-org",
    "unsloth",
    "bartowski",
    "lmstudio-community",
    "Qwen",
    "google",
    "mistralai",
    "microsoft",
    "meta-llama",
    "ibm-granite",
    "openai",
    "deepseek-ai",
    "nvidia",
    "HuggingFaceTB",
    "allenai",
    "THUDM",
    "zai-org",
    "moonshotai",
    "LiquidAI",
    "tiiuae",
];
const CATS: [&str; 4] = ["vision", "coding", "reasoning", "chat"];
// Rust's regex has no look-around: the guard chars before/after are matched instead
static RE_QUANT: LazyLock<Regex> =
    re!(r"(?i)(?:^|[^a-z0-9])((?:UD-)?(?:I?Q\d+(?:_[a-z0-9]+)*|BF16|F16|F32|MXFP4(?:_MOE)?))[-.]");
static RE_VISION: LazyLock<Regex> = re!(r"(?i)\bvl\b|-vl-|vision");
static RE_CODING: LazyLock<Regex> = re!(r"(?i)(?:^|[^a-z])coder|(?:^|[^a-z])code(?:$|[^a-z])|devstral|codestral");
static RE_REASONING: LazyLock<Regex> = re!(r"(?i)think|reason|-r1\b|qwq|magistral");

pub fn models_dir() -> PathBuf {
    BASE.join("models")
}

// -- pure helpers (tested in tests.rs) ----------------------------------------

pub fn quant_of(fname_or_path: &str) -> Option<String> {
    RE_QUANT.captures(fname(fname_or_path)).map(|m| m[1].to_uppercase())
}

pub struct Quant {
    pub quant: String,
    pub files: Vec<(String, u64)>,
    pub size: u64,
}

/// [(repo path, bytes)] -> (quants sorted by size, mmproj files). Split
/// GGUFs (-00001-of-0000N) are summed into one quant.
pub fn group_files(files: &[(String, u64)]) -> (Vec<Quant>, Vec<(String, u64)>) {
    let mut quants: Vec<Quant> = vec![];
    let mut mmprojs: Vec<(String, u64)> = vec![];
    for (path, n) in files {
        let name = fname(path);
        let Some(stem) = name.strip_suffix(".gguf") else {
            continue;
        };
        if name.to_lowercase().contains("mmproj") {
            mmprojs.push((path.clone(), *n));
            continue;
        }
        let q = quant_of(path).unwrap_or_else(|| stem.to_string());
        match quants.iter_mut().find(|g| g.quant == q) {
            Some(g) => {
                g.files.push((path.clone(), *n));
                g.size += n;
            }
            None => quants.push(Quant { quant: q, files: vec![(path.clone(), *n)], size: *n }),
        }
    }
    for g in &mut quants {
        g.files.sort();
    }
    quants.sort_by_key(|g| g.size);
    (quants, mmprojs)
}

/// Biggest non-full-precision quant whose size + overhead fits the budget.
pub fn recommend(quants: &[Quant], budget: u64, extra: u64) -> Option<usize> {
    let fits = quants
        .iter()
        .enumerate()
        .filter(|(_, q)| !FULL.contains(&q.quant.as_str()) && q.size + extra + OVERHEAD <= budget);
    fits.rev().max_by_key(|(_, q)| q.size).map(|(i, _)| i) // rev: the first of equal ones wins
}

pub fn best_mmproj<S: AsRef<str>>(files: &[S]) -> Option<usize> {
    ["f16", "bf16", "q8_0"]
        .iter()
        .find_map(|pref| files.iter().position(|f| fname(f.as_ref()).to_lowercase().contains(pref)))
        .or((!files.is_empty()).then_some(0))
}

pub fn category(repo: &str, pipeline_tag: Option<&str>) -> &'static str {
    if pipeline_tag == Some("image-text-to-text") || RE_VISION.is_match(repo) {
        "vision"
    } else if RE_CODING.is_match(repo) {
        "coding"
    } else if RE_REASONING.is_match(repo) {
        "reasoning"
    } else {
        "chat"
    }
}

pub fn fit_label(need: u64, budget: u64, ram: u64) -> (&'static str, u32) {
    if need <= budget {
        ("GPU", OK)
    } else if need <= budget + ram / 5 * 4 {
        ("GPU+RAM", WARN)
    } else {
        ("troppo grande", ERR)
    }
}

pub fn q4_need(params: u64) -> u64 {
    (params as f64 * Q4_BYTES_PER_PARAM) as u64 + OVERHEAD
}

pub fn thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push('.');
        }
        out.push(c);
    }
    out
}

// -- API ----------------------------------------------------------------------------

/// Token for gated repos, where huggingface_hub keeps it: $HF_TOKEN or `hf auth login`'s file.
pub fn token() -> Option<String> {
    let file = || {
        let cache = env::var_os("XDG_CACHE_HOME").map_or_else(|| home().join(".cache"), PathBuf::from);
        let dir = env::var_os("HF_HOME").map_or_else(|| cache.join("huggingface"), PathBuf::from);
        fs::read_to_string(dir.join("token")).ok()
    };
    env::var("HF_TOKEN").ok().or_else(file).map(|t| t.trim().to_string()).filter(|t| !t.is_empty())
}

fn hf(url: &str) -> ureq::Request {
    let req = get(url);
    match token() {
        Some(t) => req.set("Authorization", &format!("Bearer {t}")),
        None => req,
    }
}

#[derive(Clone)]
pub struct Repo {
    pub id: String,
    pub downloads: u64,
    pub params: Option<u64>,
    pub arch: Option<String>,
    pub ctx: Option<u64>,
    pub pipeline: Option<String>,
    pub created: Option<String>, // RFC 3339
}

fn list_models(search: Option<&str>, limit: usize) -> Result<Vec<Repo>, String> {
    list(search, None, "downloads", limit)
}

/// GGUF repos sorted by `sort` (downloads, createdAt, trendingScore), newest/highest first.
pub fn list(search: Option<&str>, author: Option<&str>, sort: &str, limit: usize) -> Result<Vec<Repo>, String> {
    let mut req = hf(&api()).query("filter", "gguf").query("sort", sort).query("limit", &limit.to_string());
    for (k, v) in [("search", search), ("author", author)] {
        if let Some(v) = v {
            req = req.query(k, v);
        }
    }
    for e in ["downloads", "gguf", "pipeline_tag", "createdAt"] {
        req = req.query("expand", e);
    }
    let list: Vec<Value> = get_json(req)?;
    // read field by field: one odd entry must not sink the whole list
    Ok(list
        .iter()
        .map(|m| Repo {
            id: m["id"].as_str().unwrap_or_default().into(),
            downloads: m["downloads"].as_u64().unwrap_or(0),
            params: m["gguf"]["total"].as_u64().filter(|&n| n > 0),
            arch: m["gguf"]["architecture"].as_str().map(Into::into),
            ctx: m["gguf"]["context_length"].as_u64().filter(|&n| n > 0),
            pipeline: m["pipeline_tag"].as_str().map(Into::into),
            created: m["createdAt"].as_str().map(Into::into),
        })
        .collect())
}

/// Orgs whose new GGUF repos are news (official llama.cpp org and the best-known quantizers).
pub const QUANTIZERS: [&str; 4] = ["ggml-org", "unsloth", "bartowski", "lmstudio-community"];

/// Chat/vision models llama-server can serve (not image generators, embedders, …).
pub fn is_llm(r: &Repo) -> bool {
    LLM_TAGS.contains(&r.pipeline.as_deref()) && r.params.is_some_and(|p| p >= 100_000_000)
}

pub fn is_trusted(repo: &str) -> bool {
    TRUSTED.contains(&repo.split('/').next().unwrap_or(""))
}

// -- listing --------------------------------------------------------------------------

pub fn search(q: &str) -> Result<Vec<Repo>, String> {
    list_models(Some(q), 30)
}

/// The most downloaded GGUF repos of trusted orgs that run here, up to 6 per category.
pub fn suggest(budget: u64, ram: u64) -> Result<Vec<(&'static str, Repo)>, String> {
    // ponytail: one API page (1000 repos), follow the Link header if it feels thin
    let models = list_models(None, 1000)?;
    let mut per_cat: [Vec<Repo>; 4] = Default::default();
    for m in models {
        let Some(params) = m.params else { continue };
        if params < 500_000_000
            || !LLM_TAGS.contains(&m.pipeline.as_deref())
            || m.ctx.unwrap_or(0) < 4096
            || !TRUSTED.contains(&m.id.split('/').next().unwrap_or(""))
            || fit_label(q4_need(params), budget, ram).0 == "troppo grande"
        {
            continue;
        }
        let i = CATS.iter().position(|c| *c == category(&m.id, m.pipeline.as_deref())).unwrap_or(3);
        if per_cat[i].len() < 6 {
            per_cat[i].push(m);
        }
    }
    Ok(per_cat.into_iter().zip(CATS).flat_map(|(v, c)| v.into_iter().map(move |m| (c, m))).collect())
}

/// Memory a repo needs at Q4_K_M, judged from its parameter count.
pub fn repo_need(r: &Repo) -> Option<u64> {
    r.params.map(q4_need)
}

// -- download --------------------------------------------------------------------------

pub struct RepoFiles {
    pub quants: Vec<Quant>,
    pub mmprojs: Vec<(String, u64)>,
    pub sha: HashMap<String, String>, // LFS sha256 per repo path
    pub gated: bool,
}

fn repo_info(repo: &str) -> Result<Value, String> {
    get_json(hf(&format!("{}/{repo}", api())).query("blobs", "true"))
}

/// Every file of the repo: path -> (size, LFS sha256 or "").
pub fn remote_files(repo: &str) -> Result<HashMap<String, (u64, String)>, String> {
    let info = repo_info(repo)?;
    let sib = info["siblings"].as_array().into_iter().flatten();
    Ok(sib
        .filter_map(|s| {
            let sha = s["lfs"]["sha256"].as_str().unwrap_or_default().to_string();
            Some((s["rfilename"].as_str()?.to_string(), (s["size"].as_u64().unwrap_or(0), sha)))
        })
        .collect())
}

pub fn repo_files(repo: &str) -> Result<RepoFiles, String> {
    let info = repo_info(repo)?;
    let sib = || info["siblings"].as_array().into_iter().flatten();
    let files: Vec<(String, u64)> =
        sib().filter_map(|s| Some((s["rfilename"].as_str()?.to_string(), s["size"].as_u64().unwrap_or(0)))).collect();
    let sha = sib()
        .filter_map(|s| Some((s["rfilename"].as_str()?.to_string(), s["lfs"]["sha256"].as_str()?.to_string())))
        .collect();
    let (quants, mmprojs) = group_files(&files);
    if quants.is_empty() {
        return Err("nessun file GGUF nel repo".into());
    }
    let gated = !matches!(info["gated"], Value::Bool(false) | Value::Null); // false | "auto" | "manual"
    Ok(RepoFiles { quants, mmprojs, sha, gated })
}

/// Local folder for a repo's files: models/<org>__<name>.
pub fn repo_dir(repo: &str) -> PathBuf {
    models_dir().join(repo.replace('/', "__"))
}

const ORIGIN: &str = ".corral-origin.json";

/// Where a models/<org>__<name> folder came from, and the sha256 of each file in it (repo path ->
/// hex), written by `fetch` and by the update check: models/<org>__<name>/.corral-origin.json.
#[derive(Default, Serialize, Deserialize)]
pub struct Origin {
    pub repo: String,
    #[serde(default)]
    pub sha256: BTreeMap<String, String>,
}

/// The origin of a repo folder; without the file (downloads before it existed) it is read from
/// the folder name (org__name).
pub fn origin(dir: &Path) -> Option<Origin> {
    if let Some(o) = fs::read_to_string(dir.join(ORIGIN)).ok().and_then(|s| serde_json::from_str(&s).ok()) {
        return Some(o);
    }
    // ponytail: an org containing "__" would split wrong here; the origin file covers new downloads
    let (org, name) = dir.file_name()?.to_str()?.split_once("__")?;
    Some(Origin { repo: format!("{org}/{name}"), sha256: BTreeMap::new() })
}

pub fn save_origin(dir: &Path, o: &Origin) {
    let _ = fs::write(dir.join(ORIGIN), serde_json::to_string_pretty(o).unwrap_or_default());
}

/// Repo path ("Q6_K/m-00001-of-00002.gguf") under the local model folder.
pub fn local(dest: &Path, repo_path: &str) -> PathBuf {
    repo_path.split('/').fold(dest.to_path_buf(), |p, c| p.join(c))
}

/// A repo id or file path from the server that stays inside the local folder: no empty, "." or
/// ".." part, no drive letter or backslash.
pub fn safe_repo_path(p: &str) -> bool {
    p.split('/').all(|part| !part.is_empty() && part != "." && part != ".." && !part.contains(['\\', ':']))
}

/// Download one repo file into repo_dir (blocking); Err = message for the user.
pub fn fetch(
    repo: &str,
    file: &str,
    size: u64,
    sha: Option<&str>,
    on: &mut dyn FnMut(Dl) -> bool,
) -> Result<(), String> {
    if !safe_repo_path(repo) || !safe_repo_path(file) {
        return Err(format!("nome non valido dal server: {repo}/{file}"));
    }
    let dir = repo_dir(repo);
    let path = local(&dir, file);
    fs::create_dir_all(path.parent().unwrap_or(&models_dir())).map_err(|e| e.to_string())?;
    let mut o = origin(&dir).unwrap_or_else(|| Origin { repo: repo.into(), ..Origin::default() });
    // a changed remote file of the same size must not pass for the one already here
    let size = if sha.is_some_and(|s| o.sha256.get(file).is_some_and(|old| old != s)) { 0 } else { size };
    match download(hf(&format!("{}/{repo}/resolve/main/{file}", endpoint())), &path, size, sha, on) {
        Ok(()) => {
            if let Some(s) = sha {
                o.sha256.insert(file.into(), s.into());
                save_origin(&dir, &o);
            }
            Ok(())
        }
        Err(ureq::Error::Status(401 | 403, _)) => {
            Err("repo gated: accetta la licenza sul sito, poi imposta HF_TOKEN (o `hf auth login`)".into())
        }
        Err(e) => Err(format!("download fallito: {e}")),
    }
}
