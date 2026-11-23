//! Local GGUF models: where they are (Corral's folders, extra folders, the llama.cpp / Hugging Face
//! caches, LM Studio), deleting them, checking Hugging Face for a newer file, preset export/import
//! and orphaned presets. Pure logic (blocking: run network and hashing in a thread).

use std::collections::HashSet;
use std::env;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::preset::{self, FIELDS, RE_SPLIT};
use crate::{hub, ui::*};

pub struct Local {
    pub path: PathBuf,
    pub source: &'static str,
    /// where it sits, for display ("org/repo", a folder)
    pub folder: String,
    pub size: u64,
    /// Corral's own folders and the ones the user added: files we may delete
    pub managed: bool,
}

fn cache_home() -> PathBuf {
    env::var_os("XDG_CACHE_HOME").map_or_else(|| home().join(".cache"), PathBuf::from)
}

/// Hugging Face hub cache, the way huggingface_hub (and llama.cpp's -hf since late 2025) find it.
pub fn hf_cache() -> PathBuf {
    let var = |k: &str| env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    var("HF_HUB_CACHE")
        .or_else(|| var("HUGGINGFACE_HUB_CACHE"))
        .or_else(|| var("HF_HOME").map(|h| h.join("hub")))
        .unwrap_or_else(|| cache_home().join("huggingface").join("hub"))
}

/// llama.cpp's own cache ($LLAMA_CACHE; older -hf downloads went to the platform cache dir).
pub fn llama_cache() -> PathBuf {
    if let Some(p) = env::var_os("LLAMA_CACHE").filter(|v| !v.is_empty()) {
        return PathBuf::from(p);
    }
    if cfg!(windows) {
        env::var_os("LOCALAPPDATA")
            .map_or_else(|| home().join("AppData").join("Local"), PathBuf::from)
            .join("llama.cpp")
    } else if cfg!(target_os = "macos") {
        home().join("Library").join("Caches").join("llama.cpp")
    } else {
        cache_home().join("llama.cpp")
    }
}

/// (source label, folder, managed) in scan order: a file found twice keeps the first source.
pub fn sources(state: &State) -> Vec<(&'static str, PathBuf, bool)> {
    let mut v: Vec<_> = preset::model_dirs().into_iter().map(|d| ("corral", d, true)).collect();
    v.extend(state.model_dirs.iter().map(|d| ("cartella", expand(d), true)));
    v.push(("llama.cpp", llama_cache(), false));
    v.push(("HF cache", hf_cache(), false));
    v.push(("LM Studio", home().join(".lmstudio").join("models"), false));
    v.push(("LM Studio", home().join(".cache").join("lm-studio").join("models"), false));
    v
}

/// Every main GGUF (no mmproj, first part of a split model) in every source.
pub fn scan(state: &State) -> Vec<Local> {
    let mut out = vec![];
    let mut seen = HashSet::new();
    for (source, root, managed) in sources(state) {
        let mut found = vec![];
        preset::walk(&root, &mut found);
        found.sort();
        for p in found {
            let name = fname(&ps(&p)).to_lowercase();
            if name.contains("mmproj") || RE_SPLIT.captures(&name).is_some_and(|m| &m[1] != "00001") {
                continue;
            }
            // the HF cache links snapshots to blobs: the same blob may show up in several snapshots
            if !seen.insert(fs::canonicalize(&p).unwrap_or_else(|_| p.clone())) {
                continue;
            }
            let folder = folder_label(&root, &p);
            out.push(Local { size: preset::model_size(&p), folder, path: p, source, managed });
        }
    }
    out
}

/// "org/repo" for repo-shaped folders (models--org--repo, org__repo), else the folder under the root.
fn folder_label(root: &Path, p: &Path) -> String {
    let rel = p.parent().and_then(|d| d.strip_prefix(root).ok()).map(ps).unwrap_or_default();
    let first = rel.split(['/', '\\']).next().unwrap_or("");
    if let Some(r) = first.strip_prefix("models--") {
        return r.replacen("--", "/", 1);
    }
    if root == hub::models_dir()
        && let Some((org, name)) = first.split_once("__")
    {
        return format!("{org}/{name}");
    }
    if rel.is_empty() { short(&ps(root)) } else { rel }
}

/// All the files of a model: the parts of a split GGUF, else just the file.
pub fn parts(p: &Path) -> Vec<PathBuf> {
    let name = fname(&ps(p)).to_string();
    if !RE_SPLIT.is_match(&name) {
        return vec![p.to_path_buf()];
    }
    let stem = RE_SPLIT.replace(&name, "").into_owned();
    let dir = p.parent().unwrap_or(Path::new("."));
    let mut v: Vec<PathBuf> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|f| {
            let n = fname(&ps(f)).to_string();
            RE_SPLIT.is_match(&n) && RE_SPLIT.replace(&n, "") == stem
        })
        .collect();
    v.sort();
    v
}

pub fn total_size(models: &[Local]) -> u64 {
    models.iter().map(|m| m.size).sum()
}

/// What deleting `model` removes: its files, plus the mmproj files next to it that no other model
/// in that folder and no other model's preset needs.
pub fn delete_plan(state: &State, model: &Path, all: &[Local]) -> Vec<PathBuf> {
    let dir = model.parent();
    let others_here = all.iter().any(|m| m.path != model && m.path.parent() == dir);
    let used_elsewhere = |mm: &Path| {
        state
            .presets
            .iter()
            .filter(|(k, _)| Path::new(k.as_str()) != model)
            .any(|(_, named)| named.values().any(|p| p.get("mmproj").is_some_and(|v| Path::new(v) == mm)))
    };
    let mut v = parts(model);
    if !others_here {
        v.extend(preset::mmprojs_near(model).into_iter().filter(|mm| !used_elsewhere(mm)));
    }
    v
}

/// Delete the files of `plan` and forget the model's presets / last use; an emptied repo folder
/// under models/ goes too.
pub fn delete(state: &mut State, model: &Path, plan: &[PathBuf]) -> Result<(), String> {
    for f in plan {
        fs::remove_file(f).map_err(|e| format!("{}: {e}", f.display()))?;
    }
    forget(state, &ps(model));
    if let Some(dir) = model.parent().filter(|d| d.starts_with(hub::models_dir()) && *d != hub::models_dir()) {
        let left: Vec<_> = fs::read_dir(dir).into_iter().flatten().flatten().collect();
        if left.iter().all(|e| e.file_name().to_string_lossy().starts_with('.')) {
            let _ = fs::remove_dir_all(dir);
        }
    }
    save_state(state)
}

fn forget(state: &mut State, model: &str) {
    state.presets.remove(model);
    state.preset_sel.remove(model);
    state.last_used.remove(model);
    if state.last_model.as_deref() == Some(model) {
        state.last_model = None;
    }
}

/// Models with saved presets / last use whose file is gone.
pub fn orphans(state: &State) -> Vec<String> {
    let keys = state.presets.keys().chain(state.last_used.keys()).chain(state.preset_sel.keys());
    let mut v: Vec<String> = keys.filter(|k| !Path::new(k.as_str()).exists()).cloned().collect();
    v.sort();
    v.dedup();
    v
}

pub fn drop_orphans(state: &mut State) -> Result<usize, String> {
    let o = orphans(state);
    for m in &o {
        forget(state, m);
    }
    save_state(state).map(|()| o.len())
}

// -- updates ------------------------------------------------------------------------------

pub fn sha256_file(p: &Path) -> Result<String, String> {
    let mut f = File::open(p).map_err(|e| e.to_string())?;
    let mut h = Sha256::new();
    let mut buf = vec![0; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

/// The Hugging Face repo a file under models/<org>__<name>/ came from, with its folder and repo path.
pub fn hf_origin(model: &Path) -> Option<(String, PathBuf, String)> {
    let rel = model.strip_prefix(hub::models_dir()).ok()?;
    let mut comps = rel.components();
    let dir = hub::models_dir().join(comps.next()?);
    let file = comps.map(|c| c.as_os_str().to_string_lossy().into_owned()).collect::<Vec<_>>().join("/");
    Some((hub::origin(&dir)?.repo, dir, file))
}

pub enum Update {
    Same,
    /// files that differ (or vanished) on Hugging Face
    Changed(Vec<String>),
}

/// Compare `model` (all its parts) with the repo it was downloaded from: size first, then sha256
/// (the one saved at download; otherwise the local file is hashed once and the result saved).
/// Blocking: run it in a thread.
pub fn check_update(model: &Path) -> Result<(String, Update), String> {
    let (repo, dir, _) = hf_origin(model).ok_or("non scaricato da Corral: non so da quale repo viene")?;
    let remote = hub::remote_files(&repo)?;
    let mut o = hub::origin(&dir).unwrap_or_default();
    o.repo = repo.clone();
    let mut changed = vec![];
    for part in parts(model) {
        let Some((_, _, file)) = hf_origin(&part) else { continue };
        let Some((size, sha)) = remote.get(&file) else {
            changed.push(format!("{file} (non più nel repo)"));
            continue;
        };
        let local = fs::metadata(&part).map_or(0, |m| m.len());
        if local != *size {
            changed.push(file);
            continue;
        }
        if sha.is_empty() {
            continue; // not an LFS file: the size is all we have
        }
        let mine = match o.sha256.get(&file) {
            Some(s) => s.clone(),
            None => {
                let s = sha256_file(&part)?;
                o.sha256.insert(file.clone(), s.clone());
                s
            }
        };
        if !mine.eq_ignore_ascii_case(sha) {
            changed.push(file);
        }
    }
    hub::save_origin(&dir, &o);
    Ok((repo, if changed.is_empty() { Update::Same } else { Update::Changed(changed) }))
}

// -- preset files -----------------------------------------------------------------------------

/// Write preset `name` of `model` (values that differ from the defaults) as JSON to `dest`.
/// Without the API key: an exported preset is made to be shared.
pub fn export_preset(state: &State, model: &Path, name: &str, dest: &Path) -> Result<(), String> {
    let mut values = state.presets.get(&ps(model)).and_then(|n| n.get(name)).cloned().unwrap_or_default();
    values.remove("api_key");
    let doc = serde_json::json!({
        "corral_preset": 1,
        "name": name,
        "model": fname(&ps(model)),
        "values": values,
    });
    fs::write(dest, serde_json::to_string_pretty(&doc).unwrap_or_default())
        .map_err(|e| format!("{}: {e}", dest.display()))
}

/// Default export file name: ~/<model alias>-<preset>.preset.json
pub fn export_path(model: &Path, name: &str) -> PathBuf {
    let clean = |s: &str| -> String {
        s.chars().map(|c| if c.is_alphanumeric() || "-_.".contains(c) { c } else { '_' }).collect()
    };
    home().join(format!("{}-{}.preset.json", clean(&preset::base_alias(model)), clean(name)))
}

/// A preset file's (name, values): Corral's export, or a bare {key: value} object. Unknown keys and
/// values llama-server would refuse are dropped. Returns also how many were dropped.
pub fn parse_preset(text: &str, fallback_name: &str) -> Result<(String, Preset, usize), String> {
    let v: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("JSON non valido: {e}"))?;
    let values = if v.get("values").is_some() { &v["values"] } else { &v };
    let obj = values.as_object().ok_or("il file non contiene un oggetto di parametri")?;
    let name = v["name"].as_str().filter(|n| !n.trim().is_empty()).unwrap_or(fallback_name).trim().to_string();
    let mut out = Preset::new();
    let mut dropped = 0;
    for (k, val) in obj {
        let s = match val {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::Bool(b) => (if *b { "on" } else { "off" }).into(),
            _ => String::new(),
        };
        if FIELDS.iter().any(|f| f.key == k) && preset::check(k, &s).is_ok() {
            out.insert(k.clone(), s);
        } else {
            dropped += 1;
        }
    }
    Ok((name, out, dropped))
}

/// Import a preset file for `model` as a new preset (name made unique) and select it.
pub fn import_preset(state: &mut State, model: &Path, file: &Path) -> Result<(String, usize), String> {
    let text = fs::read_to_string(file).map_err(|e| format!("{}: {e}", file.display()))?;
    let stem = fname(&ps(file)).trim_end_matches(".json").trim_end_matches(".preset").to_string();
    let (name, values, dropped) = parse_preset(&text, &stem)?;
    let taken: Vec<String> = preset::preset_names(state, model);
    let name = unique(&name, &taken);
    preset::add_preset(state, model, &name, values)?;
    save_state(state)?;
    Ok((name, dropped))
}

/// `name`, or "name 2", "name 3"… when taken.
pub fn unique(name: &str, taken: &[String]) -> String {
    let free = |n: &String| !taken.contains(n);
    std::iter::once(name.to_string()).chain((2..).map(|i| format!("{name} {i}"))).find(free).unwrap_or_default()
}

/// Short summary of a preset's changed values: "ctx=32768 ngl=all".
pub fn summary(values: &Preset) -> String {
    values.iter().filter(|(k, _)| *k != "api_key").map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(" ")
}
