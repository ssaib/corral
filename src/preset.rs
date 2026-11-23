//! Per-model llama-server presets: the parameter table (FIELDS), defaults, validation,
//! the command line, local model discovery and the llama-fit-params memory estimate.
//! Pure logic: no terminal I/O (the editor is `tui::config`).

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, SystemTime};

use regex::Regex;

use crate::installs::{self, Install};
use crate::{hub, ui::*};

pub const KV: &[&str] = &["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl", "f32"];
pub const ALWAYS: [&str; 4] = ["alias", "mmproj", "host", "port"]; // passed even when equal to the default
/// What llama-fit-params needs to estimate memory (the rest of the preset doesn't change it).
pub const MEMORY_KEYS: &[&str] = &[
    "device",
    "ngl",
    "ctx",
    "fit",
    "fit_target",
    "fit_ctx",
    "fa",
    "ctk",
    "ctv",
    "np",
    "ts",
    "cpu_moe",
    "n_cpu_moe",
    "ot",
    "moe_cache",
    "batch",
    "ubatch",
    "kv_offload",
    "load_mode",
];
pub const INTS: &[&str] = &[
    "port",
    "ctx",
    "np",
    "threads",
    "top_k",
    "seed",
    "reasoning_budget",
    "n_cpu_moe",
    "moe_cache",
    "draft_max",
    "draft_min",
    "batch",
    "ubatch",
    "cache_reuse",
    "models_max",
    "margin",
    "sleep",
    "fit_ctx",
];
pub const FLOATS: &[&str] = &["temp", "top_p", "min_p", "repeat_penalty", "lora_scale"];
pub static RE_SPLIT: LazyLock<Regex> = re!(r"-(\d{5})-of-(\d{5})\.gguf$");

pub enum Flag {
    No,
    Opt(&'static str),
    Toggle(&'static str, &'static str),
}

pub enum Choices {
    Free,
    Mmproj,
    Device,
    Model, // another local GGUF (draft)
    Hosts, // hosts of ~/.ssh/config (RPC)
    Lora,
    Ot, // OT_PRESETS or free text
    List(&'static [&'static str]),
}

pub struct Field {
    pub key: &'static str,
    pub label: &'static str,
    pub sec: &'static str,
    pub default: &'static str, // '' = server default, flag omitted
    pub flag: Flag,
    pub choices: Choices,
}

pub(crate) const fn fld(
    key: &'static str,
    label: &'static str,
    sec: &'static str,
    default: &'static str,
    flag: Flag,
    choices: Choices,
) -> Field {
    Field { key, label, sec, default, flag, choices }
}

use Choices::{Free, List};
use Flag::{Opt, Toggle};

/// Shortcuts for -ot (tensor name regex = buffer type); llama.cpp's own MoE pattern, see common/fit.cpp.
pub const OT_PRESETS: &[(&str, &str)] = &[
    (r"\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU", "tutti gli esperti MoE in RAM (come --cpu-moe)"),
    (r"\.ffn_(up|gate_up|gate)_(ch|)exps=CPU", "esperti up/gate in RAM, down su GPU"),
    (r"blk\.\d*[13579]\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU", "esperti dei layer dispari in RAM (metà)"),
];

pub(crate) const ON_OFF: Choices = List(&["on", "off"]);
const SRV_ON_OFF: Choices = List(&["", "on", "off"]);

// Opt("a|b"): the first spelling this build knows wins (renamed flags); sections "moe" (dense models)
// and "avanzate" start collapsed in the editor.
pub static FIELDS: &[Field] = &[
    fld("alias", "alias", "base", "", Opt("--alias"), Free),
    fld("mmproj", "mmproj (vision)", "base", "", Opt("--mmproj"), Choices::Mmproj),
    fld("host", "host", "base", "127.0.0.1", Opt("--host"), Free),
    fld("port", "porta", "base", "8080", Opt("--port"), Free),
    fld("api_key", "API key", "base", "", Opt("--api-key"), Free),
    fld("device", "dispositivi", "memoria", "", Opt("-dev"), Choices::Device),
    fld("ngl", "layer su GPU", "memoria", "auto", Opt("-ngl"), Free),
    fld("ctx", "contesto (0 = dal modello)", "memoria", "0", Opt("-c"), Free),
    fld("fit", "auto-fit in memoria", "memoria", "on", Opt("-fit"), ON_OFF),
    fld("fit_target", "VRAM da lasciare libera (MiB)", "memoria", "", Opt("-fitt|--fit-target"), Free),
    fld("fit_ctx", "contesto minimo dell'auto-fit", "memoria", "", Opt("-fitc|--fit-ctx"), Free),
    fld("fa", "flash attention", "memoria", "auto", Opt("-fa"), List(&["auto", "on", "off"])),
    fld("ctk", "KV cache K", "memoria", "f16", Opt("-ctk"), List(KV)),
    fld("ctv", "KV cache V", "memoria", "f16", Opt("-ctv"), List(KV)),
    fld("ts", "ripartizione tra GPU (es. 3,1)", "memoria", "", Opt("-ts"), Free),
    // hosts that lend their GPUs (llama.cpp RPC): --rpc with their tunnels, added at launch (see
    // `server::add_rpc`)
    fld("rpc", "GPU di altri host (RPC)", "memoria", "", Flag::No, Choices::Hosts),
    fld("cpu_moe", "tutti gli esperti in RAM", "moe", "off", Toggle("--cpu-moe", ""), ON_OFF),
    fld("n_cpu_moe", "esperti in RAM: primi N layer", "moe", "", Opt("--n-cpu-moe"), Free),
    fld("ot", "override tensori (-ot)", "moe", "", Opt("-ot"), Choices::Ot),
    fld("moe_cache", "cache GPU esperti (MiB)", "moe", "", Opt("--moe-cache-mib"), Free),
    fld("draft", "modello draft", "draft", "", Opt("-md"), Choices::Model),
    fld(
        "spec_type",
        "tipo speculative",
        "draft",
        "",
        Opt("--spec-type"),
        List(&["", "draft-simple", "draft-mtp", "ngram-mod", "ngram-simple", "ngram-map-k", "none"]),
    ),
    fld("ngld", "layer draft su GPU", "draft", "", Opt("-ngld"), Free),
    fld("draft_max", "token draft max", "draft", "", Opt("--spec-draft-n-max|--draft-max"), Free),
    fld("draft_min", "token draft min", "draft", "", Opt("--spec-draft-n-min|--draft-min"), Free),
    fld("np", "slot paralleli (-1 auto)", "server", "-1", Opt("-np"), Free),
    fld("threads", "thread CPU (-1 auto)", "server", "-1", Opt("-t"), Free),
    // chat / --embeddings / --rerank (see `pairs`)
    fld("usage", "uso del modello", "server", "chat", Flag::No, List(&["chat", "embedding", "rerank"])),
    fld("temp", "temperature", "sampling", "", Opt("--temp"), Free),
    fld("top_p", "top-p", "sampling", "", Opt("--top-p"), Free),
    fld("top_k", "top-k", "sampling", "", Opt("--top-k"), Free),
    fld("min_p", "min-p", "sampling", "", Opt("--min-p"), Free),
    fld("repeat_penalty", "repeat penalty", "sampling", "", Opt("--repeat-penalty"), Free),
    fld("seed", "seed", "sampling", "", Opt("-s"), Free),
    fld("jinja", "template jinja", "template", "on", Toggle("--jinja", "--no-jinja"), ON_OFF),
    fld("chat_template_file", "file template custom", "template", "", Opt("--chat-template-file"), Free),
    fld("reasoning", "reasoning", "template", "auto", Opt("-rea"), List(&["auto", "on", "off"])),
    fld(
        "reasoning_format",
        "formato reasoning",
        "template",
        "",
        Opt("--reasoning-format"),
        List(&["", "none", "deepseek", "deepseek-legacy"]),
    ),
    fld("reasoning_budget", "budget reasoning (-1 ∞)", "template", "-1", Opt("--reasoning-budget"), Free),
    fld(
        "reasoning_effort",
        "reasoning effort",
        "template",
        "default",
        Opt("--reasoning-effort"),
        List(&["default", "minimal", "low", "medium", "high", "xhigh", "max"]),
    ),
    fld("lora", "adattatore LoRA", "avanzate", "", Opt("--lora"), Choices::Lora),
    fld("lora_scale", "scala LoRA", "avanzate", "", Opt("--lora-scaled"), Free),
    fld(
        "pooling",
        "pooling embedding",
        "avanzate",
        "",
        Opt("--pooling"),
        List(&["", "none", "mean", "cls", "last", "rank"]),
    ),
    fld("batch", "batch logico (-b)", "avanzate", "", Opt("-b"), Free),
    fld("ubatch", "batch fisico (-ub)", "avanzate", "", Opt("-ub"), Free),
    fld(
        "load_mode",
        "caricamento modello",
        "avanzate",
        "auto",
        Opt("-lm"),
        List(&["auto", "none", "mmap", "mlock", "mmap+mlock", "dio"]),
    ),
    fld("kv_offload", "KV cache su GPU", "avanzate", "on", Toggle("--kv-offload", "--no-kv-offload"), ON_OFF),
    fld("cache_reuse", "cache reuse (token, 0 = no)", "avanzate", "", Opt("--cache-reuse"), Free),
    fld(
        "cont_batching",
        "continuous batching",
        "avanzate",
        "on",
        Toggle("--cont-batching", "--no-cont-batching"),
        ON_OFF,
    ),
    fld("metrics", "endpoint /metrics", "avanzate", "off", Toggle("--metrics", ""), ON_OFF),
    fld("slots", "endpoint /slots", "avanzate", "", Toggle("--slots", "--no-slots"), SRV_ON_OFF),
    fld("props", "POST /props", "avanzate", "off", Toggle("--props", ""), ON_OFF),
    fld("extra", "argomenti extra", "extra", "", Flag::No, Free),
];

// -- local models ------------------------------------------------------------------

pub(crate) fn model_dirs() -> Vec<PathBuf> {
    [Some(hub::models_dir()), BASE.parent().map(|p| p.join("models"))].into_iter().flatten().collect()
}

pub(crate) fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in fs::read_dir(dir).into_iter().flatten().flatten() {
        let p = e.path();
        if e.file_type().is_ok_and(|t| t.is_dir()) {
            walk(&p, out);
        } else if p.extension().is_some_and(|x| x == "gguf") && p.is_file() {
            out.push(p);
        }
    }
}

/// Main GGUF files (no mmproj, only the first part of a split model).
pub fn local_models() -> Vec<PathBuf> {
    let mut out = vec![];
    for d in model_dirs() {
        let mut found = vec![];
        walk(&d, &mut found);
        found.sort();
        out.extend(found.into_iter().filter(|p| {
            let name = fname(&ps(p)).to_lowercase();
            !name.contains("mmproj") && RE_SPLIT.captures(&name).is_none_or(|m| &m[1] == "00001")
        }));
    }
    out
}

pub fn model_size(p: &Path) -> u64 {
    let name = ps(p);
    let name = fname(&name);
    if !RE_SPLIT.is_match(name) {
        return fs::metadata(p).map_or(0, |m| m.len());
    }
    let stem = RE_SPLIT.replace(name, "");
    let parts = fs::read_dir(p.parent().unwrap_or(Path::new("."))).into_iter().flatten().flatten().filter(|e| {
        let n = e.file_name().to_string_lossy().into_owned();
        RE_SPLIT.is_match(&n) && RE_SPLIT.replace(&n, "") == stem
    });
    parts.map(|e| e.metadata().map_or(0, |m| m.len())).sum()
}

pub fn mmprojs_near(p: &Path) -> Vec<PathBuf> {
    let dir = fs::read_dir(p.parent().unwrap_or(Path::new("."))).into_iter().flatten().flatten();
    let mut v: Vec<PathBuf> = dir
        .map(|e| e.path())
        .filter(|f| {
            let n = fname(&ps(f)).to_lowercase();
            n.ends_with(".gguf") && n.contains("mmproj")
        })
        .collect();
    v.sort();
    v
}

// -- presets ---------------------------------------------------------------------------

pub fn base_alias(p: &Path) -> String {
    let name = ps(p);
    let name = fname(&name);
    let mut stem = RE_SPLIT.replace(name, "").into_owned();
    if let Some(s) = stem.strip_suffix(".gguf") {
        stem = s.to_string();
    }
    if let Some(q) = hub::quant_of(name)
        && let Some(i) = stem.to_ascii_uppercase().rfind(&q)
        && i > 0
    {
        stem.truncate(i);
    }
    match stem.trim_end_matches(['-', '.', '_']) {
        "" => p.file_stem().map_or(String::new(), |s| s.to_string_lossy().into_owned()),
        s => s.to_string(),
    }
}

pub fn defaults(model: &Path) -> Preset {
    let mut d: Preset = FIELDS.iter().map(|f| (f.key.to_string(), f.default.to_string())).collect();
    d.insert("alias".into(), base_alias(model));
    let near: Vec<String> = mmprojs_near(model).iter().map(|p| ps(p)).collect();
    d.insert("mmproj".into(), hub::best_mmproj(&near).map(|i| near[i].clone()).unwrap_or_default());
    d
}

/// The selected preset of `model` over the defaults.
pub fn get_preset(state: &State, model: &Path) -> Preset {
    get_named(state, model, &preset_name(state, model))
}

/// Preset `name` of `model` over the defaults (an unknown name = just the defaults).
pub fn get_named(state: &State, model: &Path, name: &str) -> Preset {
    let mut p = defaults(model);
    if let Some(saved) = state.presets.get(&ps(model)).and_then(|n| n.get(name)) {
        p.extend(saved.clone());
    }
    p
}

/// Store only what differs from the defaults into the selected preset, then write state.json.
pub fn put_preset(state: &mut State, model: &Path, preset: &Preset) -> Result<(), String> {
    let name = preset_name(state, model);
    state.presets.entry(ps(model)).or_default().insert(name, diff(model, preset));
    save_state(state)
}

/// Store `values` as preset `name` of `model` (only what differs from the defaults), replacing
/// one with that name; which preset is selected doesn't change. The caller writes state.json.
pub fn put_named(state: &mut State, model: &Path, name: &str, values: &Preset) {
    state.presets.entry(ps(model)).or_default().insert(name.into(), diff(model, values));
}

/// `preset` minus the values equal to the model's defaults.
pub fn diff(model: &Path, preset: &Preset) -> Preset {
    let d = defaults(model);
    preset.iter().filter(|(k, v)| d.get(*k) != Some(v)).map(|(k, v)| (k.clone(), v.clone())).collect()
}

/// Saved preset names of `model` ("default" when it has none yet).
pub fn preset_names(state: &State, model: &Path) -> Vec<String> {
    match state.presets.get(&ps(model)) {
        Some(n) if !n.is_empty() => n.keys().cloned().collect(),
        _ => vec![DEFAULT_PRESET.into()],
    }
}

/// Name of the preset in use for `model`.
pub fn preset_name(state: &State, model: &Path) -> String {
    let names = preset_names(state, model);
    state.preset_sel.get(&ps(model)).filter(|s| names.contains(s)).cloned().unwrap_or_else(|| names[0].clone())
}

pub fn select_preset(state: &mut State, model: &Path, name: &str) {
    state.preset_sel.insert(ps(model), name.into());
}

/// New preset `name` with `values` (only what differs from the defaults), made the selected one.
pub fn add_preset(state: &mut State, model: &Path, name: &str, values: Preset) -> Result<(), String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("il nome non può essere vuoto".into());
    }
    let named = state.presets.entry(ps(model)).or_default();
    if named.contains_key(name) {
        return Err(format!("esiste già un preset \"{name}\""));
    }
    named.insert(name.into(), values);
    select_preset(state, model, name);
    Ok(())
}

/// Rename preset `from` of `model`; the multi-model profiles that run it follow the new name.
pub fn rename_preset(state: &mut State, model: &Path, from: &str, to: &str) -> Result<(), String> {
    let values = state.presets.get(&ps(model)).and_then(|n| n.get(from)).cloned().unwrap_or_default();
    let sel = preset_name(state, model);
    add_preset(state, model, to, values)?;
    state.presets.entry(ps(model)).or_default().remove(from);
    select_preset(state, model, if sel == from { to.trim() } else { &sel });
    let key = ps(model);
    for entry in state.routers.values_mut().flat_map(|p| &mut p.models) {
        if entry.model == key && entry.preset == from {
            entry.preset = to.trim().to_string();
        }
    }
    Ok(())
}

/// Names of the multi-model profiles that run `model` with preset `name`.
pub fn profiles_using(state: &State, model: &Path, name: &str) -> Vec<String> {
    let key = ps(model);
    let uses = |p: &crate::router::Profile| p.models.iter().any(|e| e.model == key && e.preset == name);
    state.routers.iter().filter(|(_, p)| uses(p)).map(|(n, _)| n.clone()).collect()
}

/// Remove preset `name`; the last one can't go (it just goes back to the defaults), nor one that a
/// multi-model profile runs (it would silently start with the defaults).
pub fn delete_preset(state: &mut State, model: &Path, name: &str) -> Result<(), String> {
    let users = profiles_using(state, model, name);
    if !users.is_empty() {
        return Err(format!(
            "lo usa il profilo multi-modello {}: cambia lì il preset, poi eliminalo",
            users.join(", ")
        ));
    }
    let Some(named) = state.presets.get_mut(&ps(model)).filter(|n| n.len() > 1) else {
        return Err("è l'unico preset: usa \"ripristina default\" nell'editor".into());
    };
    named.remove(name);
    if state.preset_sel.get(&ps(model)).is_some_and(|s| s == name) {
        state.preset_sel.remove(&ps(model));
    }
    Ok(())
}

/// A server is starting with `model`: remember it as the last one and when it was used.
pub fn mark_used(state: &mut State, model: &Path) {
    state.last_model = Some(ps(model));
    state.last_used.insert(ps(model), stamp());
}

/// The command line for `binary` (flags this build doesn't know are left out, see `args`).
pub fn build_cmd(binary: &Path, model: &Path, p: &Preset, only: Option<&[&str]>) -> Vec<String> {
    let mut cmd = vec![ps(binary), "-m".into(), ps(model)];
    cmd.extend(args(p, only, cached_flags(binary).as_deref()));
    if only.is_none() {
        cmd.extend(shell_split(p.get("extra").map_or("", String::as_str)));
    }
    cmd
}

/// Preset -> flags. Only values that differ from the server defaults become flags; with `flags`
/// (the build's `--help`) unknown ones are dropped and renamed ones respelled, so a preset
/// never breaks another build. `only` restricts to a subset of keys. No "extra" here.
pub fn args(p: &Preset, only: Option<&[&str]>, flags: Option<&Flags>) -> Vec<String> {
    pairs(p, only, flags).into_iter().flat_map(|(f, v)| std::iter::once(f).chain(v)).collect()
}

/// A flag and its value, if it takes one.
pub type Pair = (String, Option<String>);

/// `args` as (flag, value) pairs: the router's preset INI writes them as `flag = value`.
pub fn pairs(p: &Preset, only: Option<&[&str]>, flags: Option<&Flags>) -> Vec<Pair> {
    pairs_in(FIELDS, p, only, flags)
}

/// `pairs` over another parameter table (the router's options).
pub fn pairs_in(fields: &[Field], p: &Preset, only: Option<&[&str]>, flags: Option<&Flags>) -> Vec<Pair> {
    let has = |f: &str| flags.is_none_or(|s| s.contains(f));
    let get = |k: &str| p.get(k).map_or("", String::as_str);
    let flag = |f: &str| (f.to_string(), None);
    let mut out: Vec<Pair> = vec![];
    for f in fields {
        let v = get(f.key);
        let unset = v.is_empty();
        let left_out = only.is_some_and(|o| !o.contains(&f.key));
        let server_default = v == f.default && !ALWAYS.contains(&f.key);
        if unset || left_out || server_default {
            continue;
        }
        let words: Vec<Pair> = match (f.key, &f.flag) {
            ("lora_scale", _) => continue, // goes with "lora"
            ("lora", _) if !matches!(get("lora_scale"), "" | "1") => {
                // ponytail: FNAME:SCALE syntax of current builds; very old ones took two arguments
                if !has("--lora-scaled") {
                    continue;
                }
                vec![("--lora-scaled".into(), Some(format!("{v}:{}", get("lora_scale"))))]
            }
            ("usage", _) => match v {
                "embedding" if has("--embeddings") => vec![flag("--embeddings")],
                "rerank" if has("--rerank") => vec![flag("--rerank")],
                _ => continue,
            },
            (_, Opt(spec)) => match spec.split('|').find(|s| has(s)) {
                Some(s) => vec![(s.into(), Some(v.into()))],
                None if f.key == "load_mode" => match legacy_load(v) {
                    Some(w) if w.iter().all(|x| has(x)) => w.iter().map(|x| flag(x)).collect(),
                    _ => continue,
                },
                None => continue,
            },
            (_, Toggle(on, off)) => {
                let word = if v == "on" { on } else { off };
                if word.is_empty() || !has(word) {
                    continue; // no flag for this side (or not in this build)
                }
                vec![flag(word)]
            }
            (_, Flag::No) => continue,
        };
        out.extend(words);
    }
    // newer builds only speculate with a plain draft model when asked to
    let draft = !get("draft").is_empty() && only.is_none_or(|o| o.contains(&"draft"));
    if draft && get("spec_type").is_empty() && flags.is_some_and(|s| s.contains("--spec-type")) {
        out.push(("--spec-type".into(), Some("draft-simple".into())));
    }
    out
}

/// Free-form arguments ("extra") as pairs: a flag takes the next word as its value unless that
/// is another flag (negative numbers are values).
// ponytail: guesses from the shape of the words, not from the build's --help
pub fn extra_pairs(extra: &str) -> Vec<Pair> {
    let is_flag = |w: &str| w.starts_with('-') && w.len() > 1 && w.parse::<f64>().is_err();
    let mut out: Vec<Pair> = vec![];
    for w in shell_split(extra) {
        if is_flag(&w) {
            out.push((w, None));
        } else if let Some(last) = out.last_mut()
            && last.1.is_none()
        {
            last.1 = Some(w); // the value of the flag just before
        } // else a stray value with no flag before it: llama-server would refuse it anyway
    }
    out
}

/// -lm/--load-mode values as the flags of builds from before it existed.
fn legacy_load(v: &str) -> Option<&'static [&'static str]> {
    Some(match v {
        "auto" | "mmap" => &[],
        "none" => &["--no-mmap"],
        "mlock" => &["--no-mmap", "--mlock"],
        "mmap+mlock" => &["--mlock"],
        "dio" => &["--direct-io"],
        _ => return None,
    })
}

/// Can this build express the field at all? (None = flags unknown: assume yes.)
pub fn supported(f: &Field, flags: Option<&Flags>) -> bool {
    let Some(s) = flags else { return true };
    let legacy: &[&str] = if f.key == "load_mode" { &["--no-mmap", "--mlock"] } else { &[] };
    match f.flag {
        Opt(spec) => spec.split('|').chain(legacy.iter().copied()).any(|x| s.contains(x)),
        Toggle(on, off) => s.contains(on) || s.contains(off),
        Flag::No if f.key == "usage" => s.contains("--embeddings"),
        Flag::No => true,
    }
}

/// The hosts whose GPUs the preset uses (its "rpc" field: names separated by commas or spaces).
pub fn rpc_hosts(p: &Preset) -> Vec<String> {
    let mut hosts: Vec<String> = vec![];
    for name in p.get("rpc").map_or("", String::as_str).split([',', ' ']) {
        let name = name.trim();
        // a name starting with - would reach ssh as an option
        if !name.is_empty() && !name.starts_with('-') && !hosts.iter().any(|h| h == name) {
            hosts.push(name.to_string());
        }
    }
    hosts
}

/// Reject values llama-server would refuse; empty = server default (always fine).
pub fn check(key: &str, v: &str) -> Result<(), String> {
    let int = |x: &str| x.parse::<i64>().is_ok();
    let pair = |o: &str| o.split_once('=').is_some_and(|(a, b)| !a.is_empty() && !b.is_empty());
    let ok = match key {
        _ if v.is_empty() => true,
        "ngl" | "ngld" => v == "auto" || v == "all" || int(v),
        "ts" => v.split(',').all(|x| x.trim().parse::<f64>().is_ok()), // e.g. 3,1
        "fit_target" => v.split(',').all(|x| x.trim().parse::<u64>().is_ok()), // MiB, one per GPU or one for all
        "ot" => v.split(',').all(pair),                                // pattern=buffer,…
        _ if INTS.contains(&key) => int(v),
        _ if FLOATS.contains(&key) => v.parse::<f64>().is_ok(),
        _ => true,
    };
    if ok { Ok(()) } else { Err(format!("valore non valido: {v}")) }
}

// -- flags of a build ------------------------------------------------------------------------

pub type Flags = HashSet<String>;

static RE_FLAG: LazyLock<Regex> = re!(r"(?:^|[\s,])(--?[A-Za-z][A-Za-z0-9-]*)");
static RE_ENV: LazyLock<Regex> = re!(r"\(env: ([A-Z][A-Z0-9_]*)\)");
type FlagCache = HashMap<PathBuf, (Option<SystemTime>, Arc<Flags>)>;
static FLAG_CACHE: LazyLock<Mutex<FlagCache>> = LazyLock::new(Mutex::default);

/// Every flag spelling in a `--help` text and the environment variables its options read
/// ("(env: LLAMA_API_KEY)"), minus the options it says were removed.
pub fn help_flags(help: &str) -> Flags {
    let mut out = Flags::new();
    let mut entry: Vec<String> = vec![]; // spellings of the option being read
    let mut removed = false; // its description says it's gone
    for line in help.lines() {
        if line.starts_with('-') {
            // a new option starts: keep the previous one unless it was removed
            if !removed {
                out.extend(entry.drain(..));
            }
            entry = RE_FLAG.captures_iter(line).map(|m| m[1].to_string()).collect();
            removed = false;
        }
        entry.extend(RE_ENV.captures_iter(line).map(|m| m[1].to_string()));
        removed |= line.contains("has been removed");
    }
    if !removed {
        out.extend(entry);
    }
    out
}

fn mtime(binary: &Path) -> Option<SystemTime> {
    fs::metadata(binary).and_then(|m| m.modified()).ok()
}

/// Flags already read for this binary (never blocks; None = not read yet / unreadable).
pub fn cached_flags(binary: &Path) -> Option<Arc<Flags>> {
    let cache = FLAG_CACHE.lock().ok()?;
    let (t, f) = cache.get(binary)?;
    (*t == mtime(binary)).then(|| f.clone())
}

/// Flags this binary accepts, from its `--help` (cached per path and mtime, so a rebuild is
/// read again). Blocking the first time (~10 ms); None = no usable help, nothing is filtered.
pub fn supported_flags(binary: &Path) -> Option<Arc<Flags>> {
    if let Some(f) = cached_flags(binary) {
        return Some(f);
    }
    let out = capture(Command::new(binary).arg("--help"), Duration::from_secs(20)).ok()?;
    let f = Arc::new(help_flags(&(out.stdout + &out.stderr)));
    if !f.contains("--help") {
        return None;
    }
    FLAG_CACHE.lock().ok()?.insert(binary.into(), (mtime(binary), f.clone()));
    Some(f)
}

// -- GGUF header -------------------------------------------------------------------------------

/// What the GGUF metadata says about the model's shape.
#[derive(Debug, PartialEq)]
pub struct Gguf {
    pub arch: String,
    pub layers: u32,
    pub experts: u32, // 0 = dense
    pub ctx: u32,     // training context (0 = not in the header)
}

/// `general.architecture`, `<arch>.block_count`, `.expert_count` and `.context_length` of a GGUF file
/// (the first part of a split model holds the metadata).
pub fn gguf_info(p: &Path) -> Option<Gguf> {
    gguf_read(&mut io::BufReader::new(fs::File::open(p).ok()?))
}

pub fn gguf_read(r: &mut impl Read) -> Option<Gguf> {
    fn n<const N: usize>(r: &mut impl Read) -> Option<[u8; N]> {
        let mut b = [0; N];
        r.read_exact(&mut b).ok().map(|_| b)
    }
    fn u32_(r: &mut impl Read) -> Option<u32> {
        n::<4>(r).map(u32::from_le_bytes)
    }
    fn u64_(r: &mut impl Read) -> Option<u64> {
        n::<8>(r).map(u64::from_le_bytes)
    }
    fn skip(r: &mut impl Read, len: u64) -> Option<()> {
        (io::copy(&mut r.take(len), &mut io::sink()).ok()? == len).then_some(())
    }
    fn string(r: &mut impl Read) -> Option<String> {
        let len = u64_(r)?;
        if len > 1 << 20 {
            return None;
        }
        let mut b = vec![0; len as usize];
        r.read_exact(&mut b).ok()?;
        Some(String::from_utf8_lossy(&b).into_owned())
    }
    // value types: 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32, 7 bool, 8 string, 9 array, 10 u64, 11 i64, 12 f64
    fn width(t: u32) -> Option<u64> {
        Some(match t {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            10..=12 => 8,
            _ => return None,
        })
    }
    fn value(r: &mut impl Read, t: u32) -> Option<()> {
        match t {
            8 => {
                let len = u64_(r)?;
                skip(r, len)
            }
            9 => {
                let (et, len) = (u32_(r)?, u64_(r)?);
                match width(et) {
                    Some(w) => skip(r, w.checked_mul(len)?),
                    None => (0..len).try_for_each(|_| value(r, et)),
                }
            }
            t => skip(r, width(t)?),
        }
    }
    if &n::<4>(r)? != b"GGUF" || u32_(r)? < 2 {
        return None;
    }
    let (_tensors, kvs) = (u64_(r)?, u64_(r)?);
    let mut g = Gguf { arch: String::new(), layers: 0, experts: 0, ctx: 0 };
    for _ in 0..kvs {
        let (key, t) = (string(r)?, u32_(r)?);
        let int = matches!(t, 4 | 5) && !g.arch.is_empty();
        if key == "general.architecture" && t == 8 {
            g.arch = string(r)?;
        } else if int && key == format!("{}.block_count", g.arch) {
            g.layers = u32_(r)?;
        } else if int && key == format!("{}.expert_count", g.arch) {
            g.experts = u32_(r)?;
        } else if int && key == format!("{}.context_length", g.arch) {
            g.ctx = u32_(r)?;
        } else {
            value(r, t)?;
        }
        if g.layers > 0 && g.experts > 0 && g.ctx > 0 {
            break;
        }
    }
    (!g.arch.is_empty()).then_some(g)
}

/// llama-fit-params' verdict on what fits in memory with the preset's memory settings.
pub struct Fit {
    pub ok: bool,
    pub breakdown: Vec<String>, // per-device table
    pub verdict: Vec<String>,   // "projected …", "will leave …", "cannot …"
    pub tail: Vec<String>,      // last stderr lines when it failed
    pub fitted: String,         // adapted arguments, e.g. "-c 32768 -ngl 99"
}

/// Ask llama-fit-params (same build, same memory settings) what fits. Blocking: run it in a thread.
pub fn fit_params(inst: &Install, model: &Path, preset: &Preset) -> Result<Fit, String> {
    let fp = installs::binary(inst, "llama-fit-params");
    if !fp.is_file() {
        return Err("questa installazione non ha llama-fit-params".into());
    }
    supported_flags(&fp); // fit-params has its own flag set: build_cmd filters by it
    let mut cmd = build_cmd(&fp, model, preset, Some(MEMORY_KEYS));
    cmd.extend(["-lv".into(), "4".into()]);
    let out = capture(Command::new(&cmd[0]).args(&cmd[1..]), Duration::from_secs(180))?;
    let lines = |pred: &dyn Fn(&str) -> Option<String>| out.stderr.lines().filter_map(pred).collect::<Vec<_>>();
    let tail: Vec<String> = out.stderr.lines().map(String::from).collect();
    Ok(Fit {
        ok: out.ok,
        breakdown: lines(&|l| l.split_once("common_memory_breakdown_print: ").map(|x| x.1.to_string())),
        verdict: lines(&|l| {
            let hit = l.contains("common_params_fit_impl:")
                && (l.contains("projected") || l.contains("will leave") || l.contains("cannot"));
            hit.then(|| l.split_once(": ").map_or(l, |x| x.1).to_string())
        }),
        tail: if out.ok { vec![] } else { tail[tail.len().saturating_sub(5)..].to_vec() },
        fitted: out.stdout.trim().to_string(),
    })
}

/// Copy llama-fit-params' `-c N -ngl N [-ts a,b] [-ot "pattern=buffer,…"]` into the preset
/// (for MoE models -ot is where it puts the experts that don't fit); false = nothing to apply.
pub fn apply_fit(preset: &mut Preset, fitted: &str) -> bool {
    let words = shell_split(fitted);
    let mut any = false;
    // each known flag with the word after it: an unknown flag in between can't shift the pairs
    for pair in words.windows(2) {
        let key = match pair[0].as_str() {
            "-c" => "ctx",
            "-ngl" => "ngl",
            "-ts" => "ts",
            "-ot" => "ot",
            _ => continue,
        };
        let value = &pair[1];
        preset.insert(key.into(), if key == "ngl" { ngl_word(value) } else { value.clone() });
        any = true;
    }
    any
}

/// llama.cpp's -ngl encoding (common/arg.cpp): -1 = auto, <= -2 = all, else a layer count.
/// llama-fit-params prints -1 when it left ngl alone (everything fits) and -2 for "all".
pub fn ngl_word(n: &str) -> String {
    match n.parse::<i64>() {
        Ok(-1) => "auto".into(),
        Ok(i) if i < -1 => "all".into(),
        _ => n.to_string(),
    }
}

/// First port from `port` on that can be bound on `host` (None = can't tell, e.g. bad host).
pub fn free_port(host: &str, port: u16) -> Option<u16> {
    for p in port..=port.saturating_add(100) {
        match std::net::TcpListener::bind((host, p)) {
            Ok(_) => return Some(p),
            Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => continue,
            Err(_) => return None,
        }
    }
    None
}
