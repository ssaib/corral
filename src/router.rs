//! Multi-model: llama-server's router mode. A profile lists models, each with one of its Corral
//! presets (edited in the same editor as a single server), behind one endpoint. Corral writes the
//! router's preset INI from those presets and starts `llama-server --models-preset`, which runs
//! every model as a child server, loaded at startup or on its first request; clients pick a model
//! with the request's "model" field (the entry's id).
//! Pure logic (blocking HTTP: run it in a thread); the screens are in `tui::router`.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::installs::{self, Install};
use crate::preset::{self, Choices, Field, Flag, Flags, ON_OFF, fld};
use crate::server;
use crate::ui::*;

/// One model of a profile.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    /// what clients send as "model" (and the INI section)
    pub id: String,
    /// GGUF path
    pub model: String,
    /// one of the model's named presets ("" = the one selected for it)
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub preset: String,
    /// loaded when the router starts instead of on the first request
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub startup: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Profile {
    #[serde(default)]
    pub models: Vec<Entry>,
    /// router options (OPTS keys) that differ from the defaults
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub opts: Preset,
}

/// llama-server's built-in tools that only read.
pub const READ_TOOLS: &str = "read_file,file_glob_search,grep_search,get_info";

use Choices::{Free, List};
use Flag::{Opt, Toggle};

/// Options of the router process (llama-server hands its command line on to every model too).
pub static OPTS: &[Field] = &[
    fld("host", "host", "router", "127.0.0.1", Opt("--host"), Free),
    fld("port", "porta", "router", "8080", Opt("--port"), Free),
    fld("api_key", "API key", "router", "", Opt("--api-key"), Free),
    fld("models_max", "modelli caricati insieme (0 = nessun limite)", "router", "4", Opt("--models-max"), Free),
    fld(
        "margin",
        "MiB da tenere liberi per GPU (scarica i meno usati)",
        "router",
        "",
        Opt("--models-memory-margin"),
        Free,
    ),
    fld(
        "autoload",
        "carica un modello alla prima richiesta",
        "router",
        "on",
        Toggle("--models-autoload", "--no-models-autoload"),
        ON_OFF,
    ),
    fld("sleep", "sospendi un modello inattivo dopo N secondi", "router", "", Opt("--sleep-idle-seconds"), Free),
    fld("tools", "tool integrati della Web UI", "agent", "", Opt("--tools"), List(&["", READ_TOOLS, "all"])),
    fld("tools_runtime", "dove girano i tool (es. docker:ubuntu:24.04)", "agent", "", Opt("--tools-runtime"), Free),
    fld("mcp", "server MCP (file JSON in formato Cursor)", "agent", "", Opt("--mcp-servers-config"), Free),
    fld("mcp_proxy", "proxy MCP della Web UI", "agent", "off", Toggle("--ui-mcp-proxy", ""), ON_OFF),
    fld("extra", "argomenti extra del router", "extra", "", Flag::No, Free),
];

/// The profile's router options over the defaults.
pub fn opts(p: &Profile) -> Preset {
    let mut o: Preset = OPTS.iter().map(|f| (f.key.to_string(), f.default.to_string())).collect();
    o.extend(p.opts.clone());
    o
}

/// Only what differs from the defaults (what a profile stores).
pub fn opts_diff(o: &Preset) -> Preset {
    OPTS.iter().filter_map(|f| o.get(f.key).filter(|v| *v != f.default).map(|v| (f.key.into(), v.clone()))).collect()
}

/// A name clients can send and an INI section can hold: letters, digits, `.` `_` `-`.
pub fn id_ok(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && !id.starts_with(['.', '-'])
        && id.chars().all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
}

/// `base` turned into a valid id that isn't in `taken`.
pub fn new_id(base: &str, taken: &[&str]) -> String {
    let s: String =
        base.chars().map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '-' }).collect();
    let mut s: String = s.trim_matches(['.', '-']).chars().take(48).collect();
    if s.is_empty() {
        s = "modello".into();
    }
    let mut id = s.clone();
    for n in 2.. {
        if !taken.contains(&id.as_str()) {
            break;
        }
        id = format!("{s}-{n}");
    }
    id
}

/// The preset an entry runs with.
pub fn entry_preset(state: &State, e: &Entry) -> Preset {
    let m = Path::new(&e.model);
    if e.preset.is_empty() { preset::get_preset(state, m) } else { preset::get_named(state, m, &e.preset) }
}

/// Keys the router sets itself for each model (it would drop them anyway).
const OWNED: [&str; 4] = ["host", "port", "alias", "api-key"];

/// The router's preset INI: a section per entry (its id) with the model and its preset's flags as
/// `flag = value` (`= true` for a bare flag). The format has no quoting: ; # and newlines can't
/// appear in a value.
pub fn ini(state: &State, prof: &Profile, flags: Option<&Flags>) -> Result<String, String> {
    if prof.models.is_empty() {
        return Err("il profilo non ha modelli: aggiungine almeno uno".into());
    }
    let mut seen = HashSet::new();
    let mut out = String::from("; scritto da Corral a ogni avvio del profilo: le modifiche fatte qui si perdono\n");
    out += "version = 1\n";
    for e in &prof.models {
        if !id_ok(&e.id) {
            return Err(format!("id «{}» non valido: solo lettere, cifre, . _ -", e.id));
        }
        if !seen.insert(e.id.as_str()) {
            return Err(format!("l'id «{}» è usato due volte", e.id));
        }
        if !Path::new(&e.model).is_file() {
            return Err(format!("{}: {} non esiste più", e.id, e.model));
        }
        let p = entry_preset(state, e);
        let mut pairs = preset::pairs(&p, None, flags);
        pairs.extend(preset::extra_pairs(p.get("extra").map_or("", String::as_str)));
        let mut kv = vec![("model".to_string(), e.model.clone())];
        for (flag, v) in pairs {
            let key = flag.trim_start_matches('-').to_string();
            if !OWNED.contains(&key.as_str()) {
                kv.push((key, v.unwrap_or_else(|| "true".into())));
            }
        }
        if e.startup {
            kv.push(("load-on-startup".into(), "true".into()));
        }
        out += &format!("\n[{}]\n", e.id);
        for (k, v) in kv {
            let key_ok = k.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
                && k.chars().all(|c| c.is_ascii_alphanumeric() || "_.-".contains(c));
            if !key_ok || v.contains([';', '#', '\n', '\r']) {
                return Err(format!(
                    "{}: «{k} {v}» non si può scrivere nel preset del router (niente ; # a capo)",
                    e.id
                ));
            }
            out += &format!("{k} = {v}\n");
        }
    }
    Ok(out)
}

pub fn ini_path(name: &str) -> PathBuf {
    BASE.join("routers").join(format!("{}.ini", installs::sanitize(name)))
}

/// Write the profile's INI and return the router's command line.
pub fn prepare(state: &State, name: &str, prof: &Profile, inst: &Install) -> Result<Vec<String>, String> {
    let srv = installs::server(inst);
    let flags = preset::supported_flags(&srv);
    if flags.as_ref().is_some_and(|f| !f.contains("--models-preset")) {
        return Err(format!("{}: questa build non ha il router multi-modello, aggiorna llama.cpp", inst.name));
    }
    let text = ini(state, prof, flags.as_deref())?;
    let path = ini_path(name);
    let err = |e: std::io::Error| format!("{}: {e}", path.display());
    fs::create_dir_all(path.parent().unwrap_or(&BASE)).map_err(err)?;
    fs::write(&path, text).map_err(err)?;
    Ok(command(&srv, &path, prof, flags.as_deref()))
}

/// `llama-server --models-preset <ini>` with the router options.
pub fn command(srv: &Path, ini: &Path, prof: &Profile, flags: Option<&Flags>) -> Vec<String> {
    let o = opts(prof);
    let mut cmd = vec![ps(srv), "--models-preset".into(), ps(ini)];
    cmd.extend(preset::pairs_in(OPTS, &o, None, flags).into_iter().flat_map(|(f, v)| [Some(f), v]).flatten());
    cmd.extend(shell_split(&o["extra"]));
    cmd
}

/// Environment of the router: an empty llama.cpp cache, so only the profile's models are listed
/// (by default the router also offers every model downloaded with `-hf`).
pub fn env() -> Vec<(String, String)> {
    let dir = BASE.join("cache").join("router-cache");
    let _ = fs::create_dir_all(&dir);
    vec![("LLAMA_CACHE".into(), ps(&dir))]
}

// -- the running router -------------------------------------------------------------------------

/// A model as the router reports it (GET /models), plus its /metrics when loaded.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Status {
    pub id: String,
    /// loaded, loading, unloaded, sleeping, downloading
    pub state: String,
    pub failed: bool,
    pub exit: Option<i64>,
    pub prom: BTreeMap<String, f64>,
}

pub fn parse_models(v: &Value) -> Vec<Status> {
    let list = v["data"].as_array().into_iter().flatten();
    list.map(|m| {
        let s = &m["status"];
        Status {
            id: m["id"].as_str().unwrap_or_default().into(),
            state: s["value"].as_str().unwrap_or("?").into(),
            failed: s["failed"].as_bool().unwrap_or(false),
            exit: s["exit_code"].as_i64(),
            prom: BTreeMap::new(),
        }
    })
    .collect()
}

pub fn load(base: &str, key: &str, id: &str) -> Result<(), String> {
    server::post_json(base, key, "/models/load", json!({ "model": id }), Duration::from_secs(30)).map(|_| ())
}

pub fn unload(base: &str, key: &str, id: &str) -> Result<(), String> {
    server::post_json(base, key, "/models/unload", json!({ "model": id }), Duration::from_secs(30)).map(|_| ())
}

/// "1 modello", "3 modelli".
pub fn count(n: usize) -> String {
    if n == 1 { "1 modello".into() } else { format!("{n} modelli") }
}

/// Disk size of the profile's models (what loading them all needs, roughly).
pub fn total_size(prof: &Profile) -> u64 {
    prof.models.iter().map(|e| preset::model_size(Path::new(&e.model))).sum()
}
