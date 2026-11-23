//! Agent, multi-agent and flow recipes on top of the router: a recipe gives each role a model and a
//! preset tuned for it (named after the role) and makes the multi-model profile that serves them on
//! one endpoint. Then: client configs to copy (OpenCode, Qwen Code, Python, environment variables)
//! and quick tests of a running profile (tool calling, embeddings, rerank).
//! Pure logic (the tests block on HTTP: run them in a thread); the screens are in `tui::agent`.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::preset;
use crate::router::{self, Entry, Profile};
use crate::server::post_json;
use crate::ui::*;

pub struct Role {
    /// the model's id on the router and the name of its preset
    pub id: &'static str,
    pub what: &'static str,
    /// words in a file name that make a model a good pick, most telling first
    pub hint: &'static [&'static str],
    /// preset values of the role, over the model's preset
    pub set: &'static [(&'static str, &'static str)],
}

pub struct Recipe {
    /// name of the profile it makes
    pub key: &'static str,
    pub name: &'static str,
    pub what: &'static str,
    pub desc: &'static str,
    pub roles: &'static [Role],
    /// router options
    pub opts: &'static [(&'static str, &'static str)],
}

const CHAT: &[&str] = &["qwen3", "gpt-oss", "glm", "gemma", "mistral", "llama", "deepseek", "phi"];
const CODE: &[&str] = &["coder", "devstral", "codestral", "qwen3", "gpt-oss", "glm"];

pub static RECIPES: &[Recipe] = &[
    Recipe {
        key: "coding",
        name: "agent di coding",
        what: "tool calling · contesto 64k · 1 slot",
        desc: "Un modello per un agent di programmazione (OpenCode, Qwen Code o il tuo): template jinja per le \
               chiamate ai tool, contesto da 64k token tutto per una sessione, cache dei prompt riusata tra un \
               turno e l'altro.",
        roles: &[Role {
            id: "coder",
            what: "scrive codice e chiama i tool",
            hint: CODE,
            set: &[("jinja", "on"), ("ctx", "65536"), ("np", "1"), ("cache_reuse", "256")],
        }],
        opts: &[],
    },
    Recipe {
        key: "multi-agent",
        name: "multi-agent",
        what: "orchestratore + worker in parallelo",
        desc: "Un orchestratore (il modello più capace, reasoning attivo) pianifica e delega; un worker (un \
               modello veloce) con 4 slot serve più sub-agent alla volta. Stesso endpoint: ogni agent sceglie il \
               suo modello col campo model della richiesta.",
        roles: &[
            Role {
                id: "planner",
                what: "pianifica e delega: il modello più capace",
                hint: CHAT,
                set: &[("jinja", "on"), ("reasoning", "on"), ("np", "1"), ("ctx", "32768")],
            },
            Role {
                id: "worker",
                what: "esegue i passi: veloce, 4 richieste insieme",
                hint: CODE,
                set: &[("jinja", "on"), ("reasoning", "off"), ("np", "4"), ("ctx", "65536"), ("cache_reuse", "256")],
            },
        ],
        opts: &[],
    },
    Recipe {
        key: "rag",
        name: "RAG sui documenti",
        what: "chat + embedding + reranker",
        desc: "Ricerca sui tuoi documenti: il modello di embedding trasforma i testi in vettori \
               (/v1/embeddings), il reranker riordina i passaggi trovati (/v1/rerank), il modello di chat \
               risponde con quei passaggi nel contesto.",
        roles: &[
            Role {
                id: "chat",
                what: "risponde con i passaggi trovati",
                hint: CHAT,
                set: &[("jinja", "on"), ("ctx", "32768")],
            },
            Role {
                id: "embed",
                what: "testi → vettori (es. Qwen3-Embedding, bge-m3, nomic-embed)",
                hint: &["embed", "bge", "e5-", "gte", "nomic", "minilm"],
                set: &[("usage", "embedding"), ("batch", "4096"), ("ubatch", "4096")],
            },
            Role {
                id: "rerank",
                what: "punteggio dei passaggi (es. Qwen3-Reranker, bge-reranker)",
                hint: &["rerank"],
                set: &[("usage", "rerank"), ("batch", "4096"), ("ubatch", "4096")],
            },
        ],
        opts: &[],
    },
    Recipe {
        key: "vision",
        name: "vision",
        what: "immagini + testo",
        desc: "Un modello multimodale col suo proiettore (mmproj): descrive immagini, legge documenti e schermate. \
               Corral usa il file mmproj che trova accanto al modello.",
        roles: &[Role {
            id: "vision",
            what: "un modello con il file mmproj nella sua cartella",
            hint: &[], // see `suggest`: the ones with an mmproj
            set: &[("jinja", "on"), ("ctx", "16384")],
        }],
        opts: &[],
    },
    Recipe {
        key: "webui-agent",
        name: "agent nella Web UI",
        what: "tool integrati di llama-server + MCP",
        desc: "La Web UI di llama.cpp fa da agent: usa i tool integrati del server (lettura file e ricerca; con all \
               anche shell e scrittura) e i server MCP del file indicato nelle opzioni del router. Solo su \
               127.0.0.1: i tool agiscono sul tuo computer.",
        roles: &[Role { id: "assistant", what: "usa i tool", hint: CODE, set: &[("jinja", "on"), ("ctx", "32768")] }],
        opts: &[("tools", router::READ_TOOLS), ("mcp_proxy", "on"), ("host", "127.0.0.1")],
    },
];

/// `models` ordered for `role` as (index, good pick): matching names first, by hint rank.
pub fn suggest(role: &Role, models: &[PathBuf]) -> Vec<(usize, bool)> {
    let rank = |p: &PathBuf| {
        let n = fname(&ps(p)).to_lowercase();
        match role.id {
            "vision" => (!preset::mmprojs_near(p).is_empty()).then_some(0),
            "embed" if n.contains("rerank") => None, // bge-reranker is no embedder
            _ => role.hint.iter().position(|h| n.contains(h)),
        }
    };
    let mut v: Vec<(usize, Option<usize>)> = models.iter().enumerate().map(|(i, p)| (i, rank(p))).collect();
    v.sort_by_key(|&(i, r)| (r.is_none(), r, i));
    v.into_iter().map(|(i, r)| (i, r.is_some())).collect()
}

/// Give each role its model and a preset named after the role (its earlier values, else the
/// model's current preset, with the role's values on top), then make the recipe's profile
/// (not saved: the caller writes state.json). Returns the profile name.
pub fn apply(state: &mut State, r: &Recipe, picks: &[PathBuf]) -> String {
    let mut entries = vec![];
    for (role, model) in r.roles.iter().zip(picks) {
        // pin what the model runs with on its own: a new name could sort first and take over (and the
        // virtual "default" of a model with no saved preset would vanish)
        let current = preset::preset_name(state, model);
        state.presets.entry(ps(model)).or_default().entry(current.clone()).or_default();
        preset::select_preset(state, model, &current);
        let had = preset::preset_names(state, model).iter().any(|n| n == role.id);
        let mut p = if had { preset::get_named(state, model, role.id) } else { preset::get_preset(state, model) };
        for (k, v) in role.set {
            p.insert(k.to_string(), v.to_string());
        }
        let values = preset::diff(model, &p);
        state.presets.entry(ps(model)).or_default().insert(role.id.into(), values);
        entries.push(Entry { id: role.id.into(), model: ps(model), preset: role.id.into(), startup: true });
    }
    let prof = state.routers.entry(r.key.into()).or_default();
    prof.models = entries;
    let mut o = router::opts(prof);
    for (k, v) in r.opts {
        o.insert(k.to_string(), v.to_string());
    }
    if o["models_max"].parse::<usize>().is_ok_and(|m| m != 0 && m < r.roles.len()) {
        o.insert("models_max".into(), r.roles.len().to_string());
    }
    prof.opts = router::opts_diff(&o);
    r.key.into()
}

// -- client configs ------------------------------------------------------------------------------

/// A model as a client sees it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Served {
    pub id: String,
    pub file: String,
    pub ctx: u64,
    pub vision: bool,
    /// chat, embedding, rerank
    pub usage: String,
}

/// Where a profile answers and what it serves.
pub struct Target {
    pub name: String,
    /// http://host:port
    pub base: String,
    pub key: String,
    pub models: Vec<Served>,
}

impl Target {
    fn chat(&self) -> Vec<&Served> {
        self.models.iter().filter(|m| m.usage == "chat").collect()
    }
    fn usage(&self, u: &str) -> Option<&Served> {
        self.models.iter().find(|m| m.usage == u)
    }
    fn key_or(&self, none: &str) -> String {
        if self.key.is_empty() { none.into() } else { self.key.clone() }
    }
}

/// `base`: where the profile's router runs now (None = its host and port).
pub fn target(state: &State, name: &str, prof: &Profile, base: Option<&str>) -> Target {
    let o = router::opts(prof);
    let host = if o["host"] == "0.0.0.0" { "127.0.0.1" } else { o["host"].as_str() };
    let base = base.map_or_else(|| format!("http://{host}:{}", o["port"]), String::from);
    let models = prof
        .models
        .iter()
        .map(|e| {
            let p = router::entry_preset(state, e);
            let from_file = || preset::gguf_info(Path::new(&e.model)).map(|g| u64::from(g.ctx)).filter(|&c| c > 0);
            let ctx = p["ctx"].parse::<u64>().ok().filter(|&c| c > 0).or_else(from_file).unwrap_or(32768);
            Served {
                id: e.id.clone(),
                file: fname(&e.model).to_string(),
                ctx,
                vision: !p["mmproj"].is_empty(),
                usage: p["usage"].clone(),
            }
        })
        .collect();
    Target { name: name.into(), base, key: o["api_key"].clone(), models }
}

pub const FORMATS: [&str; 4] = ["OpenCode", "Qwen Code", "Python", "variabili"];
const KEY_VAR: &str = "CORRAL_API_KEY";

/// Export format `i` of `t`: (file name, where it goes, text).
pub fn render(i: usize, t: &Target) -> (String, String, String) {
    match i {
        0 => ("opencode.json".into(), opencode_where(), opencode(t)),
        1 => ("settings.json".into(), "~/.qwen/settings.json (unisci le chiavi se il file c'è già)".into(), qwen(t)),
        2 => (
            format!(
                "corral_{}.py",
                t.name.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect::<String>()
            ),
            "uno script da adattare: pip install openai …".into(),
            python(t),
        ),
        _ => {
            let file = if cfg!(windows) { "corral.ps1" } else { "corral.env" };
            let how = if cfg!(windows) { "in PowerShell: . .\\corral.ps1" } else { "nella shell: source corral.env" };
            (file.into(), how.into(), env_vars(t))
        }
    }
}

fn opencode_where() -> String {
    "~/.config/opencode/opencode.json (globale) o opencode.json nella cartella del progetto".into()
}

fn opencode(t: &Target) -> String {
    let mut models = Map::new();
    for m in t.chat() {
        let input = if m.vision { json!(["text", "image"]) } else { json!(["text"]) };
        models.insert(
            m.id.clone(),
            json!({
                "name": format!("{} · {}", m.id, m.file),
                "limit": { "context": m.ctx, "output": (m.ctx / 4).clamp(1024, 32768) },
                "capabilities": { "tools": true, "input": input, "output": ["text"] },
            }),
        );
    }
    let mut provider = json!({
        "name": format!("Corral · {}", t.name),
        "package": "@opencode/ai/providers/openai-compatible",
        "settings": { "baseURL": format!("{}/v1", t.base) },
        "models": models,
    });
    if !t.key.is_empty() {
        provider["env"] = json!([KEY_VAR]);
    }
    let mut v = json!({ "$schema": "https://opencode.ai/config.json", "providers": { "corral": provider } });
    if let Some(m) = t.chat().first() {
        v["model"] = format!("corral/{}", m.id).into();
    }
    serde_json::to_string_pretty(&v).unwrap_or_default()
}

fn qwen(t: &Target) -> String {
    let models: Vec<Value> = t
        .chat()
        .iter()
        .map(|m| {
            json!({
                "id": m.id,
                "name": format!("{} · Corral {}", m.id, t.name),
                "baseUrl": format!("{}/v1", t.base),
                "envKey": KEY_VAR,
                "generationConfig": { "contextWindowSize": m.ctx },
            })
        })
        .collect();
    let mut v = json!({
        "modelProviders": { "openai": models },
        "security": { "auth": { "selectedType": "openai" } },
        // a local server takes any key, but the client wants one
        "env": { KEY_VAR: t.key_or("not-needed") },
    });
    if let Some(m) = t.chat().first() {
        v["model"] = json!({ "name": m.id });
    }
    serde_json::to_string_pretty(&v).unwrap_or_default()
}

// The Python example, one section per kind of model in the profile, written out as it will be
// saved; `python` fills in the $NAMES.
const PY_HEAD: &str = r##"# Corral · profilo "$NAME": modelli {$ROLES}
# pip install openai langchain langchain-openai crewai
from openai import OpenAI

BASE_URL = $BASE_URL
API_KEY = $API_KEY  # il server locale accetta qualunque chiave se non ne ha impostata una

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
"##;

const PY_TOOLS: &str = r##"
# 1) tool calling con l'SDK OpenAI
tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Meteo attuale di una città",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
r = client.chat.completions.create(model="$MODEL", tools=tools,
    messages=[{"role": "user", "content": "Che tempo fa a Roma?"}])
print(r.choices[0].message.tool_calls)

# 2) agent LangChain: create_agent gira il ciclo modello → tool → modello
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

def get_weather(city: str) -> str:
    """Meteo attuale di una città."""
    return f"A {city} ci sono 21 gradi e sole."

agent = create_agent(ChatOpenAI(base_url=BASE_URL, api_key=API_KEY, model="$MODEL"), tools=[get_weather])
out = agent.invoke({"messages": [{"role": "user", "content": "Che tempo fa a Roma?"}]})
print(out["messages"][-1].content)
"##;

const PY_CREW: &str = r##"
# 3) multi-agent con CrewAI: un agent per ruolo, ognuno col suo modello sul router
from crewai import LLM, Agent, Crew, Task

def llm(model):
    return LLM(model=f"openai/{model}", base_url=BASE_URL, api_key=API_KEY)

planner = Agent(role="$PLANNER", goal="Scomporre il compito in passi chiari", backstory="Pianifica e delega.", llm=llm("$PLANNER"))
worker = Agent(role="$WORKER", goal="Eseguire i passi del piano", backstory="Esegue con precisione.", llm=llm("$WORKER"))
plan = Task(description="Pianifica: {topic}", expected_output="Una lista di passi", agent=planner)
work = Task(description="Esegui il piano", expected_output="Il risultato", agent=worker, context=[plan])
print(Crew(agents=[planner, worker], tasks=[plan, work]).kickoff(inputs={"topic": "un README per il progetto"}))
"##;

const PY_EMBED: &str = r##"
# embedding
emb = client.embeddings.create(model="$MODEL", input=["primo testo", "secondo testo"])
print(len(emb.data[0].embedding), "dimensioni")
"##;

const PY_RERANK: &str = r##"
# rerank (endpoint di llama-server, non nell'SDK OpenAI)
import json, urllib.request
req = urllib.request.Request(BASE_URL + "/rerank", method="POST",
    headers={"Content-Type": "application/json", "Authorization": "Bearer " + API_KEY},
    data=json.dumps({"model": "$MODEL", "query": "capitale d'Italia",
                     "documents": ["Roma è la capitale d'Italia.", "Il gatto dorme."]}).encode())
print(json.load(urllib.request.urlopen(req))["results"])
"##;

/// `s` as a Python string literal (JSON's escapes are valid Python).
fn py_str(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_default()
}

fn python(t: &Target) -> String {
    let chat = t.chat();
    // model ids are plain (router::id_ok): they go inside the quotes as they are
    let roles: Vec<String> = t.models.iter().map(|m| format!("\"{}\": \"{}\"", m.id, m.usage)).collect();
    let mut s = PY_HEAD
        .replace("$NAME", &t.name)
        .replace("$ROLES", &roles.join(", "))
        .replace("$BASE_URL", &py_str(&format!("{}/v1", t.base)))
        .replace("$API_KEY", &py_str(&t.key_or("not-needed")));
    if let Some(first) = chat.first() {
        s += &PY_TOOLS.replace("$MODEL", &first.id);
    }
    if let [planner, worker, ..] = chat[..] {
        s += &PY_CREW.replace("$PLANNER", &planner.id).replace("$WORKER", &worker.id);
    }
    if let Some(e) = t.usage("embedding") {
        s += &PY_EMBED.replace("$MODEL", &e.id);
    }
    if let Some(r) = t.usage("rerank") {
        s += &PY_RERANK.replace("$MODEL", &r.id);
    }
    s
}

fn env_vars(t: &Target) -> String {
    let model = t.chat().first().map_or(String::new(), |m| m.id.clone());
    let list: Vec<String> = t.models.iter().map(|m| format!("{} ({}, ctx {})", m.id, m.usage, m.ctx)).collect();
    let vars = [
        ("OPENAI_BASE_URL", format!("{}/v1", t.base)),
        ("OPENAI_API_KEY", t.key_or("not-needed")),
        ("OPENAI_MODEL", model),
    ];
    let mut s = format!("# Corral · profilo \"{}\" · modelli: {}\n", t.name, list.join(", "));
    for (k, v) in vars {
        let line = if cfg!(windows) { format!("$env:{k} = \"{v}\"\n") } else { format!("export {k}=\"{v}\"\n") };
        s += &line;
    }
    s
}

// -- tests of a running profile --------------------------------------------------------------------

// the first request to a model not loaded yet waits for it to load
const LONG: Duration = Duration::from_secs(600);

/// One model of a running router: tool calling for chat models, else its endpoint.
pub fn test(base: &str, key: &str, m: &Served) -> Result<String, String> {
    match m.usage.as_str() {
        "embedding" => test_embed(base, key, &m.id),
        "rerank" => test_rerank(base, key, &m.id),
        _ => test_tools(base, key, &m.id),
    }
}

/// A tool call, then the answer once the tool's result is back: the two turns templates get wrong.
fn test_tools(base: &str, key: &str, id: &str) -> Result<String, String> {
    let tools = json!([{ "type": "function", "function": {
        "name": "get_weather",
        "description": "Meteo attuale di una città",
        "parameters": { "type": "object", "properties": { "city": { "type": "string" } }, "required": ["city"] },
    }}]);
    let ask = |messages: &[Value]| {
        let body = json!({ "model": id, "messages": messages, "tools": tools, "max_tokens": 1024 });
        post_json(base, key, "/v1/chat/completions", body, LONG)
    };
    let mut messages =
        vec![json!({ "role": "user", "content": "Che tempo fa a Roma adesso? Usa lo strumento get_weather." })];
    let msg = ask(&messages)?["choices"][0]["message"].clone();
    let Some(call) = msg["tool_calls"].get(0).cloned() else {
        let said = clip(msg["content"].as_str().unwrap_or("").trim(), 50);
        return Err(format!("nessuna tool call (ha risposto «{said}»): serve jinja on e un modello che usi i tool"));
    };
    let name = call["function"]["name"].as_str().unwrap_or("");
    if name != "get_weather" {
        return Err(format!("ha chiamato «{name}» invece di get_weather"));
    }
    let raw = call["function"]["arguments"].as_str().unwrap_or("");
    let args: Value = serde_json::from_str(raw).map_err(|_| format!("argomenti non JSON: {}", clip(raw, 60)))?;
    messages.push(msg);
    messages.push(json!({ "role": "tool", "tool_call_id": call["id"], "content": "{\"temperatura\": 21, \"cielo\": \"sereno\"}" }));
    let answer = ask(&messages)?["choices"][0]["message"]["content"].as_str().unwrap_or("").trim().replace('\n', " ");
    if answer.is_empty() {
        return Err("tool call ok, ma nessuna risposta dopo il risultato del tool".into());
    }
    Ok(format!("get_weather({}) e poi «{}»", args["city"].as_str().unwrap_or("?"), clip(&answer, 60)))
}

fn test_embed(base: &str, key: &str, id: &str) -> Result<String, String> {
    let body = json!({ "model": id, "input": ["Corral avvia llama.cpp", "un gatto dorme"] });
    let r = post_json(base, key, "/v1/embeddings", body, LONG)?;
    match r["data"][0]["embedding"].as_array().map_or(0, Vec::len) {
        0 => Err("nessun vettore nella risposta (il modello è di embedding? uso = embedding)".into()),
        n => Ok(format!("2 vettori da {n} dimensioni")),
    }
}

fn test_rerank(base: &str, key: &str, id: &str) -> Result<String, String> {
    let docs = ["Roma è la capitale d'Italia.", "Il gatto dorme sul divano."];
    let body = json!({ "model": id, "query": "Qual è la capitale d'Italia?", "documents": docs });
    let r = post_json(base, key, "/v1/rerank", body, LONG)?;
    let score = |v: &Value| v["relevance_score"].as_f64().unwrap_or(f64::MIN);
    let results = r["results"].as_array().cloned().unwrap_or_default();
    let best = results.iter().max_by(|a, b| score(a).total_cmp(&score(b))).ok_or("nessun risultato")?;
    match best["index"].as_u64() {
        Some(0) => Ok(format!("il documento giusto è in cima (punteggio {:.2})", score(best))),
        _ => Err("il documento giusto non è in cima: è un modello reranker? (uso = rerank)".into()),
    }
}
