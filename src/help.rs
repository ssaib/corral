//! What each parameter does, for the preset editor and the router options: one line always shown
//! under the table, and a longer explanation with a practical tip behind `?`.

use ratatui::text::Line;

use crate::preset::{Choices, Field, Flag};
use crate::ui::*;

/// (key, one line, the longer explanation). Keys of `preset::FIELDS` and `router::OPTS`; the ones
/// both have (host, port, api_key, extra) mean the same thing in both.
static HELP: &[(&str, &str, &str)] = &[
    // -- base
    (
        "alias",
        "nome del modello nelle API: quello che i client vedono in /v1/models e mandano in «model»",
        "Di default è il nome del file senza la quantizzazione. Cambialo se un client (OpenCode, Qwen Code, un \
         SDK) si aspetta un nome preciso, o per distinguere due preset dello stesso modello.",
    ),
    (
        "mmproj",
        "proiettore multimodale: il file che fa vedere al modello le immagini (e sentire l'audio)",
        "I modelli vision (Qwen2.5-VL, Gemma 3, SmolVLM…) hanno un file mmproj-*.gguf, di solito nella stessa \
         cartella. Senza, il modello lavora solo a testo. Occupa memoria in più, che la stima di \
         llama-fit-params non conta.",
    ),
    (
        "host",
        "indirizzo su cui il server ascolta: 127.0.0.1 solo da questo computer, 0.0.0.0 da tutta la rete",
        "Con 127.0.0.1 il server si raggiunge solo da qui, o da un tunnel ssh. Con 0.0.0.0 lo raggiunge \
         chiunque nella tua rete: in quel caso imposta anche una API key.",
    ),
    (
        "port",
        "porta TCP del server (http://host:porta/v1); se è occupata Corral usa la prima libera",
        "Molti client provano 8080; 11434 è quella di Ollama, utile con i client fatti per Ollama. Due server \
         insieme hanno bisogno di porte diverse: Corral sposta da solo quella occupata e te lo dice.",
    ),
    (
        "api_key",
        "chiave che i client devono mandare (Authorization: Bearer …); vuota = nessuna protezione",
        "Indispensabile se il server ascolta in rete. Corral la tiene fuori dall'elenco dei processi e dai log, \
         la salva solo in file leggibili da te e la toglie dai preset esportati.",
    ),
    // -- memoria
    (
        "device",
        "su quali GPU caricare il modello: vuoto = tutte, none = solo CPU",
        "I nomi sono quelli di llama-server --list-devices (Vulkan0, CUDA0…). Serve a escludere una GPU \
         integrata lenta, o a dare una GPU a un modello e l'altra a un secondo server. Più dispositivi: \
         separali con la virgola.",
    ),
    (
        "ngl",
        "quanti layer del modello stanno in VRAM: auto, all o un numero; gli altri girano sulla CPU",
        "Più layer in GPU = più veloce, ma più VRAM. «auto» lascia decidere a llama.cpp (con l'auto-fit), \
         «all» li mette tutti. Se il modello non entra abbassalo: ogni layer in meno sposta lavoro sulla CPU e \
         rallenta la generazione.",
    ),
    (
        "ctx",
        "lunghezza del contesto in token (prompt + risposta); 0 = quella di addestramento del modello",
        "Più contesto = conversazioni e file più lunghi, ma la KV cache cresce in proporzione e occupa VRAM. \
         8k basta per chat brevi, 32k per coding e agent, di più per documenti lunghi. Con più slot (-np) il \
         contesto si divide tra le richieste.",
    ),
    (
        "fit",
        "adatta da solo alla memoria libera i valori che non hai impostato (contesto, layer, esperti)",
        "Con «on» llama.cpp misura la memoria all'avvio e riduce ciò che hai lasciato al default finché il \
         modello entra; i valori che imposti tu restano come sono. «v» nell'editor mostra cosa farebbe, \
         «o» prova per te tre configurazioni.",
    ),
    (
        "fit_target",
        "MiB di VRAM che l'auto-fit lascia liberi su ogni GPU (default 1024)",
        "Un margine per il desktop, il browser o un secondo modello. Più alto = meno contesto o meno layer in \
         GPU, ma niente errori di memoria quando altri programmi usano la scheda. Un valore per GPU separato \
         da virgole, oppure uno solo per tutte.",
    ),
    (
        "fit_ctx",
        "il contesto più corto a cui l'auto-fit può scendere prima di spostare layer sulla CPU (default 4096)",
        "Con il contesto lasciato a 0, l'auto-fit lo accorcia finché il modello entra in VRAM, ma non sotto \
         questo valore: oltre, sposta layer sulla CPU. Alzalo se ti serve un contesto lungo anche a costo di \
         velocità.",
    ),
    (
        "fa",
        "flash attention: attenzione più veloce e con meno memoria; auto la usa se la GPU la supporta",
        "Riduce la memoria del contesto e velocizza i prompt lunghi, ed è necessaria per quantizzare la V \
         cache. Lascia «auto» salvo problemi con un backend; «on» la forza.",
    ),
    (
        "ctk",
        "tipo della K cache: f16 piena qualità, q8_0 metà memoria quasi senza perdite, q4_0 un quarto",
        "La KV cache tiene il contesto e con contesti lunghi pesa diversi GiB. q8_0 la dimezza con una perdita \
         trascurabile; q4_0 risparmia di più ma può peggiorare le risposte lunghe. La K è più sensibile della V.",
    ),
    (
        "ctv",
        "tipo della V cache: come la K, ma quantizzarla richiede la flash attention",
        "q8_0 è un buon compromesso per raddoppiare il contesto a parità di VRAM. Con la flash attention \
         spenta llama-server rifiuta una V cache quantizzata.",
    ),
    (
        "ts",
        "quanto modello va su ogni GPU, in proporzione (es. 3,1 = tre quarti sulla prima)",
        "Con più GPU llama.cpp divide i layer in base alla memoria libera. Impostalo per bilanciare a mano GPU \
         diverse, ad esempio per lasciare spazio sulla scheda che usa il desktop.",
    ),
    (
        "rpc",
        "host di ~/.ssh/config che prestano le loro GPU (RPC di llama.cpp), attraverso un tunnel ssh",
        "Per i modelli che non stanno nelle tue GPU: una parte dei layer va su quelle degli host scelti. Su \
         ogni host serve Corral con un llama.cpp che abbia l'RPC. La prima volta i pesi passano in rete, poi \
         restano in cache sull'host.",
    ),
    // -- moe
    (
        "cpu_moe",
        "tiene in RAM tutti gli esperti dei modelli MoE: la VRAM resta per attenzione e contesto",
        "Nei modelli Mixture of Experts (Qwen3-30B-A3B, gpt-oss…) ogni token usa pochi esperti: tenerli in RAM \
         costa poco in velocità e libera molta VRAM. È la prima cosa da provare quando un MoE non entra in GPU.",
    ),
    (
        "n_cpu_moe",
        "come il precedente, ma solo per gli esperti dei primi N layer: un compromesso VRAM/velocità",
        "Aumenta N finché il modello entra in VRAM: meno esperti in RAM = più veloce. Con l'auto-fit attivo \
         llama.cpp sceglie da solo quanti spostarne.",
    ),
    (
        "ot",
        "sposta tensori precisi su un dispositivo (regex=buffer, es. gli esperti sulla CPU)",
        "Controllo fine di dove finiscono i pesi: «\\.ffn_.*_exps=CPU» mette tutti gli esperti in RAM. \
         llama-fit-params propone da solo queste regole per i MoE che non stanno in VRAM. Più regole: \
         separale con la virgola.",
    ),
    (
        "moe_cache",
        "MiB di VRAM per tenere in GPU gli esperti MoE usati più spesso",
        "Con gli esperti in RAM, una cache in VRAM di quelli più richiesti recupera parte della velocità. Ha \
         senso solo quando degli esperti stanno in RAM (cpu_moe, n_cpu_moe, -ot).",
    ),
    // -- draft
    (
        "draft",
        "modello piccolo che propone i token, verificati a blocchi da quello grande (speculative decoding)",
        "Può accelerare la generazione anche di 1,5-2× su codice e testi prevedibili. Deve avere lo stesso \
         vocabolario (stessa famiglia) ed essere molto più piccolo, es. Qwen3-0.6B per Qwen3-14B. Occupa \
         memoria in più.",
    ),
    (
        "spec_type",
        "metodo di speculative decoding: modello draft, teste MTP del modello o n-gram (vuoto = server)",
        "draft-simple usa il modello draft; draft-mtp le teste multi-token dei modelli che le hanno; i metodi \
         ngram non usano altri modelli e rendono quando la risposta ripete il prompt, come nelle modifiche al \
         codice.",
    ),
    (
        "ngld",
        "layer del modello draft in GPU (come -ngl, ma per il draft)",
        "Il draft è piccolo: di solito conviene tutto in GPU (all).",
    ),
    (
        "draft_max",
        "quanti token il draft propone al massimo ogni volta",
        "Più alto = più guadagno quando il draft indovina, più lavoro sprecato quando sbaglia. 8-16 è un buon \
         inizio.",
    ),
    (
        "draft_min",
        "token minimi da proporre perché la verifica valga la pena",
        "Sotto questa soglia la speculazione non parte. Di solito 0-2.",
    ),
    // -- server
    (
        "np",
        "slot paralleli: quante richieste il server serve insieme (-1 = auto)",
        "Più slot = più utenti o agent in parallelo, ma si dividono il contesto. Per uso personale con un \
         client solo, 1 dà tutto il contesto a ogni richiesta.",
    ),
    (
        "threads",
        "thread della CPU per la generazione (-1 = auto, i core fisici)",
        "Conta soprattutto quando una parte del modello gira sulla CPU. Più thread dei core fisici di solito \
         rallenta; abbassalo per lasciare la CPU libera ad altro.",
    ),
    (
        "usage",
        "a cosa serve il modello: chat, embedding (vettori per il RAG) o rerank",
        "embedding attiva /v1/embeddings (--embeddings), rerank attiva /v1/rerank (--rerank). Un modello di \
         embedding o un reranker non sa chattare: scegli l'uso che dice la sua scheda.",
    ),
    // -- sampling
    (
        "temp",
        "temperatura: bassa = risposte prevedibili, alta = più varie (vuoto = default del modello)",
        "0-0.3 per codice e risposte fattuali, 0.6-0.8 per la chat, oltre 1 per la scrittura creativa. Molti \
         modelli indicano i valori consigliati nella scheda; i client possono cambiarla a ogni richiesta.",
    ),
    (
        "top_p",
        "sceglie solo tra i token più probabili fino a questa probabilità cumulativa (es. 0.95)",
        "Taglia le code improbabili; con la temperatura bassa conta poco. 0.8-0.95 sono valori tipici.",
    ),
    (
        "top_k",
        "sceglie solo tra i K token più probabili (es. 20 o 40; 0 = nessun limite)",
        "Un altro taglio delle code: molti modelli recenti consigliano 20.",
    ),
    (
        "min_p",
        "scarta i token meno probabili di questa frazione del migliore (es. 0.05)",
        "Si adatta a quanto il modello è sicuro meglio di top-p. 0-0.1 sono valori tipici.",
    ),
    (
        "repeat_penalty",
        "penalizza i token già usati, contro le ripetizioni (1 = nessuna penalità)",
        "1.05-1.15 riduce i loop; valori alti rovinano codice e formati ripetitivi per natura (JSON, tabelle).",
    ),
    (
        "seed",
        "seme casuale: stesso seed e stesso prompt danno la stessa risposta (-1 = casuale)",
        "Utile per test riproducibili e per confrontare due preset.",
    ),
    // -- template
    (
        "jinja",
        "usa il template di chat jinja del modello: serve per il tool calling e il reasoning",
        "Lascialo on: il template incluso nel GGUF formatta i messaggi come il modello si aspetta. Spegnilo \
         solo con modelli vecchi dal template rotto.",
    ),
    (
        "chat_template_file",
        "file .jinja che sostituisce il template di chat del modello",
        "Per correggere un template sbagliato, o usare quello aggiornato pubblicato dall'autore del modello.",
    ),
    (
        "reasoning",
        "ragionamento dei modelli «thinking»: auto = come dice il template, off = risposta diretta",
        "off risponde subito, senza la fase di ragionamento: più veloce e meno token. on la chiede ai modelli \
         che la supportano.",
    ),
    (
        "reasoning_format",
        "dove va il ragionamento nella risposta API: deepseek = campo a parte, none = nel testo",
        "deepseek lo mette in reasoning_content, separato dalla risposta: è quello che si aspettano la Web UI \
         e molti client. none lo lascia nel testo tra i tag <think>.",
    ),
    (
        "reasoning_budget",
        "token massimi di ragionamento (-1 = senza limite, 0 = nessun ragionamento)",
        "Limita quanto il modello pensa prima di rispondere: contiene i tempi e il contesto consumato.",
    ),
    (
        "reasoning_effort",
        "quanto ragionare, per i modelli che lo prevedono (es. gpt-oss: low, medium, high)",
        "default lascia scegliere al template; i livelli alti danno risposte più curate ma più lente.",
    ),
    // -- avanzate
    (
        "lora",
        "adattatore LoRA da applicare al modello (un fine-tuning leggero)",
        "Un GGUF di LoRA addestrato sullo stesso modello base: cambia stile o competenze senza scaricare un \
         modello intero.",
    ),
    ("lora_scale", "intensità della LoRA (1 = piena, 0.5 = metà)", "Valori sotto 1 attenuano l'adattatore."),
    (
        "pooling",
        "come un modello di embedding riassume i token in un solo vettore",
        "Vuoto usa quello indicato dal modello; cambialo solo se la sua scheda lo chiede (cls, mean, last…).",
    ),
    (
        "batch",
        "batch logico: quanti token di prompt il server prende per passo (default 2048)",
        "Insieme al batch fisico decide la velocità di lettura dei prompt lunghi. Raramente serve cambiarlo.",
    ),
    (
        "ubatch",
        "batch fisico: token elaborati insieme dalla GPU (default 512)",
        "1024-2048 velocizzano i prompt lunghi sulle GPU potenti ma usano più VRAM di calcolo, cioè meno \
         contesto. Abbassalo se manca memoria.",
    ),
    (
        "load_mode",
        "come caricare i pesi: mmap (dal disco su richiesta), mlock (fissi in RAM), dio (I/O diretto)",
        "auto va bene quasi sempre. mlock evita che il sistema sposti il modello nello swap; none legge tutto \
         in RAM all'avvio; dio può caricare più in fretta dagli SSD NVMe.",
    ),
    (
        "kv_offload",
        "tiene la KV cache (il contesto) in VRAM; off la sposta in RAM",
        "off libera VRAM per altri layer, ma la generazione rallenta molto: lascialo on salvo casi particolari.",
    ),
    (
        "cache_reuse",
        "riusa la cache di un prompt simile spostandone i pezzi (token minimi, 0 = no)",
        "Con agent e chat che rimandano lo stesso lungo prefisso evita di rileggerlo ogni volta: 256 è un \
         valore tipico.",
    ),
    (
        "cont_batching",
        "continuous batching: unisce al volo le richieste parallele (on consigliato)",
        "È ciò che fa rendere gli slot paralleli: lascialo on.",
    ),
    (
        "metrics",
        "endpoint /metrics in formato Prometheus",
        "Per monitorare il server con Prometheus o Grafana; la dashboard di Corral non ne ha bisogno.",
    ),
    (
        "slots",
        "endpoint /slots con lo stato di ogni slot",
        "La dashboard di Corral lo usa per mostrare cosa fa ogni slot. Vuoto = default del server.",
    ),
    (
        "props",
        "permette di cambiare le proprietà del server con POST /props",
        "Serve solo agli strumenti che lo chiedono esplicitamente: lascialo off.",
    ),
    (
        "extra",
        "argomenti in più per llama-server, scritti come in un terminale",
        "Per le opzioni che Corral non elenca, es. «--no-warmup --keep 256»: vanno in fondo al comando così \
         come sono.",
    ),
    // -- router
    (
        "models_max",
        "quanti modelli il router tiene caricati insieme (0 = nessun limite)",
        "Raggiunto il limite, per caricarne uno nuovo il router scarica quello usato meno di recente.",
    ),
    (
        "margin",
        "MiB di VRAM da tenere liberi su ogni GPU: sotto, il router scarica i modelli meno usati",
        "Evita gli errori di memoria quando più modelli grandi si alternano sulla stessa GPU.",
    ),
    (
        "autoload",
        "carica un modello alla prima richiesta che lo nomina",
        "off: i modelli si caricano solo all'avvio o a mano dalla dashboard del router.",
    ),
    (
        "sleep",
        "scarica un modello che non riceve richieste da N secondi",
        "Libera memoria per gli altri; alla richiesta successiva si ricarica, con qualche secondo di attesa.",
    ),
    (
        "tools",
        "tool integrati della Web UI di llama-server (leggere file, cercare, …)",
        "Permettono al modello di leggere i file del tuo computer dalla Web UI: la sola lettura è più prudente \
         di «all». Tieni il router su 127.0.0.1.",
    ),
    (
        "tools_runtime",
        "dove girano i tool: su questo computer o in un container (es. docker:ubuntu:24.04)",
        "Un container tiene i comandi del modello lontani dal tuo sistema.",
    ),
    (
        "mcp",
        "file JSON dei server MCP (formato Cursor) da offrire alla Web UI",
        "Aggiunge tool esterni (browser, database, GitHub…) agli agent della Web UI.",
    ),
    (
        "mcp_proxy",
        "la Web UI raggiunge i server MCP passando dal router",
        "Serve quando i server MCP non si raggiungono direttamente dal browser.",
    ),
];

/// The one-line help and the longer explanation of a parameter.
pub fn of(key: &str) -> Option<(&'static str, &'static str)> {
    HELP.iter().find(|(k, ..)| *k == key).map(|(_, short, long)| (*short, *long))
}

/// The llama-server flag behind a field, as shown to the user.
pub fn flag(f: &Field) -> String {
    match (&f.flag, f.key) {
        (Flag::Opt(spec), _) => spec.split('|').next().unwrap_or(spec).to_string(),
        (Flag::Toggle(on, ""), _) => on.to_string(),
        (Flag::Toggle(on, off), _) => format!("{on} / {off}"),
        (Flag::No, "usage") => "--embeddings / --rerank".into(),
        (Flag::No, "rpc") => "--rpc (con i tunnel)".into(),
        (Flag::No, _) => "argomenti liberi".into(),
    }
}

/// The `?` window: the one line, the explanation, then flag, default and the values it takes.
pub fn explain(f: &Field) -> Vec<Line<'static>> {
    let (short, long) = of(f.key).unwrap_or(("", ""));
    let mut lines: Vec<Line<'static>> = wrap(short, 72).into_iter().map(|l| Line::from(sp(l, bold(FG)))).collect();
    lines.push(Line::default());
    lines.extend(wrap(long, 72).into_iter().map(|l| Line::from(sp(l, st(FG)))));
    lines.push(Line::default());
    let row = |k: &str, v: String| Line::from(vec![sp(format!("{k:<9}"), st(DIM)), sp(v, st(LOAD))]);
    lines.push(row("flag", flag(f)));
    let default = if f.default.is_empty() { "quello del server".to_string() } else { f.default.to_string() };
    lines.push(row("default", default));
    if let Choices::List(values) = f.choices {
        let shown: Vec<&str> = values.iter().map(|v| if v.is_empty() { "(server)" } else { v }).collect();
        lines.push(row("valori", shown.join(" · ")));
    }
    lines
}

/// `text` in lines of at most `width` columns, broken between words.
pub fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = vec![];
    let mut line = String::new();
    for word in text.split_whitespace() {
        let fits = line.is_empty() || line.chars().count() + 1 + word.chars().count() <= width;
        if !fits {
            lines.push(line.clone());
            line.clear();
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}
