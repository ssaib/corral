# Corral

Interfaccia a schermo intero (tastiera e mouse) per usare [llama.cpp](https://github.com/ggml-org/llama.cpp)
in locale su Linux, macOS e Windows:

- **installazioni**: release precompilate per sistema e GPU, build da sorgente, cartelle importate;
- **modelli**: ricerca e download di GGUF da Hugging Face, suggeriti per il tuo hardware, preset di
  `llama-server` con nome per ogni modello (con l'aiuto di ogni parametro), stima della memoria e benchmark;
- **ottimizza**: tre preset per modello (lite, normal, max) dimensionati sulla memoria di questo computer
  con llama-fit-params e misurati con llama-bench;
- **server**: un modello sotto una dashboard live (slot, token/s, metriche, log), oppure più modelli su un
  solo endpoint con il router di llama-server;
- **agent e flussi**: ricette per coding, multi-agent, RAG, vision e agent nella Web UI, con le config
  pronte per OpenCode, Qwen Code e Python e un test di ogni modello;
- **news**: build di llama.cpp, nuovi modelli GGUF, notizie IA dai feed RSS, novità di Corral;
- **da terminale e come servizio**: avvio, elenco e arresto dei server senza interfaccia, e avvio con il
  computer con riavvio automatico se un server cade;
- **host remoti**: i server di altri computer via SSH, collegati qui con un tunnel: dashboard, prova e
  config dei client come per un server locale;
- **un modello su più computer**: le GPU di altri host prestate a llama-server con l'RPC di llama.cpp,
  attraverso i tunnel.

## Avvio

Serve Rust 1.88 o più recente ([rustup.rs](https://rustup.rs)).

```sh
./run.sh        # compila quando i sorgenti cambiano e avvia, con i dati in questa cartella
./run.sh -y     # avvia subito l'ultimo modello usato, col suo preset
```

Oppure `cargo build --release` e poi `target/release/corral`: i dati vanno in `~/.corral`, o nella cartella
indicata da `$CORRAL_HOME`. `corral --help` riassume opzioni e variabili d'ambiente (token Hugging Face, proxy).

## Ottimizza

«ottimizza» (o nel menu, o nell'editor di un preset) prepara tre preset del modello per questo computer,
diversi per quanta memoria usano:

| preset | contesto | KV cache | VRAM |
|---|---|---|---|
| lite | 8k | q8_0 | lascia libero un quarto della GPU più grande (i layer in più vanno sulla CPU) |
| normal | fino a 32k | f16, o q8_0 se così il modello sta tutto in GPU | margine di 1 GiB |
| max | il più lungo che entra | f16 se ci sta tutto, altrimenti q8_0 | margine di 512 MiB, batch fisico 1024 |

Per ognuno llama-fit-params sceglie contesto e layer in GPU (ed eventualmente gli esperti MoE da tenere in
RAM), una seconda passata dice quanta VRAM occupa, e llama-bench ne misura la velocità (pp512: lettura del
prompt, tg128: generazione). Invio salva i tre come preset «lite», «normal» e «max» del modello e usa quello
selezionato; a lo avvia subito. I risultati finiscono anche nello storico dei benchmark.

Nell'editor del preset, il riquadro sotto la tabella spiega il parametro selezionato; ? apre la spiegazione
completa.

## Da terminale

```sh
corral start qwen3-8b --preset lungo   # un modello (anche solo una parte del nome); aspetta che sia pronto
corral start --profile coding          # un profilo multi-modello
corral status                          # i server attivi (--json per gli script)
corral models                          # i modelli con i loro preset, e i profili
corral stop qwen3                      # per nome, porta o pid; --all li ferma tutti
corral service install --profile coding   # parte con il computer e riparte se cade
corral service list
corral service remove coding
```

I servizi usano systemd (Linux) e launchd (macOS); su Linux, per farli partire anche senza accedere,
serve `loginctl enable-linger`. Un server avviato da terminale compare in «server attivi» della
prossima interfaccia aperta, e viceversa.

## Via SSH

Corral funziona anche dentro una sessione SSH: la copia passa dal terminale (OSC 52), «o» mostra il
tunnel da aprire sul tuo computer (`ssh -L 8080:127.0.0.1:8080 …`), e se la connessione cade i server
restano attivi: li ritrovi alla prossima apertura.

## Host remoti

«host remoti» (h nel menu) elenca gli host di `~/.ssh/config` e usa il tuo `ssh`, con le sue chiavi,
l'agent e i ProxyJump. Di un host vedi le GPU, i server attivi, i modelli e i profili:

- invio su un modello o un profilo lo avvia sull'host e lo collega qui con un tunnel (`ssh -L`): la
  dashboard, la prova del prompt e le config esportate per OpenCode e Qwen Code usano `127.0.0.1` di
  questo computer; log e GPU arrivano sulla stessa connessione;
- invio su un server attivo lo collega, s lo ferma sull'host;
- o apre in questo terminale il Corral dell'host, per scaricare modelli o installare llama.cpp lì;
- u ci copia questo Corral in `~/.local/bin`, se l'host ha lo stesso sistema e processore.

Serve l'accesso con una chiave, senza password (`ssh-copy-id host`), e una shell POSIX sull'host
(Linux o macOS). Uscire da Corral chiude solo i tunnel: i server restano sull'host.

## Un modello su più computer (RPC)

Un modello che non sta nelle tue GPU può usare anche quelle di altri computer. Nel preset, «GPU di
altri host (RPC)» (sezione memoria) sceglie uno o più host di `~/.ssh/config`. All'avvio Corral apre
su ognuno, via ssh, il server RPC di llama.cpp (`corral rpc`, in ascolto solo su `127.0.0.1`
dell'host, perché non chiede password) e passa a llama-server `--rpc` con i tunnel: una parte dei
layer va sulle GPU degli host, che la dashboard mostra con il loro nome. La prima volta i pesi
passano in rete, poi ogni host li tiene in cache (`~/.cache/llama.cpp/rpc`).

- Sugli host serve Corral (u nella schermata dell'host) con un llama.cpp che abbia l'RPC: le release,
  o le build da sorgente di Corral 0.5.0 in poi. Meglio la stessa versione ovunque: Corral avvisa se
  è diversa.
- Se un host ha anche una GPU integrata, Corral presta solo quelle dedicate.
- Il server vive con i collegamenti: da terminale serve `corral start … --foreground` (va bene anche
  come servizio), e «lascia attivo all'uscita» non c'è. Se un host cade, il server non riparte da
  solo.

## Dati

Nella cartella dati: `state.json` (preset, profili, impostazioni), `installs/`, `models/`, `logs/`,
`routers/` (gli INI dei profili multi-modello), `export/` (le config salvate per i client), `cache/`.

## Sviluppo

```sh
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt
cargo llvm-cov --ignore-filename-regex 'fake\.rs' --summary-only   # coverage (cargo install cargo-llvm-cov)
```

I test non toccano né la rete né i tuoi dati: sotto `cargo test` la cartella dati e la home sono
cartelle temporanee, ogni richiesta web va a un finto server locale (Hugging Face, GitHub, feed RSS) e
su Linux e macOS llama.cpp e ssh sono finti anche loro (`src/fake.rs`): schermate, comandi, server,
router, host remoti e tunnel girano davvero, contro quei finti. Le schermate si provano disegnandole
nel `TestBackend` di ratatui con tasti e clic simulati (`src/tui/tests.rs`);
`CORRAL_SHOW=1 cargo test tui:: -- --nocapture --test-threads 1` stampa ogni schermata disegnata.
Con un llama.cpp e un modello veri:
`CORRAL_TEST_BIN=<cartella bin> CORRAL_TEST_MODEL=<file.gguf> cargo test real_tune -- --ignored --nocapture`
prova «ottimizza» per davvero.

La CI esegue gli stessi controlli su Linux, macOS e Windows, una compilazione con Rust 1.88 e la
coverage, che non deve scendere sotto l'85% delle righe.
Le novità di ogni versione stanno in [CHANGELOG.md](CHANGELOG.md), che Corral mostra anche nelle news.

## Licenza

A tua scelta, [MIT](LICENSE-MIT) o [Apache 2.0](LICENSE-APACHE). Salvo indicazione contraria, ogni
contributo inviato a questo progetto è concesso con le stesse due licenze.
