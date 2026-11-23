# Novità di Corral

Mostrate nella sezione news (fonte «Corral»): ogni versione è un titolo `## versione — data — in breve`
seguito dalle sue righe `- …`.

## 0.6.0 — 2026-11-23 — menu a card, ottimizza, aiuto dei parametri
- Menu principale ridisegnato: l'hardware in alto, poi le azioni come card raggruppate (modelli, server, sistema), ognuna con il suo stato (modelli locali, server attivi, news da leggere, installazioni…); frecce, Tab, la lettera della card o un clic.
- Ottimizza (o nel menu o nell'editor del preset): tre preset del modello pensati per questo computer. lite lascia libero un quarto della VRAM, normal tiene fino a 32k di contesto tutto in GPU se ci sta, max il contesto più lungo che entra. Ognuno è dimensionato da llama-fit-params e misurato con llama-bench (pp512 e tg128), poi salvato come preset «lite», «normal» e «max».
- Aiuto dei parametri: sotto la tabella del preset un riquadro spiega sempre il parametro selezionato, con flag e default; ? apre la spiegazione completa con un consiglio pratico. Anche per le opzioni dei profili multi-modello.
- Nuovi parametri del preset: «VRAM da lasciare libera» (--fit-target) e «contesto minimo dell'auto-fit» (--fit-ctx).
- Il benchmark passa a llama-bench anche batch, ubatch, ripartizione tra GPU, esperti MoE in RAM e override dei tensori del preset.
- Interfaccia rifinita: una barra in alto con il percorso, l'installazione e i server attivi, i tasti in basso; pannelli dai bordi neutri che si colorano solo quando dicono qualcosa (focus, attivo, errore), titoli in minuscolo, finestre di dialogo sopra lo schermo attenuato, notifiche con una striscia colorata. La dashboard del server usa lo stesso telaio, con lo stato in una riga.
- Quando un server non parte, il motivo mostrato è la causa vera e non il «exiting due to…» che la segue; con un'architettura che il llama.cpp installato non conosce, suggerisce di aggiornarlo.
- Test: ogni schermata, i comandi, i server, il router e gli host remoti sono provati contro un finto llama.cpp, un finto ssh e un finto web, senza rete e senza toccare i dati; la CI misura la coverage (oltre l'85% delle righe).

## 0.5.0 — 2026-11-14 — un modello su più computer (RPC)
- Un modello può usare anche le GPU di altri computer: nel preset, «GPU di altri host (RPC)» sceglie gli host di ~/.ssh/config, e all'avvio llama-server riceve `--rpc` con i tunnel verso i loro server RPC.
- Su ogni host Corral avvia il server RPC di llama.cpp solo su 127.0.0.1, raggiungibile dal tunnel ssh, con la cache dei pesi, e lo ferma quando il collegamento finisce; se l'host ha anche una GPU integrata presta solo quelle dedicate.
- Le GPU prestate compaiono nella dashboard con il nome dell'host; una versione diversa di llama.cpp sull'host viene segnalata.
- Le build da sorgente includono l'RPC (`-DGGML_RPC=ON`) e il server RPC.
- `corral start` con un preset che usa altri host vuole `--foreground` (va bene anche come servizio).

## 0.4.0 — 2026-11-14 — host remoti via SSH
- Host remoti: gli host di ~/.ssh/config, raggiunti con il tuo ssh (chiavi, agent, ProxyJump); di ognuno le GPU, i server attivi, i modelli e i profili.
- Un server di un altro computer si collega qui con un tunnel: dashboard, prova del prompt e config per OpenCode e Qwen Code come per un server locale, con log e GPU dell'host.
- Da qui si avvia e si ferma un server sull'host; «o» apre in questo terminale il Corral dell'host, «u» ci copia questo Corral se l'host ha lo stesso sistema e processore.
- Comandi: `corral models` elenca i modelli con i loro preset e i profili. `corral status --json` ora è un oggetto: la versione, il sistema, le GPU e i server in `servers` (prima era solo l'elenco dei server).

## 0.3.0 — 2026-11-14 — comandi, servizi, SSH, riavvio automatico
- Comandi senza interfaccia: `corral start`, `corral status` (anche in JSON) e `corral stop` avviano, elencano e fermano modelli e profili, anche dagli script.
- Servizi: `corral service install` fa partire un modello o un profilo con il computer e lo riavvia se cade (systemd su Linux, launchd su macOS).
- Via SSH: «o» copia l'indirizzo e mostra il tunnel da aprire sul tuo computer; se la connessione cade i server restano attivi e il prossimo Corral li riprende.
- Riavvio automatico: un server che va in crash dopo essere partito riparte da solo, al massimo 3 volte in 5 minuti.
- Chiave API: non si vede più nell'elenco dei processi né nei log; state.json, i file dei server e le config esportate sono leggibili solo da te, un preset esportato non la contiene, e Corral avvisa se un server è raggiungibile in rete senza chiave.
- GPU nella dashboard: oltre alla memoria, utilizzo, temperatura e consumo (NVIDIA e AMD).
- Notifiche sul desktop, o del terminale anche via SSH, quando finiscono download, build e benchmark e quando un server cade.

## 0.2.0 — 2026-11-13 — multi-modello, agent e flussi, news
- Multi-modello: profili con più modelli su un solo endpoint (llama-server in modalità router), ognuno col suo preset di Corral; caricati all'avvio o alla prima richiesta, con limite di modelli insieme e margine di memoria.
- Dashboard del router: stato di ogni modello, carica/scarica con un clic, token/s per modello, prova di un prompt, log dei modelli figli.
- Agent e flussi: ricette per agent di coding, multi-agent (orchestratore + worker), RAG (chat + embedding + reranker), vision e agent nella Web UI con i tool integrati e i server MCP.
- Export delle config per OpenCode, Qwen Code, Python (OpenAI SDK, LangChain, CrewAI) e variabili d'ambiente, con un test di ogni modello (tool calling, embedding, rerank).
- Preset: nuovo parametro «uso del modello» (chat, embedding, reranker) e pooling per gli embedding.
- News: build e release di llama.cpp (con installazione diretta), nuovi modelli GGUF dei quantizzatori più noti filtrati per il tuo hardware, notizie IA da feed RSS modificabili, novità di Corral.
- Rimossa la vecchia versione Python: run.sh ora compila e avvia Corral.
- Più sicuro: un nome «..» per una build non può più cancellare la cartella dati; state.json si salva senza rischio di restare a metà, e uno illeggibile viene messo da parte invece di essere sovrascritto.
- Niente più crash con terminali stretti; i link delle news si aprono solo se sono pagine web; rinominare un preset aggiorna i profili multi-modello che lo usano, ed eliminarlo è bloccato finché un profilo lo usa.

## 0.1.0 — 2026-11-13 — Corral in Rust
- Riscrittura in Rust di llama-manager per Linux, macOS e Windows, con interfaccia a schermo intero usabile da tastiera e mouse e schermata di avvio.
- Installazioni di llama.cpp: release precompilate per sistema e GPU, build da sorgente, cartelle importate, aggiornamento all'ultima build.
- Modelli: ricerca e download da Hugging Face con ripresa, suggeriti per il tuo hardware, cartelle extra e altre sorgenti (cache HF, LM Studio).
- Preset con nome per modello, stima della memoria con llama-fit-params, benchmark con llama-bench.
- Dashboard live: avvio, slot, throughput, metriche, log filtrabile, prompt di prova; server in background ripresi al riavvio e fermati in modo ordinato all'uscita.
