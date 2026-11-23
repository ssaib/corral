//! Corral: installs llama.cpp (prebuilt releases or source builds with
//! Vulkan/CUDA/Metal/CPU/custom flags), finds and downloads GGUF models from Hugging
//! Face, keeps per-model presets of llama-server parameters and launches the server
//! under the live dashboard: one model, or several behind one endpoint (router mode),
//! with recipes and client configs for agents, and a news feed. Other computers' Corral are
//! reached over ssh, their servers through tunnels.

macro_rules! re {
    ($s:expr) => {
        std::sync::LazyLock::new(|| regex::Regex::new($s).unwrap())
    };
}

mod agent;
mod bench;
mod cli;
mod dashboard;
#[cfg(test)]
mod fake;
mod help;
mod hub;
mod installs;
mod models;
mod news;
mod preset;
mod remote;
mod router;
mod server;
#[cfg(test)]
mod tests;
mod tui;
mod tune;
mod ui;

use std::fs;
use std::io::{IsTerminal, Write};
use std::process::ExitCode;

const HELP: &str = "\
Corral: installa llama.cpp (release precompilate o build da sorgente con
Vulkan/CUDA/Metal/CPU/flag custom), cerca e scarica modelli GGUF da Hugging Face,
tiene i preset di parametri llama-server di ogni modello e avvia il server
sotto una dashboard live: un modello, o più modelli su un solo endpoint (router),
con ricette e config per agent (OpenCode, Qwen Code, Python) e le news del mondo IA.
I server di altri computer arrivano via SSH, con un tunnel (menu › host remoti).
Interfaccia a schermo intero: tastiera e mouse.

uso: corral [-y]              interfaccia a schermo intero
     corral <comando> …       comandi senza interfaccia (elenco sotto)
  -y, --yes   avvia subito l'ultimo modello (installazione attiva + suo preset)
  -h, --help  questo aiuto

dati: $CORRAL_HOME (default ~/.corral; se esiste solo ~/.llama-manager usa quella,
      vale anche $LLAMA_MANAGER_HOME): state.json · installs/ · models/ · logs/
      · routers/ (preset INI dei profili multi-modello) · export/ (config dei client)
repo gated su Hugging Face: token in $HF_TOKEN o via `hf auth login`";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(code) = cli::run(&args) {
        return code;
    }
    if let Some(a) = args.iter().find(|a| !["-y", "--yes"].contains(&a.as_str())) {
        let help = a == "-h" || a == "--help";
        if !help {
            eprintln!("argomento sconosciuto: {a}\n");
        }
        println!("{HELP}\n\n{}", cli::HELP);
        return if help { ExitCode::SUCCESS } else { ExitCode::from(2) };
    }
    if !std::io::stdout().is_terminal() {
        eprintln!("corral è un'interfaccia a schermo intero: avvialo da un terminale");
        return ExitCode::FAILURE;
    }
    if let Err(e) = fs::create_dir_all(&*ui::BASE) {
        eprintln!("impossibile creare {}: {e}", ui::BASE.display());
        return ExitCode::FAILURE;
    }
    ui::rotate_logs(50);
    let (state, problem) = ui::load_state();
    match tui::run(state, problem, !args.is_empty()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            let _ = writeln!(std::io::stderr(), "terminale non utilizzabile: {e}"); // no panic if the tty is gone
            ExitCode::FAILURE
        }
    }
}
