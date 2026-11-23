#!/usr/bin/env bash
# Avvia Corral con i dati in questa cartella; se c'è cargo, ricompila quando i sorgenti sono cambiati.
set -e
cd "$(dirname "$(readlink -f "$0")")"
bin=target/release/corral
if command -v cargo > /dev/null; then
  if [ ! -x "$bin" ] || [ -n "$(find src Cargo.toml Cargo.lock CHANGELOG.md -newer "$bin" 2> /dev/null | head -n 1)" ]; then
    cargo build --release --locked
  fi
elif [ ! -x "$bin" ]; then
  echo "per compilare Corral serve Rust: installalo da https://rustup.rs e rilancia ./run.sh" >&2
  exit 1
fi
CORRAL_HOME="${CORRAL_HOME:-$PWD}" exec "$bin" "$@"
