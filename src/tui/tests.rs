//! Screens drawn into ratatui's TestBackend and driven with simulated keys, clicks and ticks,
//! with the whole chrome (header, footer, modals, toasts) as in the real loop.
//! `CORRAL_SHOW=1 cargo test tui:: -- --nocapture --test-threads 1` prints every frame drawn.

use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};

use super::*;
use crate::installs::{Device, Install};

pub fn fake_install() -> Install {
    Install {
        name: "rel-b9999-vulkan".into(),
        kind: "release".into(),
        backend: "vulkan".into(),
        bin_dir: "/nonexistent/corral-test".into(),
        git_ref: None,
        flags: vec![],
        src_dir: None,
    }
}

pub fn gpu() -> Device {
    Device { name: "Vulkan0".into(), desc: "NVIDIA GeForce RTX 4080 Laptop GPU".into(), total: 12282, free: 11474 }
}

/// A context with the detection done: one GPU, 16 GiB of RAM, an install whose binaries don't exist.
pub fn ctx() -> Ctx {
    let mut ctx = Ctx::bare(State::default());
    ctx.hw = Hw {
        step: HW_STEPS.len(),
        inst: Some(fake_install()),
        ram: 16 << 30,
        version: "b9999 · abc1234".into(),
        devs: vec![gpu()],
    };
    ctx
}

pub struct Harness {
    app: App,
    pub width: u16,
    pub height: u16,
    _servers: Option<crate::fake::SharedServers>, // declared last: dropped after the servers stop
}

impl Harness {
    pub fn new(screen: Box<dyn Screen>, ctx: Ctx) -> Self {
        let servers = Some(crate::fake::shared_servers());
        let app = App { stack: vec![screen], ctx, footer: Zones::default() };
        Self { app, width: 120, height: 36, _servers: servers }
    }

    /// For the test that holds run/ alone (`fake::all_servers`).
    pub fn alone(screen: Box<dyn Screen>, ctx: Ctx) -> Self {
        let app = App { stack: vec![screen], ctx, footer: Zones::default() };
        Self { app, width: 120, height: 36, _servers: None }
    }

    pub fn sized(mut self, width: u16, height: u16) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// One frame, as text (one line per row).
    pub fn draw(&mut self) -> String {
        let out = self.frame();
        if std::env::var_os("CORRAL_SHOW").is_some() {
            eprintln!("{out}");
        }
        out
    }

    /// A frame, as the loop draws one between two events (widgets learn their size and place).
    fn frame(&mut self) -> String {
        let mut term = Terminal::new(TestBackend::new(self.width, self.height)).unwrap();
        term.draw(|f| self.app.draw(f)).unwrap();
        let buf = term.backend().buffer();
        let mut out = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                out.push_str(buf[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    /// One turn of the loop without the terminal: background messages, then the top screen's tick.
    pub fn tick(&mut self) {
        self.app.ctx.pump();
        if let Some(top) = self.app.stack.last_mut() {
            let a = top.tick(&mut self.app.ctx);
            self.app.apply(a);
        }
        if !self.app.stack.is_empty() {
            self.frame();
        }
    }

    /// Ticks until `done` holds (background tasks), at most ~10 s.
    pub fn tick_until(&mut self, mut done: impl FnMut(&mut Self) -> bool) {
        for _ in 0..1000 {
            if done(self) {
                return;
            }
            self.tick();
            thread::sleep(Duration::from_millis(10));
        }
        let screen = self.draw();
        panic!("tick_until: still waiting after 10 s\ntoasts: {:?}\n{screen}", self.toasts());
    }

    pub fn key(&mut self, code: KeyCode) -> &mut Self {
        self.app.event(Event::Key(KeyEvent::new(code, KeyModifiers::NONE)));
        self.frame();
        self
    }

    pub fn ctrl(&mut self, c: char) -> &mut Self {
        self.app.event(Event::Key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL)));
        self.frame();
        self
    }

    pub fn typed(&mut self, text: &str) -> &mut Self {
        for c in text.chars() {
            self.key(KeyCode::Char(c));
        }
        self
    }

    fn mouse(&mut self, kind: MouseEventKind, x: u16, y: u16) {
        let m = MouseEvent { kind, column: x, row: y, modifiers: KeyModifiers::NONE };
        self.app.event(Event::Mouse(m));
        self.frame();
    }

    pub fn click(&mut self, x: u16, y: u16) -> &mut Self {
        self.mouse(MouseEventKind::Down(MouseButton::Left), x, y);
        self
    }

    pub fn right_click(&mut self) -> &mut Self {
        self.mouse(MouseEventKind::Down(MouseButton::Right), 1, 1);
        self
    }

    pub fn hover(&mut self, x: u16, y: u16) -> &mut Self {
        self.mouse(MouseEventKind::Moved, x, y);
        self
    }

    pub fn wheel(&mut self, down: bool, x: u16, y: u16) -> &mut Self {
        let kind = if down { MouseEventKind::ScrollDown } else { MouseEventKind::ScrollUp };
        self.mouse(kind, x, y);
        self
    }

    /// Where `text` first appears on screen under the header (column, row).
    pub fn find(&mut self, text: &str) -> Option<(u16, u16)> {
        let screen = self.draw();
        for (y, line) in screen.lines().enumerate().skip(2) {
            if let Some(byte) = line.find(text) {
                let x = line[..byte].chars().count();
                return Some((x as u16, y as u16));
            }
        }
        None
    }

    pub fn click_on(&mut self, text: &str) -> &mut Self {
        let (x, y) = self.find(text).unwrap_or_else(|| panic!("«{text}» is not on screen:\n{}", self.draw()));
        self.click(x, y)
    }

    /// Select the row showing `text` (a click on the row already selected opens it: back, then).
    pub fn select(&mut self, text: &str) -> &mut Self {
        let depth = self.depth();
        self.click_on(text);
        if self.depth() > depth {
            self.key(KeyCode::Esc);
        }
        self
    }

    pub fn depth(&self) -> usize {
        self.app.stack.len()
    }

    pub fn title(&self) -> String {
        self.app.stack.last().map(|s| s.title()).unwrap_or_default()
    }

    pub fn modal_open(&self) -> bool {
        self.app.ctx.modal.is_some()
    }

    pub fn toasts(&self) -> Vec<String> {
        self.app.ctx.toasts.iter().map(|t| t.msg.clone()).collect()
    }
}

mod menu {
    use super::*;
    use crate::tui::menu::Menu;

    fn menu() -> Harness {
        let mut h = Harness::new(Box::new(Menu::new()), ctx());
        h.tick(); // reads the counts
        h
    }

    #[test]
    fn cards_hardware_and_counts() {
        let mut h = menu();
        let s = h.draw();
        for title in ["avvia server", "ottimizza", "cerca su HF", "server attivi", "installazioni", "log"] {
            assert!(s.contains(title), "{title} missing:\n{s}");
        }
        for group in ["modelli", "server", "sistema", "hardware"] {
            assert!(s.contains(group), "{group} missing");
        }
        assert!(s.contains("RTX 4080") && s.contains("11474 / 12282 MiB liberi"));
        assert!(s.contains("nessuno attivo"));
        assert!(s.contains("per 11.2 GiB di VRAM"));
        assert!(s.contains("questo computer"), "descriptions are whole:\n{s}");
    }

    #[test]
    fn arrows_letters_clicks_and_back() {
        let mut h = menu();
        h.draw();
        h.key(KeyCode::Right).key(KeyCode::Right); // cerca su HF
        h.key(KeyCode::Down); // the card under it: agent e flussi
        h.key(KeyCode::Up).key(KeyCode::Left).key(KeyCode::Down); // ottimizza, then multi-modello
        h.key(KeyCode::End).key(KeyCode::Tab); // wraps to the first
        h.key(KeyCode::BackTab).key(KeyCode::Home);
        h.hover(10, 10).wheel(true, 10, 10);
        h.key(KeyCode::Enter);
        assert_eq!(h.title(), "modelli locali");
        h.key(KeyCode::Esc);
        assert_eq!(h.depth(), 1);
        h.key(KeyCode::Char('l'));
        assert_eq!(h.depth(), 2);
        h.right_click();
        assert_eq!(h.depth(), 1);
        h.click_on("server attivi");
        assert_eq!(h.depth(), 2);
        h.key(KeyCode::Char('q'));
        h.key(KeyCode::Char('x')); // no card: ignored
        assert_eq!(h.depth(), 1);
        h.ctrl('c');
        assert_eq!(h.depth(), 0, "ctrl+c closes everything");
    }

    #[test]
    fn every_card_opens_its_screen() {
        for (key, title) in [
            ('m', "multi-modello"),
            ('g', "agent e flussi"),
            ('c', "cerca su Hugging Face"),
            ('i', "installazioni"),
            ('h', "host remoti"),
            ('n', "news"),
            ('v', "server attivi"),
            ('l', "log"),
        ] {
            let mut h = menu();
            h.key(KeyCode::Char(key));
            assert_eq!(h.depth(), 2, "{key}");
            let screen = h.draw();
            assert!(screen.contains(title) || h.title().contains(title), "{key}: {}", h.title());
        }
    }

    #[test]
    fn without_install_the_models_need_one_first() {
        let mut ctx = ctx();
        ctx.hw.inst = None;
        let mut h = Harness::new(Box::new(Menu::new()), ctx);
        h.tick(); // first run: straight to the installs
        assert_eq!(h.title(), "installazioni");
        assert!(h.toasts().iter().any(|t| t.contains("nessuna installazione di llama.cpp")));
        h.key(KeyCode::Esc);
        h.tick();
        assert!(h.draw().contains("serve llama.cpp (i)"));
        h.key(KeyCode::Char('a'));
        assert_eq!(h.title(), "installazioni");
        assert!(h.toasts().iter().any(|t| t.contains("serve prima un'installazione")));
    }

    #[test]
    fn optimize_asks_for_a_model() {
        model("Menu-Tune-Q4_K_M.gguf");
        let mut h = menu();
        h.key(KeyCode::Char('o'));
        assert!(h.modal_open());
        assert!(h.draw().contains("quale modello?"));
        h.typed("1").key(KeyCode::Enter);
        assert!(h.title().starts_with("ottimizza · "), "{}", h.title());
    }

    #[test]
    fn small_terminal_keeps_every_card() {
        let mut h = menu().sized(80, 24);
        let s = h.draw();
        for title in ["avvia server", "ottimizza", "host remoti", "installazioni", "log"] {
            assert!(s.contains(title), "{title} missing at 80x24:\n{s}");
        }
    }
}

/// A model in the test data folder (see `ui::BASE` under test): models/<name>.
pub fn model(name: &str) -> std::path::PathBuf {
    let p = crate::hub::models_dir().join("test__repo").join(name);
    if !p.exists() {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, crate::fake::gguf(36, 0, 40960, 4096)).unwrap();
    }
    p
}

mod config {
    use super::*;
    use crate::tui::config::Config;

    #[test]
    fn help_follows_the_selection() {
        let ctx = ctx();
        let m = model("Help-Q4_K_M.gguf");
        let mut h = Harness::new(Box::new(Config::new(&ctx, &m, None)), ctx);
        let s = h.draw();
        assert!(s.contains("alias · --alias") && s.contains("nome del modello nelle API"), "{s}");
        h.key(KeyCode::Down).key(KeyCode::Down).key(KeyCode::Down);
        let s = h.draw();
        assert!(s.contains("porta · --port") && s.contains("porta TCP del server"), "{s}");
        h.key(KeyCode::Char('?'));
        assert!(h.modal_open());
        let s = h.draw();
        assert!(s.contains("Ollama") && s.contains("default  8080"), "{s}");
    }
}

mod tune {
    use super::*;
    use crate::tune::Tune;

    #[test]
    fn three_cards_and_a_missing_install() {
        let ctx = ctx();
        let m = model("Tune-Q8_0.gguf");
        let mut h = Harness::new(Box::new(Tune::new(&ctx, &m)), ctx);
        h.tick_until(|h| h.draw().contains("salva i preset")); // the footer once the tests are over
        let s = h.draw();
        for t in [" lite ", " normal ", " max ", "8.192", "32.768", "fino a 40.960", "q8_0"] {
            assert!(s.contains(t), "{t} missing:\n{s}");
        }
        assert!(s.contains("✖ questa installazione non ha"), "{s}");
        h.key(KeyCode::Enter);
        assert!(h.toasts().iter().any(|t| t.contains("non è stato verificato")));
    }

    /// The real thing: `CORRAL_TEST_BIN=<llama.cpp bin dir> CORRAL_TEST_MODEL=<gguf> cargo test real_tune -- --ignored`
    #[test]
    #[ignore = "needs llama.cpp and a model (CORRAL_TEST_BIN, CORRAL_TEST_MODEL)"]
    fn real_tune() {
        let (Some(bin), Some(model)) = (std::env::var_os("CORRAL_TEST_BIN"), std::env::var_os("CORRAL_TEST_MODEL"))
        else {
            return;
        };
        let mut ctx = ctx();
        let inst = Install { bin_dir: bin.to_string_lossy().into(), ..fake_install() };
        ctx.hw.devs = crate::installs::devices(&inst);
        ctx.hw.inst = Some(inst);
        let model = std::path::PathBuf::from(model);
        let mut h = Harness::new(Box::new(Tune::new(&ctx, &model)), ctx);
        let started = Instant::now();
        while !h.draw().matches("✓ pronto").count().eq(&3) && started.elapsed() < Duration::from_secs(600) {
            h.tick();
            thread::sleep(Duration::from_millis(100));
            if h.draw().contains('✖') && !h.draw().contains('⠋') {
                break;
            }
        }
        let s = h.draw();
        eprintln!("{s}");
        assert_eq!(s.matches("✓ pronto").count(), 3, "{s}");
    }
}

#[cfg(unix)]
mod installs {
    use super::*;
    use crate::fake;
    use crate::tui::installs::Installs;

    #[test]
    fn import_activate_update_and_remove() {
        let mut h = Harness::new(Box::new(Installs::new()), ctx());
        h.key(KeyCode::Char('i'));
        assert!(h.modal_open());
        h.ctrl('u').typed(&ps(&fake::llama_cpp())).key(KeyCode::Enter);
        assert!(h.toasts().iter().any(|t| t.contains("importata")), "{:?}", h.toasts());
        assert!(h.app.ctx.state.install.as_deref().is_some_and(|n| n.starts_with("ext-")));
        h.tick_until(|h| h.draw().contains("b9999 · abc1234")); // the version, read in the background
        h.select("ext-fake-llama").click_on("ext-fake-llama"); // select, then activate
        assert!(h.toasts().iter().any(|t| t.starts_with("attiva: ext-")));
        h.key(KeyCode::Char('u'));
        assert!(h.toasts().iter().any(|t| t.contains("si aggiornano a mano")));
        h.key(KeyCode::Char('x'));
        assert!(h.modal_open());
        h.key(KeyCode::Char('s'));
        assert!(h.toasts().iter().any(|t| t.starts_with("eliminata ext-")), "{:?}", h.toasts());
        assert_eq!(h.app.ctx.state.install, None);
        // a folder without llama.cpp
        h.key(KeyCode::Char('i')).ctrl('u').typed("/nonexistent/x").key(KeyCode::Enter);
        assert!(h.toasts().iter().any(|t| t.starts_with('✖')));
    }

    #[test]
    fn release_download_and_install() {
        let _turn = release_turn();
        fake::serve_release();
        let mut h = Harness::new(Box::new(Installs::new()), ctx());
        h.key(KeyCode::Char('r'));
        assert_eq!(h.depth(), 2);
        h.tick_until(|h| h.draw().contains("b9998"));
        h.key(KeyCode::Enter); // b9999: its variants for this system
        assert!(h.draw().contains("variante"), "{}", h.draw());
        h.key(KeyCode::Esc); // back to the builds
        h.key(KeyCode::Enter).key(KeyCode::Enter); // b9999, the recommended variant
        h.tick_until(|h| h.depth() == 1);
        assert!(h.toasts().iter().any(|t| t.contains("✓ installata rel-b9999")), "{:?}", h.toasts());
        assert!(h.app.ctx.state.install.as_deref().is_some_and(|n| n.starts_with("rel-b9999")));
        h.tick_until(|h| h.draw().contains("b9999 · abc1234"));
        // again: it exists, so it asks first
        h.key(KeyCode::Char('r'));
        h.tick_until(|h| h.draw().contains("b9998"));
        h.key(KeyCode::Enter).key(KeyCode::Enter);
        assert!(h.modal_open());
        h.key(KeyCode::Char('n'));
        h.key(KeyCode::Esc).key(KeyCode::Esc);
        assert_eq!(h.depth(), 1);
        // up to date: 'u' says so once the latest build is known
        h.select("rel-b9999-");
        h.tick_until(|h| {
            h.key(KeyCode::Char('u'));
            h.toasts().iter().any(|t| t.contains("già all'ultima release") || t.contains("controllo"))
        });
    }

    #[test]
    fn source_build_asks_ref_flags_and_name() {
        let mut h = Harness::new(Box::new(Installs::new()), ctx());
        h.key(KeyCode::Char('b'));
        assert_eq!(h.depth(), 2);
        let s = h.draw();
        assert!(s.contains("cpu") || s.contains("CPU"), "{s}");
        h.key(KeyCode::Down).key(KeyCode::Up).key(KeyCode::Enter);
        if h.modal_open() && h.draw().contains("Sì") {
            h.key(KeyCode::Char('s')); // prerequisites missing: go on anyway
        }
        // branch, extra flags, then the name: Esc there, before anything is cloned
        for _ in 0..2 {
            if h.modal_open() {
                h.key(KeyCode::Enter);
            }
        }
        h.draw();
        h.key(KeyCode::Esc);
        h.key(KeyCode::Esc);
        assert!(h.depth() <= 2);
    }
}

mod hugging_face {
    use super::*;
    use crate::tui::hf::{RepoScreen, Search};

    #[test]
    fn search_then_a_repo() {
        let mut h = Harness::new(Box::new(Search::new()), ctx());
        h.typed("qwen").key(KeyCode::Enter);
        h.tick_until(|h| h.draw().contains("unsloth/Qwen3-8B-GGUF"));
        let s = h.draw();
        assert!(s.contains("8.2B") && s.contains("123.456"), "{s}");
        h.key(KeyCode::Tab).key(KeyCode::Down).key(KeyCode::Up);
        h.key(KeyCode::Enter);
        assert_eq!(h.depth(), 2);
        h.tick_until(|h| h.draw().contains("Q8_0"));
        assert!(h.draw().contains("Q4_K_M"));
        h.key(KeyCode::Esc);
        h.key(KeyCode::Char('/')).typed("x").key(KeyCode::Esc); // back to the field, then out
        assert_eq!(h.depth(), 0);
    }

    #[test]
    fn suggestions_for_this_hardware() {
        let ctx = ctx();
        let mut h = Harness::new(Box::new(Search::suggest(&ctx)), ctx);
        h.tick_until(|h| h.draw().contains("Qwen2.5-Coder"));
        assert!(h.draw().contains("coding"));
    }

    #[test]
    fn download_with_mmproj_then_the_editor() {
        let mut h = Harness::new(Box::new(RepoScreen::new("test/Download-GGUF")), ctx());
        h.tick_until(|h| h.draw().contains("Q4_K_M"));
        h.key(KeyCode::Home).key(KeyCode::Enter); // Q4_K_M
        assert!(h.modal_open(), "a vision repo asks for the mmproj");
        h.key(KeyCode::Enter);
        if h.modal_open() {
            h.key(KeyCode::Char('s')); // little disk space left: go on
        }
        h.tick_until(|h| h.title().starts_with("configura"));
        let model = crate::hub::models_dir().join("test__Download-GGUF").join("Model-Q4_K_M.gguf");
        assert!(model.is_file());
        assert!(h.draw().contains("mmproj-Model-F16.gguf"), "the editor gets the mmproj");
        // downloaded again: the file is there, nothing to fetch
        h.app.apply(Act::Replace(Box::new(RepoScreen::new("test/Download-GGUF"))));
        h.tick_until(|h| h.draw().contains("Q4_K_M"));
        h.key(KeyCode::Home).key(KeyCode::Enter).key(KeyCode::Down).key(KeyCode::Enter); // no mmproj
        if h.modal_open() {
            h.key(KeyCode::Char('s'));
        }
        h.tick_until(|h| h.title().starts_with("configura"));
    }

    #[test]
    fn a_missing_repo_goes_back() {
        let mut h = Harness::new(Box::new(Search::new()), ctx());
        h.app.apply(Act::Push(Box::new(RepoScreen::new("test/missing-GGUF"))));
        h.tick_until(|h| h.depth() == 1);
        assert!(h.toasts().iter().any(|t| t.starts_with('✖')));
    }
}

mod local_models {
    use super::*;
    use crate::tui::models::{Models, Presets};

    #[test]
    fn list_presets_and_folders() {
        let m = model("Listed-Q4_K_M.gguf");
        let ctx = ctx();
        let mut h = Harness::new(Box::new(Models::new(&ctx)), ctx);
        h.click_on("Listed-Q4_K_M.gguf").click_on("Listed-Q4_K_M.gguf");
        assert!(h.title().starts_with("configura · Listed-Q4_K_M.gguf"), "{}", h.title());
        h.key(KeyCode::Esc);
        h.key(KeyCode::Char('p'));
        assert!(h.title().starts_with("preset"), "{}", h.title());
        h.key(KeyCode::Char('n')).typed("coding").key(KeyCode::Enter);
        h.key(KeyCode::Char('d')).typed(" bis").key(KeyCode::Enter);
        h.key(KeyCode::Char('r')).ctrl('u').typed("lungo").key(KeyCode::Enter);
        let names = crate::preset::preset_names(&h.app.ctx.state, &m);
        assert!(names.contains(&"lungo".to_string()) && names.contains(&"coding".to_string()), "{names:?}");
        let out = std::env::temp_dir().join(format!("corral-preset-{}.json", std::process::id()));
        h.key(KeyCode::Char('e')).ctrl('u').typed(&ps(&out)).key(KeyCode::Enter);
        assert!(out.is_file(), "{:?}", h.toasts());
        h.key(KeyCode::Char('i')).ctrl('u').typed(&ps(&out)).key(KeyCode::Enter);
        h.key(KeyCode::Char('x')).key(KeyCode::Char('s'));
        h.key(KeyCode::Enter); // the editor of the selected preset
        assert!(h.title().starts_with("configura"), "{}", h.title());
        h.key(KeyCode::Esc).key(KeyCode::Esc);
        // extra folders: add one, then remove it
        h.key(KeyCode::Char('f'));
        assert!(h.modal_open());
        h.key(KeyCode::End).key(KeyCode::Enter).ctrl('u').typed(&ps(&std::env::temp_dir())).key(KeyCode::Enter);
        assert_eq!(h.app.ctx.state.model_dirs.len(), 1, "{:?}", h.toasts());
        h.key(KeyCode::Char('f')).key(KeyCode::Home).key(KeyCode::Enter).key(KeyCode::Char('s'));
        assert!(h.app.ctx.state.model_dirs.is_empty());
        h.key(KeyCode::Char('o')).key(KeyCode::Char('s')); // orphaned presets
        h.key(KeyCode::Char('u')); // updates: asks Hugging Face
        h.tick_until(|h| h.toasts().iter().any(|t| t.contains("aggiorn") || t.contains("Hugging") || t.contains('✓')));
        let _ = std::fs::remove_file(out);
        let _ = Presets::new(&m);
    }

    #[test]
    fn delete_a_downloaded_model() {
        let m = crate::hub::models_dir().join("test__Delete-GGUF").join("Gone-Q4_K_M.gguf");
        std::fs::create_dir_all(m.parent().unwrap()).unwrap();
        std::fs::write(&m, crate::fake::gguf(36, 0, 40960, 100)).unwrap();
        let ctx = ctx();
        let mut h = Harness::new(Box::new(Models::new(&ctx)), ctx);
        h.select("Gone-Q4_K_M.gguf");
        h.key(KeyCode::Char('x'));
        assert!(h.modal_open());
        assert!(h.draw().contains("non si può annullare"));
        h.key(KeyCode::Char('s'));
        assert!(!m.exists());
        assert!(h.toasts().iter().any(|t| t.starts_with("eliminato")));
    }
}

/// The tests that install the fake build b9999 (the same folder, removed and fetched again) take
/// turns.
pub static RELEASE_B9999: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub fn release_turn() -> std::sync::MutexGuard<'static, ()> {
    RELEASE_B9999.lock().unwrap_or_else(|e| e.into_inner())
}

/// A port nobody listens on right now.
pub fn free_port() -> String {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    l.local_addr().unwrap().port().to_string()
}

/// Stop what a test started, and wait for it: no fake server outlives its test.
pub fn stop_all(h: &mut Harness) {
    for s in &mut h.app.ctx.servers {
        s.set_keep(false);
        s.stop();
    }
    h.tick_until(|h| h.app.ctx.servers.iter().all(|s| !s.running()));
}

#[cfg(unix)]
mod servers {
    use super::*;
    use crate::dashboard::Servers;
    use crate::tui::config::Config;
    use crate::{fake, preset};

    /// A context with the fake llama.cpp, and a model whose preset listens on a free port.
    pub fn served(name: &str, alias: &str) -> (Ctx, std::path::PathBuf) {
        let mut ctx = ctx();
        ctx.hw.inst = Some(fake::install());
        let m = model(name);
        let mut p = preset::defaults(&m);
        p.insert("port".into(), free_port());
        if !alias.is_empty() {
            p.insert("alias".into(), alias.into());
        }
        preset::put_named(&mut ctx.state, &m, crate::ui::DEFAULT_PRESET, &p);
        (ctx, m)
    }

    #[test]
    fn dashboard_prompt_filters_background_and_stop() {
        let (ctx, m) = served("Served-Q4_K_M.gguf", "");
        let mut h = Harness::new(Box::new(Config::new(&ctx, &m, None)), ctx);
        h.key(KeyCode::Char('a'));
        assert_eq!(h.depth(), 2);
        h.tick_until(|h| h.app.ctx.servers.first().is_some_and(|s| s.dash.m.health == "ok"));
        h.tick_until(|h| h.app.ctx.servers[0].dash.n_slots == 2); // read from the log
        h.key(KeyCode::Char('t'));
        assert!(h.modal_open());
        h.typed("ciao").key(KeyCode::Enter);
        h.tick_until(|h| h.draw().contains("modello finto"));
        h.key(KeyCode::Esc);
        for k in ['f', 'f', 'p', 'p'] {
            h.key(KeyCode::Char(k));
        }
        h.key(KeyCode::Char('/')).typed("listening").key(KeyCode::Enter);
        assert_eq!(h.app.ctx.servers[0].dash.search, "listening");
        h.key(KeyCode::Char('/')).ctrl('u').key(KeyCode::Enter);
        for k in [KeyCode::Up, KeyCode::Down, KeyCode::PageUp, KeyCode::PageDown, KeyCode::Home, KeyCode::End] {
            h.key(k);
        }
        h.wheel(true, 10, 10).wheel(false, 10, 10);
        h.key(KeyCode::Char('c'));
        assert!(h.toasts().iter().any(|t| t.contains("copiato")));
        h.key(KeyCode::Char('o'));
        assert!(h.toasts().iter().any(|t| t.contains("browser")));
        let s = h.draw();
        assert!(s.contains("Served"), "{s}");
        h.key(KeyCode::Char('b')); // background: back, the server keeps running
        assert_eq!(h.depth(), 1);
        assert!(h.app.ctx.servers[0].running());
        // the list of the running servers
        h.app.apply(Act::Push(Box::new(Servers::new())));
        assert!(h.draw().contains("Served"));
        h.key(KeyCode::Char('l')).key(KeyCode::Char('l')); // kept after exit, then not
        assert!(!h.app.ctx.servers[0].keep);
        h.key(KeyCode::Enter); // its dashboard again
        assert_eq!(h.depth(), 3);
        h.key(KeyCode::Esc);
        assert!(h.modal_open(), "leaving a running server asks first");
        h.key(KeyCode::Char('s'));
        h.tick_until(|h| h.depth() == 2);
        assert!(h.app.ctx.servers.iter().all(|s| !s.running()));
        h.key(KeyCode::Char('x')); // drop the ended ones
        assert!(h.app.ctx.servers.is_empty());
        stop_all(&mut h);
    }

    #[test]
    fn a_crash_at_start_shows_its_end() {
        let (ctx, m) = served("Crash-Q4_K_M.gguf", "crash-test");
        let mut h = Harness::new(Box::new(Config::new(&ctx, &m, None)), ctx);
        h.key(KeyCode::Char('a'));
        h.tick_until(|h| h.app.ctx.servers.first().is_some_and(|s| !s.running()));
        h.tick_until(|h| h.app.ctx.servers[0].dash.exit_code.is_some());
        assert_eq!(h.app.ctx.servers[0].dash.exit_code, Some(1));
        h.draw();
        h.key(KeyCode::Char('m')); // ended: back to the first screen
        assert_eq!(h.depth(), 1);
        stop_all(&mut h);
    }

    #[test]
    fn stop_from_the_list_twice_kills() {
        let (ctx, m) = served("Killed-Q4_K_M.gguf", "");
        let mut h = Harness::new(Box::new(Servers::new()), ctx);
        let launch =
            crate::server::model_launch(&fake::install(), "b9999", &m, &preset::get_preset(&h.app.ctx.state, &m));
        let s = crate::server::Server::start(&launch, vec![]).unwrap();
        h.app.ctx.servers.push(s);
        h.tick_until(|h| h.app.ctx.servers[0].dash.m.health == "ok");
        assert!(h.draw().contains("Killed"));
        h.key(KeyCode::Char('s')).key(KeyCode::Char('s'));
        h.tick_until(|h| !h.app.ctx.servers[0].running());
        stop_all(&mut h);
    }
}

#[cfg(unix)]
mod multi_model {
    use super::*;
    use crate::tui::router::Profiles;

    #[test]
    fn profile_edit_export_launch_and_router_dashboard() {
        let (ctx, _m) = super::servers::served("Routed-Q4_K_M.gguf", "");
        let mut h = Harness::new(Box::new(Profiles::new()), ctx);
        h.key(KeyCode::Char('n')).typed("squadra").key(KeyCode::Enter);
        assert!(h.app.ctx.state.routers.contains_key("squadra"), "{:?}", h.toasts());
        if h.depth() == 1 {
            h.key(KeyCode::Enter);
        }
        assert_eq!(h.title(), "squadra");
        h.key(KeyCode::Char('n'));
        assert!(h.modal_open());
        h.click_on("Routed-Q4_K_M.gguf").key(KeyCode::Enter);
        assert_eq!(h.app.ctx.state.routers["squadra"].models.len(), 1, "{}", h.draw());
        // the model's actions: its preset, its id, when it loads
        h.key(KeyCode::Home).key(KeyCode::Enter);
        if h.modal_open() {
            h.key(KeyCode::Esc);
        }
        // the router's options, their help, the INI
        h.key(KeyCode::Tab).key(KeyCode::Down).key(KeyCode::Char('?'));
        assert!(h.modal_open());
        h.key(KeyCode::Enter);
        h.key(KeyCode::Down).key(KeyCode::Down).key(KeyCode::Down).key(KeyCode::Enter); // models_max
        h.ctrl('u').typed("2").key(KeyCode::Enter);
        h.key(KeyCode::Char('v'));
        assert!(h.draw().contains("[Routed"), "{}", h.draw());
        h.key(KeyCode::Enter);
        // exports for the clients
        h.key(KeyCode::Char('e'));
        for k in [KeyCode::Tab, KeyCode::Right, KeyCode::Left, KeyCode::BackTab, KeyCode::Down, KeyCode::Up] {
            h.key(k);
        }
        h.key(KeyCode::PageDown).key(KeyCode::PageUp).key(KeyCode::Char('c')).key(KeyCode::Char('w'));
        h.key(KeyCode::Char('t')); // nothing running: says so
        h.key(KeyCode::Esc);
        assert_eq!(h.title(), "squadra");
        // start it: the router's dashboard with its models
        h.key(KeyCode::Char('a'));
        h.tick_until(|h| h.app.ctx.servers.first().is_some_and(|s| s.dash.m.health == "ok"));
        h.tick_until(|h| h.draw().contains("beta"));
        h.key(KeyCode::Down).key(KeyCode::Enter); // load / unload a model
        h.key(KeyCode::Char('t')).typed("ciao").key(KeyCode::Enter);
        h.tick_until(|h| h.draw().contains("modello finto"));
        h.key(KeyCode::Esc);
        h.key(KeyCode::Char('f')).key(KeyCode::Char('p')).key(KeyCode::Char('p'));
        h.key(KeyCode::Char('/')).typed("x").key(KeyCode::Enter);
        h.key(KeyCode::Char('c')).key(KeyCode::Char('o')).key(KeyCode::Char('e'));
        h.key(KeyCode::Esc);
        h.key(KeyCode::Char('s'));
        h.tick_until(|h| !h.app.ctx.servers[0].running());
        h.key(KeyCode::Char('m'));
        assert_eq!(h.depth(), 1);
        // rename and delete the profile
        h.key(KeyCode::Char('r')).ctrl('u').typed("gruppo").key(KeyCode::Enter);
        assert!(h.app.ctx.state.routers.contains_key("gruppo"));
        h.key(KeyCode::Char('x')).key(KeyCode::Char('s'));
        assert!(h.app.ctx.state.routers.is_empty());
        stop_all(&mut h);
    }

    #[test]
    fn agent_recipes_make_profiles() {
        let (ctx, _m) = super::servers::served("Agent-Q4_K_M.gguf", "");
        let mut h = Harness::new(Box::new(crate::tui::agent::Agents::new()), ctx);
        assert!(h.draw().contains("ricette"));
        for _ in 0..4 {
            h.key(KeyCode::Down);
            h.draw();
        }
        h.key(KeyCode::Home).key(KeyCode::Enter);
        // one question per role: a model for each
        for _ in 0..6 {
            if !h.modal_open() {
                break;
            }
            h.key(KeyCode::Enter);
        }
        assert!(!h.app.ctx.state.routers.is_empty(), "{:?} {}", h.toasts(), h.draw());
        h.key(KeyCode::Char('e'));
        assert!(h.title().contains("esporta") || h.depth() == 2, "{}", h.title());
        for _ in 0..4 {
            if h.depth() > 1 || h.modal_open() {
                h.key(KeyCode::Esc);
            }
        }
        h.key(KeyCode::Char('p'));
        assert_eq!(h.depth(), 2);
        stop_all(&mut h);
    }
}

mod small_screens {
    use super::*;

    #[test]
    fn logs_list_and_viewer() {
        let dir = crate::ui::logs();
        std::fs::create_dir_all(&dir).unwrap();
        let lines: String = (0..200).map(|i| format!("0.00.{i:03}.000 I srv  line {i}\n")).collect();
        std::fs::write(dir.join("llama-test-viewer.log"), lines).unwrap();
        let mut h = Harness::new(Box::new(crate::tui::logs::Logs::new()), ctx());
        h.click_on("llama-test-viewer.log");
        if h.depth() == 1 {
            h.click_on("llama-test-viewer.log");
        }
        assert_eq!(h.depth(), 2, "{}", h.draw());
        assert!(h.draw().contains("line 199"));
        for k in [KeyCode::PageUp, KeyCode::Up, KeyCode::Home, KeyCode::Down, KeyCode::PageDown, KeyCode::End] {
            h.key(k);
        }
        h.wheel(false, 5, 5).wheel(true, 5, 5);
        h.key(KeyCode::Esc);
        assert_eq!(h.depth(), 1);
    }

    #[test]
    fn news_sources_feeds_and_actions() {
        let mut c = ctx();
        let screen = crate::tui::news::News::new(&mut c);
        let mut h = Harness::new(Box::new(screen), c);
        h.tick_until(|h| h.draw().contains("A new model & more"));
        for _ in 0..5 {
            h.key(KeyCode::Tab);
        }
        h.key(KeyCode::BackTab);
        h.key(KeyCode::Enter).key(KeyCode::Char('o'));
        h.key(KeyCode::Char('f'));
        assert!(h.modal_open());
        h.key(KeyCode::Esc);
        h.key(KeyCode::Char('r')).key(KeyCode::Tab); // all the sources again
        h.tick_until(|h| h.draw().contains("A new model & more"));
        // a llama.cpp build: install it; a new model: its files
        h.key(KeyCode::Tab);
        h.tick_until(|h| h.draw().contains("b9999 · server: faster slots"));
        h.key(KeyCode::Char('i'));
        assert_eq!(h.depth(), 2);
        h.key(KeyCode::Esc);
        assert_eq!(h.depth(), 1);
        h.key(KeyCode::Tab);
        h.tick_until(|h| h.draw().contains("Qwen3-8B-GGUF"));
        h.key(KeyCode::Char('d'));
        assert_eq!(h.depth(), 2);
    }

    #[test]
    fn splash_detects_then_leaves() {
        let mut h = Harness::new(Box::new(crate::tui::splash::Splash::new(false)), ctx());
        assert!(h.draw().contains("CORRAL") || h.draw().contains("Corral") || !h.draw().is_empty());
        h.tick_until(|h| h.depth() == 0);
        // -y with no model used yet: says so, then the menu
        let mut h = Harness::new(Box::new(crate::tui::splash::Splash::new(true)), ctx());
        h.tick_until(|h| h.depth() == 0);
        assert!(h.toasts().iter().any(|t| t.contains("nessun ultimo modello")));
        let mut h = Harness::new(Box::new(crate::tui::splash::Splash::new(true)), ctx());
        h.key(KeyCode::Char('x')).key(KeyCode::Enter);
        h.sized(60, 20).draw();
    }
}

#[cfg(unix)]
mod with_fake_llama_cpp {
    use super::*;
    use crate::tui::config::Config;
    use crate::tune::Tune;
    use crate::{bench, preset};

    #[test]
    fn three_tiers_sized_measured_saved_and_launched() {
        let (ctx, m) = super::servers::served("Tiers-Q8_0.gguf", "");
        let mut h = Harness::new(Box::new(Tune::new(&ctx, &m)), ctx);
        h.tick_until(|h| h.draw().matches("✓ pronto").count() == 3);
        let s = h.draw();
        assert!(s.contains("36 di 37 layer"), "lite keeps a quarter free:\n{s}");
        assert!(s.contains("ctx    32.768"), "normal: 32k, whole in VRAM with q8_0:\n{s}");
        assert!(s.contains("ctx    37.376"), "max: the longest that fits:\n{s}");
        assert!(s.contains("3000.5 t/s") && s.contains("45.5 t/s"));
        h.key(KeyCode::Left).key(KeyCode::Right).key(KeyCode::Tab);
        h.click_on(" normal ");
        h.key(KeyCode::Enter);
        assert!(h.toasts().iter().any(|t| t.contains("preset lite, normal, max salvati · in uso: normal")));
        let mut names = preset::preset_names(&h.app.ctx.state, &m);
        names.sort();
        assert_eq!(names, ["default", "lite", "max", "normal"]);
        assert_eq!(preset::get_named(&h.app.ctx.state, &m, "normal")["ctx"], "32768");
        assert!(bench::history().iter().filter(|r| r.model == ps(&m)).count() >= 3);
        // again: replacing them asks first; then start the selected one
        h.key(KeyCode::Char('r'));
        assert!(h.draw().contains("annulla"), "running again");
        h.key(KeyCode::Enter); // still running: wait
        h.tick_until(|h| h.draw().matches("✓ pronto").count() == 3);
        h.key(KeyCode::Enter).key(KeyCode::Char('n'));
        h.key(KeyCode::Right).key(KeyCode::Char('a'));
        assert!(h.modal_open());
        h.key(KeyCode::Char('s'));
        assert_eq!(h.depth(), 2, "the dashboard");
        h.tick_until(|h| h.app.ctx.servers.first().is_some_and(|s| s.dash.m.health == "ok"));
        assert_eq!(preset::preset_name(&h.app.ctx.state, &m), "max");
        stop_all(&mut h);
    }

    #[test]
    fn a_model_that_fails_to_load() {
        let (ctx, _) = super::servers::served("broken-Q8_0.gguf", "");
        let m = model("broken-Q8_0.gguf");
        let mut h = Harness::new(Box::new(Tune::new(&ctx, &m)), ctx);
        h.tick_until(|h| h.draw().contains("salva i preset"));
        assert_eq!(h.draw().matches("✖").count(), 3, "{}", h.draw());
        h.key(KeyCode::Char('a'));
        assert!(h.toasts().iter().any(|t| t.contains("non è stato verificato")));
    }

    #[test]
    fn benchmark_and_its_history() {
        let (ctx, m) = super::servers::served("Bench-Q4_K_M.gguf", "");
        let p = preset::get_preset(&ctx.state, &m);
        let mut h = Harness::new(Box::new(bench::Bench::new(&ctx, &m, &p)), ctx);
        h.tick_until(|h| h.draw().contains("✓ pp512 3000.5 tok/s"));
        h.key(KeyCode::Char('r'));
        h.tick_until(|h| h.draw().contains("✓ pp512") && !h.draw().contains("in corso"));
        h.key(KeyCode::Down).key(KeyCode::Up);
        assert!(bench::history().iter().filter(|r| r.model == ps(&m)).count() >= 2);
        // no llama-bench result: the reason, and where the log is
        let (ctx, _) = super::servers::served("nobench-Q4_K_M.gguf", "");
        let m = model("nobench-Q4_K_M.gguf");
        let p = preset::get_preset(&ctx.state, &m);
        let mut h = Harness::new(Box::new(bench::Bench::new(&ctx, &m, &p)), ctx);
        h.tick_until(|h| h.draw().contains('✖'));
        // no install at all
        let mut c = super::ctx();
        c.hw.inst = None;
        let mut h = Harness::new(Box::new(bench::Bench::new(&c, &m, &p)), c);
        assert!(h.draw().contains("nessuna installazione"));
    }

    /// Open the editor of the row labelled `label`: scrolled into view, then clicked (the first
    /// click may only select it).
    fn open_row(h: &mut Harness, label: &str) {
        if h.find(label).is_none() {
            h.key(KeyCode::Home);
        }
        for _ in 0..80 {
            if h.find(label).is_some() {
                break;
            }
            h.key(KeyCode::Down);
        }
        h.click_on(label);
        if !h.modal_open() && !h.draw().contains("✎ invio conferma") {
            h.click_on(label);
        }
    }

    #[test]
    fn every_kind_of_field() {
        let (ctx, _) = super::servers::served("Fields-Q4_K_M.gguf", "");
        let moe = crate::hub::models_dir().join("test__moe").join("Moe-30B-A3B-Q4_K_M.gguf");
        std::fs::create_dir_all(moe.parent().unwrap()).unwrap();
        std::fs::write(&moe, crate::fake::gguf(48, 128, 262144, 100)).unwrap();
        std::fs::write(moe.with_file_name("mmproj-Moe-F16.gguf"), crate::fake::gguf(1, 0, 0, 10)).unwrap();
        let draft = model("draft-0.6B-lora-Q8_0.gguf");
        let mut h = Harness::new(Box::new(Config::new(&ctx, &moe, None)), ctx);
        let val = |h: &Harness, k: &str| preset::get_preset(&h.app.ctx.state, &moe)[k].clone();
        assert!(h.draw().contains("MoE 128 esperti"));
        // free text: a bad value, then "-" for the default
        open_row(&mut h, "contesto (0 = dal modello)");
        h.ctrl('u').typed("tanto").key(KeyCode::Enter);
        assert!(h.toasts().iter().any(|t| t.contains("valore non valido")));
        h.ctrl('u').typed("-").key(KeyCode::Enter);
        open_row(&mut h, "contesto (0 = dal modello)");
        h.ctrl('u').typed("16384").key(KeyCode::Enter);
        assert_eq!(val(&h, "ctx"), "16384");
        // choice lists: a digit picks the row at once
        open_row(&mut h, "KV cache K");
        h.typed("3");
        assert_eq!(val(&h, "ctk"), "q8_0");
        // the GPUs: none, several by hand, the one there is
        open_row(&mut h, "dispositivi");
        h.typed("3");
        assert_eq!(val(&h, "device"), "none");
        open_row(&mut h, "dispositivi");
        h.typed("4").ctrl('u').typed("Vulkan0, Vulkan1").key(KeyCode::Enter);
        assert_eq!(val(&h, "device"), "Vulkan0,Vulkan1");
        open_row(&mut h, "dispositivi");
        h.typed("1");
        assert_eq!(val(&h, "device"), "Vulkan0");
        open_row(&mut h, "dispositivi");
        h.typed("2");
        // MoE: tensor overrides from the shortcuts, or written by hand
        open_row(&mut h, "override tensori");
        h.typed("3");
        assert!(val(&h, "ot").contains("[13579]"));
        open_row(&mut h, "override tensori");
        h.typed("5").ctrl('u').typed(r"blk\.1\.ffn_.*=CPU").key(KeyCode::Enter);
        assert_eq!(val(&h, "ot"), r"blk\.1\.ffn_.*=CPU");
        open_row(&mut h, "override tensori");
        h.typed("4");
        // the mmproj next to the model; the draft and the LoRA by path
        open_row(&mut h, "mmproj (vision)");
        h.typed("2");
        assert_eq!(val(&h, "mmproj"), "");
        open_row(&mut h, "mmproj (vision)");
        h.typed("1");
        open_row(&mut h, "modello draft");
        h.key(KeyCode::End).key(KeyCode::Enter).ctrl('u').typed("/nonexistent/draft.gguf").key(KeyCode::Enter);
        assert!(h.toasts().iter().any(|t| t.contains("non esiste")));
        open_row(&mut h, "modello draft");
        h.key(KeyCode::End).key(KeyCode::Enter).ctrl('u').typed(&ps(&draft)).key(KeyCode::Enter);
        assert_eq!(val(&h, "draft"), ps(&draft));
        // other hosts' GPUs, by hand (the last rows: other tests may add hosts to the ssh config)
        open_row(&mut h, "GPU di altri host");
        h.key(KeyCode::End).key(KeyCode::Enter).ctrl('u').typed("gpu1,gpu2").key(KeyCode::Enter);
        assert_eq!(val(&h, "rpc"), "gpu1,gpu2");
        open_row(&mut h, "GPU di altri host");
        h.key(KeyCode::End).key(KeyCode::Up).key(KeyCode::Enter);
        assert_eq!(val(&h, "rpc"), "");
        // the advanced ones, the LoRA among them
        h.key(KeyCode::Char('t'));
        open_row(&mut h, "adattatore LoRA");
        h.key(KeyCode::End).key(KeyCode::Enter).ctrl('u').typed(&ps(&draft)).key(KeyCode::Enter);
        open_row(&mut h, "caricamento modello");
        h.typed("4");
        h.key(KeyCode::Char('t'));
        assert!(h.draw().contains("parametri avanzati"), "hidden again: the toggle row is selected");
        // help, estimate (applied), defaults
        h.key(KeyCode::Home).key(KeyCode::Char('?'));
        assert!(h.modal_open());
        h.key(KeyCode::Enter);
        h.key(KeyCode::Char('v'));
        h.tick_until(|h| h.modal_open());
        assert!(h.draw().contains("applico questi valori"), "{}", h.draw());
        h.key(KeyCode::Char('s'));
        assert!(h.toasts().iter().any(|t| t.contains("llama-fit-params applicati")));
        h.key(KeyCode::Char('d')).key(KeyCode::Char('s'));
        assert!(h.toasts().iter().any(|t| t.contains("riportato ai default")));
        // the tools it opens
        h.key(KeyCode::Char('o'));
        assert!(h.title().starts_with("ottimizza"));
        h.key(KeyCode::Esc).key(KeyCode::Char('b'));
        assert_eq!(h.title(), "benchmark");
        h.key(KeyCode::Esc);
        assert!(h.title().starts_with("configura"));
    }

    #[test]
    fn crashed_after_start_comes_back() {
        let (ctx, m) = super::servers::served("Restart-Q4_K_M.gguf", "flaky");
        let mut h = Harness::new(Box::new(Config::new(&ctx, &m, None)), ctx);
        h.key(KeyCode::Char('a'));
        h.tick_until(|h| h.app.ctx.servers.first().is_some_and(|s| s.dash.m.health == "ok"));
        h.tick_until(|h| h.toasts().iter().any(|t| t.contains("lo riavvio")));
        stop_all(&mut h);
    }
}

#[cfg(unix)]
mod commands {
    use std::process::ExitCode;

    use super::*;
    use crate::{cli, fake, installs, preset};

    fn run(args: &[&str]) -> Option<ExitCode> {
        let args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
        cli::run(&args)
    }

    /// `corral start` leaves its server running on purpose: this stops it even if the test fails.
    struct StopCli;

    impl Drop for StopCli {
        fn drop(&mut self) {
            for r in crate::server::running_now().into_iter().filter(|r| r.info.alias == "Cli") {
                crate::server::stop_pid(r.pid, Duration::from_secs(5));
            }
        }
    }

    #[test]
    fn start_status_stop_models_and_services() {
        let _disk = fake::disk();
        let _servers = fake::shared_servers();
        let _stop = StopCli;
        let inst = installs::import_dir(&fake::llama_cpp()).unwrap();
        let m = model("Cli-Q4_K_M.gguf");
        let mut state = crate::ui::State { install: Some(inst.name.clone()), ..Default::default() };
        let mut p = preset::defaults(&m);
        p.insert("port".into(), free_port());
        preset::put_named(&mut state, &m, "default", &p);
        preset::put_named(&mut state, &m, "lungo", &p);
        crate::ui::save_state(&state).unwrap();
        let ok = Some(ExitCode::SUCCESS);
        assert_eq!(run(&["status"]), ok);
        assert_eq!(run(&["status", "--json"]), ok);
        assert_eq!(run(&["models"]), ok);
        assert_eq!(run(&["models", "--json"]), ok);
        assert_eq!(run(&["start"]), Some(ExitCode::from(2)), "no model: the usage");
        assert_eq!(run(&["start", "nessuno-cosi"]), Some(ExitCode::FAILURE));
        assert_eq!(run(&["start", "Cli-Q4", "--preset", "manca"]), Some(ExitCode::FAILURE));
        assert_eq!(run(&["start", "--profile", "manca"]), Some(ExitCode::FAILURE));
        assert_eq!(run(&["start", "Cli-Q4", "--preset", "lungo"]), ok);
        let mine = || crate::server::running_now().into_iter().filter(|r| r.info.alias == "Cli").count();
        assert_eq!(mine(), 1, "other tests may run servers of their own");
        assert_eq!(run(&["status"]), ok);
        assert_eq!(run(&["status", "--json"]), ok);
        assert_eq!(run(&["stop", "nessuno"]), Some(ExitCode::FAILURE));
        assert_eq!(run(&["stop", "Cli"]), ok);
        assert_eq!(mine(), 0);
        // services: the unit file in this home, systemctl left alone
        assert_eq!(run(&["service", "install", "Cli-Q4"]), ok);
        assert_eq!(run(&["service", "list"]), ok);
        assert_eq!(run(&["service", "remove", "cli"]), ok);
        assert_eq!(run(&["service"]), Some(ExitCode::from(2)));
        assert_eq!(run(&["service", "boh"]), Some(ExitCode::from(2)));
        assert_eq!(run(&["--bogus-flag"]), None, "not a command: the interface's own options");
        let _ = installs::remove(&inst);
    }
}

#[cfg(unix)]
mod remote_hosts {
    use super::*;
    use crate::tui::hosts::Hosts;
    use crate::{fake, remote};

    /// The servers a test started on a fake host stop with the test, even when it fails.
    struct StopHost(&'static str);

    impl Drop for StopHost {
        fn drop(&mut self) {
            if let Ok(s) = remote::status(self.0) {
                for r in s.servers {
                    let _ = remote::stop(self.0, r.pid);
                }
            }
        }
    }

    #[test]
    fn status_models_start_stop_and_copy() {
        fake::host("box1");
        let _stop = StopHost("box1");
        let s = remote::status("box1").unwrap();
        assert_eq!(s.version, crate::remote::VERSION);
        assert!(s.servers.is_empty());
        let m = remote::models("box1").unwrap();
        assert_eq!(m.models.len(), 1);
        assert!(m.models[0].path.ends_with("Remote-Q4_K_M.gguf"));
        assert_eq!(m.profiles[0].name, "lavoro");
        let line = remote::start("box1", &["Remote".to_string()]).unwrap();
        assert!(line.contains("http://127.0.0.1:"), "{line}");
        let running = remote::status("box1").unwrap().servers;
        assert_eq!(running.len(), 1);
        assert_eq!(running[0].health, "ok");
        // its tunnel and its log, as the dashboard gets them
        let link = remote::link("box1", &running[0]).unwrap();
        let base = format!("http://127.0.0.1:{}", link.port);
        assert_eq!(crate::server::health_of(&base, ""), "ok");
        drop(link);
        remote::stop("box1", running[0].pid).unwrap();
        assert!(remote::status("box1").unwrap().servers.is_empty());
        let (os, arch) = remote::platform("box1").unwrap();
        assert_eq!((os.as_str(), arch.as_str()), (std::env::consts::OS, std::env::consts::ARCH));
        remote::copy_corral("box1").unwrap();
        // its GPUs, lent over RPC
        let rpc = remote::rpc_link("box1", 51200).unwrap();
        assert_eq!(rpc.host, "box1");
        drop(rpc);
    }

    #[test]
    fn hosts_that_cant_answer() {
        fake::host("nocorral-box");
        assert!(matches!(remote::status("nocorral-box"), Err(remote::Problem::Missing)));
        assert!(matches!(remote::status("down-box"), Err(remote::Problem::Ssh(_))));
        assert!(remote::copy_corral("down-box").is_err());
        assert!(remote::rpc_link("down-box", 51300).is_err());
        assert!(remote::rpc_links(&["down-box".to_string()]).is_err());
    }

    #[test]
    fn the_hosts_screen() {
        fake::host("box2");
        fake::host("nocorral-box2");
        let _stop = StopHost("box2");
        let ssh = crate::ui::home().join(".ssh");
        std::fs::create_dir_all(&ssh).unwrap();
        std::fs::write(ssh.join("config"), "Host box2\nHost down-box2\nHost nocorral-box2\nHost *\n  User x\n")
            .unwrap();
        let mut h = Harness::new(Box::new(Hosts::new()), ctx());
        assert!(h.draw().contains("down-box2"));
        // one that can't be reached, one without Corral
        h.click_on("down-box2").key(KeyCode::Enter);
        h.tick_until(|h| h.draw().contains("Connection refused"));
        h.key(KeyCode::Esc);
        h.click_on("nocorral-box2").key(KeyCode::Enter);
        h.tick_until(|h| h.draw().contains("non trovato") || h.draw().contains("Corral"));
        h.key(KeyCode::Char('u'));
        h.tick_until(|h| h.toasts().iter().any(|t| t.contains("copiato") || t.contains('✓') || t.contains('✖')));
        h.key(KeyCode::Esc);
        // the one with everything: its models and profiles, a server started there and attached here
        h.click_on("box2").key(KeyCode::Enter);
        assert_eq!(h.depth(), 2);
        h.tick_until(|h| h.draw().contains("Remote-Q4_K_M.gguf"));
        assert!(h.draw().contains("lavoro"));
        h.click_on("Remote-Q4_K_M.gguf");
        if !h.modal_open() && h.app.ctx.servers.is_empty() {
            h.click_on("Remote-Q4_K_M.gguf");
        }
        if h.modal_open() {
            h.key(KeyCode::Enter); // its preset
        }
        h.tick_until(|h| h.app.ctx.servers.first().is_some_and(|s| s.dash.m.health == "ok"));
        assert_eq!(h.app.ctx.servers[0].host(), Some("box2"));
        h.key(KeyCode::Char('t')).typed("ciao").key(KeyCode::Enter);
        h.tick_until(|h| h.draw().contains("modello finto"));
        h.key(KeyCode::Esc);
        h.key(KeyCode::Esc); // a host's server: back, it keeps running there
        if h.modal_open() {
            h.key(KeyCode::Char('n'));
        }
        h.tick_until(|h| h.depth() == 2 && !h.modal_open());
        h.key(KeyCode::Char('r'));
        h.tick_until(|h| h.draw().contains("qui su") || h.draw().contains("127.0.0.1"));
        h.key(KeyCode::Char('o'));
        assert!(h.app.ctx.foreground.take().is_some(), "o opens the host's Corral in this terminal");
        h.key(KeyCode::Char('s'));
        if h.modal_open() {
            h.key(KeyCode::Char('s'));
        }
        h.tick_until(|h| h.toasts().iter().any(|t| t.contains("fermato") || t.contains('✓')) || !h.modal_open());
        stop_all(&mut h);
    }
}

#[cfg(unix)]
mod more_flows {
    use super::*;
    use crate::tui::config::Config;
    use crate::{agent, fake, models, preset, server};

    #[test]
    fn model_tests_of_the_export() {
        let _servers = fake::shared_servers(); // its server starts before the harness
        let (ctx, m) = super::servers::served("Probe-Q4_K_M.gguf", "");
        let p = preset::get_preset(&ctx.state, &m);
        let launch = server::model_launch(&fake::install(), "b9999", &m, &p);
        let s = server::Server::start(&launch, vec![]).unwrap();
        let base = s.base.clone();
        let mut h = Harness::new(Box::new(Config::new(&ctx, &m, None)), ctx);
        h.app.ctx.servers.push(s);
        h.tick_until(|h| h.app.ctx.servers[0].dash.m.health == "ok");
        let served = |usage: &str| agent::Served {
            id: "x".into(),
            file: String::new(),
            ctx: 4096,
            vision: false,
            usage: usage.into(),
        };
        let chat = agent::test(&base, "", &served("chat")).unwrap();
        assert!(chat.contains("get_weather"), "{chat}");
        assert!(agent::test(&base, "", &served("embedding")).unwrap().contains('3'));
        assert!(!agent::test(&base, "", &served("rerank")).unwrap().is_empty());
        assert!(agent::test("http://127.0.0.1:1", "", &served("chat")).is_err());
        stop_all(&mut h);
    }

    #[test]
    fn a_preset_with_other_hosts_gpus() {
        fake::host("gpu-box");
        let (mut ctx, m) = super::servers::served("Lent-Q4_K_M.gguf", "");
        let mut p = preset::get_preset(&ctx.state, &m);
        p.insert("rpc".into(), "gpu-box".into());
        preset::put_named(&mut ctx.state, &m, "default", &p);
        let mut h = Harness::new(Box::new(Config::new(&ctx, &m, None)), ctx);
        h.key(KeyCode::Char('a'));
        assert_eq!(h.title(), "GPU degli host");
        assert!(h.draw().contains("collego le GPU di gpu-box"));
        h.tick_until(|h| h.app.ctx.servers.first().is_some_and(|s| s.dash.m.health == "ok"));
        assert!(h.app.ctx.servers[0].uses_rpc());
        h.app.apply(Act::Push(Box::new(crate::dashboard::Servers::new())));
        h.key(KeyCode::Char('l')); // can't be kept: it lives with the links
        assert!(!h.app.ctx.servers[0].keep);
        stop_all(&mut h);
        // a host that can't be reached: back to the editor, with why
        let (mut ctx, m) = super::servers::served("Unlent-Q4_K_M.gguf", "");
        let mut p = preset::get_preset(&ctx.state, &m);
        p.insert("rpc".into(), "down-gpu".into());
        preset::put_named(&mut ctx.state, &m, "default", &p);
        let mut h = Harness::new(Box::new(Config::new(&ctx, &m, None)), ctx);
        h.key(KeyCode::Char('a'));
        h.tick_until(|h| h.depth() == 1);
        assert!(h.toasts().iter().any(|t| t.contains("Connection refused")), "{:?}", h.toasts());
    }

    #[test]
    fn a_source_build_that_fails() {
        let mut h = Harness::new(Box::new(crate::tui::installs::Installs::new()), ctx());
        h.key(KeyCode::Char('b'));
        h.key(KeyCode::Home).key(KeyCode::Enter);
        for _ in 0..4 {
            if !h.modal_open() {
                break;
            }
            if h.draw().contains("nome installazione") {
                h.ctrl('u').typed("src-test-fails");
            }
            if h.draw().contains("Sì (s)") {
                h.key(KeyCode::Char('s'));
            } else {
                h.key(KeyCode::Enter);
            }
        }
        h.tick_until(|h| h.draw().contains("✖ git fetch"));
        h.key(KeyCode::Esc);
        let _ = crate::installs::all_installs()
            .into_iter()
            .filter(|i| i.name == "src-test-fails")
            .map(|i| crate::installs::remove(&i));
    }

    #[test]
    fn router_editor_model_actions() {
        let (ctx, m) = super::servers::served("Actions-Q4_K_M.gguf", "");
        let mut ctx = ctx;
        let current = preset::get_preset(&ctx.state, &m);
        preset::put_named(&mut ctx.state, &m, "lungo", &current);
        let entry = crate::router::Entry { id: "actions".into(), model: ps(&m), preset: String::new(), startup: false };
        ctx.state.routers.insert("azioni".into(), crate::router::Profile { models: vec![entry], ..Default::default() });
        let mut h = Harness::new(Box::new(crate::tui::router::Editor::new("azioni")), ctx);
        let open = |h: &mut Harness| {
            h.key(KeyCode::Home).key(KeyCode::Enter);
            assert!(h.modal_open());
        };
        open(&mut h);
        h.typed("2"); // its preset
        if h.modal_open() {
            h.typed("2");
        }
        open(&mut h);
        h.typed("3").ctrl('u').typed("nuovo-id").key(KeyCode::Enter); // its id
        assert_eq!(h.app.ctx.state.routers["azioni"].models[0].id, "nuovo-id", "{:?}", h.toasts());
        open(&mut h);
        h.typed("3").ctrl('u').typed("non valido!").key(KeyCode::Enter);
        assert!(h.toasts().iter().any(|t| t.starts_with('✖')));
        open(&mut h);
        h.typed("4"); // loaded at the start
        assert!(h.app.ctx.state.routers["azioni"].models[0].startup);
        open(&mut h);
        h.typed("1"); // its preset in the editor
        assert!(h.title().starts_with("configura"), "{}", h.title());
        h.key(KeyCode::Esc);
        h.key(KeyCode::Home).key(KeyCode::Char('p'));
        h.key(KeyCode::Esc);
        h.key(KeyCode::Home).key(KeyCode::Char('x'));
        if h.modal_open() {
            h.key(KeyCode::Char('s'));
        }
        assert!(h.app.ctx.state.routers["azioni"].models.is_empty());
        h.key(KeyCode::Char('a')); // no models: says so
        assert!(h.toasts().iter().any(|t| t.starts_with('✖') || t.contains("modell")));
    }

    #[test]
    fn news_feeds_added_and_removed() {
        let mut c = ctx();
        let screen = crate::tui::news::News::new(&mut c);
        let mut h = Harness::new(Box::new(screen), c);
        h.key(KeyCode::Char('f')).key(KeyCode::End);
        h.key(KeyCode::Up).key(KeyCode::Enter); // + add
        if h.modal_open() && h.draw().contains("Annulla (esc)") {
            h.ctrl('u').typed("https://example.net/rss.xml").key(KeyCode::Enter);
        }
        let feeds = h.app.ctx.state.feeds.clone().unwrap_or_default();
        assert!(feeds.iter().any(|f| f.contains("example.net")), "{feeds:?} {:?}", h.toasts());
        h.key(KeyCode::Char('f')).key(KeyCode::Home).key(KeyCode::Enter).key(KeyCode::Char('s'));
        assert_eq!(h.app.ctx.state.feeds.as_ref().map(Vec::len), Some(feeds.len() - 1));
        h.key(KeyCode::Char('f')).key(KeyCode::End).key(KeyCode::Enter); // back to the defaults
        assert_eq!(h.app.ctx.state.feeds, None);
        h.tick_until(|h| h.draw().contains("A new model"));
    }

    #[test]
    fn a_downloaded_model_is_checked_for_updates() {
        // as the download leaves it: the repo's folder, its origin file
        let mut h = Harness::new(Box::new(crate::tui::hf::RepoScreen::new("test/Update-GGUF")), ctx());
        h.tick_until(|h| h.draw().contains("Q4_K_M"));
        h.key(KeyCode::Home).key(KeyCode::Enter).key(KeyCode::Down).key(KeyCode::Enter);
        if h.modal_open() {
            h.key(KeyCode::Char('s'));
        }
        h.tick_until(|h| h.title().starts_with("configura"));
        let model = crate::hub::models_dir().join("test__Update-GGUF").join("Model-Q4_K_M.gguf");
        assert!(matches!(models::check_update(&model).unwrap().1, models::Update::Same));
        std::fs::write(&model, b"changed").unwrap();
        assert!(matches!(models::check_update(&model).unwrap().1, models::Update::Changed(_)));
        let ctx = ctx();
        let mut h = Harness::new(Box::new(crate::tui::models::Models::new(&ctx)), ctx);
        h.select("test/Update-GGUF").key(KeyCode::Char('u'));
        h.tick_until(|h| h.modal_open());
        h.key(KeyCode::Char('s'));
        assert_eq!(h.depth(), 2, "the repo, to download it again");
    }

    #[test]
    fn build_prerequisites() {
        for (backend, _, _) in crate::installs::build_options() {
            let (_ok, what) = crate::installs::prereqs(backend);
            assert!(!what.is_empty() || backend == "cpu" || backend == "custom", "{backend}");
        }
    }
}

#[cfg(unix)]
mod last_flows {
    use std::process::ExitCode;

    use super::*;
    use crate::{cli, fake, installs, models, preset, server};

    fn run(args: &[&str]) -> Option<ExitCode> {
        let args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
        cli::run(&args)
    }

    /// Stop the servers with these aliases, even if the test fails.
    struct Stop(&'static [&'static str]);

    impl Drop for Stop {
        fn drop(&mut self) {
            for r in server::running_now().into_iter().filter(|r| self.0.contains(&r.info.alias.as_str())) {
                server::stop_pid(r.pid, Duration::from_secs(5));
            }
        }
    }

    #[test]
    fn split_models_orphans_and_their_files() {
        let dir = crate::hub::models_dir().join("test__Split-GGUF");
        std::fs::create_dir_all(&dir).unwrap();
        let first = dir.join("Big-Q4_K_M-00001-of-00002.gguf");
        std::fs::write(&first, fake::gguf(36, 0, 40960, 10)).unwrap();
        std::fs::write(dir.join("Big-Q4_K_M-00002-of-00002.gguf"), b"rest").unwrap();
        std::fs::write(dir.join("mmproj-Big-F16.gguf"), b"mm").unwrap();
        assert_eq!(models::parts(&first).len(), 2);
        assert_eq!(preset::model_size(&first), fake::gguf(36, 0, 40960, 10).len() as u64 + 4);
        let mut state = crate::ui::State::default();
        preset::put_named(&mut state, &first, "default", &preset::defaults(&first));
        state.presets.entry("/nonexistent/gone.gguf".into()).or_default().insert("default".into(), Default::default());
        state.last_used.insert("/nonexistent/gone.gguf".into(), "2026".into());
        assert_eq!(models::orphans(&state), ["/nonexistent/gone.gguf"]);
        assert_eq!(models::drop_orphans(&mut state).unwrap(), 1);
        assert!(models::orphans(&state).is_empty());
        let rows = models::scan(&state);
        let plan = models::delete_plan(&state, &first, &rows);
        assert_eq!(plan.len(), 3, "both parts and the mmproj no other model uses: {plan:?}");
        models::delete(&mut state, &first, &plan).unwrap();
        assert!(!first.exists() && !state.presets.contains_key(&ps(&first)));
    }

    #[test]
    fn commands_in_the_foreground_profiles_and_failures() {
        let _disk = fake::disk();
        let _servers = fake::shared_servers();
        let _stop = Stop(&["Fg", "crash-cli", "gruppo"]);
        let inst = installs::import_dir(&fake::llama_cpp()).unwrap();
        let mut state = crate::ui::State { install: Some(inst.name.clone()), ..Default::default() };
        let add = |state: &mut crate::ui::State, name: &str, alias: &str, rpc: &str| {
            let m = model(name);
            let mut p = preset::defaults(&m);
            p.insert("port".into(), free_port());
            p.insert("alias".into(), alias.into());
            p.insert("rpc".into(), rpc.into());
            preset::put_named(state, &m, "default", &p);
            m
        };
        let fg = add(&mut state, "Fg-Q4_K_M.gguf", "Fg", "");
        add(&mut state, "CrashCli-Q4_K_M.gguf", "crash-cli", "");
        add(&mut state, "Lending-Q4_K_M.gguf", "lending", "some-host");
        let entry = crate::router::Entry { id: "fg".into(), model: ps(&fg), preset: String::new(), startup: true };
        let mut profile = crate::router::Profile { models: vec![entry], ..Default::default() };
        profile.opts.insert("port".into(), free_port());
        state.routers.insert("gruppo".into(), profile);
        crate::ui::save_state(&state).unwrap();
        assert_eq!(run(&["start", "CrashCli"]), Some(ExitCode::FAILURE), "it ends before it's ready");
        assert_eq!(run(&["start", "Lending"]), Some(ExitCode::FAILURE), "other hosts' GPUs want --foreground");
        assert_eq!(run(&["start", "Q4_K_M"]), Some(ExitCode::FAILURE), "more than one model matches");
        // in the foreground: until the server ends (here: `corral stop` from another thread)
        let fg_thread = thread::spawn(|| run(&["start", "Fg-Q4", "--foreground"]));
        let started = Instant::now();
        while !server::running_now().iter().any(|r| r.info.alias == "Fg") && started.elapsed().as_secs() < 10 {
            thread::sleep(Duration::from_millis(50));
        }
        let pid = server::running_now().into_iter().find(|r| r.info.alias == "Fg").map(|r| r.pid).unwrap();
        thread::sleep(Duration::from_millis(500));
        assert_eq!(run(&["stop", &pid.to_string()]), Some(ExitCode::SUCCESS));
        assert!(fg_thread.join().unwrap().is_some());
        // a profile: the router, stopped by its port
        assert_eq!(run(&["start", "--profile", "gruppo"]), Some(ExitCode::SUCCESS));
        let r = server::running_now().into_iter().find(|r| r.info.alias == "gruppo").unwrap();
        let port = r.base.rsplit(':').next().unwrap().to_string();
        assert_eq!(run(&["stop", "--all", "gruppo"]), Some(ExitCode::from(2)));
        assert_eq!(run(&["stop"]), Some(ExitCode::from(2)));
        assert_eq!(run(&["stop", &port]), Some(ExitCode::SUCCESS));
        assert_eq!(run(&["watch"]), Some(ExitCode::from(2)));
        assert_eq!(run(&["watch", "x"]), Some(ExitCode::from(2)));
        assert_eq!(run(&["watch", "999999"]), Some(ExitCode::FAILURE));
        assert_eq!(run(&["rpc", "--port", "x"]), Some(ExitCode::from(2)));
        let _ = installs::remove(&inst);
    }

    #[test]
    fn kept_servers_are_adopted_by_the_next_corral() {
        let _alone = fake::all_servers(); // it adopts every server in run/: no other test's may be there
        let _stop = Stop(&["Kept"]);
        let (ctx, m) = super::servers::served("Kept-Q4_K_M.gguf", "Kept");
        let p = preset::get_preset(&ctx.state, &m);
        let launch = server::model_launch(&fake::install(), "b9999", &m, &p);
        let mut s = server::Server::start(&launch, vec![]).unwrap();
        let started = Instant::now();
        while s.dash.m.health != "ok" && started.elapsed().as_secs() < 10 {
            s.pump();
            thread::sleep(Duration::from_millis(50));
        }
        s.set_keep(true);
        drop(s); // Corral quits: the server stays
        let next = Ctx::new(crate::ui::State::default());
        let kept = next.servers.iter().find(|s| s.dash.alias() == "Kept").expect("adopted");
        assert!(kept.keep, "kept again by the next Corral");
        assert!(next.toasts.iter().any(|t| t.msg.contains("ripresi")));
        let pid = kept.pid;
        let mut h = Harness::alone(Box::new(crate::dashboard::Servers::new()), next);
        h.tick();
        let screen = h.draw();
        assert!(screen.contains("Kept") && screen.contains("resta attivo"), "{screen}");
        stop_all_adopted(&mut h, pid);
    }

    /// Stop the adopted server: here (unlike a real next Corral) this process is its parent, so
    /// once it ends it must be reaped, or it stays a zombie that looks alive.
    fn stop_all_adopted(h: &mut Harness, pid: u32) {
        for s in &mut h.app.ctx.servers {
            s.set_keep(false);
            s.stop();
        }
        for _ in 0..200 {
            let reaped = unsafe { libc::waitpid(pid as i32, std::ptr::null_mut(), libc::WNOHANG) };
            if reaped == pid as i32 {
                break;
            }
            thread::sleep(Duration::from_millis(50));
        }
        h.tick_until(|h| h.app.ctx.servers.iter().all(|s| !s.running()));
    }

    #[cfg(target_os = "linux")] // the old install is a Vulkan build: Linux has that variant
    #[test]
    fn updating_a_release_and_a_source_install() {
        let _turn = super::release_turn();
        fake::serve_release();
        // an older release of the same variant, and a source build
        let old = installs::Install {
            name: "rel-b9000-vulkan".into(),
            kind: "release".into(),
            backend: "vulkan".into(),
            bin_dir: ps(&fake::llama_cpp()),
            git_ref: Some("b9000".into()),
            flags: vec![],
            src_dir: None,
        };
        let dir = crate::ui::BASE.join("installs").join(&old.name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("install.json"), serde_json::to_string(&old).unwrap()).unwrap();
        let mut h = Harness::new(Box::new(crate::tui::installs::Installs::new()), ctx());
        h.tick_until(|h| h.draw().contains("↑ b9999"));
        h.select("rel-b9000-vulkan").key(KeyCode::Char('u'));
        assert!(h.modal_open(), "{:?}", h.toasts());
        h.key(KeyCode::Char('s'));
        assert_eq!(h.depth(), 2);
        h.tick_until(|h| h.depth() == 1);
        assert!(h.toasts().iter().any(|t| t.contains("installata rel-b9999-vulkan")), "{:?}", h.toasts());
        h.select("rel-b9000-vulkan").key(KeyCode::Char('x')).key(KeyCode::Char('s'));
        assert!(!dir.exists());
        // a source install rebuilt at another ref (it fails at the fetch: no network in the tests)
        let src = installs::new_source("src-update-test", "cpu", "master", vec![]).unwrap();
        h.app.apply(Act::Push(Box::new(crate::tui::installs::Build::update(src.clone()))));
        h.tick_until(|h| h.draw().contains("✖ git fetch"));
        h.key(KeyCode::Esc);
        let _ = installs::remove(&src);
    }

    #[test]
    fn leaving_the_router_dashboard() {
        let _stop = Stop(&["uscita"]);
        let (mut ctx, m) = super::servers::served("Leave-Q4_K_M.gguf", "");
        let entry = crate::router::Entry { id: "leave".into(), model: ps(&m), preset: String::new(), startup: false };
        let mut profile = crate::router::Profile { models: vec![entry], ..Default::default() };
        profile.opts.insert("port".into(), free_port());
        ctx.state.routers.insert("uscita".into(), profile);
        let mut h = Harness::new(Box::new(crate::tui::router::Profiles::new()), ctx);
        h.key(KeyCode::Char('a'));
        h.tick_until(|h| h.app.ctx.servers.first().is_some_and(|s| s.dash.m.health == "ok"));
        h.key(KeyCode::Char('b')); // in the background
        assert_eq!(h.depth(), 1);
        h.key(KeyCode::Char('r')); // running: can't be renamed
        assert!(h.toasts().iter().any(|t| t.contains("in esecuzione")));
        h.key(KeyCode::Enter).key(KeyCode::Char('a')); // its editor: start = its dashboard again
        h.key(KeyCode::Esc);
        if h.modal_open() {
            h.key(KeyCode::Char('s'));
        }
        h.tick_until(|h| h.app.ctx.servers.iter().all(|s| !s.running()));
        stop_all(&mut h);
    }
}
