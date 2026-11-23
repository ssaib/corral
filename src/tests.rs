//! Self-checks for the parsing/decision logic: `cargo test`.

use std::path::{Path, PathBuf};

use crate::installs::{Asset, Release, recommended, release_variants};
use crate::preset::{base_alias, build_cmd, defaults, free_port, ngl_word};
use crate::{dashboard, hub, preset, ui};

const GIB: u64 = 1 << 30;

#[test]
fn quant_names() {
    // unsloth UD-, mradermacher i1, split files in subfolders
    assert_eq!(hub::quant_of("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf").as_deref(), Some("Q4_K_M"));
    assert_eq!(hub::quant_of("model.i1-IQ4_XS.gguf").as_deref(), Some("IQ4_XS"));
    assert_eq!(hub::quant_of("Qwen3-30B-A3B-UD-Q4_K_XL.gguf").as_deref(), Some("UD-Q4_K_XL"));
    assert_eq!(hub::quant_of("gemma-3-12b-it-f16.gguf").as_deref(), Some("F16"));
    assert_eq!(hub::quant_of("Q8_0/big-Q8_0-00001-of-00002.gguf").as_deref(), Some("Q8_0"));
    assert_eq!(hub::quant_of("Qwen3-8B.gguf"), None);
}

#[test]
fn grouping_and_choice() {
    let files: Vec<(String, u64)> = [
        ("README.md", 1),
        ("m-Q4_K_M.gguf", 4 * GIB),
        ("m-Q8_0.gguf", 8 * GIB),
        ("m-f16.gguf", 15 * GIB),
        ("Q6_K/m-Q6_K-00001-of-00002.gguf", 3 * GIB),
        ("Q6_K/m-Q6_K-00002-of-00002.gguf", 3 * GIB),
        ("mmproj-m-f16.gguf", GIB),
        ("mmproj-m-Q8_0.gguf", GIB / 2),
    ]
    .map(|(f, n)| (f.to_string(), n))
    .into();
    let (quants, mm) = hub::group_files(&files);
    assert_eq!(quants.iter().map(|q| q.quant.as_str()).collect::<Vec<_>>(), ["Q4_K_M", "Q6_K", "Q8_0", "F16"]);
    assert!(quants[1].size == 6 * GIB && quants[1].files.len() == 2);
    assert_eq!(hub::best_mmproj(&mm.iter().map(|m| &m.0).collect::<Vec<_>>()), Some(0)); // f16 preferred
    assert_eq!(hub::recommend(&quants, 10 * GIB, 0), Some(2)); // Q8_0: 8 + 1.5 fits in 10
    assert_eq!(hub::recommend(&quants, 10 * GIB, GIB), Some(1)); // with mmproj only Q6_K fits
    assert_eq!(hub::recommend(&quants, 30 * GIB, 0), Some(2)); // never picks F16 when a quant fits
    assert_eq!(hub::recommend(&quants, 2 * GIB, 0), None);
    assert_eq!(hub::category("unsloth/Qwen2.5-VL-7B-Instruct-GGUF", None), "vision");
    assert_eq!(hub::category("bartowski/Qwen2.5-Coder-14B-GGUF", Some("text-generation")), "coding");
    assert_eq!(hub::category("unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF", None), "reasoning");
    assert_eq!(hub::category("ggml-org/gemma-3-4b-it-GGUF", None), "chat");
    assert_eq!(hub::category("city96/umt5-xxl-encoder-gguf", None), "chat"); // "encoder" is not "coder"
}

fn release(names: &[&str]) -> Release {
    let assets = names
        .iter()
        .map(|n| Asset { name: n.to_string(), size: 1, browser_download_url: String::new(), digest: None })
        .collect();
    Release { tag_name: "b1".into(), published_at: String::new(), assets }
}

#[test]
fn release_choice() {
    // CUDA only if the driver supports that runtime
    let rel = release(&[
        "llama-b1-bin-ubuntu-x64.tar.gz",
        "llama-b1-bin-ubuntu-vulkan-x64.tar.gz",
        "llama-b1-bin-ubuntu-cuda-12.8-x64.tar.gz",
        "llama-b1-bin-ubuntu-cuda-13.5-x64.tar.gz",
        "cudart-llama-b1-bin-ubuntu-cuda-12.8-x64.tar.gz",
        "llama-b1-bin-ubuntu-arm64.tar.gz",
        "llama-b1-bin-win-cpu-x64.zip",
    ]);
    let v = release_variants(&rel, Some((13, 2)), "ubuntu", "x64");
    assert_eq!(v.iter().map(|x| x.variant.as_str()).collect::<Vec<_>>(), ["cpu", "vulkan", "cuda-12.8", "cuda-13.5"]);
    assert_eq!(v.iter().map(|x| x.ok).collect::<Vec<_>>(), [true, true, true, false]);
    assert_eq!(v[2].assets.len(), 2); // + cudart
    assert_eq!(v[recommended(&v)].variant, "cuda-12.8");
    let v = release_variants(&rel, None, "ubuntu", "x64");
    assert_eq!(v[recommended(&v)].variant, "vulkan");

    // Windows zips name the CPU build and ship cudart without the tag; Apple silicon = Metal
    let rel = release(&[
        "llama-b1-bin-win-cpu-x64.zip",
        "llama-b1-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
        "llama-b1-bin-win-rocm-10.0-x64.zip",
        "llama-b1-bin-win-vulkan-arm64.zip",
        "llama-b1-bin-macos-arm64.tar.gz",
    ]);
    let v = release_variants(&rel, Some((12, 8)), "win", "x64");
    assert_eq!(v.iter().map(|x| x.variant.as_str()).collect::<Vec<_>>(), ["cpu", "cuda-12.4", "rocm-10.0"]);
    assert_eq!(v[1].assets.len(), 2);
    assert!(!v[2].ok); // needs the ROCm runtime
    assert_eq!(v[recommended(&v)].variant, "cuda-12.4");
    assert_eq!(release_variants(&rel, None, "macos", "arm64")[0].variant, "metal");
}

#[test]
fn command_line() {
    // server defaults are omitted, alias/host/port always passed
    let model = Path::new("/x/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf");
    let m = model.to_string_lossy().to_string();
    assert_eq!(base_alias(model), "Qwen2.5-VL-3B-Instruct");
    assert_eq!(base_alias(Path::new("/x/big-UD-Q4_K_XL-00001-of-00002.gguf")), "big");
    let mut p = defaults(model);
    let srv = PathBuf::from("srv");
    assert_eq!(
        build_cmd(&srv, model, &p, None),
        ["srv", "-m", &m, "--alias", "Qwen2.5-VL-3B-Instruct", "--host", "127.0.0.1", "--port", "8080"]
    );
    for (k, v) in
        [("ctx", "16384"), ("ctk", "q8_0"), ("jinja", "off"), ("temp", "0.6"), ("extra", "--metrics --alias 'a b'")]
    {
        p.insert(k.into(), v.into());
    }
    let cmd = build_cmd(&srv, model, &p, None);
    assert_eq!(cmd[9..], ["-c", "16384", "-ctk", "q8_0", "--temp", "0.6", "--no-jinja", "--metrics", "--alias", "a b"]);
    assert_eq!(
        build_cmd(Path::new("fp"), model, &p, Some(&["ctx", "ctk"])),
        ["fp", "-m", &m, "-c", "16384", "-ctk", "q8_0"]
    );
    assert_eq!(ui::shell_split(r#"-a "x y" 'it''s' "q\"z""#), ["-a", "x y", "its", "q\"z"]);
    // a quoted UNC path keeps its double backslash on Windows; elsewhere \\ is an escaped backslash
    assert_eq!(ui::shell_split(r#""\\nas\t.jinja""#), [if cfg!(windows) { r"\\nas\t.jinja" } else { r"\nas\t.jinja" }]);
    assert_eq!(ui::shell_join(&["a".into(), "b c".into(), "it's".into()]), r#"a 'b c' 'it'"'"'s'"#);
}

#[test]
fn build_compat() {
    // real --help lines: aliases on one line, description on the next, removed flags
    let help = "-cmoe, --cpu-moe                        keep all MoE weights in the CPU\n\
        -ot,   --override-tensor <tensor name pattern>=<buffer type>,...\n\
        \x20                                       override tensor buffer type\n\
        --spec-draft-n-max N                    number of tokens to draft (default: 3)\n\
        --spec-draft-model, -md, --model-draft FNAME\n\
        --spec-type none,draft-simple,ngram-mod\n\
        --draft, --draft-n, --draft-max N       the argument has been removed. use --spec-draft-n-max or\n\
        \x20                                       --spec-ngram-mod-n-max\n\
        -lm,   --load-mode MODE                 model loading mode (default: auto)\n\
        --jinja, --no-jinja                     whether to use jinja (default: enabled)\n\
        --lora-scaled FNAME:SCALE,...           path to LoRA adapter with user defined scaling\n\
        ----- server -----\n\
        -b,    --batch-size N                   logical maximum batch size (default: 2048)";
    let new = preset::help_flags(help);
    for f in ["--cpu-moe", "-cmoe", "-ot", "--override-tensor", "-md", "--spec-draft-n-max", "-lm", "--no-jinja", "-b"]
    {
        assert!(new.contains(f), "{f}");
    }
    assert!(!new.contains("--draft-max") && !new.contains("--draft") && !new.contains("-----"));
    let old: preset::Flags = ["-md", "--draft-max", "--no-mmap", "--mlock", "-b"].map(String::from).into();

    let mut p = defaults(Path::new("/x/m.gguf"));
    for (k, v) in [
        ("cpu_moe", "on"),
        ("draft", "/x/d.gguf"),
        ("draft_max", "8"),
        ("load_mode", "mlock"),
        ("batch", "1024"),
        ("lora", "/x/l.gguf"),
        ("lora_scale", "0.5"),
        ("metrics", "off"),
    ] {
        p.insert(k.into(), v.into());
    }
    let only = Some(&["cpu_moe", "draft", "draft_max", "load_mode", "batch", "lora", "metrics"][..]);
    // unknown build: everything passes, no implied --spec-type
    assert_eq!(
        preset::args(&p, only, None),
        [
            "--cpu-moe",
            "-md",
            "/x/d.gguf",
            "--spec-draft-n-max",
            "8",
            "--lora-scaled",
            "/x/l.gguf:0.5",
            "-b",
            "1024",
            "-lm",
            "mlock"
        ]
    );
    // new build: draft-simple is requested explicitly
    assert_eq!(
        preset::args(&p, only, Some(&new)),
        [
            "--cpu-moe",
            "-md",
            "/x/d.gguf",
            "--spec-draft-n-max",
            "8",
            "--lora-scaled",
            "/x/l.gguf:0.5",
            "-b",
            "1024",
            "-lm",
            "mlock",
            "--spec-type",
            "draft-simple"
        ]
    );
    // old build: no --cpu-moe/--lora-scaled, renamed --draft-max, load mode as --no-mmap --mlock
    assert_eq!(
        preset::args(&p, only, Some(&old)),
        ["-md", "/x/d.gguf", "--draft-max", "8", "-b", "1024", "--no-mmap", "--mlock"]
    );
    let f = |k: &str| preset::FIELDS.iter().find(|f| f.key == k).unwrap();
    assert!(preset::supported(f("load_mode"), Some(&old)) && !preset::supported(f("cpu_moe"), Some(&old)));
    assert!(preset::supported(f("jinja"), Some(&new)) && !preset::supported(f("jinja"), Some(&old)));

    // llama-fit-params output for a MoE model: experts of some layers kept in RAM
    let fitted = r#"-c 32768 -ngl -2 -ts 9,1 -ot "blk\.30\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.31\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU""#;
    assert!(preset::apply_fit(&mut p, fitted));
    assert_eq!((p["ctx"].as_str(), p["ngl"].as_str(), p["ts"].as_str()), ("32768", "all", "9,1"));
    assert_eq!(
        p["ot"],
        r"blk\.30\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.31\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU"
    );
    assert!(preset::check("ot", &p["ot"]).is_ok() && preset::check("ot", "blk").is_err());
    assert!(preset::check("ts", "3,1").is_ok() && preset::check("ts", "3;1").is_err());
    assert!(preset::check("ngld", "all").is_ok() && preset::check("n_cpu_moe", "x").is_err());
}

#[test]
fn gguf_header() {
    let s = |b: &mut Vec<u8>, x: &str| {
        b.extend((x.len() as u64).to_le_bytes());
        b.extend(x.as_bytes());
    };
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend(0u64.to_le_bytes()); // tensors
    b.extend(6u64.to_le_bytes()); // key/values
    s(&mut b, "general.architecture");
    b.extend(8u32.to_le_bytes());
    s(&mut b, "qwen3moe");
    s(&mut b, "general.tags"); // array of strings
    b.extend(9u32.to_le_bytes());
    b.extend(8u32.to_le_bytes());
    b.extend(2u64.to_le_bytes());
    s(&mut b, "a");
    s(&mut b, "bc");
    s(&mut b, "qwen3moe.rope.freq_base"); // f32
    b.extend(6u32.to_le_bytes());
    b.extend(1e6f32.to_le_bytes());
    s(&mut b, "qwen3moe.block_count");
    b.extend(4u32.to_le_bytes());
    b.extend(48u32.to_le_bytes());
    s(&mut b, "qwen3moe.expert_count");
    b.extend(4u32.to_le_bytes());
    b.extend(128u32.to_le_bytes());
    s(&mut b, "qwen3moe.context_length");
    b.extend(4u32.to_le_bytes());
    b.extend(40960u32.to_le_bytes());
    let g = preset::gguf_read(&mut &b[..]).unwrap();
    assert_eq!(g, preset::Gguf { arch: "qwen3moe".into(), layers: 48, experts: 128, ctx: 40960 });
    assert_eq!(preset::gguf_read(&mut &b[..20]).map(|g| g.experts), None); // truncated header
    assert!(preset::gguf_read(&mut &b"GGUX"[..]).is_none());
}

#[test]
fn dashboard_parsing() {
    // real llama-server lines
    let info = dashboard::Info {
        alias: "a".into(),
        model: "m.gguf".into(),
        mmproj: "x".into(),
        endpoint: "h:1".into(),
        install: "i".into(),
        backend: "vulkan".into(),
        version: "b1 · abc".into(),
        settings: vec![("ngl".into(), "auto".into())],
        router: false,
        host: String::new(),
    };
    let mut d = dashboard::Dashboard::new(info, PathBuf::from("x.log"));
    for line in [
        "0.00.105.703 I srv    load_model: loading model 'm.gguf'",
        "0.04.314.735 I srv    load_model: loaded multimodal model, 'mm.gguf'",
        "0.04.482.821 I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 128000, kv_unified = 'true'",
        "0.04.486.940 I srv  llama_server: listening on http://127.0.0.1:18080",
        "0.05.074.646 I slot launch_slot_: id  3 | task 0 | processing task, is_child = 0",
        "0.05.118.983 I slot print_timing: id  3 | task 0 | prompt eval time =      21.22 ms /    25 tokens \
         (    0.85 ms per token,  1178.02 tokens per second)",
        "0.05.118.986 I slot print_timing: id  3 | task 0 |        eval time =      22.86 ms /     4 tokens \
         (    7.62 ms per token,   131.22 tokens per second)",
        "0.05.119.017 I slot      release: id  3 | task 0 | stop processing: n_tokens = 28, truncated = 0",
        "0.05.200.000 W srv  llama_server: something",
    ] {
        d.feed(line);
    }
    assert!(d.done == 5 && d.url.as_deref() == Some("http://127.0.0.1:18080") && (d.n_slots, d.n_ctx) == (4, 128000));
    assert!(
        d.requests == 1 && (d.tok_prompt, d.tok_gen, d.last_tg) == (25, 4, 131.22) && d.busy.is_empty() && d.warns == 1
    );
    assert_eq!(d.lines.len(), 9);
}

#[test]
fn data_dir_fallback() {
    let home = std::env::temp_dir().join(format!("corral-home-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&home);
    std::fs::create_dir_all(home.join(".llama-manager")).unwrap();
    assert_eq!(ui::data_dir(&home), home.join(".llama-manager")); // only the old dir: keep using it
    std::fs::create_dir_all(home.join(".corral")).unwrap();
    assert_eq!(ui::data_dir(&home), home.join(".corral"));
    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn port_in_use() {
    let busy = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = busy.local_addr().unwrap().port();
    assert!(free_port("127.0.0.1", port).is_some_and(|p| p > port));
    assert_eq!(free_port("not a host", 8080), None);
}

#[test]
fn fit_ngl_values() {
    // llama-fit-params output: -1 = left at auto, -2 = all layers
    assert_eq!(ngl_word("-1"), "auto");
    assert_eq!(ngl_word("-2"), "all");
    assert_eq!(ngl_word("17"), "17");
}

#[test]
fn download_resume_and_hash() {
    use std::io::{Read, Write};
    // Serves "hello world" (ETag "v1"); honours Range only when If-Range matches.
    let srv = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let url = format!("http://{}/f", srv.local_addr().unwrap());
    std::thread::spawn(move || {
        for mut c in srv.incoming().flatten() {
            let (mut req, mut b) = (Vec::new(), [0; 1]);
            while !req.ends_with(b"\r\n\r\n") && c.read(&mut b).unwrap() == 1 {
                req.push(b[0]);
            }
            let req = String::from_utf8_lossy(&req).to_lowercase();
            let (code, body) =
                if req.contains("if-range: \"v1\"") { ("206 Partial", "world") } else { ("200 OK", "hello world") };
            let head = format!("HTTP/1.1 {code}\r\nETag: \"v1\"\r\nContent-Length: {}\r\n\r\n", body.len());
            let _ = c.write_all((head + body).as_bytes());
        }
    });
    let dir = std::env::temp_dir().join(format!("lm-dl-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (dest, part, etag) = (dir.join("f"), dir.join("f.part"), dir.join("f.part.etag"));
    let sha = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"; // sha256("hello world")
    let get = |sha| ui::download(ureq::get(&url), &dest, 0, Some(sha), &mut |_| true).is_ok();
    // same ETag: resumed; stale ETag: whole file again; both end verified
    for (stale, tag) in [("hello ", "\"v1\""), ("XXXXXXXX", "\"v0\"")] {
        std::fs::write(&part, stale).unwrap();
        std::fs::write(&etag, tag).unwrap();
        assert!(get(sha));
        assert_eq!(std::fs::read_to_string(&dest).unwrap(), "hello world");
        assert!(!part.exists() && !etag.exists());
        std::fs::remove_file(&dest).unwrap();
    }
    assert!(!get(&sha.replace('b', "c")));
    assert!(!dest.exists() && !part.exists());
    let _ = std::fs::remove_dir_all(&dir);
}

// -- TUI logic: navigation and mouse hit-testing ---------------------------------------------

mod tui_logic {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
    use ratatui::layout::{Position, Rect};
    use ratatui::text::Span;

    use crate::tui::widgets::{Edit, Hit, List, Modal, TextInput, widths};
    use crate::tui::{Answer, Ctx, Ev, draw_footer, draw_toasts};
    use crate::ui::{ERR, State, cell};

    fn k(c: KeyCode) -> Ev {
        Ev::Key(KeyEvent::new(c, KeyModifiers::NONE))
    }
    fn ch(c: char) -> Ev {
        k(KeyCode::Char(c))
    }
    fn mouse(kind: MouseEventKind, x: u16, y: u16) -> Ev {
        Ev::Mouse(MouseEvent { kind, column: x, row: y, modifiers: KeyModifiers::NONE })
    }
    fn click(x: u16, y: u16) -> Ev {
        mouse(MouseEventKind::Down(MouseButton::Left), x, y)
    }
    fn rows(n: usize) -> Vec<Vec<Span<'static>>> {
        (0..n).map(|i| vec![cell(format!("row {i}"))]).collect()
    }
    /// Draw `list` with `n` rows into a 40x`h` area at the origin.
    fn draw(list: &mut List, n: usize, h: u16, ctx: &Ctx) {
        let mut t = Terminal::new(TestBackend::new(40, 20)).unwrap();
        t.draw(|f| list.draw(f, Rect::new(0, 0, 40, h), &["name"], rows(n), ctx)).unwrap();
    }
    /// Where `text` was drawn in the backend's buffer.
    fn find(t: &Terminal<TestBackend>, text: &str) -> Option<Position> {
        let b = t.backend().buffer();
        (0..b.area.height).find_map(|y| {
            let line: String = (0..b.area.width).map(|x| b[(x, y)].symbol().to_string()).collect();
            line.find(text).map(|i| Position::new(line[..i].chars().count() as u16, y))
        })
    }

    #[test]
    fn list_keys_and_digits() {
        let mut l = List::new(0);
        l.set_len(12);
        assert_eq!(l.on(&k(KeyCode::Down)), Some(Hit::Moved));
        assert_eq!(l.sel, 1);
        l.on(&k(KeyCode::End));
        assert_eq!(l.sel, 11);
        l.on(&k(KeyCode::Down)); // clamped
        assert_eq!(l.sel, 11);
        l.on(&k(KeyCode::Home));
        assert_eq!(l.on(&k(KeyCode::Enter)), Some(Hit::Open(0)));
        // 12 rows: "1" could still become 10-12, so it only selects; "12" opens
        assert_eq!(l.on(&ch('1')), Some(Hit::Moved));
        assert_eq!(l.on(&ch('2')), Some(Hit::Open(11)));
        assert_eq!(l.on(&ch('5')), Some(Hit::Open(4))); // 50 > 12: no longer number possible
        assert_eq!(l.on(&ch('0')), Some(Hit::Moved)); // 0 is no row
        assert_eq!(l.sel, 4);
        assert_eq!(l.on(&ch('x')), None); // letters are the screen's shortcuts
        let mut short = List::new(0);
        short.set_len(4);
        assert_eq!(short.on(&ch('3')), Some(Hit::Open(2))); // ≤ 9 rows: a digit opens at once
    }

    #[test]
    fn list_mouse() {
        let ctx = Ctx::bare(State::default());
        let mut l = List::new(0);
        draw(&mut l, 10, 6, &ctx); // header + rule, then rows 0..3 at y = 2..5
        assert_eq!(l.on(&click(5, 1)), None); // header
        assert_eq!(l.on(&click(5, 3)), Some(Hit::Moved)); // first click selects row 1
        assert_eq!(l.sel, 1);
        assert_eq!(l.on(&click(5, 3)), Some(Hit::Open(1))); // a click on the selected row opens it
        assert_eq!(l.on(&mouse(MouseEventKind::ScrollDown, 5, 4)), Some(Hit::Moved));
        assert_eq!(l.sel, 2);
        assert_eq!(l.on(&mouse(MouseEventKind::ScrollDown, 5, 15)), None); // wheel outside the list
        l.on(&k(KeyCode::End));
        draw(&mut l, 10, 6, &ctx); // scrolled: rows 6..9 visible
        assert_eq!(l.on(&click(5, 2)), Some(Hit::Moved));
        assert_eq!(l.sel, 6);
        assert_eq!(l.cell_rect(6, 0).map(|r| r.y), Some(2));
        assert_eq!(l.cell_rect(0, 0), None); // scrolled out
        let mut m = List::new(0);
        m.single_click = true;
        draw(&mut m, 3, 6, &ctx);
        assert_eq!(m.on(&click(5, 4)), Some(Hit::Open(2)));
        assert_eq!(m.on(&click(5, 5)), None); // below the last row
    }

    #[test]
    fn column_widths() {
        let cells = vec![vec![4, 6], vec![30, 3], vec![12, 20]];
        assert_eq!(widths(&cells, 100), [30, 20]);
        let w = widths(&cells, 34); // 30 columns of text + 2x2 gap
        assert_eq!(w.iter().sum::<usize>(), 30);
        assert_eq!(widths(&cells, 0), [8, 8]); // never below 8
    }

    #[test]
    fn text_input() {
        let mut t = TextInput::new("port");
        for c in "à1".chars() {
            t.on(&ch(c));
        }
        assert_eq!(t.value, "portà1");
        t.on(&k(KeyCode::Left));
        t.on(&k(KeyCode::Backspace));
        assert_eq!(t.value, "port1");
        t.on(&k(KeyCode::Home));
        t.on(&k(KeyCode::Delete));
        assert_eq!(t.value, "ort1");
        assert!(matches!(t.on(&k(KeyCode::Enter)), Some(Edit::Submit)));
        assert!(matches!(t.on(&k(KeyCode::Esc)), Some(Edit::Cancel)));
        t.on(&Ev::Key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL)));
        assert_eq!(t.value, "");
        let ctrl_c = Ev::Key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(t.on(&ctrl_c).is_none()); // left to the app (quit)
    }

    #[test]
    fn modal_answers() {
        let ctx = Ctx::bare(State::default());
        let mut m = Modal::ask("ok?", false);
        assert_eq!(m.on(&k(KeyCode::Enter)), Some(Answer::No)); // the default
        m.on(&k(KeyCode::Tab));
        assert_eq!(m.on(&k(KeyCode::Enter)), Some(Answer::Yes));
        assert_eq!(m.on(&ch('n')), Some(Answer::No));
        assert_eq!(m.on(&k(KeyCode::Esc)), Some(Answer::No));
        // the buttons are clickable once drawn
        let mut t = Terminal::new(TestBackend::new(80, 24)).unwrap();
        t.draw(|f| m.draw(f, f.area(), &ctx)).unwrap();
        let yes = find(&t, "Sì (s)").unwrap();
        let no = find(&t, "No (n)").unwrap();
        assert_eq!(m.on(&click(yes.x, yes.y)), Some(Answer::Yes));
        assert_eq!(m.on(&click(no.x + 2, no.y)), Some(Answer::No));
        assert_eq!(m.on(&click(0, 0)), None); // outside: ignored
        let mut p = Modal::prompt("nome", "", "abc");
        p.on(&ch('d'));
        assert_eq!(p.on(&k(KeyCode::Enter)), Some(Answer::Text("abcd".into())));
        let mut c = Modal::choose("x", &["v"], rows(3), 0);
        t.draw(|f| c.draw(f, f.area(), &ctx)).unwrap();
        assert_eq!(c.on(&ch('2')), Some(Answer::Pick(1)));
        let r = find(&t, "row 2").unwrap();
        assert_eq!(c.on(&click(r.x, r.y)), None); // first click selects
        assert_eq!(c.on(&click(r.x, r.y)), Some(Answer::Pick(2))); // second click on the selected row
        assert_eq!(c.on(&k(KeyCode::Esc)), Some(Answer::Cancel));
    }

    #[test]
    fn tiny_terminal() {
        // a modal and a toast on a terminal narrower than they want to be: drawn smaller, no panic
        let mut ctx = Ctx::bare(State::default());
        ctx.toast("✖ un messaggio d'errore piuttosto lungo per uno schermo così stretto", ERR);
        for (w, h) in [(39, 12), (20, 6), (3, 2)] {
            let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
            let mut m = Modal::ask("fermo il server e torno indietro?", true);
            t.draw(|f| {
                m.draw(f, f.area(), &ctx);
                draw_toasts(f, f.area(), &ctx);
            })
            .unwrap();
        }
    }

    #[test]
    fn footer_buttons() {
        let mut ctx = Ctx::bare(State::default());
        let keys = [(KeyCode::Enter, "invio", "apri"), (KeyCode::Char('q'), "q", "esci")];
        let mut t = Terminal::new(TestBackend::new(60, 1)).unwrap();
        let mut zones = None;
        t.draw(|f| zones = Some(draw_footer(f, f.area(), &keys, &ctx))).unwrap();
        let zones = zones.unwrap();
        let q = find(&t, " q ").unwrap();
        // badge and description are both clickable and send the key
        assert_eq!(zones.at(Position::new(2, 0)), Some(KeyCode::Enter));
        assert_eq!(zones.at(Position::new(10, 0)), Some(KeyCode::Enter));
        assert_eq!(zones.at(Position::new(q.x + 1, 0)), Some(KeyCode::Char('q')));
        assert_eq!(zones.at(Position::new(50, 0)), None);
        ctx.set_mouse(Some(Position::new(2, 0)));
        assert!(ctx.hovered(Rect::new(1, 0, 10, 1)));
    }
}

#[test]
fn state_migration_and_named_presets() {
    use crate::preset::*;
    use crate::ui::Preset;
    let m = Path::new("/nope/m-Q4_K_M.gguf");
    // old layout: one flat preset per model -> "default"
    let old = r#"{"install":"x","presets":{"/nope/m-Q4_K_M.gguf":{"ctx":"32768"},"/nope/e.gguf":{}},"last_model":"/nope/m-Q4_K_M.gguf"}"#;
    let mut s: ui::State = serde_json::from_str(old).unwrap();
    assert_eq!(s.presets["/nope/m-Q4_K_M.gguf"]["default"]["ctx"], "32768");
    assert_eq!(preset_names(&s, Path::new("/nope/e.gguf")), ["default"]); // empty = virtual default
    assert_eq!(get_preset(&s, m)["ctx"], "32768");
    // the new layout round-trips
    let back: ui::State = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
    assert_eq!(back.presets, s.presets);

    add_preset(&mut s, m, "coding 32k", [("ctx".into(), "65536".into())].into()).unwrap();
    assert_eq!(preset_name(&s, m), "coding 32k"); // the new one is selected
    assert_eq!(get_preset(&s, m)["ctx"], "65536");
    assert!(add_preset(&mut s, m, "default", Preset::new()).is_err());
    assert!(add_preset(&mut s, m, "  ", Preset::new()).is_err());
    rename_preset(&mut s, m, "default", "chat").unwrap();
    assert_eq!(preset_names(&s, m), ["chat", "coding 32k"]);
    assert_eq!(preset_name(&s, m), "coding 32k"); // renaming another keeps the selection
    rename_preset(&mut s, m, "coding 32k", "code").unwrap();
    assert_eq!(preset_name(&s, m), "code"); // renaming the selected one follows it
    delete_preset(&mut s, m, "code").unwrap();
    assert_eq!(preset_name(&s, m), "chat");
    assert_eq!(get_preset(&s, m)["ctx"], "32768");
    assert!(delete_preset(&mut s, m, "chat").is_err()); // the last one stays
    mark_used(&mut s, m);
    assert!(s.last_used.contains_key("/nope/m-Q4_K_M.gguf"));
    assert_eq!(crate::models::orphans(&s), ["/nope/e.gguf", "/nope/m-Q4_K_M.gguf"]);
}

#[test]
fn preset_files() {
    use crate::models::{parse_preset, unique};
    let text = r#"{"corral_preset":1,"name":"fast","values":{"ctx":"4096","bogus":"1","port":"x"}}"#;
    let (n, v, dropped) = parse_preset(text, "f").unwrap();
    assert_eq!((n.as_str(), v.len(), dropped), ("fast", 1, 2)); // unknown key and bad port dropped
    let (n, v, _) = parse_preset(r#"{"ctx": 8192, "jinja": false}"#, "file").unwrap();
    assert_eq!((n.as_str(), v["ctx"].as_str(), v["jinja"].as_str()), ("file", "8192", "off"));
    assert!(parse_preset("[1]", "x").is_err());
    let taken = ["a".to_string(), "a 2".to_string()];
    assert_eq!(unique("a", &taken), "a 3");
    assert_eq!(unique("b", &taken), "b");
}

#[test]
fn proxy_rules() {
    fn env(vars: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |k| vars.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string())
    }
    let e = env(&[("HTTPS_PROXY", "http://p:3128"), ("http_proxy", "http://h:80")]);
    assert_eq!(ui::proxy_env(true, &e).as_deref(), Some("http://p:3128"));
    assert_eq!(ui::proxy_env(false, &e).as_deref(), Some("http://h:80"));
    assert_eq!(ui::proxy_env(true, &env(&[("ALL_PROXY", "socks5://s:1")])).as_deref(), Some("socks5://s:1"));
    assert_eq!(ui::proxy_env(true, &env(&[("HTTP_PROXY", "http://h:80")])), None); // curl: not for https
    assert!(ui::no_proxy("127.0.0.1", "") && ui::no_proxy("localhost", "") && ui::no_proxy("[::1]", ""));
    assert!(ui::no_proxy("cdn.huggingface.co", "foo.org, .huggingface.co"));
    assert!(ui::no_proxy("huggingface.co", "huggingface.co:443"));
    assert!(ui::no_proxy("x.y", "*"));
    assert!(!ui::no_proxy("evilhuggingface.co", "huggingface.co"));
    assert!(!ui::no_proxy("github.com", ""));
}

#[test]
fn release_updates() {
    use crate::installs::{Install, newer_release, release_variant};
    let i = Install {
        name: "rel-b6000-cuda-12.4".into(),
        kind: "release".into(),
        backend: "cuda".into(),
        bin_dir: String::new(),
        git_ref: Some("b6000".into()),
        flags: vec![],
        src_dir: None,
    };
    assert_eq!(release_variant(&i).as_deref(), Some("cuda-12.4"));
    assert!(newer_release(&i, "b6001") && !newer_release(&i, "b6000") && !newer_release(&i, "b5999"));
    assert!(!newer_release(&Install { kind: "source".into(), ..i.clone() }, "b7000"));
}

/// Processes, signals and host facts, on whatever OS runs the tests (CI: Linux, macOS, Windows).
mod system {
    use std::process::Command;
    use std::time::{Duration, Instant};

    use regex::Regex;

    use crate::ui;

    #[cfg(windows)]
    unsafe extern "system" {
        fn GenerateConsoleCtrlEvent(event: u32, group: u32) -> i32;
    }

    /// `script` through sh -c / cmd /C.
    fn shell(script: &str) -> Command {
        #[cfg(windows)]
        {
            let mut c = Command::new("cmd");
            std::os::windows::process::CommandExt::raw_arg(c.arg("/C"), script);
            c
        }
        #[cfg(not(windows))]
        {
            let mut c = Command::new("sh");
            c.args(["-c", script]);
            c
        }
    }

    #[test]
    fn merged_output_keeps_order() {
        let (mut child, rx) = ui::spawn_merged(shell("echo one&& echo two 1>&2&& echo three")).unwrap();
        assert_eq!(rx.iter().collect::<Vec<_>>(), ["one", "two", "three"]);
        assert!(child.wait().unwrap().success());
    }

    #[test]
    fn capture_and_timeout() {
        let out = ui::capture(&mut shell("echo out&& echo err 1>&2"), Duration::from_secs(20)).unwrap();
        assert!(out.ok);
        assert_eq!((out.stdout.trim(), out.stderr.trim()), ("out", "err"));
        assert!(!ui::capture(&mut shell("exit 3"), Duration::from_secs(20)).unwrap().ok);
        let mut slow = Command::new(if cfg!(windows) { "ping" } else { "sleep" });
        slow.args(if cfg!(windows) { &["-n", "30", "127.0.0.1"][..] } else { &["30"][..] });
        let t0 = Instant::now();
        assert_eq!(ui::capture(&mut slow, Duration::from_millis(300)).err().as_deref(), Some("timeout"));
        assert!(t0.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn exit_codes() {
        let code = |s: &str| ui::exit_code(&shell(s).status().unwrap());
        assert_eq!(code("exit 3"), 3);
        if cfg!(unix) {
            assert_eq!(code("kill -TERM $$"), -15); // killed by a signal: -signal, like Popen
        }
    }

    #[test]
    fn which_finds_the_shell() {
        assert!(ui::which(if cfg!(windows) { "cmd" } else { "sh" }).is_some_and(|p| p.is_file()));
        assert_eq!(ui::which("surely-not-a-command-4242"), None);
    }

    #[test]
    fn host_facts() {
        assert!(ui::ram_available() > 0);
        assert!(ui::disk_free(&std::env::temp_dir()).is_some_and(|n| n > 0));
        let ok = |re: &str, s: String| assert!(Regex::new(re).unwrap().is_match(&s), "{s}");
        ok(r"^20\d\d-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01]) ([01]\d|2[0-3]):[0-5]\d:[0-6]\d$", ui::stamp());
        ok(r"^([01]\d|2[0-3]):[0-5]\d:[0-6]\d$", ui::clock());
        ok(r"^20\d{6}-\d{6}$", ui::file_stamp());
    }

    #[cfg(unix)]
    #[test]
    fn stop_interrupts_then_kills() {
        // one child exits 7 on SIGINT, the other ignores it and is killed after the grace
        for (trap, want) in [("exit 7", 7), ("", -9)] {
            let script = format!("trap '{trap}' INT; echo ready; while :; do sleep 0.1; done");
            let (child, rx) = ui::spawn_merged(shell(&script)).unwrap();
            assert_eq!(rx.recv().unwrap(), "ready");
            let child = std::sync::Mutex::new(child);
            ui::stop(&child, Duration::from_millis(500));
            assert_eq!(ui::exit_code(&child.lock().unwrap().wait().unwrap()), want);
        }
    }

    /// Child half of `quit_signal_runs_stops`: does nothing unless that test starts it.
    #[test]
    fn quit_child() {
        if std::env::var_os("CORRAL_QUIT_CHILD").is_none() {
            return;
        }
        let _q = ui::on_quit(|| println!("stopped"));
        println!("ready");
        std::thread::sleep(Duration::from_secs(20));
    }

    #[test]
    fn quit_signal_runs_stops() {
        let mut c = Command::new(std::env::current_exe().unwrap());
        c.args(["--exact", "tests::system::quit_child", "--nocapture"]).env("CORRAL_QUIT_CHILD", "1");
        #[cfg(windows)] // its own group, so the Ctrl+Break below reaches only the child
        std::os::windows::process::CommandExt::creation_flags(&mut c, 0x200); // CREATE_NEW_PROCESS_GROUP
        let (mut child, rx) = ui::spawn_merged(c).unwrap();
        assert!(rx.iter().any(|l| l.ends_with("ready")));
        #[cfg(unix)]
        let (sent, want) = (unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGTERM) } == 0, 128 + libc::SIGTERM);
        #[cfg(windows)]
        let (sent, want) = (unsafe { GenerateConsoleCtrlEvent(1, child.id()) } != 0, 128 + 1); // CTRL_BREAK_EVENT
        if !sent {
            // a Windows host without a console has nothing to carry the event: can't test it there
            let _ = child.kill();
            eprintln!("quit_signal_runs_stops: no console, skipped");
            return;
        }
        assert!(rx.iter().any(|l| l.contains("stopped")));
        assert_eq!(ui::exit_code(&child.wait().unwrap()), want);
    }
}

mod multi {
    use std::path::{Path, PathBuf};

    use crate::router::{self, Entry, Profile};
    use crate::{agent, news, preset, ui};

    /// Two empty files posing as models (the INI only needs them to exist).
    fn models(tag: &str) -> (PathBuf, PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!("corral-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (a, b) = (dir.join("Qwen3-Coder-30B-A3B-Q4_K_M.gguf"), dir.join("Qwen3-Embedding-0.6B-Q8_0.gguf"));
        std::fs::write(&a, b"").unwrap();
        std::fs::write(&b, b"").unwrap();
        (dir, a, b)
    }

    #[test]
    fn router_ini_and_command() {
        let (dir, a, b) = models("ini");
        let mut s = ui::State::default();
        let p = ui::Preset::from([
            ("ctx".into(), "4096".into()),
            ("np".into(), "2".into()),
            ("extra".into(), "--temp 0.7 --no-warmup -ngl -1 stray --x".into()),
        ]);
        s.presets.entry(ui::ps(&a)).or_default().insert("coding".into(), p);
        s.presets
            .entry(ui::ps(&b))
            .or_default()
            .insert("default".into(), [("usage".into(), "embedding".into())].into());
        let e = |id: &str, m: &Path, preset: &str, startup| Entry {
            id: id.into(),
            model: ui::ps(m),
            preset: preset.into(),
            startup,
        };
        let mut prof =
            Profile { models: vec![e("coder", &a, "coding", true), e("embed", &b, "", false)], ..Default::default() };
        let ini = router::ini(&s, &prof, None).unwrap();
        let coder = ini.split("[embed]").next().unwrap();
        for want in ["version = 1", "[coder]", &format!("model = {}", ui::ps(&a)), "c = 4096", "np = 2"] {
            assert!(coder.contains(want), "{want} in {ini}");
        }
        // extra flags become pairs; the router owns host, port and alias
        for want in ["temp = 0.7", "no-warmup = true", "ngl = -1", "x = true", "load-on-startup = true"] {
            assert!(coder.contains(want), "{want} in {ini}");
        }
        assert!(!ini.contains("stray") && !ini.contains("host =") && !ini.contains("alias ="));
        let embed = ini.split("[embed]").nth(1).unwrap();
        assert!(
            !embed.contains("load-on-startup") && embed.contains("Embedding") && embed.contains("embeddings = true")
        );
        // what the INI can't hold
        prof.models[1].id = "coder".into();
        assert!(router::ini(&s, &prof, None).unwrap_err().contains("due volte"));
        prof.models[1].id = "bad id".into();
        assert!(router::ini(&s, &prof, None).is_err());
        prof.models[1].id = "embed".into();
        s.presets.get_mut(&ui::ps(&a)).unwrap().get_mut("coding").unwrap().insert("ot".into(), "x;y=CPU".into());
        assert!(router::ini(&s, &prof, None).unwrap_err().contains("coder"));
        // the router's own command line: only what differs from the server, host and port always
        prof.opts = router::opts_diff(&ui::Preset::from([
            ("models_max".into(), "2".into()),
            ("autoload".into(), "off".into()),
            ("tools".into(), router::READ_TOOLS.into()),
            ("port".into(), "8080".into()),
        ]));
        assert!(!prof.opts.contains_key("port")); // default: not stored
        let cmd = router::command(Path::new("srv"), Path::new("p.ini"), &prof, None);
        let want = ["srv", "--models-preset", "p.ini", "--host", "127.0.0.1", "--port", "8080", "--models-max", "2"];
        assert_eq!(cmd[..9], want);
        assert!(cmd.contains(&"--no-models-autoload".into()) && cmd.contains(&router::READ_TOOLS.to_string()));
        assert!(!cmd.contains(&"--ui-mcp-proxy".into()));
        // ids
        assert_eq!(router::new_id("Qwen3 Coder:30B", &["Qwen3-Coder-30B"]), "Qwen3-Coder-30B-2");
        assert_eq!(router::new_id("..", &[]), "modello");
        assert!(router::id_ok("a.b_c-1") && !router::id_ok("-a") && !router::id_ok("a/b") && !router::id_ok(""));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn router_status() {
        let v: serde_json::Value = serde_json::from_str(
            r#"{"data":[{"id":"coder","status":{"value":"loaded","args":["llama-server"]}},
                        {"id":"embed","status":{"value":"unloaded","failed":true,"exit_code":1}},
                        {"id":"x"}]}"#,
        )
        .unwrap();
        let s = router::parse_models(&v);
        assert_eq!((s[0].id.as_str(), s[0].state.as_str(), s[0].failed), ("coder", "loaded", false));
        assert_eq!((s[1].failed, s[1].exit), (true, Some(1)));
        assert_eq!(s[2].state, "?");
        // the router's log: a child's lines go under its model's id
        let info = crate::dashboard::Info {
            alias: "p".into(),
            model: "2 modelli".into(),
            mmproj: String::new(),
            endpoint: "h:1".into(),
            install: "i".into(),
            backend: "b".into(),
            version: "v".into(),
            settings: vec![],
            router: true,
            host: String::new(),
        };
        let mut d = crate::dashboard::Dashboard::new(info, std::path::PathBuf::from("x.log"));
        d.feed("0.00.1 I srv  load: spawning server instance with name=coder on port 39531");
        d.feed("[39531] 0.00.2 W srv    load_model: something odd");
        let l = d.lines.back().unwrap();
        assert_eq!((l.level, l.text.as_str()), ('W', "coder srv load_model: something odd"));
    }

    #[test]
    fn recipes_and_exports() {
        let (dir, a, b) = models("recipe");
        let all = vec![b.clone(), a.clone()];
        let rag = agent::RECIPES.iter().find(|r| r.key == "rag").unwrap();
        // the embedder is suggested for "embed", not the coder
        assert_eq!(agent::suggest(&rag.roles[1], &all)[0], (0, true));
        let mut s = ui::State::default();
        let multi = agent::RECIPES.iter().find(|r| r.key == "multi-agent").unwrap();
        assert_eq!(agent::apply(&mut s, multi, &[a.clone(), a.clone()]), "multi-agent");
        assert_eq!(preset::preset_name(&s, &a), "default"); // on its own it still runs as before
        let prof = &s.routers["multi-agent"];
        assert_eq!(prof.models.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), ["planner", "worker"]);
        assert!(prof.models.iter().all(|e| e.startup));
        assert_eq!(preset::get_named(&s, &a, "worker")["np"], "4");
        assert_eq!(preset::get_named(&s, &a, "planner")["reasoning"], "on");
        // re-applying keeps the user's own changes to a role's preset
        s.presets.get_mut(&ui::ps(&a)).unwrap().get_mut("worker").unwrap().insert("ctk".into(), "q8_0".into());
        agent::apply(&mut s, multi, &[a.clone(), a.clone()]);
        assert_eq!(preset::get_named(&s, &a, "worker")["ctk"], "q8_0");
        // exports
        let t = agent::target(&s, "multi-agent", &s.routers["multi-agent"], None);
        assert_eq!(t.base, "http://127.0.0.1:8080");
        let (_, _, oc) = agent::render(0, &t);
        let v: serde_json::Value = serde_json::from_str(&oc).unwrap();
        let p = &v["providers"]["corral"];
        assert_eq!(p["settings"]["baseURL"], "http://127.0.0.1:8080/v1");
        assert_eq!(p["models"]["worker"]["limit"]["context"], 65536);
        assert_eq!(p["models"]["planner"]["capabilities"]["tools"], true);
        assert!(p.get("env").is_none()); // no API key, no variable
        assert_eq!(v["model"], "corral/planner");
        let (_, _, qw) = agent::render(1, &t);
        let v: serde_json::Value = serde_json::from_str(&qw).unwrap();
        assert_eq!(v["modelProviders"]["openai"][1]["id"], "worker");
        assert_eq!(v["model"]["name"], "planner");
        let (_, _, py) = agent::render(2, &t);
        assert!(py.contains("create_agent") && py.contains("Crew(") && !py.contains("embeddings"));
        let (_, _, env) = agent::render(3, &t);
        assert!(env.contains("OPENAI_BASE_URL") && env.contains("http://127.0.0.1:8080/v1"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn feeds_and_dates() {
        let rss = r#"<?xml version="1.0"?><rss><channel><title>Blog &amp; news</title>
            <item><title><![CDATA[Speech <b>models</b>]]></title><link>https://x.org/a</link>
              <description>&lt;p&gt;Hello &amp;amp; welcome&lt;/p&gt;</description>
              <pubDate>Fri, 13 Nov 2026 14:05:22 GMT</pubDate></item>
            <item><title>Second</title><link>https://x.org/b</link><pubDate>Sat, 24 Oct 2026 00:00:00 +0200</pubDate></item>
            </channel></rss>"#;
        let (title, items) = news::parse_feed(rss);
        assert_eq!(title, "Blog & news");
        assert_eq!(items.len(), 2);
        assert_eq!((items[0].title.as_str(), items[0].url.as_str()), ("Speech models", "https://x.org/a"));
        assert_eq!(items[0].text, "Hello & welcome");
        assert_eq!(items[0].date, 1_794_578_722);
        assert_eq!(items[1].date, 1_792_792_800); // 2026-10-23 22:00 UTC
        let atom = r#"<feed><title>Simon</title><entry><title>llm 0.37</title>
            <link href="https://s.net/self" rel="self"/><link href="https://s.net/a" rel="alternate"/>
            <published>2026-11-12T22:55:19+00:00</published><summary type="html">&lt;p&gt;Gemini&lt;/p&gt;</summary>
            </entry></feed>"#;
        let (title, items) = news::parse_feed(atom);
        assert_eq!(
            (title.as_str(), items[0].url.as_str(), items[0].text.as_str()),
            ("Simon", "https://s.net/a", "Gemini")
        );
        assert_eq!(news::parse_date("2026-11-13T10:20:19Z"), Some(1_794_565_219));
        assert_eq!(news::parse_date("2026-11-12T12:00:53.000Z"), news::parse_date("2026-11-12T13:00:53+01:00"));
        assert_eq!(news::parse_date("Thu, 01 Jan 1970 00:00:00 GMT"), Some(0));
        assert_eq!(news::parse_date("non è una data"), None);
        let now = 1_794_565_219;
        assert_eq!(news::ago(now, now - 125), "2 min fa");
        assert_eq!(news::ago(now, now - 3 * 3600), "3 h fa");
        assert_eq!(news::ago(now, now - 2 * 86_400), "2 g fa");
        assert_eq!(news::ago(now, 1_792_792_800), "23/10/26");
        assert_eq!(news::unescape("a &#8211; b &#x2014; &bogus; c"), "a – b — &bogus; c");
        // a build's body: the commit message, not the download links
        let body = "<details open>\n\nCUDA: split the grid (#32179)\n\nThe kernel\n\n</details>\n\n**Website:**\n- <https://llama.app>";
        assert_eq!(news::release_text(body), "CUDA: split the grid (#32179)\nThe kernel");
    }

    #[test]
    fn changelog_matches_the_version() {
        let items = news::corral();
        assert!(items.len() >= 2);
        assert_eq!(items[0].tag, env!("CARGO_PKG_VERSION"), "CHANGELOG.md: add the new version on top");
        assert!(items[0].date > 0 && items[0].text.starts_with("• "));
        assert!(items[0].title.starts_with(&format!("Corral {}", env!("CARGO_PKG_VERSION"))));
    }
}

mod fixes {
    use std::path::Path;

    use crate::router::{Entry, Profile};
    use crate::{hub, installs, preset, server, ui};

    #[test]
    fn names_that_would_point_outside() {
        // "." and ".." as an install or export name would be the folder itself or its parent
        assert_eq!(installs::sanitize(".."), "_..");
        assert_eq!(installs::sanitize("."), "_.");
        assert_eq!(installs::sanitize(""), "_");
        assert_eq!(installs::sanitize("src-cuda b9000/x"), "src-cuda_b9000_x");
        assert_eq!(installs::sanitize(".config"), ".config");
        assert!(!installs::exists("..") && !installs::exists(".") && !installs::exists("a/b"));
        // file names from the Hugging Face API
        assert!(hub::safe_repo_path("Q4_K_M/model-00001-of-00002.gguf") && hub::safe_repo_path("org/name"));
        for bad in ["../x.gguf", "a/../b.gguf", "/etc/x.gguf", "a//b.gguf", "C:x.gguf", "a\\b.gguf", ".."] {
            assert!(!hub::safe_repo_path(bad), "{bad}");
        }
        // links from feeds
        assert!(server::browsable("https://huggingface.co/x") && server::browsable("http://127.0.0.1:8080"));
        assert!(
            !server::browsable("file:///etc/passwd") && !server::browsable("javascript:x") && !server::browsable("")
        );
    }

    #[test]
    fn state_file_survives() {
        let dir = std::env::temp_dir().join(format!("corral-state-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.json");
        assert!(ui::load_state_at(&path).1.is_none()); // first run: nothing to say
        let s = ui::State { last_model: Some("m.gguf".into()), ..ui::State::default() };
        ui::save_state_at(&s, &path).unwrap();
        assert!(!dir.join("state.json.tmp").exists());
        assert_eq!(ui::load_state_at(&path).0.last_model.as_deref(), Some("m.gguf"));
        // a broken file is set aside, not overwritten by the next save
        std::fs::write(&path, "{\"presets\": tronc").unwrap();
        let (fresh, problem) = ui::load_state_at(&path);
        assert!(fresh.last_model.is_none() && problem.is_some_and(|p| p.contains("state.json.bad")));
        assert_eq!(std::fs::read_to_string(dir.join("state.json.bad")).unwrap(), "{\"presets\": tronc");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn presets_used_by_profiles() {
        let m = Path::new("/modelli/qwen.gguf");
        let mut s = ui::State::default();
        s.presets.insert(ui::ps(m), [("coder".into(), ui::Preset::new()), ("lungo".into(), ui::Preset::new())].into());
        let entry = Entry { id: "coder".into(), model: ui::ps(m), preset: "coder".into(), startup: true };
        s.routers.insert("coding".into(), Profile { models: vec![entry], ..Profile::default() });
        // deleting it would leave the profile running the model with the defaults
        let err = preset::delete_preset(&mut s, m, "coder").unwrap_err();
        assert!(err.contains("coding"));
        // a rename carries over to the profile
        preset::rename_preset(&mut s, m, "coder", "coder-64k").unwrap();
        assert_eq!(s.routers["coding"].models[0].preset, "coder-64k");
        assert_eq!(preset::profiles_using(&s, m, "coder-64k"), ["coding"]);
        assert!(preset::delete_preset(&mut s, m, "lungo").is_ok());
        assert!(preset::delete_preset(&mut s, m, "coder-64k").is_err()); // still used (and the last one)
    }

    #[test]
    fn fit_output_with_an_unknown_flag() {
        let mut p = ui::Preset::new();
        assert!(preset::apply_fit(&mut p, "-c 32768 --no-mmap -ngl -2 -ot \"exps=CPU\""));
        assert_eq!((p["ctx"].as_str(), p["ngl"].as_str(), p["ot"].as_str()), ("32768", "all", "exps=CPU"));
        assert!(!preset::apply_fit(&mut ui::Preset::new(), "--no-mmap"));
    }
}

mod operations {
    use crate::server::{self, Gpu};
    use crate::{models, preset, ui};

    fn words(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn api_key_stays_off_the_command_line() {
        // the build tells which variables it reads, next to each option
        let help = "--api-key KEY      API key (default: none)\n                   (env: LLAMA_API_KEY)\n\
                    --old              gone\n                   (env: LLAMA_ARG_OLD) has been removed\n";
        let flags = preset::help_flags(help);
        assert!(flags.contains("--api-key") && flags.contains("LLAMA_API_KEY"));
        assert!(!flags.contains("LLAMA_ARG_OLD") && !flags.contains("--old"));
        // both spellings leave the command line; the last one is the one llama-server uses
        let mut cmd = words("srv --api-key first -m x.gguf --api-key=second --port 8080");
        assert_eq!(server::flag_value(&cmd, "--api-key").as_deref(), Some("second"));
        let mut env = vec![];
        server::key_to_env(&mut cmd, &mut env);
        assert_eq!(cmd, words("srv -m x.gguf --port 8080"));
        assert_eq!(env, [("LLAMA_API_KEY".to_string(), "second".to_string())]);
        // no key: nothing changes
        let mut env = vec![];
        server::key_to_env(&mut cmd, &mut env);
        assert!(env.is_empty() && cmd.len() == 5);
        // reachable from other computers without a key
        assert!(server::exposed_without_key(&words("srv --host 0.0.0.0")));
        assert!(server::exposed_without_key(&words("srv --host 192.168.1.5 --port 8080")));
        assert!(!server::exposed_without_key(&words("srv --host 0.0.0.0 --api-key k")));
        assert!(!server::exposed_without_key(&words("srv --host 0.0.0.0 --api-key-file /k")));
        for local in ["srv", "srv --host 127.0.0.1", "srv --host localhost", "srv --host ::1"] {
            assert!(!server::exposed_without_key(&words(local)), "{local}");
        }
    }

    #[test]
    fn files_with_the_key_are_private() {
        let dir = std::env::temp_dir().join(format!("corral-private-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // an exported preset is made to be shared: no key in it
        let model = std::path::Path::new("/modelli/qwen.gguf");
        let mut s = ui::State::default();
        let values = ui::Preset::from([("api_key".into(), "sk-segreto".into()), ("ctx".into(), "8192".into())]);
        s.presets.insert(ui::ps(model), [("default".to_string(), values)].into());
        let out = dir.join("qwen.preset.json");
        models::export_preset(&s, model, "default", &out).unwrap();
        let text = std::fs::read_to_string(&out).unwrap();
        assert!(text.contains("8192") && !text.contains("sk-segreto"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |p: &std::path::Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
            // new file, and one that was readable by everyone before
            let fresh = dir.join("nuovo.json");
            ui::write_private(&fresh, "x").unwrap();
            assert_eq!(mode(&fresh), 0o600);
            let old = dir.join("vecchio.json");
            std::fs::write(&old, "x").unwrap();
            std::fs::set_permissions(&old, std::fs::Permissions::from_mode(0o644)).unwrap();
            ui::write_private(&old, "nuovo").unwrap();
            assert_eq!((mode(&old), std::fs::read_to_string(&old).unwrap().as_str()), (0o600, "nuovo"));
            // state.json holds the presets, keys included
            let state = dir.join("state.json");
            ui::save_state_at(&s, &state).unwrap();
            assert_eq!(mode(&state), 0o600);
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn tunnel_over_ssh() {
        assert_eq!(server::local_address("http://127.0.0.1:8080"), Some(("127.0.0.1", "8080")));
        assert_eq!(server::local_address("http://localhost:9000/v1/models"), Some(("localhost", "9000")));
        assert_eq!(server::local_address("https://huggingface.co/x"), None); // not on this machine
        assert_eq!(server::local_address("http://192.168.1.5:8080"), None);
        let ssh = "192.168.1.20 51234 192.168.1.10 22";
        assert_eq!(
            server::tunnel_command("127.0.0.1", "8080", ssh, "ada"),
            "ssh -L 8080:127.0.0.1:8080 ada@192.168.1.10"
        );
        // another ssh port; 0.0.0.0 is reached through the loopback
        let ssh = "10.0.0.2 40000 10.0.0.1 2222";
        assert_eq!(server::tunnel_command("0.0.0.0", "9000", ssh, ""), "ssh -p 2222 -L 9000:127.0.0.1:9000 10.0.0.1");
    }

    #[test]
    fn gpu_counters() {
        // a laptop's line as nvidia-smi prints it, a card without a power reading, a name with a comma
        let csv = "NVIDIA GeForce RTX 4080 Laptop GPU, 338, 12282, 0, 40, 6.43\n\
                   Tesla K80, 100, 11441, 97, 71, [N/A]\n\
                   Acme, Inc. GPU, 1, 2, 3, 4, 5.6\n";
        let g = server::parse_nvidia(csv);
        assert_eq!(g.len(), 3);
        assert_eq!((g[0].used, g[0].total, g[0].busy, g[0].temp, g[0].watts), (338, 12282, Some(0), Some(40), Some(6)));
        assert_eq!(g[0].load(), "0% · 40°C · 6 W");
        assert_eq!((g[1].watts, g[1].load().as_str()), (None, "97% · 71°C"));
        assert_eq!(g[2].name, "Acme, Inc. GPU");
        assert_eq!(Gpu { name: "x".into(), ..Gpu::default() }.load(), "");
    }

    #[test]
    fn end_of_a_log() {
        let dir = std::env::temp_dir().join(format!("corral-log-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let log = dir.join("llama.log");
        let mut text = String::from("# 2026-11-14 10:00:00  llama-server -m x.gguf\n");
        text += "0.00.100.000 I srv    load_model: loading model\n";
        text += "0.05.000.000 E srv    operator(): got exception: out of memory\n";
        text += "0.05.000.100 I srv    cleaning up\n";
        std::fs::write(&log, &text).unwrap();
        assert_eq!(server::last_error(&log), "0.05.000.000 E srv    operator(): got exception: out of memory");
        // an abort prints a bare line: still the reason
        std::fs::write(&log, "0.00.1 I srv ok\n/src/ggml.c:123: GGML_ASSERT(n > 0) failed\n").unwrap();
        assert!(server::last_error(&log).contains("GGML_ASSERT"));
        // a model this llama.cpp can't read: the reason, not the "exiting" that follows it
        let mut text =
            String::from("0.00.27 E llama_model_load: error loading model: unknown model architecture: 'x'\n");
        text += "0.00.33 E srv    load_model: failed to load model, 'x.gguf'\n";
        text += "0.00.33 E srv  server_run_i: exiting due to model loading error\n";
        std::fs::write(&log, text).unwrap();
        let error = server::last_error(&log);
        assert!(error.contains("unknown model architecture") && error.contains("aggiornalo"), "{error}");
        // only repeats of the failure: still better than nothing
        std::fs::write(&log, "0.00.33 E srv  server_run_i: exiting due to model loading error\n").unwrap();
        assert!(server::last_error(&log).contains("exiting"));
        // nothing that looks like an error (a segfault writes nothing): no reason rather than a wrong one
        std::fs::write(&log, "uno\ndue\n").unwrap();
        assert_eq!(server::last_error(&log), "");
        // only the end is read: the first, cut line is dropped
        assert_eq!(server::read_tail(&log, 5), "due\n");
        assert_eq!(server::last_error(&dir.join("non-esiste.log")), "");
        let _ = std::fs::remove_dir_all(dir);
    }
}

mod remote_hosts {
    use std::path::Path;
    use std::process::Command;

    use crate::remote::{self, Problem};
    use crate::server::RunFile;
    use crate::{cli, dashboard, ui};

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("corral-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn info(alias: &str, router: bool) -> dashboard::Info {
        dashboard::Info {
            alias: alias.into(),
            model: "/m/qwen3-8b-Q4_K_M.gguf".into(),
            mmproj: String::new(),
            endpoint: "localhost:18300".into(),
            install: "rel-b1".into(),
            backend: "vulkan".into(),
            version: "b1 · abc".into(),
            settings: vec![("ctx".into(), "8192".into())],
            router,
            host: String::new(),
        }
    }

    /// A run file for a process that has already ended (nothing answers on its port either).
    fn gone_run(log: &Path) -> RunFile {
        let mut quick = Command::new(if cfg!(windows) { "cmd" } else { "true" });
        if cfg!(windows) {
            quick.args(["/C", "exit"]);
        }
        let mut child = quick.spawn().unwrap();
        let pid = child.id();
        child.wait().unwrap();
        RunFile {
            pid,
            base: "http://127.0.0.1:1".into(),
            api_key: "sk-segreto".into(),
            log: log.to_path_buf(),
            started: 1_700_000_000,
            keep: true,
            info: info("qwen3", false),
        }
    }

    #[test]
    fn hosts_from_the_ssh_config() {
        let dir = temp_dir("ssh-config");
        std::fs::create_dir_all(dir.join("conf.d")).unwrap();
        let config = "# commento\nHost workstation gpu-box\n    HostName 192.168.1.20\n\
                      Host *.lan !nas ?x\nHost=pi\nInclude conf.d/*.conf\nMatch all\nhost  workstation\n";
        std::fs::write(dir.join("config"), config).unwrap();
        std::fs::write(dir.join("conf.d").join("b.conf"), "Host laptop\n").unwrap();
        std::fs::write(dir.join("conf.d").join("a.conf"), "Host \"server due\" -oProxyCommand=x\n").unwrap();
        std::fs::write(dir.join("conf.d").join("note.txt"), "Host no\n").unwrap();
        let mut hosts = vec![];
        remote::read_config(&dir.join("config"), &dir, 0, &mut hosts);
        // patterns are not hosts, a name that would be an ssh option is dropped, includes in place and sorted
        assert_eq!(hosts, ["workstation", "gpu-box", "pi", "server due", "laptop"]);
        // a file that includes itself ends
        std::fs::write(dir.join("loop"), "Host a\nInclude loop\n").unwrap();
        let mut hosts = vec![];
        remote::read_config(&dir.join("loop"), &dir, 0, &mut hosts);
        assert_eq!(hosts, ["a"]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn versions_and_tunnels() {
        assert!(remote::older("0.3.0", "0.4.0") && remote::older("0.4.0", "0.10.0"));
        assert!(!remote::older("0.4.0", "0.4.0") && !remote::older("1.0.0", "0.4.0"));
        assert_eq!(remote::tunnel_target("http://localhost:8080"), Some(("127.0.0.1".into(), 8080)));
        assert_eq!(remote::tunnel_target("http://0.0.0.0:9000/v1"), Some(("127.0.0.1".into(), 9000)));
        assert_eq!(remote::tunnel_target("http://192.168.1.5:8080"), Some(("192.168.1.5".into(), 8080)));
        assert_eq!(remote::tunnel_target("http://senza-porta"), None);
        assert!(Problem::Ssh("ada@ws: Permission denied (publickey).".into()).hint("ws").contains("ssh-copy-id ws"));
        assert!(Problem::Missing.hint("ws").starts_with("u: "));
    }

    /// The command line that finds Corral on the host, run by a real sh: ~/.local/bin first.
    #[cfg(unix)]
    #[test]
    fn corral_on_the_host() {
        use std::os::unix::fs::PermissionsExt;
        let dir = temp_dir("remote-home");
        let bin = dir.join(".local").join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::write(bin.join("corral"), "#!/bin/sh\necho \"args: $*\"\n").unwrap();
        std::fs::set_permissions(bin.join("corral"), std::fs::Permissions::from_mode(0o755)).unwrap();
        let line = remote::corral_command(&["start", "/m/my model.gguf", "--preset", "it's"]);
        let run = |home: &Path| {
            let out = Command::new("sh")
                .arg("-c")
                .arg(&line)
                .env("HOME", home)
                .env("PATH", "/usr/bin:/bin")
                .output()
                .unwrap();
            (out.status.code(), String::from_utf8_lossy(&out.stdout).trim().to_string())
        };
        assert_eq!(run(&dir), (Some(0), "args: start /m/my model.gguf --preset it's".to_string()));
        // nowhere: 127, what the screens read as "no Corral there"
        assert_eq!(run(&dir.join("vuota")).0, Some(127));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// What `corral status --json` and `models --json` print is what the other computer reads.
    #[test]
    fn the_host_answers() {
        let dir = temp_dir("status-json");
        let mut state = ui::State::default();
        let run = gone_run(&dir.join("llama.log"));
        let mut router = gone_run(&dir.join("router.log"));
        router.info = info("coding", true);
        let profile = crate::router::Profile {
            models: vec![crate::router::Entry {
                id: "coder".into(),
                model: "/m/coder.gguf".into(),
                ..Default::default()
            }],
            ..Default::default()
        };
        state.routers.insert("coding".into(), profile);
        let json = cli::status_json(&[run, router], &state);
        let text = serde_json::to_string(&json).unwrap();
        assert!(!text.contains("sk-segreto")); // the key only goes through watch
        let status: remote::Status = serde_json::from_str(&text).unwrap();
        assert_eq!(status.version, remote::VERSION);
        assert_eq!((status.servers[0].info.alias.as_str(), status.servers[0].health.as_str()), ("qwen3", "no-answer"));
        assert!(status.servers[0].served.is_empty());
        // a router lists its models as clients see them
        assert_eq!(status.servers[1].served.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(), ["coder"]);
        let models: remote::Models = serde_json::from_str(&cli::models_json(&state).to_string()).unwrap();
        assert_eq!((models.profiles[0].name.as_str(), models.profiles[0].models), ("coding", 1));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// `corral watch` on one side, the link on the other: the hello with the key, the log
    /// (a greeting of the host's shell before the hello is not log), the GPUs, the end.
    #[cfg(unix)]
    #[test]
    fn watch_and_link() {
        use crate::server::Server;
        use std::time::{Duration, Instant};

        let dir = temp_dir("watch");
        let run = gone_run(&dir.join("llama.log"));
        let (tx, lines) = std::sync::mpsc::channel();
        for l in [
            "0.00.1 I srv    load_model: loading model",
            "0.00.2 I srv  llama_server: listening on http://127.0.0.1:18300",
        ] {
            tx.send(l.to_string()).unwrap();
        }
        let mut sent = b"Benvenuto!\n".to_vec();
        cli::stream(&mut sent, &run, &lines, "sk-segreto").unwrap();
        let text = String::from_utf8_lossy(&sent).into_owned();
        assert!(text.contains("\u{1e}hello ") && text.ends_with("\u{1e}end\n"));
        let wire = dir.join("wire.txt");
        std::fs::write(&wire, &sent).unwrap();
        let mut cat = Command::new("cat");
        cat.arg(&wire);
        let link = remote::connect(cat, 18399).unwrap();
        assert_eq!(link.hello["key"], "sk-segreto");
        let host_run = remote::Running {
            pid: run.pid,
            url: "http://localhost:18300".into(),
            health: "ok".into(),
            started: run.started,
            keep: true,
            log: "/home/ada/.corral/logs/llama.log".into(),
            info: info("qwen3", false),
            served: vec![],
        };
        let mut s = Server::remote("ws", &host_run, link);
        let t0 = Instant::now();
        while s.running() && t0.elapsed() < Duration::from_secs(10) {
            s.pump();
            std::thread::sleep(Duration::from_millis(20));
        }
        // the host said it ended: "terminato", not a lost link
        assert_eq!(s.dash.state().0, "terminato");
        assert_eq!(
            (s.host(), s.base.as_str(), s.api_key.as_str()),
            (Some("ws"), "http://127.0.0.1:18399", "sk-segreto")
        );
        assert!(s.dash.url.is_some()); // the log arrived: "listening on" was read
        assert!(!s.dash.lines.iter().any(|l| l.text.contains("benvenuto")));
        // the same stream cut before the end: the link dropped, the server may still run there
        let cut = text.split("\u{1e}end").next().unwrap();
        std::fs::write(&wire, cut).unwrap();
        let mut cat = Command::new("cat");
        cat.arg(&wire);
        let mut s = Server::remote("ws", &host_run, remote::connect(cat, 18399).unwrap());
        let t0 = Instant::now();
        while s.running() && t0.elapsed() < Duration::from_secs(10) {
            s.pump();
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(s.dash.state().0, "scollegato");
        // no hello: ssh's last words are the error
        let mut fail = Command::new("sh");
        fail.args(["-c", "echo 'ssh: connect to host ws port 22: Connection refused' >&2; exit 255"]);
        let e = remote::connect(fail, 18399).err().unwrap();
        assert_eq!(e, "ssh: connect to host ws port 22: Connection refused");
        let _ = std::fs::remove_dir_all(dir);
    }
}

mod rpc {
    use crate::{installs, preset, ui};

    #[test]
    fn devices_to_lend() {
        // a laptop's Vulkan devices: the dedicated GPU only (llama.cpp here leaves the integrated one out too)
        let both = "ggml_vulkan: Found 2 Vulkan devices:\n\
            ggml_vulkan: 0 = NVIDIA GeForce RTX 4080 Laptop GPU (NVIDIA) | uma: 0 | rebar: 1 | fp16: 1 |\n\
            ggml_vulkan: 1 = AMD Radeon 610M (RADV RAPHAEL_MENDOCINO) (radv) | uma: 1 | rebar: 1 | fp16: 1 |\n";
        assert_eq!(installs::dedicated_gpus(both), Some(vec!["Vulkan0".to_string()]));
        // only one kind (or not Vulkan at all): everything, as llama.cpp would
        let mini_pc = "ggml_vulkan: 0 = AMD Radeon 780M (RADV PHOENIX) (radv) | uma: 1 | fp16: 1 |\n";
        assert_eq!(installs::dedicated_gpus(mini_pc), None);
        assert_eq!(installs::dedicated_gpus("ggml_cuda_init: found 1 CUDA devices:\n"), None);
        let d =
            installs::device_line("  Vulkan0: NVIDIA GeForce RTX 4080 Laptop GPU (12282 MiB, 11237 MiB free)").unwrap();
        assert_eq!((d.name.as_str(), d.total, d.free), ("Vulkan0", 12282, 11237));
        // the preset's hosts: commas or spaces, no repeats, nothing ssh would read as an option
        let p = ui::Preset::from([("rpc".to_string(), "ws, nas ws -oProxyCommand=x".to_string())]);
        assert_eq!(preset::rpc_hosts(&p), ["ws", "nas"]);
        assert!(preset::rpc_hosts(&ui::Preset::new()).is_empty());
    }

    /// `corral rpc`'s stream on one side, the link on the other, and the command it changes.
    #[cfg(unix)]
    #[test]
    fn lend_and_add_rpc() {
        use crate::remote::{self, RpcLink};
        use crate::server::{self, Launch};

        let dir = std::env::temp_dir().join(format!("corral-rpc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // the RPC server has ended already: hello, then the end
        let mut child = std::process::Command::new("true").spawn().unwrap();
        child.wait().unwrap();
        let (_tx, output) = std::sync::mpsc::channel::<String>();
        let mut log = std::fs::File::create(dir.join("rpc.log")).unwrap();
        let devices = ["Vulkan0: NVIDIA GeForce RTX 4090 (24210 MiB, 23700 MiB free)"];
        let hello = serde_json::json!({ "version": "b12300 · abc", "backend": "cuda", "devices": devices });
        let mut sent = vec![];
        crate::cli::lend(&mut sent, &mut child, &output, &mut log, &hello).unwrap();
        let wire = dir.join("wire.txt");
        std::fs::write(&wire, &sent).unwrap();
        let mut cat = std::process::Command::new("cat");
        cat.arg(&wire);
        let link = RpcLink::new("ws", remote::connect(cat, 50060).unwrap());
        assert_eq!((link.host.as_str(), link.port, link.version.as_str()), ("ws", 50060, "b12300 · abc"));
        assert_eq!(link.devices, devices);
        // --rpc with the tunnel here; an explicit device list gets the lent ones (RPC0, …)
        let cmd: Vec<String> =
            "llama-server -m /m/x.gguf -dev Vulkan0 --port 8080".split(' ').map(String::from).collect();
        let info = crate::dashboard::Info {
            alias: "x".into(),
            model: "/m/x.gguf".into(),
            mmproj: String::new(),
            endpoint: "127.0.0.1:8080".into(),
            install: "i".into(),
            backend: "vulkan".into(),
            version: "b12347".into(),
            settings: vec![],
            router: false,
            host: String::new(),
        };
        let mut launch = Launch { cmd, env: vec![], info, notes: vec![] };
        server::add_rpc(&mut launch, std::slice::from_ref(&link), "b12347 · 8ca5a80").unwrap();
        assert_eq!(server::flag_value(&launch.cmd, "--rpc").as_deref(), Some("127.0.0.1:50060"));
        assert_eq!(server::flag_value(&launch.cmd, "-dev").as_deref(), Some("Vulkan0,RPC0"));
        assert_eq!(launch.info.settings.last().map(|s| s.1.as_str()), Some("ws"));
        // another llama.cpp build on the host: a warning (RPC wants the same version)
        assert!(launch.notes.iter().any(|n| n.contains("b12300") && n.contains("b12347")));
        let _ = std::fs::remove_dir_all(dir);
    }
}

mod tuning {
    use crate::preset::{self, FIELDS, Fit};
    use crate::router::OPTS;
    use crate::tune::{self, TIERS, Tier};
    use crate::ui::{Preset, State};
    use crate::{bench, help};

    fn p(pairs: &[(&str, &str)]) -> Preset {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn every_parameter_has_help() {
        for f in FIELDS.iter().chain(OPTS) {
            let (short, long) = help::of(f.key).unwrap_or_else(|| panic!("no help for {}", f.key));
            assert!(!short.is_empty() && !long.is_empty(), "{}", f.key);
            assert!(!help::flag(f).is_empty());
            assert!(!help::explain(f).is_empty());
        }
        assert_eq!(help::of("nope"), None);
        assert_eq!(help::wrap("uno due tre quattro", 9), ["uno due", "tre", "quattro"]);
        assert_eq!(help::wrap("", 9), Vec::<String>::new());
        let ctx = FIELDS.iter().find(|f| f.key == "ctx").unwrap();
        assert_eq!(help::flag(ctx), "-c");
        let jinja = FIELDS.iter().find(|f| f.key == "jinja").unwrap();
        assert_eq!(help::flag(jinja), "--jinja / --no-jinja");
        let metrics = FIELDS.iter().find(|f| f.key == "metrics").unwrap();
        assert_eq!(help::flag(metrics), "--metrics");
        let usage = FIELDS.iter().find(|f| f.key == "usage").unwrap();
        assert!(help::flag(usage).contains("--rerank"));
        // a choice list shows its values, the server's default included
        let reasoning_format = FIELDS.iter().find(|f| f.key == "reasoning_format").unwrap();
        let text: String = help::explain(reasoning_format).iter().map(|l| l.to_string()).collect();
        assert!(text.contains("(server) · none · deepseek"), "{text}");
    }

    #[test]
    fn fit_flags_in_presets() {
        assert!(preset::check("fit_target", "1024").is_ok());
        assert!(preset::check("fit_target", "2048,512").is_ok());
        assert!(preset::check("fit_target", "molta").is_err());
        assert!(preset::check("fit_ctx", "8192").is_ok());
        assert!(preset::check("fit_ctx", "8k").is_err());
        let args = preset::args(&p(&[("fit_target", "3072"), ("fit_ctx", "16384")]), None, None);
        assert_eq!(args, ["-fitt", "3072", "-fitc", "16384"]);
        assert!(preset::MEMORY_KEYS.contains(&"fit_target") && preset::MEMORY_KEYS.contains(&"fit_ctx"));
    }

    #[test]
    fn tier_plans() {
        assert_eq!(tune::quarter(12282), 3072);
        assert_eq!(tune::quarter(8192), 2048);
        assert_eq!(tune::quarter(2048), 1024); // never under llama.cpp's own margin
        assert_eq!(TIERS.map(Tier::name), ["lite", "normal", "max"]);
        assert!(TIERS.iter().all(|t| !t.desc().is_empty()));

        // the base's own settings stay; the memory ones are the tier's
        let base = p(&[("alias", "mio"), ("port", "11434"), ("temp", "0.2"), ("ngl", "99"), ("ctx", "65536")]);
        let lite = tune::plan(Tier::Lite, &base, 40960, 12282);
        assert_eq!(lite.attempts.len(), 1);
        let a = &lite.attempts[0];
        assert_eq!((a["alias"].as_str(), a["port"].as_str(), a["temp"].as_str()), ("mio", "11434", "0.2"));
        assert_eq!((a["ctx"].as_str(), a["ctk"].as_str(), a["ctv"].as_str()), ("8192", "q8_0", "q8_0"));
        assert_eq!((a["fa"].as_str(), a["np"].as_str(), a["fit_target"].as_str()), ("on", "1", "3072"));
        assert_eq!(a["ngl"], "auto", "the auto-fit picks the layers");
        // a short model context caps the lite one; no GPU: no margin to keep
        let lite = tune::plan(Tier::Lite, &base, 4096, 0);
        assert_eq!(lite.attempts[0]["ctx"], "4096");
        assert_eq!(lite.attempts[0]["fit_target"], "");

        let normal = tune::plan(Tier::Normal, &base, 40960, 12282);
        assert_eq!(normal.max_ctx, Some(32768));
        let kv: Vec<(&str, &str)> =
            normal.attempts.iter().map(|a| (a["ctk"].as_str(), a["fit_ctx"].as_str())).collect();
        assert_eq!(kv, [("f16", "32768"), ("q8_0", "32768"), ("q8_0", "8192")]);
        assert!(normal.attempts.iter().all(|a| a["ctx"] == "0"), "the auto-fit sizes the context");
        let fa: Vec<&str> = normal.attempts.iter().map(|a| a["fa"].as_str()).collect();
        assert_eq!(fa, ["auto", "on", "on"], "flash attention forced only for a quantized cache");

        let max = tune::plan(Tier::Max, &base, 131072, 12282);
        assert_eq!(max.max_ctx, None);
        assert_eq!(max.attempts.len(), 2);
        assert_eq!((max.attempts[0]["ctk"].as_str(), max.attempts[0]["fit_ctx"].as_str()), ("f16", "131072"));
        assert_eq!((max.attempts[1]["ctk"].as_str(), max.attempts[1]["fit_ctx"].as_str()), ("q8_0", ""));
        assert_eq!((max.attempts[1]["ubatch"].as_str(), max.attempts[1]["fit_target"].as_str()), ("1024", "512"));
        assert_eq!(tune::plan(Tier::Max, &base, 0, 12282).attempts.len(), 1, "unknown context: q8_0 only");
    }

    #[test]
    fn whole_model_in_vram() {
        assert!(tune::all_on_gpu(&p(&[("ngl", "auto")]), 36));
        assert!(tune::all_on_gpu(&p(&[("ngl", "all")]), 36));
        assert!(tune::all_on_gpu(&p(&[("ngl", "37")]), 36), "36 blocks + the output layer");
        assert!(!tune::all_on_gpu(&p(&[("ngl", "35")]), 36));
        assert!(!tune::all_on_gpu(&p(&[("ngl", "40")]), 0), "unknown shape: a number proves nothing");
        assert!(!tune::all_on_gpu(&p(&[("ngl", "auto"), ("ot", r"\.ffn_.*_exps=CPU")]), 36));
    }

    #[test]
    fn projected_memory() {
        let fit = |verdict: &[&str]| Fit {
            ok: true,
            breakdown: vec![],
            verdict: verdict.iter().map(|s| s.to_string()).collect(),
            tail: vec![],
            fitted: String::new(),
        };
        let v = fit(&[
            "projected to use 14167 MiB of device memory vs. 11390 MiB of free device memory",
            "cannot meet free memory target of 512 MiB, need to reduce device memory by 3288 MiB",
            "projected to use 8204 MiB of device memory vs. 11390 MiB of free device memory",
        ]);
        assert_eq!(tune::projected(&v), Some((8204, 11390)), "the last pass counts");
        assert_eq!(tune::projected(&fit(&["context size reduced from 40960 to 17664"])), None);
    }

    #[test]
    fn saved_as_presets() {
        let dir = std::env::temp_dir().join(format!("corral-tune-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("Qwen3-8B-Q8_0.gguf");
        std::fs::write(&model, b"GGUF").unwrap();
        let mut state = State::default();
        let lite = p(&[("ctx", "8192"), ("ctk", "q8_0")]);
        let max = p(&[("ctx", "33024"), ("ctk", "q8_0"), ("ubatch", "1024")]);
        tune::save(&mut state, &model, &[(Tier::Lite, &lite), (Tier::Max, &max)], Tier::Max).unwrap();
        let mut names = preset::preset_names(&state, &model);
        names.sort();
        assert_eq!(names, ["default", "lite", "max"], "a model with no preset keeps its default");
        assert_eq!(preset::preset_name(&state, &model), "max");
        assert_eq!(preset::get_preset(&state, &model)["ubatch"], "1024");
        // again: the tiers are replaced, the selection follows
        tune::save(&mut state, &model, &[(Tier::Lite, &lite)], Tier::Lite).unwrap();
        assert_eq!(preset::preset_name(&state, &model), "lite");
        assert_eq!(preset::preset_names(&state, &model).len(), 3);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn bench_gets_the_memory_settings() {
        let tuned = p(&[
            ("ngl", "30"),
            ("batch", "4096"),
            ("ubatch", "1024"),
            ("ts", "3,1"),
            ("n_cpu_moe", "8"),
            ("cpu_moe", "on"),
            ("ot", "a=CPU,b=CPU"),
            ("kv_offload", "off"),
        ]);
        let args = bench::params(&tuned);
        let joined = args.join(" ");
        assert!(joined.starts_with("-ngl 30 -b 4096 -ub 1024 -ncmoe 8 -ts 3/1 -ot "), "{joined}");
        assert!(joined.ends_with(";a=CPU;b=CPU -nkvo 1"), "{joined}");
        assert!(bench::params(&p(&[("ngl", "auto"), ("kv_offload", "on")])).is_empty());
    }
}

/// Against the fakes of `crate::fake`: the web and, on Unix, llama.cpp.
mod fakes {
    use crate::{fake, hub, installs, news, preset};

    #[test]
    fn hugging_face_search_and_files() {
        let found = hub::search("qwen").unwrap();
        assert_eq!(found.len(), 5);
        assert_eq!(found[0].id, "unsloth/Qwen3-8B-GGUF");
        assert_eq!(found[0].params, Some(8_190_000_000));
        assert_eq!(found[4].params, None, "an odd entry doesn't sink the list");
        let files = hub::repo_files("unsloth/Qwen3-8B-GGUF").unwrap();
        assert_eq!(files.quants.iter().map(|q| q.quant.as_str()).collect::<Vec<_>>(), ["Q4_K_M", "Q8_0"]);
        assert_eq!(files.mmprojs.len(), 1);
        assert!(!files.gated);
        assert!(hub::repo_files("x/missing-GGUF").is_err());
        let suggested = hub::suggest(16 << 30, 32 << 30).unwrap();
        assert!(suggested.iter().any(|(cat, r)| *cat == "coding" && r.id.contains("Coder")), "{}", suggested.len());
        assert!(suggested.iter().all(|(_, r)| !r.id.starts_with("someone/")), "untrusted orgs stay out");
    }

    #[test]
    fn releases_and_news() {
        let rels = installs::releases().unwrap();
        assert_eq!(rels[0].tag_name, "b9999");
        assert_eq!(installs::latest_tag().as_deref(), Some("b9999"));
        let items = news::llama().unwrap();
        assert!(items[0].title.starts_with("b9999 · server: faster slots"), "{}", items[0].title);
        assert!(items.iter().any(|i| i.title.contains("release stabile")));
        let feed = news::feed("https://example.org/feed.xml").unwrap();
        assert_eq!(feed.len(), 2);
        assert_eq!(feed[0].title, "A new model & more");
        let models = news::models().unwrap();
        assert!(models.iter().any(|m| m.tag == "unsloth/Qwen3-8B-GGUF"));
        let all = news::fetch(&["https://example.org/feed.xml".to_string()]);
        assert!(all.items.len() >= 4 && all.errors.is_empty(), "{:?}", all.errors);
        let _ = fake::local("http://127.0.0.1:1/x");
    }

    #[cfg(unix)]
    #[test]
    fn llama_cpp_tools() {
        let inst = fake::install();
        assert_eq!(installs::version(&inst), "b9999 · abc1234");
        let devs = installs::devices(&inst);
        assert_eq!(devs.len(), 1);
        assert_eq!((devs[0].name.as_str(), devs[0].total, devs[0].free), ("Vulkan0", 12282, 11474));
        let flags = preset::supported_flags(&installs::server(&inst)).unwrap();
        assert!(flags.contains("--alias") && flags.contains("-fitt") && flags.contains("LLAMA_API_KEY"));
        assert!(!flags.contains("--old-thing"), "removed options are left out");
    }
}
