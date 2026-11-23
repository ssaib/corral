//! News: llama.cpp builds and releases (GitHub), new GGUF models from the best-known quantizers
//! (Hugging Face), AI news from RSS/Atom feeds and Corral's own changelog. Fetched together and
//! cached for an hour in cache/news.json. Pure logic (blocking HTTP: run it in a thread).

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::sync::LazyLock;
use std::thread;
use std::time::Duration;

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ui::*;
use crate::{hub, installs};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Item {
    /// SOURCES key
    pub src: String,
    /// shown: "llama.cpp", the org, the feed's title, "Corral"
    pub source: String,
    /// unix seconds (0 = unknown)
    pub date: u64,
    pub title: String,
    pub text: String,
    pub url: String,
    /// llama.cpp: the release tag; models: the repo
    pub tag: String,
    /// models: memory needed at Q4_K_M (0 = unknown)
    pub need: u64,
}

pub const SOURCES: [(&str, &str); 4] =
    [("llama", "llama.cpp"), ("models", "modelli"), ("feeds", "notizie IA"), ("corral", "Corral")];

pub const DEFAULT_FEEDS: [&str; 6] = [
    "https://huggingface.co/blog/feed.xml",
    "https://www.reddit.com/r/LocalLLaMA/top/.rss?t=week",
    "https://simonwillison.net/atom/everything/",
    "https://the-decoder.com/feed/",
    "https://ollama.com/blog/rss.xml",
    "https://lmstudio.ai/rss.xml",
];

const TTL: u64 = 3600;
const PER_FEED: usize = 25;

#[derive(Default, Serialize, Deserialize)]
pub struct News {
    pub at: u64,
    pub items: Vec<Item>,
    pub errors: Vec<String>,
}

fn cache_file() -> PathBuf {
    BASE.join("cache").join("news.json")
}

pub fn cached() -> Option<News> {
    fs::read_to_string(cache_file()).ok().and_then(|s| serde_json::from_str(&s).ok())
}

pub fn stale(n: &News) -> bool {
    unix_now().saturating_sub(n.at) > TTL
}

/// Items newer than the last visit to the news screen (cached ones and Corral's changelog).
pub fn unseen(seen: Option<u64>) -> usize {
    let Some(seen) = seen else { return 0 };
    let cached = cached().map(|n| n.items).unwrap_or_default();
    cached.iter().chain(&corral()).filter(|i| i.date > seen).count()
}

/// Every source at once (Corral's changelog is built in, see `corral`); the result is cached.
pub fn fetch(feeds: &[String]) -> News {
    let mut out = News { at: unix_now(), ..News::default() };
    thread::scope(|sc| {
        let llama = sc.spawn(llama);
        let models = sc.spawn(models);
        let feeds: Vec<_> = feeds.iter().map(|u| (u, sc.spawn(move || feed(u)))).collect();
        let mut add = |r: thread::Result<Result<Vec<Item>, String>>, what: &str| match r {
            Ok(Ok(v)) => out.items.extend(v),
            Ok(Err(e)) => out.errors.push(format!("{what}: {e}")),
            Err(_) => out.errors.push(format!("{what}: errore interno")),
        };
        add(llama.join(), "GitHub");
        add(models.join(), "Hugging Face");
        for (u, h) in feeds {
            add(h.join(), u);
        }
    });
    out.items.sort_by_key(|a| std::cmp::Reverse(a.date));
    let _ = fs::create_dir_all(BASE.join("cache"));
    let _ = fs::write(cache_file(), serde_json::to_string(&out).unwrap_or_default());
    out
}

// -- sources ---------------------------------------------------------------------------------------

/// The latest releases: a build per merged change (bNNNN, its commit message) and the stable ones.
pub fn llama() -> Result<Vec<Item>, String> {
    let rels: Vec<Value> = get_json(installs::gh(&format!("{}/releases?per_page=30", installs::GH_API)))?;
    Ok(rels
        .iter()
        .map(|r| {
            let tag = r["tag_name"].as_str().unwrap_or_default().to_string();
            let text = release_text(r["body"].as_str().unwrap_or_default());
            let head = text.lines().next().unwrap_or_default();
            let title = if !r["prerelease"].as_bool().unwrap_or(true) {
                format!("llama.cpp {tag} · release stabile")
            } else if head.is_empty() {
                tag.clone()
            } else {
                format!("{tag} · {head}")
            };
            Item {
                src: "llama".into(),
                source: "llama.cpp".into(),
                date: parse_date(r["published_at"].as_str().unwrap_or_default()).unwrap_or(0),
                title,
                text,
                url: r["html_url"].as_str().unwrap_or_default().into(),
                tag,
                need: 0,
            }
        })
        .collect())
}

/// A release body as text: a build's commit message (inside <details>), else the notes; no HTML,
/// no download link lists.
pub fn release_text(body: &str) -> String {
    let body = match (body.find("<details"), body.find("</details>")) {
        (Some(a), Some(b)) if a < b => &body[a..b],
        _ => body.split("**Website:**").next().unwrap_or(body),
    };
    let lines: Vec<String> = strip_tags(body)
        .lines()
        .map(|l| unescape(l.trim().trim_start_matches('#').trim()))
        .filter(|l| !l.is_empty())
        .take(40)
        .collect();
    lines.join("\n")
}

/// New repos of each quantizer and the trending ones among trusted orgs (chat/vision models only).
pub fn models() -> Result<Vec<Item>, String> {
    let lists: Vec<(bool, Result<Vec<hub::Repo>, String>)> = thread::scope(|sc| {
        let mut hs: Vec<_> = hub::QUANTIZERS
            .iter()
            .map(|&a| (false, sc.spawn(move || hub::list(None, Some(a), "createdAt", 15))))
            .collect();
        hs.push((true, sc.spawn(|| hub::list(None, None, "trendingScore", 50))));
        hs.into_iter().map(|(t, h)| (t, h.join().unwrap_or_else(|_| Err("errore interno".into())))).collect()
    });
    let mut out = vec![];
    let mut seen = HashSet::new(); // a repo can be both new and trending
    let mut err = None;
    for (trending, r) in lists {
        let repos = match r {
            Ok(v) => v,
            Err(e) => {
                err = Some(e);
                continue;
            }
        };
        for r in repos.into_iter().filter(|r| hub::is_llm(r) && hub::is_trusted(&r.id)) {
            if !seen.insert(r.id.clone()) {
                continue;
            }
            let (org, name) = r.id.split_once('/').unwrap_or(("", &r.id));
            let params = r.params.unwrap_or(0);
            let mut facts = vec![r.arch.clone().unwrap_or_default(), format!("{} parametri", params_label(params))];
            if let Some(c) = r.ctx {
                facts.push(format!("contesto {}", hub::thousands(c)));
            }
            facts.push(format!("{} download", hub::thousands(r.downloads)));
            out.push(Item {
                src: "models".into(),
                source: org.into(),
                date: r.created.as_deref().and_then(parse_date).unwrap_or(0),
                title: if trending { format!("{name} · in tendenza") } else { name.into() },
                text: facts.into_iter().filter(|f| !f.is_empty()).collect::<Vec<_>>().join(" · "),
                url: format!("{}/{}", hub::endpoint(), r.id),
                tag: r.id.clone(),
                need: hub::q4_need(params),
            });
        }
    }
    match err {
        Some(e) if out.is_empty() => Err(e),
        _ => Ok(out),
    }
}

fn params_label(n: u64) -> String {
    match n {
        0 => "?".into(),
        n if n >= 1_000_000_000 => format!("{:.1}B", n as f64 / 1e9).replace(".0B", "B"),
        n => format!("{}M", n / 1_000_000),
    }
}

pub fn feed(url: &str) -> Result<Vec<Item>, String> {
    let xml = get(url).timeout(Duration::from_secs(20)).call().map_err(|e| e.to_string())?;
    let xml = xml.into_string().map_err(|e| e.to_string())?;
    let (title, mut items) = parse_feed(&xml);
    let host = url.split('/').nth(2).unwrap_or(url).trim_start_matches("www.");
    let sub = url.split("/r/").nth(1).and_then(|r| r.split('/').next()).filter(|_| host.ends_with("reddit.com"));
    let source = match (sub, title.is_empty()) {
        (Some(r), _) => format!("r/{r}"), // its own title is "top scoring links : …"
        (None, true) => host.to_string(),
        (None, false) => clip(&title, 28),
    };
    // some feeds carry their whole history: the newest ones are the news
    items.sort_by_key(|i| std::cmp::Reverse(i.date));
    items.truncate(PER_FEED);
    for i in &mut items {
        i.source = source.clone();
    }
    Ok(items)
}

/// RSS 2.0 or Atom: (channel title, items). Lenient: a missing field is left empty.
pub fn parse_feed(xml: &str) -> (String, Vec<Item>) {
    let mut blocks = all(xml, "item");
    if blocks.is_empty() {
        blocks = all(xml, "entry");
    }
    let head = xml.find("<item").or_else(|| xml.find("<entry")).map_or(xml, |i| &xml[..i]);
    let title = inner(head, "title").map(text).unwrap_or_default();
    let items = blocks
        .iter()
        .map(|b| {
            let date = ["pubDate", "published", "updated", "dc:date"].iter().find_map(|t| inner(b, t));
            let body = ["description", "summary", "content:encoded", "content"].iter().find_map(|t| inner(b, t));
            Item {
                src: "feeds".into(),
                source: String::new(),
                date: date.and_then(|d| parse_date(d.trim())).unwrap_or(0),
                title: inner(b, "title").map(text).unwrap_or_default(),
                text: clip(&body.map(text).unwrap_or_default(), 700),
                url: link(b),
                tag: String::new(),
                need: 0,
            }
        })
        .filter(|i| !i.title.is_empty())
        .collect();
    (title, items)
}

/// RSS <link>url</link>, else Atom's <link href="…"> (rel="alternate" or no rel).
fn link(b: &str) -> String {
    if let Some(l) = inner(b, "link").map(text).filter(|l| l.starts_with("http")) {
        return l;
    }
    static RE_LINK: LazyLock<Regex> = re!(r"<link\b([^>]*)>");
    static RE_HREF: LazyLock<Regex> = re!(r#"href\s*=\s*"([^"]*)""#);
    let links = RE_LINK.captures_iter(b).map(|m| m[1].to_string());
    let pick = links.filter(|a| !a.contains("rel=") || a.contains("alternate")).find_map(|a| {
        let h = RE_HREF.captures(&a)?;
        Some(unescape(&h[1]))
    });
    pick.unwrap_or_default()
}

/// Byte range of the body of the first `<name …>…</name>` in `s` (not a self-closing `<name/>`).
fn element(s: &str, name: &str) -> Option<(usize, usize)> {
    let (open, close) = (format!("<{name}"), format!("</{name}>"));
    let mut from = 0;
    while let Some(i) = s[from..].find(&open) {
        let at = from + i + open.len(); // just after "<name"
        let rest = &s[at..];
        if !rest.starts_with(['>', ' ', '\t', '\r', '\n']) {
            from = at; // a longer tag name (<titles>, <content:encoded> for <content>)
            continue;
        }
        let gt = rest.find('>')?;
        if rest[..gt].ends_with('/') {
            from = at; // <name/>: no body
            continue;
        }
        let start = at + gt + 1;
        let end = start + s[start..].find(&close)?;
        return Some((start, end));
    }
    None
}

/// Inner text of the first `<name …>…</name>`.
fn inner<'a>(s: &'a str, name: &str) -> Option<&'a str> {
    element(s, name).map(|(start, end)| &s[start..end])
}

/// Every `<name …>…</name>` body.
fn all<'a>(s: &'a str, name: &str) -> Vec<&'a str> {
    let mut out = vec![];
    let mut from = 0;
    while let Some((start, end)) = element(&s[from..], name) {
        out.push(&s[from + start..from + end]);
        from += end; // the next search starts at this one's closing tag
    }
    out
}

/// Feed text: CDATA as is, else entities decoded; then no tags, entities decoded, one line.
fn text(raw: &str) -> String {
    let raw = raw.trim();
    let html = match raw.strip_prefix("<![CDATA[").and_then(|r| r.strip_suffix("]]>")) {
        Some(c) => c.to_string(),
        None => unescape(raw),
    };
    unescape(&strip_tags(&html)).split_whitespace().collect::<Vec<_>>().join(" ")
}

fn strip_tags(s: &str) -> String {
    static RE_TAG: LazyLock<Regex> = re!(r"(?s)<[^>]*>");
    RE_TAG.replace_all(s, " ").into_owned()
}

pub fn unescape(s: &str) -> String {
    static RE_ENT: LazyLock<Regex> = re!(r"&(#[xX][0-9a-fA-F]+|#[0-9]+|[a-zA-Z]+);");
    RE_ENT
        .replace_all(s, |m: &regex::Captures| {
            let e = &m[1];
            let code = match e.strip_prefix('#') {
                Some(n) => match n.strip_prefix(['x', 'X']) {
                    Some(h) => u32::from_str_radix(h, 16).ok(),
                    None => n.parse().ok(),
                },
                None => match e {
                    "amp" => Some(38),
                    "lt" => Some(60),
                    "gt" => Some(62),
                    "quot" => Some(34),
                    "apos" => Some(39),
                    "nbsp" => Some(32),
                    "hellip" => Some(0x2026),
                    "mdash" => Some(0x2014),
                    "ndash" => Some(0x2013),
                    "rsquo" | "lsquo" => Some(0x2019),
                    "rdquo" | "ldquo" => Some(0x201c),
                    _ => None,
                },
            };
            code.and_then(char::from_u32).map_or(m[0].to_string(), String::from)
        })
        .into_owned()
}

// -- Corral's changelog ----------------------------------------------------------------------------

const CHANGELOG: &str = include_str!("../CHANGELOG.md");

pub fn corral() -> Vec<Item> {
    parse_changelog(CHANGELOG)
}

/// `## 0.2.0 — 2026-11-13 — what` headings, each followed by its "- …" lines.
pub fn parse_changelog(md: &str) -> Vec<Item> {
    md.split("\n## ")
        .skip(1)
        .map(|sec| {
            let (head, body) = sec.split_once('\n').unwrap_or((sec, ""));
            let mut parts = head.split(" — ").map(str::trim);
            let (ver, date, what) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""), parts.next());
            let text: Vec<String> = body
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(|l| l.strip_prefix("- ").map_or(l.to_string(), |b| format!("• {b}")))
                .collect();
            Item {
                src: "corral".into(),
                source: "Corral".into(),
                date: parse_date(&format!("{date}T12:00:00Z")).unwrap_or(0),
                title: what.map_or(format!("Corral {ver}"), |w| format!("Corral {ver} · {w}")),
                text: text.join("\n"),
                url: String::new(),
                tag: ver.into(),
                need: 0,
            }
        })
        .collect()
}

// -- dates -------------------------------------------------------------------------------------------

/// Days since 1970-01-01 of a proleptic Gregorian date (H. Hinnant's days_from_civil).
fn days(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468
}

/// (year, month, day) of a day count since 1970-01-01 (civil_from_days).
fn civil(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// RFC 3339 (Atom, GitHub, Hugging Face) or RFC 2822 (RSS) to unix seconds.
pub fn parse_date(s: &str) -> Option<u64> {
    static RE_3339: LazyLock<Regex> =
        re!(r"^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?\s*(Z|[+-]\d{2}:?\d{2})?)?");
    static RE_2822: LazyLock<Regex> =
        re!(r"(\d{1,2})\s+([A-Za-z]{3})[a-z]*\s+(\d{4})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([+-]\d{4}|[A-Za-z]+)?");
    const MONTHS: [&str; 12] = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
    let n = |c: Option<regex::Match>| c.map_or(0, |m| m.as_str().parse::<i64>().unwrap_or(0));
    let (y, mo, d, h, mi, sec, zone) = if let Some(c) = RE_3339.captures(s.trim()) {
        (n(c.get(1)), n(c.get(2)), n(c.get(3)), n(c.get(4)), n(c.get(5)), n(c.get(6)), c.get(7))
    } else {
        let c = RE_2822.captures(s)?;
        let mo = MONTHS.iter().position(|m| c[2].eq_ignore_ascii_case(m))? as i64 + 1;
        (n(c.get(3)), mo, n(c.get(1)), n(c.get(4)), n(c.get(5)), n(c.get(6)), c.get(7))
    };
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) {
        return None;
    }
    let off = match zone.map(|z| z.as_str()) {
        Some(z) if z.starts_with(['+', '-']) => {
            let digits: String = z[1..].chars().filter(char::is_ascii_digit).collect();
            let v = digits.get(..2).and_then(|x| x.parse::<i64>().ok()).unwrap_or(0) * 3600
                + digits.get(2..4).and_then(|x| x.parse::<i64>().ok()).unwrap_or(0) * 60;
            if z.starts_with('-') { -v } else { v }
        }
        Some(z) => match z.to_ascii_uppercase().as_str() {
            "EST" | "CDT" => -5 * 3600,
            "EDT" => -4 * 3600,
            "CST" | "MDT" => -6 * 3600,
            "MST" | "PDT" => -7 * 3600,
            "PST" => -8 * 3600,
            _ => 0, // Z, GMT, UT, UTC and the unknown ones
        },
        None => 0,
    };
    let t = days(y, mo, d) * 86_400 + h * 3600 + mi * 60 + sec - off;
    u64::try_from(t).ok()
}

/// "5 min fa", "3 h fa", "2 g fa", then the date (dd/mm/yy, UTC).
pub fn ago(now: u64, t: u64) -> String {
    if t == 0 {
        return "—".into();
    }
    match now.saturating_sub(t) {
        d @ 0..3600 => format!("{} min fa", (d / 60).max(1)),
        d @ 3600..86_400 => format!("{} h fa", d / 3600),
        d @ 86_400..604_800 => format!("{} g fa", d / 86_400),
        _ => {
            let (y, m, d) = civil((t / 86_400) as i64);
            format!("{d:02}/{m:02}/{:02}", y % 100)
        }
    }
}
