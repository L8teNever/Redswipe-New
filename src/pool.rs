//! Gescannt-Modus ("Vorrat"): lädt im Hintergrund laufend Beiträge aus deiner Quelle, aus allen Sortierungen
//! gemischt (hot, new, rising, top Tag bis alle), lässt sie scannen und gibt nur fertig gescannte, noch nicht
//! gesehene Beiträge aus, nach Geschmack sortiert. Läuft auch, wenn niemand die Seite offen hat.
//!
//! Gesehene Beiträge (einmal im Bild gewesen) werden gemerkt und nie wieder vorgeschlagen, auch nicht im
//! normalen Live-Feed.

use std::{
    collections::{HashMap, HashSet},
    env, fs,
    path::PathBuf,
    sync::{Mutex, OnceLock},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Notify;

use crate::{AppState, Post, err, fetch_listing, listing_url, optimize, parse_source, redgifs, scan, taste};

pub const DEFAULT_SOURCE: &str = "https://www.reddit.com/user/Brilliant-Nothing684/m/redswipe/";
/// Höchstens so viele Beiträge im Vorrat.
const MAX_POOL: usize = 1500;
/// Bis to so viele fertig gescannte Videos im Vorrat sammeln (über Nacht).
const TARGET_READY: usize = 500;
/// Höchstens so viele noch nicht fertig gescannte Beiträge, sonst wird erst einmal nichts nachgeladen.
const BACKLOG: usize = 120;
/// Unter so vielen fertigen Beiträgen wird zügig nachgeladen.
const LOW_READY: usize = 500;
const MAX_SEEN: usize = 50_000;
/// Scan-Ergebnisse liegen 3 Tage im Cache, Vorrats-Einträge fliegen etwas früher raus.
const MAX_AGE: u64 = 60 * 60 * 60;
const PAGES: usize = 3;
/// Alle Sortierungen, aus denen gemischt wird: (Sortierung, Zeitraum).
const LISTINGS: &[(&str, &str)] =
    &[("hot", "day"), ("new", "day"), ("rising", "day"), ("top", "day"), ("top", "week"), ("top", "month"), ("top", "year"), ("top", "all")];

#[derive(Serialize, Deserialize, Clone)]
struct Entry {
    post: Value,
    added: u64,
    /// Merkmale für die Geschmacks-Bewertung, sobald alle Scans fertig sind.
    #[serde(default)]
    feats: Option<taste::Feats>,
    /// Nicht scanbar (z. B. nur eingebettet) oder alle Scans fehlgeschlagen: wird nie angezeigt.
    #[serde(default)]
    dead: bool,
}

#[derive(Serialize, Deserialize, Default)]
struct Store {
    #[serde(default)]
    source: String,
    #[serde(default)]
    posts: HashMap<String, Entry>,
    #[serde(default)]
    seen: Vec<String>,
    #[serde(default)]
    last_fetch: u64,
    /// RedGIFs-Nischen mischen (Standard an) und welche; Seite pro Nische (0 = erschöpft).
    #[serde(default)]
    rg_init: bool,
    #[serde(default)]
    rg_on: bool,
    #[serde(default)]
    rg_niches: Vec<String>,
    #[serde(default)]
    rg_pages: HashMap<String, u32>,
    #[serde(skip)]
    seen_set: HashSet<String>,
    #[serde(skip)]
    dirty: bool,
    #[serde(skip)]
    fetching: bool,
}

fn path() -> PathBuf {
    PathBuf::from(env::var("VPN_DIR").unwrap_or_else(|_| "/vpn".into())).join("pool.json")
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn store() -> &'static Mutex<Store> {
    static S: OnceLock<Mutex<Store>> = OnceLock::new();
    S.get_or_init(|| {
        let mut s = fs::read_to_string(path()).ok().and_then(|t| serde_json::from_str::<Store>(&t).ok()).unwrap_or_default();
        if s.source.trim().is_empty() {
            s.source = DEFAULT_SOURCE.into();
        }
        if !s.rg_init {
            s.rg_init = true;
            s.rg_on = true;
            s.rg_niches = redgifs::DEFAULT_NICHES.iter().map(|x| x.to_string()).collect();
        }
        s.seen_set = s.seen.iter().cloned().collect();
        Mutex::new(s)
    })
}

fn lock() -> std::sync::MutexGuard<'static, Store> {
    store().lock().unwrap_or_else(|e| e.into_inner())
}

fn wake() -> &'static Notify {
    static N: OnceLock<Notify> = OnceLock::new();
    N.get_or_init(Notify::new)
}

fn save_if_dirty() {
    let text = {
        let mut s = lock();
        if !s.dirty {
            return;
        }
        s.dirty = false;
        serde_json::to_string(&*s).ok()
    };
    if let Some(t) = text {
        let p = path();
        let tmp = p.with_extension("tmp");
        if fs::write(&tmp, t).is_ok() {
            let _ = fs::rename(&tmp, &p);
        }
    }
}

/// Wurde der Beitrag schon einmal angezeigt?
pub fn is_seen(id: &str) -> bool {
    lock().seen_set.contains(id)
}

fn mark_seen(id: &str) {
    let mut s = lock();
    if s.seen_set.insert(id.to_string()) {
        s.seen.push(id.to_string());
        if s.seen.len() > MAX_SEEN {
            let cut = s.seen.len() - MAX_SEEN;
            let old: Vec<String> = s.seen.drain(..cut).collect();
            for o in old {
                s.seen_set.remove(&o);
            }
        }
    }
    s.posts.remove(id);
    s.dirty = true;
    scan::cancel_post(id);
}

pub fn start(st: AppState) {
    tokio::spawn(crawl_loop(st.clone()));
    tokio::spawn(maint_loop(st));
}

// ---------- Nachladen ----------

async fn crawl_loop(st: AppState) {
    tokio::time::sleep(Duration::from_secs(5)).await;
    loop {
        let (source, due) = {
            let s = lock();
            let backlog = s.posts.values().filter(|e| e.feats.is_none() && !e.dead).count();
            let ready = s.posts.values().filter(|e| e.feats.is_some()).count();
            let wait = if ready < TARGET_READY { 45 } else { 15 * 60 };
            (
                s.source.clone(),
                ready < TARGET_READY
                    && backlog < BACKLOG
                    && s.posts.len() < MAX_POOL
                    && now().saturating_sub(s.last_fetch) >= wait,
            )
        };
        if due {
            fill(&st, &source).await;
        }
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_secs(15)) => {}
            _ = wake().notified() => {}
        }
    }
}

/// Alle Sortierungen der Quelle durchgehen (je bis zu 3 Seiten) und Neues in den Vorrat legen.
async fn fill(st: &AppState, source: &str) {
    let Some(src) = parse_source(source) else { return };
    lock().fetching = true;
    let mut added_total = fill_redgifs(st, source).await;
    'outer: for (sort, t) in LISTINGS {
        let mut after: Option<String> = None;
        for _ in 0..PAGES {
            let url = listing_url(&src, sort, t, after.as_deref());
            match fetch_listing(st, &url).await {
                Ok((posts, next)) => {
                    let Some(n) = add(st, source, posts) else { break 'outer };
                    added_total += n;
                    after = next;
                }
                Err((_, m)) => {
                    eprintln!("Vorrat: {sort}/{t} nicht ladbar: {m}");
                    break;
                }
            }
            if after.is_none() {
                break;
            }
            {
                let s = lock();
                let backlog = s.posts.values().filter(|e| e.feats.is_none() && !e.dead).count();
                if backlog >= BACKLOG || s.posts.len() >= MAX_POOL {
                    break 'outer;
                }
            }
            tokio::time::sleep(Duration::from_millis(1500)).await;
        }
    }
    {
        let mut s = lock();
        s.fetching = false;
        if s.source == source {
            s.last_fetch = now();
        }
        s.dirty = true;
    }
    eprintln!("Vorrat: {added_total} neue Beiträge aus {source}");
    save_if_dirty();
}

fn backlog_full() -> bool {
    let s = lock();
    s.posts.values().filter(|e| e.feats.is_none() && !e.dead).count() >= BACKLOG || s.posts.len() >= MAX_POOL
}

/// RedGIFs-Nischen in zufälliger Reihenfolge, je eine Seite; höchstens die Hälfte des Rückstaus, damit Reddit
/// auch noch Platz hat.
async fn fill_redgifs(st: &AppState, source: &str) -> usize {
    let (on, mut niches, mut pages) = {
        let s = lock();
        (s.rg_on, s.rg_niches.clone(), s.rg_pages.clone())
    };
    if !on || niches.is_empty() {
        return 0;
    }
    // Alle erschöpft: wieder von vorn
    if niches.iter().all(|n| pages.get(n) == Some(&0)) {
        pages.clear();
    }
    let mut rng = SystemTime::now().duration_since(UNIX_EPOCH).map_or(7, |d| d.as_nanos() as u64) | 1;
    for i in (1..niches.len()).rev() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        niches.swap(i, (rng as usize) % (i + 1));
    }
    let mut added = 0;
    for niche in niches {
        if added >= BACKLOG / 2 || backlog_full() {
            break;
        }
        let page = pages.get(&niche).copied().unwrap_or(1);
        if page == 0 {
            continue;
        }
        match redgifs::niche_page(st, &niche, page).await {
            Ok((posts, more)) => {
                let Some(n) = add(st, source, posts) else { break };
                added += n;
                pages.insert(niche, if more { page + 1 } else { 0 });
            }
            Err(m) => eprintln!("Vorrat: RedGIFs #{niche} nicht ladbar: {m}"),
        }
        tokio::time::sleep(Duration::from_millis(1200)).await;
    }
    let mut s = lock();
    s.rg_pages = pages;
    s.dirty = true;
    eprintln!("Vorrat: {added} neue RedGIFs-Clips");
    added
}

/// Neue Beiträge übernehmen und zum Scannen vormerken. None, wenn die Quelle inzwischen gewechselt hat.
fn add(st: &AppState, source: &str, posts: Vec<Post>) -> Option<usize> {
    let mut fresh = Vec::new();
    {
        let mut s = lock();
        if s.source != source {
            return None;
        }
        for p in posts {
            if s.posts.len() >= MAX_POOL {
                break;
            }
            if s.seen_set.contains(&p.id) || s.posts.contains_key(&p.id) {
                continue;
            }
            let Ok(v) = serde_json::to_value(&p) else { continue };
            s.posts.insert(p.id.clone(), Entry { post: v.clone(), added: now(), feats: None, dead: false });
            fresh.push(v);
        }
        if !fresh.is_empty() {
            s.dirty = true;
        }
    }
    for v in &fresh {
        scan::enqueue_value(st, v);
    }
    Some(fresh.len())
}

// ---------- Pflege: Scan-Stand prüfen, Merkmale berechnen, Altes entfernen ----------

async fn maint_loop(st: AppState) {
    let mut tick: u64 = 0;
    loop {
        tokio::time::sleep(Duration::from_secs(10)).await;
        tick += 1;
        // Alle 3 Minuten noch Ungescanntes erneut vormerken (die Warteschlange hat eine Obergrenze).
        let requeue = tick % 18 == 1;
        let st2 = st.clone();
        let _ = tokio::task::spawn_blocking(move || maintain(&st2, requeue)).await;
    }
}

fn maintain(st: &AppState, requeue: bool) {
    let cutoff = now().saturating_sub(MAX_AGE);
    let todo: Vec<(String, Value)> = {
        let mut s = lock();
        let before = s.posts.len();
        let seen = std::mem::take(&mut s.seen_set);
        s.posts.retain(|id, e| e.added >= cutoff && !seen.contains(id));
        s.seen_set = seen;
        if s.posts.len() != before {
            s.dirty = true;
        }
        s.posts.iter().filter(|(_, e)| e.feats.is_none() && !e.dead).map(|(k, e)| (k.clone(), e.post.clone())).collect()
    };
    let mut results: Vec<(String, Option<taste::Feats>)> = Vec::new();
    for (id, post) in &todo {
        match scan::post_state(post) {
            scan::PostScan::Pending => {
                if requeue {
                    scan::enqueue_value(st, post);
                }
            }
            scan::PostScan::Dead => results.push((id.clone(), None)),
            scan::PostScan::Ready => {
                results.push((id.clone(), Some(taste::features_value(post))));
                prewarm_post(st, post);
            }
        }
    }
    if !results.is_empty() {
        let mut s = lock();
        for (id, f) in results {
            if let Some(e) = s.posts.get_mut(&id) {
                match f {
                    Some(f) => e.feats = Some(f),
                    None => e.dead = true,
                }
            }
        }
        s.dirty = true;
    }
    save_if_dirty();
}

fn prewarm_post(st: &AppState, post: &Value) {
    if let Some(media) = post["media"].as_array() {
        for m in media {
            let kind = m["type"].as_str().unwrap_or_default();
            let raw = m["src"].as_str().unwrap_or_default();
            if (kind == "video" || kind == "image") && !raw.is_empty() {
                let audio: Vec<String> = m["audio"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|a| a.as_str().map(String::from))
                    .collect();
                optimize::prewarm(st, kind, raw, &audio);
            }
        }
    }
}

// ---------- API ----------

fn stats_value(s: &Store) -> Value {
    let ready = s.posts.values().filter(|e| e.feats.is_some()).count();
    let scanning = s.posts.values().filter(|e| e.feats.is_none() && !e.dead).count();
    let dead = s.posts.values().filter(|e| e.dead).count();
    let ready_rg = s.posts.values().filter(|e| e.feats.is_some() && e.post["origin"].as_str() == Some("redgifs")).count();
    json!({
        "ready_rg": ready_rg,
        "rg_on": s.rg_on,
        "source": s.source,
        "ready": ready,
        "scanning": scanning,
        "skipped": dead,
        "seen": s.seen.len(),
        "fetching": s.fetching,
        "last_fetch": s.last_fetch,
    })
}

pub async fn stats() -> Response {
    Json(stats_value(&lock())).into_response()
}

#[derive(Deserialize)]
pub struct NextReq {
    #[serde(default)]
    n: usize,
    /// Schon im Feed geladene Beiträge (noch nicht als gesehen markiert).
    #[serde(default)]
    exclude: Vec<String>,
}

/// Die nächsten fertig gescannten Beiträge, die am besten zu dir passen (mit etwas Zufall gemischt).
pub async fn next(Json(r): Json<NextReq>) -> Response {
    let out = tokio::task::spawn_blocking(move || {
        let n = r.n.clamp(1, 20);
        let ex: HashSet<String> = r.exclude.into_iter().take(500).collect();
        let (cands, stats): (Vec<(Value, taste::Feats)>, Value) = {
            let s = lock();
            let c = s
                .posts
                .iter()
                .filter(|(id, e)| !e.dead && !s.seen_set.contains(*id) && !ex.contains(*id))
                .filter_map(|(_, e)| e.feats.as_ref().map(|f| (e.post.clone(), f.clone())))
                .collect();
            (c, stats_value(&s))
        };
        let feats: Vec<&taste::Feats> = cands.iter().map(|c| &c.1).collect();
        let scores = taste::score_feats(&feats);
        let mut ranked: Vec<(f64, Vec<Value>, Value)> =
            cands.into_iter().zip(scores).map(|((post, _), (pct, why))| (pct, why, post)).collect();
        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Jeder vierte Platz ist ein Zufallsgriff aus dem Rest, damit das Modell auch Neues kennenlernt.
        let mut rng = SystemTime::now().duration_since(UNIX_EPOCH).map_or(7, |d| d.as_nanos() as u64) | 1;
        let mut pick = Vec::new();
        for i in 0..n {
            if ranked.is_empty() {
                break;
            }
            let idx = if i % 4 == 3 && ranked.len() > 2 {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                1 + (rng as usize) % (ranked.len() - 1)
            } else {
                0
            };
            let (pct, why, mut post) = ranked.remove(idx);
            post["match"] = json!({ "pct": pct, "why": why, "ready": true });
            pick.push(post);
        }
        json!({ "posts": pick, "stats": stats })
    })
    .await
    .unwrap_or_else(|_| json!({ "posts": [], "stats": {} }));
    Json(out).into_response()
}

#[derive(Deserialize)]
pub struct SeenReq {
    id: String,
}

/// Beitrag war im Bild: nie wieder vorschlagen.
pub async fn seen(Json(r): Json<SeenReq>) -> Response {
    if r.id.is_empty() || r.id.len() > 64 || !r.id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return err(StatusCode::BAD_REQUEST, "Ungültige Beitrags-ID");
    }
    mark_seen(&r.id);
    StatusCode::NO_CONTENT.into_response()
}

#[derive(Deserialize)]
pub struct SourceReq {
    q: String,
}

/// Quelle des Vorrats wechseln. Der alte Vorrat wird verworfen, die Liste der gesehenen Beiträge bleibt.
pub async fn set_source(Json(r): Json<SourceReq>) -> Response {
    let q = r.q.trim().to_string();
    if parse_source(&q).is_none() {
        return err(StatusCode::BAD_REQUEST, "Kein gültiger Reddit-Link");
    }
    let changed = {
        let mut s = lock();
        let norm = |x: &str| x.trim().trim_end_matches('/').to_lowercase();
        if norm(&s.source) != norm(&q) {
            s.source = q;
            s.posts.clear();
            s.last_fetch = 0;
            s.dirty = true;
            true
        } else {
            false
        }
    };
    if changed {
        wake().notify_one();
        tokio::task::spawn_blocking(save_if_dirty);
    }
    Json(stats_value(&lock())).into_response()
}

// ---------- RedGIFs-Einstellungen ----------

pub async fn rg_get() -> Response {
    let s = lock();
    Json(json!({ "on": s.rg_on, "niches": s.rg_niches, "defaults": redgifs::DEFAULT_NICHES })).into_response()
}

#[derive(Deserialize)]
pub struct RgReq {
    on: bool,
    /// Nischen, durch Komma oder Zeilenumbruch getrennt (Namen oder Links).
    niches: String,
}

pub async fn rg_set(Json(r): Json<RgReq>) -> Response {
    let mut list: Vec<String> = Vec::new();
    for n in r.niches.split([',', '\n']).filter_map(redgifs::parse_niche) {
        if !list.contains(&n) && list.len() < 40 {
            list.push(n);
        }
    }
    {
        let mut s = lock();
        let turned_on = r.on && !s.rg_on;
        s.rg_on = r.on;
        if s.rg_niches != list {
            s.rg_pages.retain(|k, _| list.contains(k));
            s.rg_niches = list;
        }
        if turned_on {
            s.last_fetch = 0;
        }
        s.dirty = true;
    }
    wake().notify_one();
    tokio::task::spawn_blocking(save_if_dirty);
    rg_get().await
}
