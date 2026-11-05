//! NudeNet-Scan im Hintergrund: Sobald ein Feed geladen ist, wird jedes Bild und jedes Video
//! (Videos Frame für Frame) vom Scanner-Container geprüft. Die Ergebnisse liegen als JSON im
//! Cache-Ordner; das Frontend holt sie nur ab, wenn "Anzeigen" an ist.

use std::{
    collections::{HashMap, HashSet},
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
    env,
    hash::{DefaultHasher, Hash, Hasher},
    sync::{Arc, OnceLock},
    time::Duration,
};

use axum::{
    Json,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use reqwest::Url;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::{Mutex, Notify};

use crate::{AppState, Post, err, media_allowed, optimize, pool};

const MAX_QUEUE: usize = 1200;

fn scanner() -> Option<String> {
    let v = env::var("NSFW_URL").ok()?;
    let v = v.trim().trim_end_matches('/').to_string();
    (!v.is_empty()).then_some(v)
}

fn inflight() -> &'static Mutex<HashSet<String>> {
    static S: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Laufende Aufträge, die wegen deiner Position gestartet wurden (prio < IDLE): Schlüssel -> Abbruch-Signal.
fn running() -> &'static Mutex<HashMap<String, Arc<Notify>>> {
    static S: OnceLock<Mutex<HashMap<String, Arc<Notify>>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Wartende Aufträge. Kleinere `prio` kommt zuerst: 0 = der Beitrag, den du gerade siehst, 1-3 = die
/// nächsten Beiträge, 1000 = alles andere (in Feed-Reihenfolge).
struct Job {
    k: String,
    kind: &'static str,
    raw: String,
    prio: u32,
    seq: u64,
    post_id: Option<String>,
}

struct Queue {
    jobs: Mutex<Vec<Job>>,
    wake: Notify,
}

fn queue() -> &'static Queue {
    static S: OnceLock<Queue> = OnceLock::new();
    S.get_or_init(|| Queue { jobs: Mutex::new(Vec::new()), wake: Notify::new() })
}

static SEQ: AtomicU64 = AtomicU64::new(0);
const IDLE: u32 = 1000;

const MAX_WORKERS: usize = 12;
static PARTS: AtomicUsize = AtomicUsize::new(0);

fn parts_file() -> std::path::PathBuf {
    optimize::cache_dir().join("scan_parts.txt")
}

/// In wie viele Zeitabschnitte ein Video geteilt und parallel gescannt wird (1-16, Standard 1).
fn parts() -> usize {
    let v = PARTS.load(Ordering::Relaxed);
    if v > 0 {
        return v;
    }
    let n = std::fs::read_to_string(parts_file()).ok().and_then(|t| t.trim().parse().ok()).unwrap_or(1usize).clamp(1, 16);
    PARTS.store(n, Ordering::Relaxed);
    n
}
static LIMIT: AtomicUsize = AtomicUsize::new(0);

fn limit_file() -> std::path::PathBuf {
    optimize::cache_dir().join("scan_workers.txt")
}

/// Wie viele Medien gleichzeitig gescannt werden (1-12): gespeicherter Wert, sonst SCAN_JOBS, sonst 2.
fn limit() -> usize {
    let v = LIMIT.load(Ordering::Relaxed);
    if v > 0 {
        return v;
    }
    let n = std::fs::read_to_string(limit_file())
        .ok()
        .and_then(|t| t.trim().parse().ok())
        .or_else(|| env::var("SCAN_JOBS").ok().and_then(|v| v.parse().ok()))
        .unwrap_or(2usize)
        .clamp(1, MAX_WORKERS);
    LIMIT.store(n, Ordering::Relaxed);
    n
}

/// Alle Arbeiter laufen von Anfang an; Nummer >= Limit pausiert, so wirkt eine Änderung sofort.
fn start_workers(st: &AppState) {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        for idx in 0..MAX_WORKERS {
            let st = st.clone();
            tokio::spawn(async move {
                loop {
                    if idx >= limit() {
                        tokio::time::sleep(Duration::from_secs(1)).await;
                        continue;
                    }
                    let job = {
                        let mut jobs = queue().jobs.lock().await;
                        let best = jobs.iter().enumerate().min_by_key(|(_, j)| (j.prio, j.seq)).map(|(i, _)| i);
                        let job = best.map(|i| jobs.remove(i));
                        if let Some(j) = &job {
                            inflight().lock().await.insert(j.k.clone());
                        }
                        job
                    };
                    match job {
                        Some(j) => work(&st, j).await,
                        None => queue().wake.notified().await,
                    }
                }
            });
        }
    });
}

pub(crate) fn key(kind: &str, url: &str) -> String {
    let mut a = DefaultHasher::new();
    ("scan", kind, url).hash(&mut a);
    let mut b = DefaultHasher::new();
    (url, kind, "scan", 18u8).hash(&mut b);
    format!("{:016x}{:016x}", a.finish(), b.finish())
}

pub(crate) fn result_path(k: &str) -> std::path::PathBuf {
    optimize::cache_dir().join(format!("{k}.scan.json"))
}

/// Alle Medien der Beiträge für den Scan vormerken (läuft im Hintergrund).
pub fn enqueue_posts(st: &AppState, posts: &[Post]) {
    if scanner().is_none() {
        return;
    }
    for p in posts {
        if pool::is_seen(&p.id) {
            continue;
        }
        for m in &p.media {
            if m.src.is_empty() || (m.kind != "image" && m.kind != "video") {
                continue;
            }
            enqueue(st.clone(), m.kind, m.src.clone(), IDLE, Some(p.id.clone()));
        }
    }
}

/// Scanbare Medien eines Beitrags (JSON wie aus dem Feed): (Art, Originaladresse). GIF-Bilder werden nie gescannt.
fn scannable(post: &Value) -> Vec<(&'static str, String)> {
    post["media"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|m| {
            let kind: &'static str = match m["type"].as_str()? {
                "image" => "image",
                "video" => "video",
                _ => return None,
            };
            let src = m["src"].as_str().filter(|s| !s.is_empty())?;
            let url = Url::parse(src).ok()?;
            if !media_allowed(&url) || (kind == "image" && url.path().to_lowercase().ends_with(".gif")) {
                return None;
            }
            Some((kind, src.to_string()))
        })
        .collect()
}

pub enum PostScan {
    /// Mindestens ein Medium wartet noch auf den Scan.
    Pending,
    /// Alles gescannt, mindestens ein Ergebnis ist brauchbar.
    Ready,
    /// Nichts scanbar oder alle Scans fehlgeschlagen.
    Dead,
}

/// Wie weit sind die Scans eines Beitrags?
pub fn post_state(post: &Value) -> PostScan {
    let media = scannable(post);
    let mut done = 0;
    for (kind, src) in &media {
        let path = result_path(&key(kind, src));
        let Ok(t) = std::fs::read_to_string(&path) else { return PostScan::Pending };
        if serde_json::from_str::<Value>(&t).ok().is_some_and(|v| v["status"].as_str() == Some("done")) {
            done += 1;
        }
    }
    if done > 0 { PostScan::Ready } else { PostScan::Dead }
}

/// Alle noch nicht gescannten Medien eines Beitrags (JSON) im Hintergrund vormerken.
pub fn enqueue_value(st: &AppState, post: &Value) {
    if scanner().is_none() {
        return;
    }
    let pid = post["id"].as_str().map(String::from);
    if let Some(ref id) = pid {
        if pool::is_seen(id) {
            return;
        }
    }
    for (kind, src) in scannable(post) {
        enqueue(st.clone(), kind, src, IDLE, pid.clone());
    }
}

fn enqueue(st: AppState, kind: &'static str, raw: String, prio: u32, post_id: Option<String>) {
    tokio::spawn(async move {
        if let Some(ref pid) = post_id {
            if pool::is_seen(pid) {
                return;
            }
        }
        let Ok(url) = Url::parse(&raw) else { return };
        if !media_allowed(&url) {
            return;
        }
        // GIFs sind Bilder mit Animation: als Bild scannen reicht für das erste Bild nicht, daher überspringen.
        if kind == "image" && url.path().to_lowercase().ends_with(".gif") {
            return;
        }
        let k = key(kind, &raw);
        if result_path(&k).exists() {
            return;
        }
        start_workers(&st);
        {
            // Läuft schon oder wartet schon: nur ggf. vorziehen.
            let mut jobs = queue().jobs.lock().await;
            if let Some(j) = jobs.iter_mut().find(|j| j.k == k) {
                j.prio = j.prio.min(prio);
                if j.post_id.is_none() {
                    j.post_id = post_id;
                }
                return;
            }
            if inflight().lock().await.contains(&k) || jobs.len() >= MAX_QUEUE {
                return;
            }
            jobs.push(Job { k, kind, raw, prio, seq: SEQ.fetch_add(1, Ordering::Relaxed), post_id });
        }
        queue().wake.notify_one();
    });
}

/// "Jetzt scannen": Auftrag ganz nach vorn und zusätzlich ein eigener Arbeiter, der sofort startet,
/// auch wenn die normalen Arbeiter gerade mit anderen Medien beschäftigt sind.
fn rush(st: AppState, kind: &'static str, raw: String, post_id: Option<String>) {
    let k = key(kind, &raw);
    tokio::spawn(async move {
        {
            let jobs = queue().jobs.lock().await;
            // Läuft schon (ggf. mit niedriger Priorität): nichts doppelt starten.
            if inflight().lock().await.contains(&k) && !jobs.iter().any(|j| j.k == k) {
                return;
            }
        }
        // Altes Ergebnis verwerfen: es wird komplett neu gescannt.
        let _ = tokio::fs::remove_file(result_path(&k)).await;
        enqueue(st.clone(), kind, raw, 0, post_id);
        // Kurz warten, bis der Auftrag in der Warteschlange liegt, dann einen Sonder-Arbeiter starten.
        tokio::time::sleep(Duration::from_millis(50)).await;
        let job = {
            let mut jobs = queue().jobs.lock().await;
            let Some(i) = jobs.iter().position(|j| j.k == k) else { return };
            let job = jobs.remove(i);
            inflight().lock().await.insert(job.k.clone());
            job
        };
        work(&st, job).await;
    });
}

/// Bricht alle Scans zu diesem Beitrag ab und entfernt ihn aus der Warteschlange (z. B. weil der Beitrag gesehen wurde).
pub fn cancel_post(post_id: &str) {
    let pid = post_id.to_string();
    tokio::spawn(async move {
        let to_cancel: Vec<String> = {
            let mut jobs = queue().jobs.lock().await;
            let mut keys = Vec::new();
            jobs.retain(|j| {
                if j.post_id.as_deref() == Some(&pid) {
                    keys.push(j.k.clone());
                    false
                } else {
                    true
                }
            });
            keys
        };
        let run = running().lock().await;
        for k in to_cancel {
            if let Some(notify) = run.get(&k) {
                notify.notify_waiters();
            }
        }
    });
}

async fn work(st: &AppState, job: Job) {
    let Job { k, kind, raw, prio, post_id, .. } = job;
    if let Some(ref pid) = post_id {
        if pool::is_seen(pid) {
            inflight().lock().await.remove(&k);
            return;
        }
    }
    let cancel = Arc::new(Notify::new());
    if prio < IDLE {
        running().lock().await.insert(k.clone(), cancel.clone());
    }
    let Ok(url) = Url::parse(&raw) else {
        inflight().lock().await.remove(&k);
        return;
    };
    // Abbruch: wird der Auftrag verworfen, schließt reqwest die Verbindung, und der Scanner hört auf.
    let res = tokio::select! {
        r = run(st, kind, &url, &k) => Some(r),
        _ = cancel.notified() => None,
    };
    running().lock().await.remove(&k);
    let Some(res) = res else {
        let _ = tokio::fs::remove_file(optimize::cache_dir().join(format!("{k}.scan.src"))).await;
        eprintln!("Scan abgebrochen ({kind} {raw}): zu weit zurück");
        inflight().lock().await.remove(&k);
        return;
    };
    match res {
        Ok(v) => {
            let _ = tokio::fs::create_dir_all(optimize::cache_dir()).await;
            let _ = tokio::fs::write(result_path(&k), v.to_string()).await;
        }
        // Scanner nicht erreichbar: nichts merken, beim nächsten Abruf wird neu versucht.
        Err(Fail::Retry(m)) => eprintln!("Scan später erneut ({kind} {raw}): {m}"),
        Err(Fail::Final(m)) => {
            eprintln!("Scan fehlgeschlagen ({kind} {raw}): {m}");
            let _ = tokio::fs::write(result_path(&k), json!({ "status": "failed", "error": m }).to_string()).await;
        }
    }
    inflight().lock().await.remove(&k);
}

enum Fail {
    Retry(String),
    Final(String),
}

async fn run(st: &AppState, kind: &str, url: &Url, k: &str) -> Result<Value, Fail> {
    let t0 = std::time::Instant::now();
    let base = scanner().ok_or_else(|| Fail::Retry("NSFW_URL nicht gesetzt".into()))?;
    let dir = optimize::cache_dir();
    tokio::fs::create_dir_all(&dir).await.map_err(|e| Fail::Retry(e.to_string()))?;
    // Eigene Namensendung, damit der Aufräumer sie wie alles andere nach 3 Tagen entfernt.
    let tmp = dir.join(format!("{k}.scan.src"));
    optimize::download(st, url, &tmp).await.map_err(Fail::Final)?;

    let timeout = if kind == "video" { 900 } else { 60 };
    let resp = st
        .client
        .post(format!("{base}/scan"))
        .timeout(Duration::from_secs(timeout))
        .json(&json!({ "path": tmp.to_string_lossy(), "kind": kind, "parts": if kind == "video" { parts() } else { 1 }, "workers": limit() }))
        .send()
        .await;
    let _ = tokio::fs::remove_file(&tmp).await;
    let resp = resp.map_err(|e| Fail::Retry(format!("Scanner nicht erreichbar: {e}")))?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        let msg = format!("Scanner HTTP {}: {}", status.as_u16(), body.chars().take(200).collect::<String>());
        return Err(if status.is_server_error() { Fail::Retry(msg) } else { Fail::Final(msg) });
    }
    let mut v: Value = resp.json().await.map_err(|e| Fail::Final(e.to_string()))?;
    v["status"] = json!("done");
    crate::nipples::process_scan_result(&mut v);
    // Dauer (Download + Scan) in Sekunden, plus Teile pro Video, für die Anzeige.
    v["took"] = json!((t0.elapsed().as_secs_f64() * 10.0).round() / 10.0);
    if kind == "video" {
        v["parts"] = json!(parts());
    }
    Ok(v)
}

#[derive(Deserialize)]
pub struct ScanQuery {
    k: Option<String>,
    url: Option<String>,
    /// Abstand zum aktuellen Beitrag (0 = sichtbar, 1-3 = die nächsten); fehlt er, bleibt die Reihenfolge.
    p: Option<u32>,
    /// 1 = sofort scannen, vor allem anderen (auch wenn alle Arbeiter belegt sind).
    now: Option<u8>,
}

/// Ergebnis abholen: {"status":"pending"} solange der Scan läuft, sonst die Erkennungen.
pub async fn handler(State(st): State<AppState>, Query(q): Query<ScanQuery>) -> Response {
    let (Some(kind), Some(raw)) = (q.k.as_deref(), q.url.as_deref()) else {
        return err(StatusCode::BAD_REQUEST, "Parameter fehlen");
    };
    let kind: &'static str = match kind {
        "image" => "image",
        "video" => "video",
        _ => return err(StatusCode::BAD_REQUEST, "Unbekannte Art"),
    };
    let Ok(url) = Url::parse(raw) else {
        return err(StatusCode::BAD_REQUEST, "Ungültige URL");
    };
    if !media_allowed(&url) {
        return err(StatusCode::FORBIDDEN, "Host nicht erlaubt");
    }
    if scanner().is_none() {
        return Json(json!({ "status": "off" })).into_response();
    }
    let k = key(kind, raw);
    let force = q.now == Some(1);
    if let (false, Ok(text)) = (force, tokio::fs::read_to_string(result_path(&k)).await) {
        if let Ok(v) = serde_json::from_str::<Value>(&text) {
            return Json(v).into_response();
        }
    }
    // Noch nichts da: sicherstellen, dass der Scan läuft.
    if q.now == Some(1) {
        rush(st, kind, raw.to_string(), None);
    } else {
        enqueue(st, kind, raw.to_string(), q.p.map_or(IDLE, |p| p.min(IDLE)), None);
    }
    Json(json!({ "status": "pending" })).into_response()
}

#[derive(Deserialize)]
pub struct Focus {
    /// Medien (Art + URL) von Beitrag 0 bis 3, also die, die gerade wichtig sind.
    items: Vec<(String, String)>,
}

/// Position melden: Alles, was priorisiert war und nicht mehr in den nächsten 3 Beiträgen liegt, wird
/// zurückgestuft (wartend) oder abgebrochen (laufend).
pub async fn focus(Json(f): Json<Focus>) -> Response {
    let keep: HashSet<String> = f
        .items
        .iter()
        .take(64)
        .filter_map(|(kind, url)| match kind.as_str() {
            "image" | "video" => Some(key(kind, url)),
            _ => None,
        })
        .collect();
    {
        let mut jobs = queue().jobs.lock().await;
        for j in jobs.iter_mut().filter(|j| j.prio < IDLE && !keep.contains(&j.k)) {
            j.prio = IDLE;
        }
    }
    for (k, n) in running().lock().await.iter() {
        if !keep.contains(k) {
            n.notify_one();
        }
    }
    StatusCode::NO_CONTENT.into_response()
}

/// Auslastung: laufende und wartende Scans, Arbeiter-Limit, Rechner-Last.
pub async fn status(State(st): State<AppState>) -> Response {
    if scanner().is_none() {
        return Json(json!({ "off": true })).into_response();
    }
    start_workers(&st);
    let queued = queue().jobs.lock().await.len();
    let running = inflight().lock().await.len();
    let load: f64 = std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|t| t.split_whitespace().next().and_then(|v| v.parse().ok()))
        .unwrap_or(0.0);
    let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
    Json(json!({ "queued": queued, "running": running, "workers": limit(), "max": MAX_WORKERS, "parts": parts(), "load": load, "cpus": cpus }))
        .into_response()
}

#[derive(Deserialize)]
pub struct Workers {
    n: usize,
}

/// Anzahl gleichzeitiger Scans setzen (bleibt nach Neustart erhalten).
pub async fn set_workers(Json(w): Json<Workers>) -> Response {
    let n = w.n.clamp(1, MAX_WORKERS);
    LIMIT.store(n, Ordering::Relaxed);
    let _ = tokio::fs::create_dir_all(optimize::cache_dir()).await;
    let _ = tokio::fs::write(limit_file(), n.to_string()).await;
    queue().wake.notify_waiters();
    Json(json!({ "workers": n })).into_response()
}

#[derive(Deserialize)]
pub struct PartsReq {
    n: usize,
}

/// Teile pro Video setzen (1-8, bleibt nach Neustart erhalten).
pub async fn set_parts(Json(w): Json<PartsReq>) -> Response {
    let n = w.n.clamp(1, 16);
    PARTS.store(n, Ordering::Relaxed);
    let _ = tokio::fs::create_dir_all(optimize::cache_dir()).await;
    let _ = tokio::fs::write(parts_file(), n.to_string()).await;
    Json(json!({ "parts": n })).into_response()
}
