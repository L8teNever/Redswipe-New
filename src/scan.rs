//! NudeNet-Scan im Hintergrund: Sobald ein Feed geladen ist, wird jedes Bild und jedes Video
//! (Videos Frame für Frame) vom Scanner-Container geprüft. Die Ergebnisse liegen als JSON im
//! Cache-Ordner; das Frontend holt sie nur ab, wenn "Anzeigen" an ist.

use std::{
    collections::{HashMap, HashSet},
    sync::atomic::{AtomicU64, Ordering},
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

use crate::{AppState, Post, err, media_allowed, optimize};

const MAX_QUEUE: usize = 300;

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

/// Die Arbeiter starten beim ersten Auftrag (SCAN_JOBS Stück, Standard 2).
fn start_workers(st: &AppState) {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let n: usize = env::var("SCAN_JOBS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
        for _ in 0..n.max(1) {
            let st = st.clone();
            tokio::spawn(async move {
                loop {
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

fn key(kind: &str, url: &str) -> String {
    let mut a = DefaultHasher::new();
    ("scan", kind, url).hash(&mut a);
    let mut b = DefaultHasher::new();
    (url, kind, "scan", 10u8).hash(&mut b);
    format!("{:016x}{:016x}", a.finish(), b.finish())
}

fn result_path(k: &str) -> std::path::PathBuf {
    optimize::cache_dir().join(format!("{k}.scan.json"))
}

/// Alle Medien der Beiträge für den Scan vormerken (läuft im Hintergrund).
pub fn enqueue_posts(st: &AppState, posts: &[Post]) {
    if scanner().is_none() {
        return;
    }
    for p in posts {
        for m in &p.media {
            if m.src.is_empty() || (m.kind != "image" && m.kind != "video") {
                continue;
            }
            enqueue(st.clone(), m.kind, m.src.clone(), IDLE);
        }
    }
}

fn enqueue(st: AppState, kind: &'static str, raw: String, prio: u32) {
    tokio::spawn(async move {
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
                return;
            }
            if inflight().lock().await.contains(&k) || jobs.len() >= MAX_QUEUE {
                return;
            }
            jobs.push(Job { k, kind, raw, prio, seq: SEQ.fetch_add(1, Ordering::Relaxed) });
        }
        queue().wake.notify_one();
    });
}

async fn work(st: &AppState, job: Job) {
    let Job { k, kind, raw, prio, .. } = job;
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
        .json(&json!({ "path": tmp.to_string_lossy(), "kind": kind }))
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
    Ok(v)
}

#[derive(Deserialize)]
pub struct ScanQuery {
    k: Option<String>,
    url: Option<String>,
    /// Abstand zum aktuellen Beitrag (0 = sichtbar, 1-3 = die nächsten); fehlt er, bleibt die Reihenfolge.
    p: Option<u32>,
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
    if let Ok(text) = tokio::fs::read_to_string(result_path(&k)).await {
        if let Ok(v) = serde_json::from_str::<Value>(&text) {
            return Json(v).into_response();
        }
    }
    // Noch nichts da: sicherstellen, dass der Scan läuft.
    enqueue(st, kind, raw.to_string(), q.p.map_or(IDLE, |p| p.min(IDLE)));
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
