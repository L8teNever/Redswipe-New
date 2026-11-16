//! Eigene Videos: am Handy oder PC hochladen, auf dem Server ablegen und wie die anderen Videos abspielen.
//! Je Video liegen drei Dateien im Ordner `uploads`: das Video, ein Vorschaubild (`.jpg`, vom Browser
//! erzeugt) und die Angaben (`.json`). Schlüssel in der App: `my:<id>`.

use std::{
    env,
    path::PathBuf,
    sync::atomic::{AtomicU32, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Json,
    body::{Body, Bytes},
    extract::{Query, Request},
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;
use tower::ServiceExt;
use tower_http::services::ServeFile;

use super::err;

/// Größte erlaubte Videodatei (8 GB).
const MAX_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const EXTS: [&str; 5] = ["mp4", "webm", "mov", "m4v", "mkv"];

fn dir() -> PathBuf {
    PathBuf::from(env::var("VPN_DIR").unwrap_or_else(|_| "/vpn".into())).join("uploads")
}

pub fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 32 && id.chars().all(|c| c.is_ascii_hexdigit())
}

fn new_id() -> String {
    static N: AtomicU32 = AtomicU32::new(0);
    let t = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_micros()).unwrap_or(0);
    format!("{t:x}{:02x}", N.fetch_add(1, Ordering::Relaxed) % 256)
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn mmss(s: u64) -> String {
    if s >= 3600 { format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60) } else { format!("{}:{:02}", s / 60, s % 60) }
}

async fn meta(id: &str) -> Option<Value> {
    let raw = tokio::fs::read(dir().join(format!("{id}.json"))).await.ok()?;
    serde_json::from_slice(&raw).ok()
}

fn file_url(id: &str) -> String {
    format!("/api/my/file?id={id}")
}

/// Kachel wie bei den anderen Quellen.
fn item(id: &str, m: &Value) -> Value {
    let dur = m["duration"].as_u64().unwrap_or(0);
    json!({
        "key": format!("my:{id}"),
        "src": "my",
        "title": m["title"].as_str().unwrap_or("Eigenes Video"),
        "thumb": if m["thumb"].as_bool() == Some(true) { format!("{}&t=1&v={}", file_url(id), m["tv"].as_u64().unwrap_or(0)) } else { String::new() },
        "duration": if dur > 0 { mmss(dur) } else { String::new() },
        "w": m["w"], "h": m["h"],
        "size": m["size"],
        "added": m["added"],
    })
}

/// Alle eigenen Videos, neueste zuerst.
pub async fn list_items() -> Vec<Value> {
    let mut out: Vec<(u64, Value)> = Vec::new();
    let Ok(mut rd) = tokio::fs::read_dir(dir()).await else { return Vec::new() };
    while let Ok(Some(e)) = rd.next_entry().await {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(id) = name.strip_suffix(".json") else { continue };
        if !valid_id(id) {
            continue;
        }
        if let Some(m) = meta(id).await {
            out.push((m["added"].as_u64().unwrap_or(0), item(id, &m)));
        }
    }
    out.sort_by(|a, b| b.0.cmp(&a.0));
    out.into_iter().map(|x| x.1).collect()
}

pub async fn list() -> Response {
    Json(json!({ "items": list_items().await, "page": 1, "login": true })).into_response()
}

/// Antwort wie `/api/ph/video` für ein eigenes Video.
pub async fn video(id: &str) -> Response {
    let Some(m) = meta(id).await else {
        return err(StatusCode::NOT_FOUND, "Video nicht gefunden");
    };
    let it = item(id, &m);
    Json(json!({
        "title": it["title"],
        "duration": m["duration"].as_u64().unwrap_or(0),
        "thumb": it["thumb"],
        "uploader": null,
        "sources": [{ "q": m["h"].as_u64().unwrap_or(0).min(m["w"].as_u64().unwrap_or(u64::MAX)), "url": file_url(id) }],
    }))
    .into_response()
}

#[derive(Deserialize)]
pub struct UploadQuery {
    title: Option<String>,
    ext: Option<String>,
    duration: Option<f64>,
    w: Option<u32>,
    h: Option<u32>,
}

/// Video als Rohdaten im Body (kein Formular), damit auch große Dateien direkt auf die Platte gehen.
pub async fn upload(Query(q): Query<UploadQuery>, body: Body) -> Response {
    let ext = q.ext.unwrap_or_default().trim().trim_start_matches('.').to_ascii_lowercase();
    let ext = if EXTS.contains(&ext.as_str()) { ext } else { "mp4".to_string() };
    let d = dir();
    if let Err(e) = tokio::fs::create_dir_all(&d).await {
        return err(StatusCode::INTERNAL_SERVER_ERROR, format!("Ordner nicht anlegbar: {e}"));
    }
    let id = new_id();
    let tmp = d.join(format!("{id}.part"));
    let mut f = match tokio::fs::File::create(&tmp).await {
        Ok(f) => f,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, format!("Datei nicht anlegbar: {e}")),
    };
    let mut size: u64 = 0;
    let mut stream = body.into_data_stream();
    while let Some(chunk) = stream.next().await {
        let fail = match chunk {
            Ok(c) => {
                size += c.len() as u64;
                if size > MAX_BYTES {
                    Some((StatusCode::PAYLOAD_TOO_LARGE, "Video ist größer als 8 GB".to_string()))
                } else if let Err(e) = f.write_all(&c).await {
                    Some((StatusCode::INTERNAL_SERVER_ERROR, format!("Schreiben fehlgeschlagen: {e}")))
                } else {
                    None
                }
            }
            Err(e) => Some((StatusCode::BAD_REQUEST, format!("Upload abgebrochen: {e}"))),
        };
        if let Some((code, msg)) = fail {
            drop(f);
            let _ = tokio::fs::remove_file(&tmp).await;
            return err(code, msg);
        }
    }
    if size == 0 {
        let _ = tokio::fs::remove_file(&tmp).await;
        return err(StatusCode::BAD_REQUEST, "Leere Datei");
    }
    if let Err(e) = f.flush().await {
        return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string());
    }
    drop(f);
    if let Err(e) = tokio::fs::rename(&tmp, d.join(format!("{id}.{ext}"))).await {
        return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string());
    }
    let title: String = q.title.unwrap_or_default().trim().chars().take(200).collect();
    let m = json!({
        "title": if title.is_empty() { "Eigenes Video".to_string() } else { title },
        "ext": ext,
        "duration": q.duration.filter(|x| x.is_finite() && *x > 0.0).map(|x| x.round() as u64).unwrap_or(0),
        "w": q.w.unwrap_or(0), "h": q.h.unwrap_or(0),
        "size": size,
        "added": now(),
        "thumb": false,
    });
    if let Err(e) = tokio::fs::write(d.join(format!("{id}.json")), m.to_string()).await {
        return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string());
    }
    Json(json!({ "key": format!("my:{id}"), "id": id, "item": item(&id, &m) })).into_response()
}

#[derive(Deserialize)]
pub struct IdQuery {
    id: String,
    t: Option<String>,
    dl: Option<String>,
}

/// Vorschaubild (JPEG aus dem Browser) zu einem Video ablegen.
pub async fn thumb(Query(q): Query<IdQuery>, body: Bytes) -> Response {
    if !valid_id(&q.id) || body.len() < 4 || body[..3] != [0xFF, 0xD8, 0xFF] {
        return err(StatusCode::BAD_REQUEST, "Ungültiges Bild");
    }
    let Some(mut m) = meta(&q.id).await else {
        return err(StatusCode::NOT_FOUND, "Video nicht gefunden");
    };
    let d = dir();
    if let Err(e) = tokio::fs::write(d.join(format!("{}.jpg", q.id)), &body).await {
        return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string());
    }
    m["thumb"] = json!(true);
    m["tv"] = json!(now());
    let _ = tokio::fs::write(d.join(format!("{}.json", q.id)), m.to_string()).await;
    Json(item(&q.id, &m)).into_response()
}

#[derive(Deserialize)]
pub struct EditBody {
    id: String,
    title: String,
}

pub async fn edit(Json(b): Json<EditBody>) -> Response {
    if !valid_id(&b.id) {
        return err(StatusCode::BAD_REQUEST, "Ungültiges Video");
    }
    let Some(mut m) = meta(&b.id).await else {
        return err(StatusCode::NOT_FOUND, "Video nicht gefunden");
    };
    let title: String = b.title.trim().chars().take(200).collect();
    if title.is_empty() {
        return err(StatusCode::BAD_REQUEST, "Titel fehlt");
    }
    m["title"] = json!(title);
    if let Err(e) = tokio::fs::write(dir().join(format!("{}.json", b.id)), m.to_string()).await {
        return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string());
    }
    Json(item(&b.id, &m)).into_response()
}

pub async fn delete(Query(q): Query<IdQuery>) -> Response {
    if !valid_id(&q.id) {
        return err(StatusCode::BAD_REQUEST, "Ungültiges Video");
    }
    let Some(m) = meta(&q.id).await else {
        return err(StatusCode::NOT_FOUND, "Video nicht gefunden");
    };
    let d = dir();
    let ext = m["ext"].as_str().unwrap_or("mp4");
    for f in [format!("{}.{ext}", q.id), format!("{}.jpg", q.id), format!("{}.json", q.id)] {
        let _ = tokio::fs::remove_file(d.join(f)).await;
    }
    Json(json!({ "ok": true })).into_response()
}

/// Video (mit Spulen über Range) oder Vorschaubild ausliefern; mit `dl=Name` als Download.
pub async fn file(Query(q): Query<IdQuery>, req: Request) -> Response {
    if !valid_id(&q.id) {
        return err(StatusCode::BAD_REQUEST, "Ungültiges Video");
    }
    let Some(m) = meta(&q.id).await else {
        return err(StatusCode::NOT_FOUND, "Video nicht gefunden");
    };
    let is_thumb = q.t.is_some();
    let ext = if is_thumb { "jpg" } else { m["ext"].as_str().unwrap_or("mp4") };
    let path = dir().join(format!("{}.{ext}", q.id));
    let mut res = match ServeFile::new(path).oneshot(req).await {
        Ok(r) => r.map(Body::new),
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let h = res.headers_mut();
    if ext == "mkv" {
        h.insert(header::CONTENT_TYPE, HeaderValue::from_static("video/x-matroska"));
    }
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static(if is_thumb { "private, max-age=31536000, immutable" } else { "private, max-age=3600" }));
    if let Some(name) = q.dl.filter(|_| !is_thumb) {
        let base = if name.trim().is_empty() { m["title"].as_str().unwrap_or("video").to_string() } else { name };
        let safe: String = base.chars().filter(|c| !c.is_control() && !"\\/:*?\"<>|".contains(*c)).take(120).collect();
        let safe = safe.trim();
        let safe = if safe.is_empty() { "video" } else { safe };
        let ascii: String = safe.chars().map(|c| if c.is_ascii_alphanumeric() || " .-_()".contains(c) { c } else { '_' }).collect();
        if let Ok(v) = HeaderValue::from_str(&format!("attachment; filename=\"{ascii}.{ext}\"; filename*=UTF-8''{}.{ext}", urlencoding::encode(safe))) {
            h.insert(header::CONTENT_DISPOSITION, v);
        }
    }
    res
}
