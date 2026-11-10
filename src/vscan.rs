//! Video-Scans (Pose, Finger, Gesicht, Körperteil-Masken): werden am PC im Browser berechnet
//! und hier nur abgelegt, damit Handy, Quest & Co. sie später ohne eigene Rechnung abspielen können.
//! Die Datei ist schon im Browser gzip-komprimiert und wird unverändert wieder ausgeliefert.

use std::{env, fs, path::PathBuf, time::UNIX_EPOCH};

use axum::{
    Json,
    body::Bytes,
    extract::Query,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::Deserialize;
use serde_json::json;

use super::err;

#[derive(Deserialize)]
pub struct KeyQuery {
    key: Option<String>,
}

fn dir() -> PathBuf {
    PathBuf::from(env::var("VPN_DIR").unwrap_or_else(|_| "/vpn".into())).join("vscans")
}

/// Nur kurze, harmlose Schlüssel zulassen (kein Pfad-Trick).
fn file(key: &str) -> Option<PathBuf> {
    let ok = !key.is_empty() && key.len() <= 80 && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    ok.then(|| dir().join(format!("{key}.json.gz")))
}

pub async fn get(Query(q): Query<KeyQuery>) -> Response {
    let Some(p) = q.key.as_deref().and_then(file) else {
        return err(StatusCode::BAD_REQUEST, "Ungültiger Schlüssel");
    };
    match tokio::fs::read(&p).await {
        Ok(b) => (
            [
                (header::CONTENT_TYPE, "application/json"),
                (header::CONTENT_ENCODING, "gzip"),
                (header::CACHE_CONTROL, "no-cache"),
            ],
            b,
        )
            .into_response(),
        Err(_) => err(StatusCode::NOT_FOUND, "Kein Scan vorhanden"),
    }
}

pub async fn put(Query(q): Query<KeyQuery>, body: Bytes) -> Response {
    let Some(p) = q.key.as_deref().and_then(file) else {
        return err(StatusCode::BAD_REQUEST, "Ungültiger Schlüssel");
    };
    // gzip-Kopf prüfen, damit nicht beliebiger Müll abgelegt wird
    if body.len() < 20 || body[0] != 0x1f || body[1] != 0x8b {
        return err(StatusCode::BAD_REQUEST, "Erwartet wird ein gzip-komprimierter Scan");
    }
    if let Err(e) = tokio::fs::create_dir_all(dir()).await {
        return err(StatusCode::INTERNAL_SERVER_ERROR, format!("Ordner nicht anlegbar: {e}"));
    }
    let tmp = p.with_extension("tmp");
    if let Err(e) = tokio::fs::write(&tmp, &body).await {
        return err(StatusCode::INTERNAL_SERVER_ERROR, format!("Speichern fehlgeschlagen: {e}"));
    }
    if let Err(e) = tokio::fs::rename(&tmp, &p).await {
        return err(StatusCode::INTERNAL_SERVER_ERROR, format!("Speichern fehlgeschlagen: {e}"));
    }
    Json(json!({ "ok": true, "bytes": body.len() })).into_response()
}

fn meta_file(key: &str) -> Option<PathBuf> {
    file(key).map(|p| p.with_file_name(format!("{key}.meta.json")))
}

/// Kleine Beschreibung zu einem Scan (Video-Adresse, Vorschaubild), damit die Seite eine Liste zeigen kann.
pub async fn put_meta(Query(q): Query<KeyQuery>, body: Bytes) -> Response {
    let Some(p) = q.key.as_deref().and_then(meta_file) else {
        return err(StatusCode::BAD_REQUEST, "Ungültiger Schlüssel");
    };
    if serde_json::from_slice::<serde_json::Value>(&body).map_or(true, |v| !v.is_object()) {
        return err(StatusCode::BAD_REQUEST, "Erwartet wird ein JSON-Objekt");
    }
    if let Err(e) = tokio::fs::create_dir_all(dir()).await {
        return err(StatusCode::INTERNAL_SERVER_ERROR, format!("Ordner nicht anlegbar: {e}"));
    }
    match tokio::fs::write(&p, &body).await {
        Ok(()) => Json(json!({ "ok": true })).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("Speichern fehlgeschlagen: {e}")),
    }
}

pub async fn del(Query(q): Query<KeyQuery>) -> Response {
    let Some(key) = q.key.as_deref().filter(|k| file(k).is_some()) else {
        return err(StatusCode::BAD_REQUEST, "Ungültiger Schlüssel");
    };
    if let Some(p) = file(key) {
        let _ = tokio::fs::remove_file(p).await;
    }
    if let Some(p) = meta_file(key) {
        let _ = tokio::fs::remove_file(p).await;
    }
    Json(json!({ "ok": true })).into_response()
}

pub async fn list() -> Response {
    let mut out = Vec::new();
    if let Ok(rd) = fs::read_dir(dir()) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            let Some(key) = name.strip_suffix(".json.gz") else { continue };
            let Ok(m) = e.metadata() else { continue };
            let t = m.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
            let meta = meta_file(key)
                .and_then(|p| fs::read(p).ok())
                .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
            out.push(json!({ "key": key, "bytes": m.len(), "time": t, "meta": meta }));
        }
    }
    Json(json!({ "scans": out })).into_response()
}
