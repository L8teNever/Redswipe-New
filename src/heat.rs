//! Gesehen-Heatmap je Video in einer SQLite-Datenbank (geräteübergreifend).
//! Das Gerät schickt nur neu gesehene Anteile je Abschnitt, der Server zählt sie auf.

use std::{
    env,
    path::PathBuf,
    sync::{Mutex, OnceLock},
};

use axum::{
    extract::Query,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use rusqlite::{params, Connection, OptionalExtension};
use serde::Deserialize;
use serde_json::json;

/// Abschnitte je Video (muss zur Seite passen).
const BINS: usize = 120;

fn path() -> PathBuf {
    PathBuf::from(env::var("VPN_DIR").unwrap_or_else(|_| "/vpn".into())).join("heat.sqlite")
}

fn db() -> Result<&'static Mutex<Connection>, String> {
    static DB: OnceLock<Result<Mutex<Connection>, String>> = OnceLock::new();
    DB.get_or_init(|| {
        let c = Connection::open(path()).map_err(|e| e.to_string())?;
        c.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS heat (
               key TEXT PRIMARY KEY,
               bins TEXT NOT NULL,
               updated INTEGER NOT NULL
             );",
        )
        .map_err(|e| e.to_string())?;
        Ok(Mutex::new(c))
    })
    .as_ref()
    .map_err(|e| e.clone())
}

fn valid_key(s: &str) -> bool {
    !s.is_empty() && s.len() <= 80 && s.chars().all(|c| c.is_ascii_alphanumeric() || "_-:.".contains(c))
}

fn load(c: &Connection, key: &str) -> rusqlite::Result<Option<Vec<f64>>> {
    let raw: Option<String> = c.query_row("SELECT bins FROM heat WHERE key = ?1", [key], |r| r.get(0)).optional()?;
    Ok(raw
        .and_then(|s| serde_json::from_str::<Vec<f64>>(&s).ok())
        .filter(|v| v.len() == BINS))
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn fail(e: impl ToString) -> Response {
    (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response()
}

#[derive(Deserialize)]
pub struct GetQuery {
    k: String,
}

pub async fn get(Query(q): Query<GetQuery>) -> Response {
    if !valid_key(&q.k) {
        return (StatusCode::BAD_REQUEST, "Ungültiger Schlüssel").into_response();
    }
    let r = tokio::task::spawn_blocking(move || {
        let c = db()?.lock().map_err(|_| "Datenbank gesperrt".to_string())?;
        load(&c, &q.k).map_err(|e| e.to_string())
    })
    .await;
    match r {
        Ok(Ok(bins)) => Json(json!({ "bins": bins })).into_response(),
        Ok(Err(e)) => fail(e),
        Err(e) => fail(e),
    }
}

#[derive(Deserialize)]
pub struct AddBody {
    k: String,
    add: Vec<f64>,
}

pub async fn add(Json(b): Json<AddBody>) -> Response {
    if !valid_key(&b.k) || b.add.len() != BINS || b.add.iter().any(|v| !v.is_finite() || *v < 0.0 || *v > 50.0) {
        return (StatusCode::BAD_REQUEST, "Ungültige Daten").into_response();
    }
    let r = tokio::task::spawn_blocking(move || -> Result<Vec<f64>, String> {
        let mut c = db()?.lock().map_err(|_| "Datenbank gesperrt".to_string())?;
        let tx = c.transaction().map_err(|e| e.to_string())?;
        let mut bins = load(&tx, &b.k).map_err(|e| e.to_string())?.unwrap_or_else(|| vec![0.0; BINS]);
        for (x, d) in bins.iter_mut().zip(&b.add) {
            *x = ((*x + d) * 100.0).round() / 100.0;
        }
        let text = serde_json::to_string(&bins).map_err(|e| e.to_string())?;
        tx.execute(
            "INSERT INTO heat (key, bins, updated) VALUES (?1, ?2, ?3)
             ON CONFLICT(key) DO UPDATE SET bins = excluded.bins, updated = excluded.updated",
            params![b.k, text, now()],
        )
        .map_err(|e| e.to_string())?;
        tx.commit().map_err(|e| e.to_string())?;
        Ok(bins)
    })
    .await;
    match r {
        Ok(Ok(bins)) => Json(json!({ "bins": bins })).into_response(),
        Ok(Err(e)) => fail(e),
        Err(e) => fail(e),
    }
}
