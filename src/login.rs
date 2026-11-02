//! Reddit-Login per Browser-Cookie: Reddit gibt anonym kein JSON mehr heraus.
//! Der Cookie ist wie ein Passwort: er wird nur auf dem Server gespeichert (Rechte 0600)
//! und nie wieder an den Browser zurückgegeben.

use std::{env, fs, io::Write, os::unix::fs::OpenOptionsExt, path::PathBuf};

use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use serde_json::json;

use super::{AppState, err, vpn::authorized};

const FILE: &str = "reddit-cookie";

fn path() -> PathBuf {
    PathBuf::from(env::var("VPN_DIR").unwrap_or_else(|_| "/vpn".into())).join(FILE)
}

/// Gespeicherter Cookie-Header (falls vorhanden).
pub fn load() -> Option<String> {
    let s = fs::read_to_string(path()).ok()?;
    let s = s.trim();
    (!s.is_empty()).then(|| s.to_string())
}

/// Akzeptiert "reddit_session=…; token_v2=…" oder eine Zeile "Cookie: …".
pub fn clean(input: &str) -> Result<String, String> {
    let s = input.trim();
    let s = s.strip_prefix("Cookie:").or_else(|| s.strip_prefix("cookie:")).unwrap_or(s).trim();
    if s.is_empty() || s.len() > 8192 {
        return Err("Der Cookie ist leer oder zu lang.".into());
    }
    if s.chars().any(|c| !(c.is_ascii_graphic() || c == ' ')) {
        return Err("Der Cookie enthält ungültige Zeichen (Zeilenumbrüche?). Bitte als eine Zeile einfügen.".into());
    }
    let has = |n: &str| s.split(';').any(|p| p.trim().starts_with(&format!("{n}=")) && p.trim().len() > n.len() + 1);
    if !has("reddit_session") && !has("token_v2") {
        return Err("Im Cookie fehlt reddit_session (oder token_v2). Das ist der Login-Cookie von reddit.com.".into());
    }
    Ok(s.to_string())
}

pub async fn status(State(_st): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(r) = authorized(&headers) {
        return r;
    }
    Json(json!({ "loggedIn": load().is_some() })).into_response()
}

pub async fn save(State(_st): State<AppState>, headers: HeaderMap, body: String) -> Response {
    if let Err(r) = authorized(&headers) {
        return r;
    }
    let cookie = match clean(&body) {
        Ok(c) => c,
        Err(m) => return err(StatusCode::BAD_REQUEST, m),
    };
    let p = path();
    let tmp = p.with_extension("tmp");
    let ok = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(&tmp)
        .and_then(|mut f| f.write_all(cookie.as_bytes()))
        .and_then(|_| fs::rename(&tmp, &p));
    if ok.is_err() {
        return err(StatusCode::INTERNAL_SERVER_ERROR, "Der Cookie konnte nicht gespeichert werden.");
    }
    Json(json!({ "ok": true, "message": "Login gespeichert. Du kannst jetzt einen Subreddit laden." })).into_response()
}

pub async fn remove(State(_st): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(r) = authorized(&headers) {
        return r;
    }
    let _ = fs::remove_file(path());
    Json(json!({ "ok": true, "message": "Abgemeldet." })).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_cookie_line_and_prefix() {
        assert_eq!(clean("reddit_session=abc123; loid=x").unwrap(), "reddit_session=abc123; loid=x");
        assert_eq!(clean("Cookie: token_v2=eyJ.a.b").unwrap(), "token_v2=eyJ.a.b");
    }

    #[test]
    fn rejects_wrong_or_multiline_cookie() {
        assert!(clean("").is_err());
        assert!(clean("loid=x").is_err());
        assert!(clean("reddit_session=").is_err());
        assert!(clean("reddit_session=a\nb=c").is_err());
    }
}
