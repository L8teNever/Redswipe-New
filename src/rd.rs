//! Reddit-Bereich: nur Bilder, direkt von Reddit, ohne Scan und ohne Vorrat.
//! Dazu die Suche nach Subreddits.

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use reqwest::header;
use serde::Deserialize;
use serde_json::{json, Value};
use std::time::Duration;

use super::{browser, err, fetch_listing, listing_url, login, parse_source, AppState, FeedResponse, Source, SORTS, TIMES};

#[derive(Deserialize)]
pub struct FeedQ {
    /// Subreddit (r/aww, aww, aww+cats); leer = r/popular.
    sub: Option<String>,
    /// Suchtext für Beiträge (im Subreddit, oder überall bei leerem sub).
    q: Option<String>,
    sort: Option<String>,
    t: Option<String>,
    after: Option<String>,
}

fn cursor_ok(a: &str) -> bool {
    a.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Beiträge nur mit Bildern (Galerien eingeschlossen), ohne Videos und Einbettungen.
pub async fn feed(State(st): State<AppState>, Query(p): Query<FeedQ>) -> Response {
    let sort = p.sort.unwrap_or_else(|| "hot".into());
    if !SORTS.contains(&sort.as_str()) {
        return err(StatusCode::BAD_REQUEST, "Unbekannte Sortierung");
    }
    let t = p.t.unwrap_or_else(|| "day".into());
    if !TIMES.contains(&t.as_str()) {
        return err(StatusCode::BAD_REQUEST, "Unbekannter Zeitraum");
    }
    let mut after = p.after.filter(|a| !a.is_empty());
    if after.as_deref().is_some_and(|a| !cursor_ok(a)) {
        return err(StatusCode::BAD_REQUEST, "Ungültiger Seiten-Cursor");
    }
    let sub = p.sub.as_deref().map(str::trim).filter(|s| !s.is_empty()).unwrap_or("popular");
    let Some(Source::Sub(sub)) = parse_source(sub) else {
        return err(StatusCode::BAD_REQUEST, "Kein gültiges Subreddit");
    };
    let text = p.q.as_deref().map(str::trim).filter(|s| !s.is_empty()).map(|s| s.chars().take(100).collect::<String>());
    // Subreddit "popular" ohne Suchtext: normales Listing; mit Suchtext: Suche über alles
    let global = sub.eq_ignore_ascii_case("popular") || sub.eq_ignore_ascii_case("all");

    let mut out = Vec::new();
    // Viele Beiträge sind Videos oder Text: bis zu drei Seiten holen, damit genug Bilder zusammenkommen.
    for _ in 0..3 {
        let url = match &text {
            Some(q) => {
                let ssort = match sort.as_str() {
                    "hot" => "hot",
                    "new" => "new",
                    "top" => "top",
                    _ => "relevance",
                };
                let base = if global { "https://www.reddit.com/search.json".to_string() } else { format!("https://www.reddit.com/r/{sub}/search.json") };
                let mut u = format!("{base}?q={}&sort={ssort}&t={t}&limit=50&raw_json=1&include_over_18=on", urlencoding::encode(q));
                if !global {
                    u.push_str("&restrict_sr=on");
                }
                if let Some(a) = &after {
                    u.push_str(&format!("&after={a}"));
                }
                u
            }
            None => listing_url(&Source::Sub(sub.clone()), if sort == "best" { "hot" } else { &sort }, &t, after.as_deref()),
        };
        match fetch_listing(&st, &url).await {
            Ok((posts, next)) => {
                out.extend(posts.into_iter().filter(|p| !p.media.is_empty() && p.media.iter().all(|m| m.kind == "image")));
                after = next;
            }
            Err((code, m)) => {
                if out.is_empty() {
                    return err(code, m);
                }
                break;
            }
        }
        if after.is_none() || out.len() >= 12 {
            break;
        }
    }
    Json(FeedResponse { posts: out, after }).into_response()
}

#[derive(Deserialize)]
pub struct SubsQ {
    q: Option<String>,
}

/// Subreddits suchen (leer = beliebte).
pub async fn subs(State(st): State<AppState>, Query(p): Query<SubsQ>) -> Response {
    let q = p.q.as_deref().map(str::trim).unwrap_or("");
    let url = if q.is_empty() {
        "https://www.reddit.com/subreddits/popular.json?limit=24&raw_json=1".to_string()
    } else {
        format!(
            "https://www.reddit.com/subreddits/search.json?q={}&limit=24&raw_json=1&include_over_18=on",
            urlencoding::encode(&q.chars().take(60).collect::<String>())
        )
    };
    match fetch_json(&st, &url).await {
        Ok(v) => {
            let list: Vec<Value> = v["data"]["children"]
                .as_array()
                .map(|a| a.iter().map(|c| &c["data"]).filter_map(sub_item).collect())
                .unwrap_or_default();
            Json(json!({ "subs": list })).into_response()
        }
        Err((code, m)) => err(code, m),
    }
}

fn sub_item(d: &Value) -> Option<Value> {
    let name = d["display_name"].as_str()?;
    if d["subreddit_type"].as_str() == Some("private") {
        return None;
    }
    let icon = d["community_icon"].as_str().filter(|s| !s.is_empty()).or_else(|| d["icon_img"].as_str().filter(|s| !s.is_empty()));
    Some(json!({
        "name": name,
        "title": d["title"].as_str().unwrap_or_default(),
        "subs": d["subscribers"].as_i64().unwrap_or(0),
        "nsfw": d["over18"].as_bool().unwrap_or(false),
        "icon": icon.map(|u| super::proxy(u.split('?').next().unwrap_or(u))),
    }))
}

/// Beliebiges Reddit-JSON holen: bevorzugt über den Browser auf dem Server, sonst mit Cookie.
async fn fetch_json(st: &AppState, url: &str) -> Result<Value, (StatusCode, String)> {
    if browser::configured() {
        match browser::fetch_text(url).await {
            Ok(text) => return serde_json::from_str(&text).map_err(|e| (StatusCode::BAD_GATEWAY, format!("Ungültige Antwort von Reddit: {e}"))),
            Err(browser::BrowserError::NeedsLogin(m)) | Err(browser::BrowserError::Failed(m)) => return Err((StatusCode::BAD_GATEWAY, m)),
            Err(browser::BrowserError::Unavailable(_)) => {}
        }
    }
    let mut req = st.client.get(url).timeout(Duration::from_secs(20));
    if let Some(c) = login::load() {
        req = req.header(header::COOKIE, c);
    }
    let resp = req.send().await.map_err(|e| (StatusCode::BAD_GATEWAY, format!("Reddit nicht erreichbar: {e}")))?;
    if !resp.status().is_success() {
        return Err((StatusCode::BAD_GATEWAY, format!("Reddit hat die Anfrage abgelehnt (HTTP {})", resp.status().as_u16())));
    }
    resp.json().await.map_err(|e| (StatusCode::BAD_GATEWAY, format!("Ungültige Antwort von Reddit: {e}")))
}
