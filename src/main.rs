use std::{env, time::Duration};

use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Query, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use reqwest::Url;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tower_http::services::{ServeDir, ServeFile};

mod actions;
mod browser;
mod login;
mod optimize;
mod scan;
mod vpn;

// Reddit antwortet auf unbekannte Clients oft mit 403/429, daher ein Browser-User-Agent.
const DEFAULT_UA: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36";
const SORTS: &[&str] = &["hot", "new", "top", "rising", "best"];
const TIMES: &[&str] = &["hour", "day", "week", "month", "year", "all"];
/// Nur diese Hosts werden über /api/media weitergereicht (Schutz vor offenem Proxy).
const ALLOWED_HOSTS: &[&str] = &[
    "i.redd.it",
    "preview.redd.it",
    "external-preview.redd.it",
    "v.redd.it",
    "i.imgur.com",
];

#[derive(Clone)]
pub struct AppState {
    pub client: reqwest::Client,
}

#[derive(Deserialize)]
struct FeedQuery {
    q: Option<String>,
    sort: Option<String>,
    t: Option<String>,
    after: Option<String>,
}

#[derive(Deserialize)]
struct MediaQuery {
    url: Option<String>,
}

#[derive(Serialize)]
struct FeedResponse {
    posts: Vec<Post>,
    after: Option<String>,
}

#[derive(Serialize, Debug, PartialEq)]
pub struct Post {
    id: String,
    title: String,
    subreddit: String,
    author: String,
    score: i64,
    comments: i64,
    permalink: String,
    nsfw: bool,
    /// Dein aktueller Vote: "up", "down" oder "none".
    vote: &'static str,
    media: Vec<Media>,
}

#[derive(Serialize, Debug, PartialEq)]
pub struct Media {
    #[serde(rename = "type")]
    kind: &'static str,
    /// Bereits auf /api/media umgeschrieben.
    url: String,
    width: Option<i64>,
    height: Option<i64>,
    /// Kandidaten für die Tonspur (Reddit-Videos haben Ton in einer eigenen Datei).
    audio: Vec<String>,
    /// Originaladresse des Mediums (für den NudeNet-Scan); leer bei Einbettungen.
    src: String,
}

#[tokio::main]
async fn main() {
    let port = env::var("PORT").unwrap_or_else(|_| "8080".into());
    let static_dir = env::var("STATIC_DIR").unwrap_or_else(|_| "static".into());
    let user_agent = env::var("USER_AGENT").unwrap_or_else(|_| DEFAULT_UA.into());

    // reqwest liest HTTPS_PROXY automatisch aus der Umgebung.
    // Kein Gesamt-Timeout hier, sonst würden lange Videos abgeschnitten; Timeouts stehen pro Anfrage.
    let client = reqwest::Client::builder()
        .user_agent(user_agent)
        .connect_timeout(Duration::from_secs(10))
        .build()
        .expect("HTTP-Client konnte nicht erstellt werden");

    tokio::spawn(vpn::keep_endpoint_fresh());
    tokio::spawn(optimize::cleanup_loop());

    let index = format!("{static_dir}/index.html");
    let app = Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/api/feed", get(feed))
        .route("/api/media", get(media))
        .route("/api/opt", get(optimize::handler))
        .route("/api/scan", get(scan::handler))
        .route("/api/vote", post(actions::vote).layer(DefaultBodyLimit::max(1024)))
        .route("/api/follow", post(actions::follow).layer(DefaultBodyLimit::max(1024)))
        .route(
            "/api/login",
            get(login::status).post(login::save).delete(login::remove).layer(DefaultBodyLimit::max(16 * 1024)),
        )
        .route("/api/vpn", get(vpn::status).post(vpn::upload).layer(DefaultBodyLimit::max(16 * 1024)))
        .fallback_service(ServeDir::new(&static_dir).fallback(ServeFile::new(index)))
        .with_state(AppState { client });

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port.parse::<u16>().unwrap_or(8080)))
        .await
        .expect("Port konnte nicht gebunden werden");
    println!("reddit-swipe läuft auf Port {port}");
    axum::serve(listener, app).await.expect("Server-Fehler");
}

pub fn err(code: StatusCode, msg: impl Into<String>) -> Response {
    (code, Json(json!({ "error": msg.into() }))).into_response()
}

async fn feed(State(st): State<AppState>, Query(p): Query<FeedQuery>) -> Response {
    let Some(source) = parse_source(p.q.as_deref().unwrap_or("")) else {
        return err(
            StatusCode::BAD_REQUEST,
            "Kein gültiger Reddit-Link (leer = Startseite, sonst z. B. https://www.reddit.com/r/aww/)",
        );
    };
    let sort = p.sort.unwrap_or_else(|| "best".into());
    if !SORTS.contains(&sort.as_str()) {
        return err(StatusCode::BAD_REQUEST, "Unbekannte Sortierung");
    }
    let t = p.t.unwrap_or_else(|| "day".into());
    if !TIMES.contains(&t.as_str()) {
        return err(StatusCode::BAD_REQUEST, "Unbekannter Zeitraum");
    }
    let after = p.after.filter(|a| !a.is_empty());
    if let Some(a) = &after {
        if !a.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
            return err(StatusCode::BAD_REQUEST, "Ungültiger Seiten-Cursor");
        }
    }

    // Startseite = der personalisierte Feed des eingeloggten Browsers (wie im Addon).
    let base = match &source {
        Source::Home => format!("https://www.reddit.com/{sort}.json"),
        Source::Sub(sub) => format!("https://www.reddit.com/r/{sub}/{sort}.json"),
    };
    let mut url = format!("{base}?limit=25&raw_json=1");
    if sort == "top" {
        url.push_str(&format!("&t={t}"));
    }
    if let Some(a) = &after {
        url.push_str(&format!("&after={a}"));
    }

    // Bevorzugt: der eingeloggte Browser (Chromium). Nur wenn er nicht erreichbar ist, geht es mit dem Cookie weiter.
    if browser::configured() {
        match browser::fetch_text(&url).await {
            Ok(text) => match serde_json::from_str::<Value>(&text) {
                Ok(value) => {
                    let (posts, after) = parse_listing(&value);
                    scan::enqueue_posts(&st, &posts);
                    return Json(FeedResponse { posts, after }).into_response();
                }
                Err(e) => return err(StatusCode::BAD_GATEWAY, format!("Ungültige Antwort von Reddit: {e}")),
            },
            Err(browser::BrowserError::NeedsLogin(m)) => {
                return err(
                    StatusCode::BAD_GATEWAY,
                    format!("{m}. Bitte im Browser neu bei Reddit einloggen (https://redswipe-login.kulbarts.com)."),
                );
            }
            Err(browser::BrowserError::Failed(m)) => return err(StatusCode::BAD_GATEWAY, m),
            Err(browser::BrowserError::Unavailable(_)) => {}
        }
    }

    let cookie = login::load();
    let mut req = st.client.get(&url).timeout(Duration::from_secs(20));
    if let Some(c) = &cookie {
        req = req.header(header::COOKIE, c.as_str());
    }
    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => return err(StatusCode::BAD_GATEWAY, format!("Reddit nicht erreichbar: {e}")),
    };
    let status = resp.status();
    if !status.is_success() {
        let hint = if cookie.is_some() {
            " – ist der Login-Cookie abgelaufen? Im Dialog neu einfügen."
        } else {
            " – Reddit verlangt einen Login (Dialog \"VPN\" → Reddit-Login)."
        };
        return err(
            StatusCode::BAD_GATEWAY,
            format!("Reddit hat die Anfrage abgelehnt (HTTP {}){hint}", status.as_u16()),
        );
    }
    let value: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => return err(StatusCode::BAD_GATEWAY, format!("Ungültige Antwort von Reddit: {e}")),
    };

    let (posts, after) = parse_listing(&value);
    scan::enqueue_posts(&st, &posts);
    Json(FeedResponse { posts, after }).into_response()
}

pub fn media_allowed(url: &Url) -> bool {
    url.scheme() == "https" && url.host_str().is_some_and(|h| ALLOWED_HOSTS.contains(&h))
}

async fn media(State(st): State<AppState>, headers: HeaderMap, Query(q): Query<MediaQuery>) -> Response {
    let Some(raw) = q.url else {
        return err(StatusCode::BAD_REQUEST, "Parameter url fehlt");
    };
    let Ok(url) = Url::parse(&raw) else {
        return err(StatusCode::BAD_REQUEST, "Ungültige URL");
    };
    if !media_allowed(&url) {
        return err(StatusCode::FORBIDDEN, "Host nicht erlaubt");
    }

    let mut req = st.client.get(url).timeout(Duration::from_secs(60));
    // Range durchreichen, damit Videos auf dem iPhone/Safari spulen können.
    if let Some(range) = headers.get(header::RANGE) {
        req = req.header(header::RANGE, range.as_bytes());
    }
    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => return err(StatusCode::BAD_GATEWAY, format!("Medium nicht ladbar: {e}")),
    };
    let upstream = resp.status();
    if upstream != reqwest::StatusCode::OK && upstream != reqwest::StatusCode::PARTIAL_CONTENT {
        return err(StatusCode::BAD_GATEWAY, format!("Medium liefert HTTP {}", upstream.as_u16()));
    }

    let mut builder = Response::builder()
        .status(StatusCode::from_u16(upstream.as_u16()).unwrap_or(StatusCode::OK))
        .header(header::CACHE_CONTROL, "public, max-age=86400");
    for name in [
        header::CONTENT_TYPE,
        header::CONTENT_LENGTH,
        header::CONTENT_RANGE,
        header::ACCEPT_RANGES,
    ] {
        if let Some(v) = resp.headers().get(&name) {
            builder = builder.header(name, v.as_bytes());
        }
    }
    builder
        .body(Body::from_stream(resp.bytes_stream()))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

#[derive(Debug, PartialEq)]
enum Source {
    /// Startseite des eingeloggten Accounts.
    Home,
    /// Ein Subreddit oder mehrere mit "+" verbundene.
    Sub(String),
}

/// Akzeptiert "" / "home" / "https://www.reddit.com/" (Startseite) sowie
/// "https://www.reddit.com/r/aww/", "r/aww", "/r/aww", "aww" und "r/aww+cats".
fn parse_source(input: &str) -> Option<Source> {
    let s = input.trim();
    let name = if s.contains("reddit.com") {
        let rest = s.split_once("reddit.com")?.1;
        let mut segs = rest.split(['/', '?', '#']).filter(|x| !x.is_empty());
        match segs.next() {
            None => return Some(Source::Home),
            Some(first) if SORTS.contains(&first) => return Some(Source::Home),
            Some("r") => segs.next()?.to_string(),
            Some(_) => return None,
        }
    } else if s.is_empty() || s.eq_ignore_ascii_case("home") {
        return Some(Source::Home);
    } else {
        s.strip_prefix("r/")
            .or_else(|| s.strip_prefix("/r/"))
            .unwrap_or(s)
            .trim_end_matches('/')
            .to_string()
    };
    let valid = !name.is_empty()
        && name.split('+').all(|part| {
            (1..=21).contains(&part.len()) && part.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        });
    valid.then_some(Source::Sub(name))
}

fn parse_listing(v: &Value) -> (Vec<Post>, Option<String>) {
    let after = v["data"]["after"].as_str().map(String::from);
    let posts = v["data"]["children"]
        .as_array()
        .map(|children| {
            children
                .iter()
                .filter_map(|c| to_post(&c["data"]))
                .collect()
        })
        .unwrap_or_default();
    (posts, after)
}

/// Gibt nur Beiträge mit mindestens einem Medium zurück.
fn to_post(d: &Value) -> Option<Post> {
    // Beworbene Beiträge werden wie im Addon übersprungen.
    if d["promoted"].as_bool() == Some(true)
        || d["is_created_from_ads_ui"].as_bool() == Some(true)
        || d["promoted_by"].is_string()
    {
        return None;
    }
    let mut media = extract_media(d);
    if media.is_empty() {
        // Crossposts tragen das Medium im Original-Beitrag.
        if let Some(parent) = d["crosspost_parent_list"].get(0).filter(|p| p.is_object()) {
            media = extract_media(parent);
        }
    }
    if media.is_empty() {
        return None;
    }
    Some(Post {
        id: d["id"].as_str()?.to_string(),
        title: d["title"].as_str().unwrap_or_default().to_string(),
        subreddit: d["subreddit"].as_str().unwrap_or_default().to_string(),
        author: d["author"].as_str().unwrap_or_default().to_string(),
        score: d["score"].as_i64().unwrap_or(0),
        comments: d["num_comments"].as_i64().unwrap_or(0),
        permalink: format!(
            "https://www.reddit.com{}",
            d["permalink"].as_str().unwrap_or_default()
        ),
        nsfw: d["over_18"].as_bool().unwrap_or(false),
        vote: match d["likes"].as_bool() {
            Some(true) => "up",
            Some(false) => "down",
            None => "none",
        },
        media,
    })
}

fn extract_media(d: &Value) -> Vec<Media> {
    let mut out = Vec::new();

    // Galerie: Reihenfolge laut gallery_data.items
    if d["is_gallery"].as_bool() == Some(true) {
        let (Some(items), Some(meta)) = (
            d["gallery_data"]["items"].as_array(),
            d["media_metadata"].as_object(),
        ) else {
            return out;
        };
        for item in items {
            let Some(id) = item["media_id"].as_str() else { continue };
            let Some(m) = meta.get(id) else { continue };
            if m["status"].as_str() != Some("valid") {
                continue;
            }
            match m["e"].as_str() {
                Some("Image") => {
                    if let Some(u) = m["s"]["u"].as_str() {
                        out.push(image(u, &m["s"]));
                    }
                }
                Some("AnimatedImage") => {
                    if let Some(mp4) = m["s"]["mp4"].as_str() {
                        out.push(video(mp4, Vec::new(), &m["s"]));
                    } else if let Some(gif) = m["s"]["gif"].as_str() {
                        out.push(image(gif, &m["s"]));
                    }
                }
                _ => {}
            }
        }
        return out;
    }

    // Reddit-Video: Ton liegt in einer eigenen Datei neben dem Video
    let rv = if d["is_video"].as_bool() == Some(true) {
        let sm = &d["secure_media"]["reddit_video"];
        let m = &d["media"]["reddit_video"];
        if sm.is_object() { sm } else { m }
    } else {
        &Value::Null
    };
    if let Some(fallback) = rv["fallback_url"].as_str() {
        let clean = fallback.split('?').next().unwrap_or(fallback);
        let base = clean.rsplit_once('/').map(|(b, _)| b).unwrap_or(clean);
        let audio = ["CMAF_AUDIO_128.mp4", "CMAF_AUDIO_64.mp4", "DASH_AUDIO_128.mp4", "DASH_audio.mp4", "DASH_AUDIO_64.mp4"]
            .iter()
            .map(|f| format!("{base}/{f}"))
            .collect();
        out.push(video(fallback, audio, rv));
        return out;
    }

    let url = d["url"].as_str().unwrap_or_default();
    let path = url.split(['?', '#']).next().unwrap_or_default().to_lowercase();
    if [".jpg", ".jpeg", ".png", ".webp", ".gif"].iter().any(|e| path.ends_with(e)) {
        out.push(image(url, &Value::Null));
    } else if path.ends_with(".gifv") {
        let mp4 = &url[..url.len() - ".gifv".len()];
        out.push(video(&format!("{mp4}.mp4"), Vec::new(), &Value::Null));
    } else if let Some(fb) = d["preview"]["reddit_video_preview"]["fallback_url"].as_str() {
        // Animierte Beiträge (RedGIFs, Giphy, …): Reddit hostet eine stumme MP4-Vorschau.
        out.push(video(fb, Vec::new(), &d["preview"]["reddit_video_preview"]));
    } else if let Some(id) = redgifs_id(url) {
        // Ohne Vorschau: RedGIFs-Player einbetten, wie das Addon es mit dem iframe macht.
        out.push(embed(&format!("https://www.redgifs.com/ifr/{id}")));
    } else if d["post_hint"].as_str() == Some("image") {
        let src = &d["preview"]["images"][0]["source"];
        if let Some(u) = src["url"].as_str() {
            out.push(image(u, src));
        }
    }
    out
}

fn image(raw: &str, dims: &Value) -> Media {
    Media {
        kind: "image",
        url: if optimize::enabled() { optimize::opt_url("image", &raw.replace("&amp;", "&"), &[]) } else { proxy(raw) },
        width: dims["x"].as_i64().or(dims["width"].as_i64()),
        height: dims["y"].as_i64().or(dims["height"].as_i64()),
        audio: Vec::new(),
        src: raw.replace("&amp;", "&"),
    }
}

fn video(raw: &str, audio: Vec<String>, dims: &Value) -> Media {
    Media {
        kind: "video",
        url: if optimize::enabled() { optimize::opt_url("video", &raw.replace("&amp;", "&"), &audio) } else { proxy(raw) },
        width: dims["width"].as_i64(),
        height: dims["height"].as_i64(),
        // Optimierte Videos haben den Ton schon eingebaut.
        audio: if optimize::enabled() { Vec::new() } else { audio.iter().map(|a| proxy(a)).collect() },
        src: raw.replace("&amp;", "&"),
    }
}

fn embed(url: &str) -> Media {
    Media { kind: "embed", url: url.to_string(), width: None, height: None, audio: Vec::new(), src: String::new() }
}

/// ID aus "https://www.redgifs.com/watch/<id>" (nur Buchstaben/Ziffern, sonst None).
fn redgifs_id(url: &str) -> Option<String> {
    let u = Url::parse(url).ok()?;
    let host = u.host_str()?;
    if host != "redgifs.com" && !host.ends_with(".redgifs.com") {
        return None;
    }
    let id = u.path_segments()?.filter(|x| !x.is_empty()).last()?.to_lowercase();
    (!id.is_empty() && id.chars().all(|c| c.is_ascii_alphanumeric())).then_some(id)
}

fn proxy(raw: &str) -> String {
    format!("/api/media?url={}", urlencoding::encode(&raw.replace("&amp;", "&")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_parsing() {
        let sub = |n: &str| Some(Source::Sub(n.to_string()));
        assert_eq!(parse_source("https://www.reddit.com/r/aww/"), sub("aww"));
        assert_eq!(parse_source("https://old.reddit.com/r/EarthPorn/top?t=week"), sub("EarthPorn"));
        assert_eq!(parse_source("r/aww"), sub("aww"));
        assert_eq!(parse_source("/r/aww"), sub("aww"));
        assert_eq!(parse_source("aww"), sub("aww"));
        assert_eq!(parse_source("r/aww+cats"), sub("aww+cats"));
        assert_eq!(parse_source(""), Some(Source::Home));
        assert_eq!(parse_source("home"), Some(Source::Home));
        assert_eq!(parse_source("https://www.reddit.com/"), Some(Source::Home));
        assert_eq!(parse_source("https://www.reddit.com/hot/?redswipe=app"), Some(Source::Home));
        assert_eq!(parse_source("https://example.com/r/aww"), None);
        assert_eq!(parse_source("https://www.reddit.com/user/x"), None);
        assert_eq!(parse_source("a b"), None);
        assert_eq!(parse_source("aww++cats"), None);
        assert_eq!(parse_source(&"x".repeat(22)), None);
    }

    #[test]
    fn skips_ads_and_handles_redgifs_and_crossposts() {
        let ad = json!({ "id": "a", "promoted": true, "url": "https://i.redd.it/a.jpg" });
        assert!(to_post(&ad).is_none());

        let rg = json!({ "id": "r", "url": "https://www.redgifs.com/watch/SomeId" });
        let p = to_post(&rg).unwrap();
        assert_eq!(p.media[0].kind, "embed");
        assert_eq!(p.media[0].url, "https://www.redgifs.com/ifr/someid");

        let rg_prev = json!({ "id": "r2", "url": "https://www.redgifs.com/watch/x",
            "preview": { "reddit_video_preview": { "fallback_url": "https://v.redd.it/q/DASH_720.mp4", "width": 1, "height": 2 } } });
        assert_eq!(to_post(&rg_prev).unwrap().media[0].kind, "video");

        let xp = json!({ "id": "x", "url": "https://www.reddit.com/r/a/comments/1/",
            "crosspost_parent_list": [ { "id": "o", "url": "https://i.redd.it/o.png" } ] });
        assert_eq!(to_post(&xp).unwrap().media[0].kind, "image");
        assert!(redgifs_id("https://evil.com/watch/abc").is_none());
    }

    #[test]
    fn listing_filters_text_posts_and_maps_video_audio() {
        let v = json!({
            "data": {
                "after": "t3_abc",
                "children": [
                    { "data": { "id": "t1", "title": "Text", "url": "https://www.reddit.com/r/aww/x", "subreddit": "aww", "author": "a", "score": 1, "num_comments": 0, "permalink": "/r/aww/x", "over_18": false } },
                    { "data": {
                        "id": "t2", "title": "Video", "subreddit": "aww", "author": "b", "score": 42,
                        "num_comments": 7, "permalink": "/r/aww/comments/t2/", "over_18": true,
                        "is_video": true, "url": "https://v.redd.it/xyz",
                        "secure_media": { "reddit_video": { "fallback_url": "https://v.redd.it/xyz/DASH_720.mp4?source=fallback", "width": 720, "height": 1280 } }
                    } }
                ]
            }
        });
        let (posts, after) = parse_listing(&v);
        assert_eq!(after.as_deref(), Some("t3_abc"));
        assert_eq!(posts.len(), 1);
        let p = &posts[0];
        assert_eq!(p.id, "t2");
        assert!(p.nsfw);
        assert_eq!(p.permalink, "https://www.reddit.com/r/aww/comments/t2/");
        assert_eq!(p.media[0].kind, "video");
        assert!(p.media[0].url.starts_with("/api/opt?k=video&url=https%3A%2F%2Fv.redd.it%2Fxyz%2FDASH_720.mp4%3Fsource%3Dfallback&a="));
        assert!(p.media[0].url.contains("CMAF_AUDIO_128.mp4"));
        // Ton ist im optimierten Video schon enthalten.
        assert!(p.media[0].audio.is_empty());
        assert_eq!(p.media[0].width, Some(720));
    }

    #[test]
    fn gallery_keeps_order_and_skips_invalid() {
        let v = json!({
            "is_gallery": true,
            "gallery_data": { "items": [ { "media_id": "b" }, { "media_id": "a" }, { "media_id": "gone" } ] },
            "media_metadata": {
                "a": { "status": "valid", "e": "Image", "s": { "u": "https://i.redd.it/a.jpg?x=1&amp;y=2", "x": 10, "y": 20 } },
                "b": { "status": "valid", "e": "Image", "s": { "u": "https://i.redd.it/b.png", "x": 1, "y": 2 } },
                "gone": { "status": "failed" }
            }
        });
        let m = extract_media(&v);
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].url, "/api/opt?k=image&url=https%3A%2F%2Fi.redd.it%2Fb.png");
        assert_eq!(m[1].url, "/api/opt?k=image&url=https%3A%2F%2Fi.redd.it%2Fa.jpg%3Fx%3D1%26y%3D2");
        assert_eq!(m[1].width, Some(10));
    }

    #[test]
    fn host_allowlist() {
        assert!(ALLOWED_HOSTS.contains(&"i.redd.it"));
        assert!(!ALLOWED_HOSTS.contains(&"evil.com"));
    }
}
