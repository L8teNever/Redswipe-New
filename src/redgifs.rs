//! RedGIFs-Nischen als zweite Quelle für den Vorrat (wie im Addon): offizielle API mit temporärem Token.
//! Die Clips landen wie Reddit-Beiträge im Vorrat und werden erst gezeigt, wenn sie fertig gescannt sind.

use std::{sync::OnceLock, time::Duration};

use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::{AppState, Media, Post, proxy};

/// Standard-Nischen aus dem Addon (redgifs.com/niches/<name>).
pub const DEFAULT_NICHES: &[&str] = &[
    "just-boobs", "amateur-girls", "transparent-clothing", "girls-showering", "bigtits", "natural-tits", "fake-boobs",
    "flashing-boobs", "grabbing-boobs", "perfect-boobs", "bouncing-boobs", "clapping-boobs", "middle-aged-boobs",
    "tit-slapping", "pulling-tits-out",
];
const EXCLUDED: &[&str] = &["legal-teens"];
const API: &str = "https://api.redgifs.com/v2";

fn token_cell() -> &'static Mutex<Option<String>> {
    static T: OnceLock<Mutex<Option<String>>> = OnceLock::new();
    T.get_or_init(|| Mutex::new(None))
}

async fn token(st: &AppState, fresh: bool) -> Option<String> {
    let mut t = token_cell().lock().await;
    if fresh {
        *t = None;
    }
    if t.is_none() {
        let v: Value = st.client.get(format!("{API}/auth/temporary")).timeout(Duration::from_secs(15)).send().await.ok()?.json().await.ok()?;
        *t = v["token"].as_str().map(String::from);
    }
    t.clone()
}

/// GET mit Token; bei 401 einmal mit frischem Token wiederholen.
async fn get(st: &AppState, url: &str) -> Result<Value, (u16, String)> {
    for attempt in 0..2 {
        let Some(tok) = token(st, attempt == 1).await else { return Err((0, "kein RedGIFs-Token".into())) };
        let r = st.client.get(url).bearer_auth(tok).timeout(Duration::from_secs(20)).send().await.map_err(|e| (0, e.to_string()))?;
        let code = r.status().as_u16();
        if code == 401 {
            continue;
        }
        if !r.status().is_success() {
            return Err((code, format!("HTTP {code}")));
        }
        return r.json().await.map_err(|e| (0, e.to_string()));
    }
    Err((401, "Token abgelehnt".into()))
}

/// Nischen-Namen aus Eingabe oder Link (https://www.redgifs.com/niches/just-boobs?gif=… -> "just-boobs").
pub fn parse_niche(s: &str) -> Option<String> {
    let s = s.trim();
    let rest = s.find("/niches/").map_or(s, |i| &s[i + 8..]);
    let slug: String = rest
        .trim()
        .replace(' ', "-")
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect::<String>()
        .to_lowercase();
    (!slug.is_empty() && slug.len() <= 60 && !EXCLUDED.contains(&slug.as_str())).then_some(slug)
}

/// Details zu einem einzelnen RedGIFs-Clip abrufen (z. B. aus einem Reddit-Link oder Embed).
/// Liefert die direkte MP4-Adresse (sd oder hd) und Maße.
pub async fn fetch_clip(st: &AppState, id: &str) -> Result<(String, Value), String> {
    let clean = id.trim().to_lowercase();
    if clean.is_empty() || clean.len() > 60 || !clean.chars().all(|c| c.is_ascii_alphanumeric()) {
        return Err("Ungültige Clip-ID".into());
    }
    let url = format!("{API}/gifs/{clean}");
    let v = get(st, &url).await.map_err(|e| e.1)?;
    let g = &v["gif"];
    let sd = g["urls"]["sd"]
        .as_str()
        .or_else(|| g["urls"]["hd"].as_str())
        .ok_or_else(|| "Keine MP4-Adresse".to_string())?;
    let poster = g["urls"]["poster"].as_str().or_else(|| g["urls"]["thumbnail"].as_str());
    let dims = json!({ "width": g["width"], "height": g["height"], "poster": poster });
    Ok((sd.to_string(), dims))
}

/// Eine Seite Clips einer Nische (beliebteste zuerst). Zweiter Wert: gibt es weitere Seiten?
/// Ist der Name keine echte Nische, wird stattdessen gesucht (wie im Addon).
pub async fn niche_page(st: &AppState, niche: &str, page: u32) -> Result<(Vec<Post>, bool), String> {
    let url = format!("{API}/niches/{niche}/gifs?order=hot&count=20&page={page}");
    let v = match get(st, &url).await {
        Ok(v) => v,
        Err((400 | 404, _)) => {
            let q = urlencoding::encode(&niche.replace('-', " ")).into_owned();
            get(st, &format!("{API}/gifs/search?search_text={q}&order=trending&count=20&page={page}")).await.map_err(|e| e.1)?
        }
        Err((_, m)) => return Err(m),
    };
    let pages = v["pages"].as_u64().unwrap_or(0);
    let posts = v["gifs"].as_array().into_iter().flatten().filter_map(|g| to_post(g, niche)).collect();
    Ok((posts, pages == 0 || u64::from(page) < pages))
}

fn to_post(g: &Value, niche: &str) -> Option<Post> {
    let id = g["id"].as_str()?.to_lowercase();
    if id.is_empty() || id.len() > 60 || !id.chars().all(|c| c.is_ascii_alphanumeric()) {
        return None;
    }
    let sd = g["urls"]["sd"].as_str().or_else(|| g["urls"]["hd"].as_str())?;
    // Nur Videos (Bilder haben keine MP4)
    if !sd.to_lowercase().contains(".mp4") {
        return None;
    }
    let tags: Vec<&str> = g["tags"].as_array().into_iter().flatten().filter_map(|t| t.as_str()).take(16).collect();
    let dims = json!({ "width": g["width"], "height": g["height"] });
    let poster = g["urls"]["poster"].as_str().or_else(|| g["urls"]["thumbnail"].as_str()).map(proxy);
    Some(Post {
        id: format!("rg_{id}"),
        title: g["description"].as_str().unwrap_or_default().chars().take(300).collect(),
        subreddit: format!("RedGIFs #{niche}"),
        author: g["userName"].as_str().unwrap_or_default().to_string(),
        score: g["likes"].as_i64().unwrap_or(0),
        comments: 0,
        permalink: format!("https://www.redgifs.com/watch/{id}"),
        nsfw: true,
        // Tags fließen wie ein Flair als Wörter in die Geschmacks-Bewertung ein
        flair: tags.join(", "),
        text: String::new(),
        vote: "none",
        origin: "redgifs",
        media: vec![Media {
            kind: "video",
            url: proxy(sd),
            poster,
            width: dims["width"].as_i64(),
            height: dims["height"].as_i64(),
            audio: Vec::new(),
            src: sd.to_string(),
        }],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn niche_names_and_links() {
        assert_eq!(parse_niche("https://www.redgifs.com/niches/just-boobs?gif=abc").as_deref(), Some("just-boobs"));
        assert_eq!(parse_niche(" Big Tits ").as_deref(), Some("big-tits"));
        assert_eq!(parse_niche("legal-teens"), None);
        assert_eq!(parse_niche(""), None);
    }

    #[test]
    fn clip_becomes_video_post() {
        let g = json!({ "id": "OddWhitesmokeDrongo", "userName": "x", "tags": ["Blonde", "Big Tits"], "description": "hi",
            "width": 720, "height": 1280, "urls": { "sd": "https://media.redgifs.com/OddWhitesmokeDrongo-mobile.mp4", "poster": "https://media.redgifs.com/OddWhitesmokeDrongo-poster.jpg" } });
        let p = to_post(&g, "just-boobs").unwrap();
        assert_eq!(p.id, "rg_oddwhitesmokedrongo");
        assert_eq!(p.origin, "redgifs");
        assert_eq!(p.media[0].kind, "video");
        assert_eq!(p.media[0].url, proxy("https://media.redgifs.com/OddWhitesmokeDrongo-mobile.mp4"));
        assert_eq!(p.media[0].poster, Some(proxy("https://media.redgifs.com/OddWhitesmokeDrongo-poster.jpg")));
        assert_eq!(p.media[0].src, "https://media.redgifs.com/OddWhitesmokeDrongo-mobile.mp4");
        assert!(p.flair.contains("Blonde"));
    }
}
