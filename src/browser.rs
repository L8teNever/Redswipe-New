//! Reddit-Daten aus einem echten, eingeloggten Chromium holen (Fernsteuer-Schnittstelle CDP).
//! Reddit blockt Skript-Clients, lässt aber den Browser durch, in dem man sich selbst eingeloggt hat.
//! Wir öffnen die JSON-Adresse dort wie eine normale Seite und lesen den Text aus.
//! Das Login bleibt im Browser-Profil; hier wird kein Cookie gespeichert.

use std::{env, time::Duration};

use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::time::{Instant, sleep, timeout};
use tokio_tungstenite::{connect_async, tungstenite::Message};

pub enum BrowserError {
    /// Browser nicht erreichbar oder nicht eingerichtet: Aufrufer darf auf den Cookie-Weg zurückfallen.
    Unavailable(String),
    /// Reddit zeigt Login- oder Sperrseite: im Browser neu einloggen.
    NeedsLogin(String),
    /// Sonstiger Fehler (Timeout, kaputte Antwort).
    Failed(String),
}

fn base() -> Option<String> {
    let v = env::var("BROWSER_CDP").ok()?;
    let v = v.trim().trim_end_matches('/').to_string();
    (!v.is_empty()).then_some(v)
}

pub fn configured() -> bool {
    base().is_some()
}

/// Chromium akzeptiert nur IP oder "localhost" als Host-Header, daher den Namen vorher auflösen.
async fn resolve(base: &str) -> Result<String, String> {
    let rest = base.strip_prefix("http://").unwrap_or(base);
    let (host, port) = rest.rsplit_once(':').unwrap_or((rest, "80"));
    let port: u16 = port.parse().map_err(|_| "BROWSER_CDP: ungültiger Port".to_string())?;
    let addrs = tokio::net::lookup_host((host, port))
        .await
        .map_err(|e| format!("Browser-Name nicht auflösbar: {e}"))?;
    addrs
        .filter(|a| a.is_ipv4())
        .map(|a| a.to_string())
        .next()
        .ok_or_else(|| "Browser hat keine IPv4-Adresse".to_string())
}

/// Lädt `url` als normale Seite im Browser und liefert den Seitentext (bei JSON-Adressen das JSON).
pub async fn fetch_text(url: &str) -> Result<String, BrowserError> {
    let Some(b) = base() else {
        return Err(BrowserError::Unavailable("BROWSER_CDP nicht gesetzt".into()));
    };
    let addr = resolve(&b).await.map_err(BrowserError::Unavailable)?;
    let http = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|e| BrowserError::Failed(e.to_string()))?;

    let new_url = format!("http://{addr}/json/new?{}", urlencoding::encode(url));
    let target: Value = http
        .put(&new_url)
        .send()
        .await
        .map_err(|e| BrowserError::Unavailable(format!("Browser nicht erreichbar: {e}")))?
        .json()
        .await
        .map_err(|e| BrowserError::Unavailable(format!("Browser antwortet ungültig: {e}")))?;
    let Some(id) = target["id"].as_str().map(str::to_string) else {
        return Err(BrowserError::Unavailable("Browser hat keinen Tab geöffnet".into()));
    };

    let result = timeout(Duration::from_secs(40), read_page(&addr, &id)).await;

    // Tab immer wieder schließen.
    let _ = http.get(format!("http://{addr}/json/close/{id}")).send().await;

    match result {
        Ok(r) => r,
        Err(_) => Err(BrowserError::Failed("Der Browser hat zu lange gebraucht".into())),
    }
}

async fn read_page(addr: &str, id: &str) -> Result<String, BrowserError> {
    let (mut ws, _) = connect_async(format!("ws://{addr}/devtools/page/{id}"))
        .await
        .map_err(|e| BrowserError::Unavailable(format!("Browser-Steuerung nicht erreichbar: {e}")))?;

    let expr = "JSON.stringify({h:location.href,s:document.readyState,t:document.readyState==='complete'&&document.body?document.body.innerText:''})";
    let started = Instant::now();
    let mut next_id = 1u64;
    loop {
        if started.elapsed() > Duration::from_secs(30) {
            return Err(BrowserError::Failed("Reddit hat im Browser nicht rechtzeitig geladen".into()));
        }
        let this = next_id;
        next_id += 1;
        let msg = json!({"id": this, "method": "Runtime.evaluate", "params": {"expression": expr, "returnByValue": true}});
        ws.send(Message::text(msg.to_string()))
            .await
            .map_err(|e| BrowserError::Failed(format!("Browser-Steuerung: {e}")))?;

        let reply = loop {
            let Some(m) = ws.next().await else {
                return Err(BrowserError::Failed("Browser hat die Verbindung geschlossen".into()));
            };
            let m = m.map_err(|e| BrowserError::Failed(format!("Browser-Steuerung: {e}")))?;
            if let Message::Text(t) = m {
                if let Ok(v) = serde_json::from_str::<Value>(&t) {
                    if v["id"].as_u64() == Some(this) {
                        break v;
                    }
                }
            }
        };

        let raw = reply["result"]["result"]["value"].as_str().unwrap_or("");
        let page: Value = serde_json::from_str(raw).unwrap_or(Value::Null);
        let href = page["h"].as_str().unwrap_or("");
        let text = page["t"].as_str().unwrap_or("");

        if href.contains("/login") || href.contains("/account/") {
            return Err(BrowserError::NeedsLogin("Reddit will einen Login".into()));
        }
        if page["s"].as_str() == Some("complete") && href.starts_with("http") {
            let trimmed = text.trim_start();
            if trimmed.starts_with('{') || trimmed.starts_with('[') {
                return Ok(text.to_string());
            }
            if !trimmed.is_empty() {
                let lower = trimmed.to_lowercase();
                let hint = if lower.contains("blocked") || lower.contains("log in") || lower.contains("login") {
                    "Reddit zeigt eine Sperr- oder Login-Seite"
                } else {
                    "Reddit hat kein JSON geliefert"
                };
                return Err(BrowserError::NeedsLogin(hint.into()));
            }
        }
        sleep(Duration::from_millis(400)).await;
    }
}

/// Öffnet `page_url` im Browser, wartet bis die Seite geladen ist und führt dort `expression` aus
/// (darf ein Promise sein). So laufen Aktionen wie Voten oder Folgen mit dem echten Login und
/// den echten Cookies, genau wie das Addon es im Reddit-Tab macht.
pub async fn eval_in_page(page_url: &str, expression: &str) -> Result<Value, BrowserError> {
    let Some(b) = base() else {
        return Err(BrowserError::Unavailable("BROWSER_CDP nicht gesetzt".into()));
    };
    let addr = resolve(&b).await.map_err(BrowserError::Unavailable)?;
    let http = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|e| BrowserError::Failed(e.to_string()))?;

    let new_url = format!("http://{addr}/json/new?{}", urlencoding::encode(page_url));
    let target: Value = http
        .put(&new_url)
        .send()
        .await
        .map_err(|e| BrowserError::Unavailable(format!("Browser nicht erreichbar: {e}")))?
        .json()
        .await
        .map_err(|e| BrowserError::Unavailable(format!("Browser antwortet ungültig: {e}")))?;
    let Some(id) = target["id"].as_str().map(str::to_string) else {
        return Err(BrowserError::Unavailable("Browser hat keinen Tab geöffnet".into()));
    };

    let result = timeout(Duration::from_secs(70), run_expression(&addr, &id, expression, page_url)).await;
    let _ = http.get(format!("http://{addr}/json/close/{id}")).send().await;
    match result {
        Ok(r) => r,
        Err(_) => Err(BrowserError::Failed("Der Browser hat zu lange gebraucht".into())),
    }
}

async fn cdp_call(
    ws: &mut tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    id: u64,
    method: &str,
    params: Value,
) -> Result<Value, BrowserError> {
    ws.send(Message::text(json!({"id": id, "method": method, "params": params}).to_string()))
        .await
        .map_err(|e| BrowserError::Failed(format!("Browser-Steuerung: {e}")))?;
    loop {
        let Some(m) = ws.next().await else {
            return Err(BrowserError::Failed("Browser hat die Verbindung geschlossen".into()));
        };
        let m = m.map_err(|e| BrowserError::Failed(format!("Browser-Steuerung: {e}")))?;
        if let Message::Text(t) = m {
            if let Ok(v) = serde_json::from_str::<Value>(&t) {
                if v["id"].as_u64() == Some(id) {
                    return Ok(v);
                }
            }
        }
    }
}

async fn run_expression(addr: &str, id: &str, expression: &str, page_url: &str) -> Result<Value, BrowserError> {
    // Reddit wird wie bisher bis „complete“ abgewartet; andere Seiten (z. B. Pornhub mit viel Werbung)
    // reichen, sobald das Dokument geparst ist.
    let host = reqwest::Url::parse(page_url).ok().and_then(|u| u.host_str().map(|h| h.trim_start_matches("www.").to_string())).unwrap_or_default();
    let is_reddit = host.ends_with("reddit.com");
    let (mut ws, _) = connect_async(format!("ws://{addr}/devtools/page/{id}"))
        .await
        .map_err(|e| BrowserError::Unavailable(format!("Browser-Steuerung nicht erreichbar: {e}")))?;

    // Warten, bis die Seite geladen ist.
    let started = Instant::now();
    let mut n = 1u64;
    loop {
        if started.elapsed() > Duration::from_secs(25) {
            let site = if is_reddit { "Reddit" } else { "Die Seite" };
            return Err(BrowserError::Failed(format!("{site} hat im Browser nicht rechtzeitig geladen")));
        }
        let r = cdp_call(
            &mut ws,
            n,
            "Runtime.evaluate",
            json!({"expression": "location.href+'|'+document.readyState", "returnByValue": true}),
        )
        .await?;
        n += 1;
        let s = r["result"]["result"]["value"].as_str().unwrap_or("");
        if is_reddit && (s.contains("/login") || s.contains("/account/")) {
            return Err(BrowserError::NeedsLogin("Reddit will einen Login".into()));
        }
        if is_reddit {
            if s.starts_with("https://") && s.contains("reddit.com") && s.ends_with("|complete") {
                break;
            }
        } else if s.starts_with("https://") && s.contains(&host) && (s.ends_with("|complete") || s.ends_with("|interactive")) {
            break;
        }
        sleep(Duration::from_millis(300)).await;
    }

    let r = cdp_call(
        &mut ws,
        n,
        "Runtime.evaluate",
        json!({"expression": expression, "awaitPromise": true, "returnByValue": true}),
    )
    .await?;
    if let Some(ex) = r["result"]["exceptionDetails"].as_object() {
        let text = ex.get("text").and_then(Value::as_str).unwrap_or("Fehler im Browser");
        return Err(BrowserError::Failed(text.to_string()));
    }
    Ok(r["result"]["result"]["value"].clone())
}

/// Löst die Video-Quellen einer PimpBunny-Videoseite über den Server-Browser auf.
/// Fängt die Weiterleitung (302) auf den Speicher-Server (st*.pimpbunny.com) ab.
pub async fn resolve_pb_video(key: &str) -> Result<Value, BrowserError> {
    let Some(b) = base() else {
        return Err(BrowserError::Unavailable("BROWSER_CDP nicht gesetzt".into()));
    };
    let addr = resolve(&b).await.map_err(BrowserError::Unavailable)?;
    let http = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|e| BrowserError::Failed(e.to_string()))?;

    let page_url = format!("https://pimpbunny.com/videos/{key}/");
    let new_url = format!("http://{addr}/json/new?{}", urlencoding::encode(&page_url));
    let target: Value = http
        .put(&new_url)
        .send()
        .await
        .map_err(|e| BrowserError::Unavailable(format!("Browser nicht erreichbar: {e}")))?
        .json()
        .await
        .map_err(|e| BrowserError::Unavailable(format!("Browser antwortet ungültig: {e}")))?;
    let Some(id) = target["id"].as_str().map(str::to_string) else {
        return Err(BrowserError::Unavailable("Browser hat keinen Tab geöffnet".into()));
    };

    let result = timeout(Duration::from_secs(55), run_pb_video(&addr, &id, &page_url)).await;
    let _ = http.get(format!("http://{addr}/json/close/{id}")).send().await;
    match result {
        Ok(r) => r,
        Err(_) => Err(BrowserError::Failed("Auflösen von PimpBunny hat zu lange gedauert".into())),
    }
}

async fn run_pb_video(addr: &str, id: &str, _page_url: &str) -> Result<Value, BrowserError> {
    let (mut ws, _) = connect_async(format!("ws://{addr}/devtools/page/{id}"))
        .await
        .map_err(|e| BrowserError::Unavailable(format!("Browser-Steuerung nicht erreichbar: {e}")))?;

    let mut n = 1u64;
    let _ = cdp_call(&mut ws, n, "Network.enable", json!({})).await?;
    n += 1;
    let _ = cdp_call(
        &mut ws,
        n,
        "Network.setCookie",
        json!({
            "name": "is_age_ok",
            "value": "1",
            "domain": ".pimpbunny.com",
            "path": "/"
        }),
    )
    .await?;
    n += 1;

    let started = Instant::now();
    loop {
        if started.elapsed() > Duration::from_secs(20) {
            return Err(BrowserError::Failed("PimpBunny hat nicht rechtzeitig geladen".into()));
        }
        let r = cdp_call(
            &mut ws,
            n,
            "Runtime.evaluate",
            json!({"expression": "location.href+'|'+document.readyState", "returnByValue": true}),
        )
        .await?;
        n += 1;
        let s = r["result"]["result"]["value"].as_str().unwrap_or("");
        if s.starts_with("https://") && s.contains("pimpbunny.com") && (s.ends_with("|complete") || s.ends_with("|interactive")) {
            break;
        }
        sleep(Duration::from_millis(300)).await;
    }

    // Bot-Sperre abwarten: warten, bis der Player auf der Seite da ist
    let wait_started = Instant::now();
    loop {
        let r = cdp_call(
            &mut ws,
            n,
            "Runtime.evaluate",
            json!({"expression": "!!(window.kvsplayer && window.kvsplayer.kt_player && window.kvsplayer.kt_player.conf && window.kvsplayer.kt_player.conf.video_url)", "returnByValue": true}),
        )
        .await?;
        n += 1;
        if r["result"]["result"]["value"].as_bool() == Some(true) {
            break;
        }
        if wait_started.elapsed() > Duration::from_secs(22) {
            break;
        }
        sleep(Duration::from_millis(500)).await;
    }

    let expr = r#"(()=>{
        const conf = (window.kvsplayer && window.kvsplayer.kt_player && window.kvsplayer.kt_player.conf) || {};
        const items = [
            { q: 360, u: conf.video_url },
            { q: 480, u: conf.video_alt_url },
            { q: 720, u: conf.video_alt_url2 },
            { q: 1080, u: conf.video_alt_url3 },
            { q: 1440, u: conf.video_alt_url4 }
        ].filter(x => x.u);
        return {
            title: (conf.video_title || document.title || '').trim(),
            uploader: (conf.video_models || '').trim(),
            model: (() => {
                const a = document.querySelector('[class*="pages-view-video-model-title"] a[href], [class*="pages-view-video-models"] a[href*="/onlyfans-creators/"], [class*="pages-view-video-models"] a[href*="/models/"]');
                if (!a) return null;
                const m = new URL(a.href, location.href).pathname.match(/^\/(onlyfans-creators|models|male-models)\/[^/]+/);
                return m ? { name: a.innerText.trim(), path: '/pb' + m[0] } : null;
            })(),
            items: items.map(it => ({ q: it.q, u: it.u }))
        };
    })()"#;

    let r = cdp_call(
        &mut ws,
        n,
        "Runtime.evaluate",
        json!({"expression": expr, "returnByValue": true}),
    )
    .await?;

    let meta = r["result"]["result"]["value"].clone();
    let title = meta["title"].as_str().unwrap_or("PimpBunny Video");
    // Erster Creator mit Link (Name und Creator-Seite), sonst der Name aus dem Player
    let model_name = meta["model"]["name"].as_str().unwrap_or("").trim();
    let upath = meta["model"]["path"].as_str().filter(|p| crate::pb::valid_user_path(p)).unwrap_or("");
    let uploader = if model_name.is_empty() { meta["uploader"].as_str().unwrap_or("") } else { model_name };
    let items = meta["items"].as_array().cloned().unwrap_or_default();

    // Die Links selbst sind für den Server gesperrt; `/api/ph/media` löst sie beim Abspielen
    // über den Browser auf den Speicher-Server auf (nur die Qualität, die wirklich läuft).
    let mut sources: Vec<Value> = items
        .iter()
        .filter_map(|it| {
            let u = it["u"].as_str()?;
            let u = u.find("https://").map(|i| &u[i..])?;
            // Qualität steht im Dateinamen (…_1080p.mp4), sonst gilt der Platz in der Liste
            let named = u.split(".mp4").next().and_then(|a| a.rsplit('_').next()).and_then(|a| a.strip_suffix('p')).and_then(|a| a.parse::<u64>().ok());
            Some(json!({ "q": named.unwrap_or_else(|| it["q"].as_u64().unwrap_or(0)), "url": u }))
        })
        .collect();
    sources.sort_by_key(|s| std::cmp::Reverse(s["q"].as_u64().unwrap_or(0)));
    if sources.is_empty() {
        return Err(BrowserError::Failed("PimpBunny hat kein Video geliefert (Seite nicht fertig geladen oder Video entfernt)".into()));
    }

    Ok(json!({
        "title": title,
        "uploader": if upath.is_empty() { json!({ "name": uploader }) } else { json!({ "name": uploader, "path": upath }) },
        "sources": sources
    }))
}

/// Folgt einer Weiterleitung im Server-Browser und gibt die Zieladresse zurück
/// (PimpBunny: `get_file`-Links führen auf den Speicher-Server, den der Server direkt laden darf).
pub async fn resolve_redirect(url: &str) -> Result<String, BrowserError> {
    let Some(b) = base() else {
        return Err(BrowserError::Unavailable("BROWSER_CDP nicht gesetzt".into()));
    };
    let addr = resolve(&b).await.map_err(BrowserError::Unavailable)?;
    let http = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|e| BrowserError::Failed(e.to_string()))?;
    let target: Value = http
        .put(format!("http://{addr}/json/new?{}", urlencoding::encode(url)))
        .send()
        .await
        .map_err(|e| BrowserError::Unavailable(format!("Browser nicht erreichbar: {e}")))?
        .json()
        .await
        .map_err(|e| BrowserError::Unavailable(format!("Browser antwortet ungültig: {e}")))?;
    let Some(id) = target["id"].as_str().map(str::to_string) else {
        return Err(BrowserError::Unavailable("Browser hat keinen Tab geöffnet".into()));
    };
    let started = Instant::now();
    let mut found = None;
    while started.elapsed() < Duration::from_secs(9) {
        sleep(Duration::from_millis(400)).await;
        let Ok(resp) = http.get(format!("http://{addr}/json/list")).send().await else { continue };
        let Ok(list) = resp.json::<Value>().await else { continue };
        let cur = list
            .as_array()
            .and_then(|a| a.iter().find(|t| t["id"].as_str() == Some(id.as_str())))
            .and_then(|t| t["url"].as_str())
            .unwrap_or("")
            .to_string();
        let moved = reqwest::Url::parse(&cur)
            .ok()
            .is_some_and(|u| u.scheme() == "https" && u.host_str().is_some_and(|h| h != "pimpbunny.com" && h.ends_with(".pimpbunny.com")));
        if moved {
            found = Some(cur);
            break;
        }
    }
    let _ = http.get(format!("http://{addr}/json/close/{id}")).send().await;
    found.ok_or_else(|| BrowserError::Failed("Weiterleitung nicht auflösbar".into()))
}
