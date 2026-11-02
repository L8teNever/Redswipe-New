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

    let result = timeout(Duration::from_secs(40), run_expression(&addr, &id, expression)).await;
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

async fn run_expression(addr: &str, id: &str, expression: &str) -> Result<Value, BrowserError> {
    let (mut ws, _) = connect_async(format!("ws://{addr}/devtools/page/{id}"))
        .await
        .map_err(|e| BrowserError::Unavailable(format!("Browser-Steuerung nicht erreichbar: {e}")))?;

    // Warten, bis die Seite geladen ist.
    let started = Instant::now();
    let mut n = 1u64;
    loop {
        if started.elapsed() > Duration::from_secs(25) {
            return Err(BrowserError::Failed("Reddit hat im Browser nicht rechtzeitig geladen".into()));
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
        if s.contains("/login") || s.contains("/account/") {
            return Err(BrowserError::NeedsLogin("Reddit will einen Login".into()));
        }
        if s.starts_with("https://") && s.contains("reddit.com") && s.ends_with("|complete") {
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
