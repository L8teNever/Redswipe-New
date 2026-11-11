//! PimpBunny-Anbindung: Listen und Suche über den Server-Browser, Videos als MP4.

use serde_json::Value;

use std::{
    collections::HashMap,
    sync::Mutex,
    time::{Duration, Instant},
};

use crate::browser;

static VIDEOS: Mutex<Option<HashMap<String, (Instant, Value)>>> = Mutex::new(None);
static CACHE: Mutex<Option<HashMap<String, (Instant, Vec<Value>)>>> = Mutex::new(None);

/// Holt die Videoliste oder Suchergebnisse von PimpBunny über den Server-Browser.
pub async fn list(page: u32, q: Option<&str>) -> Result<Vec<Value>, browser::BrowserError> {
    let q_clean = q.map(str::trim).filter(|s| !s.is_empty());
    let url = match q_clean {
        Some(query) => {
            let enc = urlencoding::encode(query);
            if page <= 1 {
                format!("https://pimpbunny.com/search/{enc}/")
            } else {
                format!("https://pimpbunny.com/search/{enc}/{page}/")
            }
        }
        None => {
            if page <= 1 {
                "https://pimpbunny.com/videos/".to_string()
            } else {
                format!("https://pimpbunny.com/videos/{page}/")
            }
        }
    };
    Ok(fetch(&url).await?.0)
}

/// Gültiger Creator-Pfad in der App: `/pb/onlyfans-creators/name` (oder `/pb/models/…`, `/pb/male-models/…`).
pub fn valid_user_path(p: &str) -> bool {
    let Some(rest) = ["/pb/onlyfans-creators/", "/pb/models/", "/pb/male-models/"].iter().find_map(|pre| p.strip_prefix(pre)) else {
        return false;
    };
    !rest.is_empty() && rest.len() <= 100 && rest.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// Alle Videos eines Creators (Seite für Seite) samt Name, Bild und Anzahl.
pub async fn user(path: &str, page: u32) -> Result<(Vec<Value>, Value), browser::BrowserError> {
    if !valid_user_path(path) {
        return Err(browser::BrowserError::Failed("Ungültiger Creator".into()));
    }
    let site = path.trim_start_matches("/pb");
    let url = if page <= 1 { format!("https://pimpbunny.com{site}/") } else { format!("https://pimpbunny.com{site}/{page}/") };
    fetch(&url).await
}

static INFO: Mutex<Option<HashMap<String, Value>>> = Mutex::new(None);

/// Seite im Server-Browser laden und die Kacheln auslesen, 5 Minuten zwischengespeichert.
async fn fetch(url: &str) -> Result<(Vec<Value>, Value), browser::BrowserError> {
    let url = url.to_string();
    let info_of = |u: &str| INFO.lock().ok().and_then(|g| g.as_ref().and_then(|m| m.get(u).cloned())).unwrap_or(Value::Null);
    if let Some((t, v)) = CACHE.lock().ok().and_then(|g| g.as_ref().and_then(|m| m.get(&url).cloned())) {
        if t.elapsed() < Duration::from_secs(300) {
            return Ok((v, info_of(&url)));
        }
    }

    let js = r#"(async()=>{
        if (!document.cookie.includes('is_age_ok')) {
            document.cookie = 'is_age_ok=1; path=/; max-age=31536000';
            const btn = document.querySelector('[class*="age-verification"] a, [class*="age-verification"] button');
            if (btn) btn.click();
        }
        // Creator-Seite: nur die eigene Videoliste, nicht die Empfehlungen darunter
        const creator = /^\/(onlyfans-creators|models|male-models)\/[^/]+/.test(location.pathname);
        const cPath = creator ? '/pb' + location.pathname.match(/^\/(onlyfans-creators|models|male-models)\/[^/]+/)[0] : '';
        const cName = creator ? ((document.querySelector('h1') || {}).innerText || '').trim() : '';
        const extract = () => {
        const list = [];
        const root = creator ? document.querySelector('#list_videos_model_video_list') : document;
        if (!root) return list;
        root.querySelectorAll('a[href*="/videos/"]').forEach(a => {
            const href = a.getAttribute('href');
            if (!href || href === '/videos/' || href.includes('/search/') || href.includes('/categories/')) return;
            const m = href.match(/\/videos?\/([^/]+)/);
            if (!m) return;
            const img = a.querySelector('img');
            if (!img) return;
            const card = a.closest('[class*="ui-card-video"]') || a.closest('.item, [class*="item"], [class*="video"], div') || a;
            const title = (a.getAttribute('title') || img.getAttribute('alt') || a.innerText).trim();
            const dur = (card.querySelector('[class*="duration"], [class*="time"], time') || {}).innerText || '';
            // Erster verlinkter Creator der Kachel (bei Collabs stehen mehrere da)
            const ml = card.querySelector('[class*="related-models"] a[href]');
            let uploader = ml ? ml.innerText : ((card.querySelector('[class*="model"], [class*="author"], a[href*="/models/"]') || {}).innerText || '');
            let upath = '';
            if (ml) { try { const pm = new URL(ml.href).pathname.match(/^\/(onlyfans-creators|models|male-models)\/[^/]+/); if (pm) upath = '/pb' + pm[0]; } catch (e) {} }
            if (creator && cName) { uploader = cName; upath = cPath; }
            const views = (card.querySelector('[class*="views"]') || {}).innerText || '';
            const attrUrl = (el, re) => { for (const at of el.attributes) { if (/^https?:\/\//.test(at.value) && re.test(at.value)) return at.value; } return ''; };
            let thumb = attrUrl(img, /\.(jpe?g|webp|png)/i);
            if (!thumb && img.src && !img.src.startsWith('data:')) thumb = img.src;
            let preview = attrUrl(a, /\.mp4/i) || attrUrl(img, /\.mp4/i);
            if (!preview) { const pe = card.querySelector('[data-preview], [data-video], video'); if (pe) preview = attrUrl(pe, /\.mp4/i) || ''; }
            list.push({
                key: 'pb:' + m[1],
                title,
                thumb: thumb.replace(/\?.*$/, ''),
                duration: dur.trim().replace(/\s+/g, ' '),
                uploader: uploader.trim().replace(/\s+/g, ' '),
                upath,
                views: views.trim(),
                preview,
                src: 'pb'
            });
        });
        const unique = [];
        const seen = new Set();
        for (const x of list) {
            if (!seen.has(x.key)) { seen.add(x.key); unique.push(x); }
        }
        return unique;
        };
        // Bot-Sperre (Rechenaufgabe) abwarten: so lange prüfen, bis die Kacheln da sind
        let items = extract();
        // Creator-Seite hinter der letzten Seite: Überschrift da, aber keine Videoliste
        for (let i = 0; i < 40 && !items.length && !(creator && cName && i > 6 && !document.querySelector('#list_videos_model_video_list')); i++) { await new Promise(r => setTimeout(r, 500)); items = extract(); }
        // Vorschaubilder liegen hinter der Bot-Sperre: im Browser holen, verkleinern und einbetten
        const shrink = async (it) => {
            if (!it.thumb) return;
            try {
                const r = await fetch(it.thumb, { credentials: 'include' });
                if (!r.ok) return;
                const bmp = await createImageBitmap(await r.blob());
                const w = Math.min(420, bmp.width), h = Math.round(bmp.height * w / bmp.width);
                const c = document.createElement('canvas'); c.width = w; c.height = h;
                c.getContext('2d').drawImage(bmp, 0, 0, w, h);
                it.thumb = c.toDataURL('image/jpeg', 0.72);
            } catch (e) { it.thumb = ''; }
        };
        const queue = items.slice();
        await Promise.all(Array.from({ length: 6 }, async () => { for (let it; (it = queue.shift());) await shrink(it); }));
        const info = {};
        if (creator) {
            info.name = ((document.querySelector('h1') || {}).innerText || '').trim();
            const av = document.querySelector('img[src*="/contents/models/"]');
            if (av) { const x = { thumb: av.getAttribute('src') || '' }; await shrink(x); info.avatar = x.thumb; }
            const n = (document.body.innerText.match(/(\d[\d.,]*)\s*\n?\s*Videos?\b/i) || [])[1];
            if (n) info.stats = [{ l: 'Videos', v: n }];
        }
        return { ok: true, items, info, title: document.title };
    })()"#;

    let _permit = PAGES.acquire().await.map_err(|e| browser::BrowserError::Failed(e.to_string()))?;
    let mut items = Vec::new();
    let mut info = Value::Null;
    for _ in 0..2 {
        let v = browser::eval_in_page(&url, js).await?;
        items = v["items"].as_array().cloned().unwrap_or_default();
        info = v["info"].clone();
        // Creator-Seite hinter der letzten Seite ist leer: kein zweiter Versuch
        if !items.is_empty() || info["name"].as_str().is_some_and(|n| !n.is_empty()) {
            break;
        }
    }
    if !items.is_empty() {
        if let Ok(mut g) = CACHE.lock() {
            let m = g.get_or_insert_with(HashMap::new);
            if m.len() > 40 {
                m.clear();
            }
            m.insert(url.clone(), (Instant::now(), items.clone()));
        }
        if let Ok(mut g) = INFO.lock() {
            let m = g.get_or_insert_with(HashMap::new);
            if m.len() > 40 {
                m.clear();
            }
            m.insert(url, info.clone());
        }
    }
    Ok((items, info))
}

/// Löst einen `get_file`-Link (Vorschau-Clip) auf den Speicher-Server auf, 20 Minuten zwischengespeichert.
pub async fn resolve_file(url: &str) -> Result<String, browser::BrowserError> {
    static FILES: Mutex<Option<HashMap<String, (Instant, String)>>> = Mutex::new(None);
    if let Some((t, v)) = FILES.lock().ok().and_then(|g| g.as_ref().and_then(|m| m.get(url).cloned())) {
        if t.elapsed() < Duration::from_secs(1200) {
            return Ok(v);
        }
    }
    // Nacheinander auflösen: der Player fragt dieselbe Datei mehrfach gleichzeitig an.
    // Als eigene Aufgabe, damit ein abgebrochener Abruf den Tab im Browser trotzdem wieder schließt.
    static GATE: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    let owned = url.to_string();
    let v = tokio::spawn(async move {
        let _gate = GATE.lock().await;
        if let Some((t, v)) = FILES.lock().ok().and_then(|g| g.as_ref().and_then(|m| m.get(&owned).cloned())) {
            if t.elapsed() < Duration::from_secs(1200) {
                return Ok(v);
            }
        }
        let v = match browser::resolve_redirect(&owned).await {
            Ok(v) => v,
            Err(_) => browser::resolve_redirect(&owned).await?,
        };
        if let Ok(mut g) = FILES.lock() {
            let m = g.get_or_insert_with(HashMap::new);
            if m.len() > 500 {
                m.clear();
            }
            m.insert(owned, (Instant::now(), v.clone()));
        }
        Ok::<String, browser::BrowserError>(v)
    })
    .await
    .map_err(|e| browser::BrowserError::Failed(e.to_string()))??;
    Ok(v)
}

/// Höchstens zwei PimpBunny-Seiten gleichzeitig im Server-Browser: bei vielen Tabs auf einmal laden sie nicht fertig.
static PAGES: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(2);

/// Videoseite auflösen (Titel, Creator, Qualitäten), 10 Minuten zwischengespeichert, ein zweiter Versuch bei Fehler.
pub async fn video(key: &str) -> Result<Value, browser::BrowserError> {
    let hit = |k: &str| VIDEOS.lock().ok().and_then(|g| g.as_ref().and_then(|m| m.get(k).cloned())).filter(|(t, _)| t.elapsed() < Duration::from_secs(600)).map(|(_, v)| v);
    if let Some(v) = hit(key) {
        return Ok(v);
    }
    let owned = key.to_string();
    tokio::spawn(async move {
        let _permit = PAGES.acquire().await.map_err(|e| browser::BrowserError::Failed(e.to_string()))?;
        if let Some(v) = hit(&owned) {
            return Ok(v);
        }
        let v = match browser::resolve_pb_video(&owned).await {
            Ok(v) => v,
            Err(_) => browser::resolve_pb_video(&owned).await?,
        };
        if let Ok(mut g) = VIDEOS.lock() {
            let m = g.get_or_insert_with(HashMap::new);
            if m.len() > 300 {
                m.clear();
            }
            m.insert(owned, (Instant::now(), v.clone()));
        }
        Ok::<Value, browser::BrowserError>(v)
    })
    .await
    .map_err(|e| browser::BrowserError::Failed(e.to_string()))?
}

// ---------- Eigene Likes, Favoriten und Abos (lokal, PimpBunny kennt hier kein Konto) ----------

use std::{env, path::PathBuf, sync::OnceLock};

use rusqlite::{params, Connection, OptionalExtension};
use serde_json::json;

fn db() -> Result<&'static Mutex<Connection>, String> {
    static DB: OnceLock<Result<Mutex<Connection>, String>> = OnceLock::new();
    DB.get_or_init(|| {
        let path = PathBuf::from(env::var("VPN_DIR").unwrap_or_else(|_| "/vpn".into())).join("pb.sqlite");
        let c = Connection::open(path).map_err(|e| e.to_string())?;
        c.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS likes (key TEXT PRIMARY KEY, ts INTEGER NOT NULL);
             CREATE TABLE IF NOT EXISTS favs (key TEXT PRIMARY KEY, item TEXT NOT NULL, ts INTEGER NOT NULL);
             CREATE TABLE IF NOT EXISTS subs (name TEXT PRIMARY KEY, label TEXT NOT NULL, ts INTEGER NOT NULL);",
        )
        .map_err(|e| e.to_string())?;
        // Creator-Pfad zum Abo (ältere Datenbanken haben die Spalte noch nicht)
        let _ = c.execute("ALTER TABLE subs ADD COLUMN path TEXT NOT NULL DEFAULT ''", []);
        Ok(Mutex::new(c))
    })
    .as_ref()
    .map_err(|e| e.clone())
}

fn now() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

fn norm(name: &str) -> String {
    name.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

/// Kachel eines Videos aus dem Listen-Zwischenspeicher.
fn cached_item(key: &str) -> Option<Value> {
    let g = CACHE.lock().ok()?;
    g.as_ref()?
        .values()
        .filter_map(|(t, v)| v.iter().find(|it| it["key"].as_str() == Some(key)).map(|it| (*t, it)))
        .max_by_key(|(t, _)| *t)
        .map(|(_, it)| it.clone())
}

/// Creator eines Videos: aus der Kachel, der Videoseite oder den Favoriten.
fn uploader_of(key: &str, meta: Option<&Value>) -> String {
    let from_item = |v: &Value| v["uploader"].as_str().unwrap_or("").trim().to_string();
    let mut name = cached_item(key).map(|v| from_item(&v)).unwrap_or_default();
    if name.is_empty() {
        if let Some(v) = VIDEOS.lock().ok().and_then(|g| g.as_ref().and_then(|m| m.get(key.trim_start_matches("pb:")).cloned())) {
            name = v.1["uploader"]["name"].as_str().unwrap_or("").trim().to_string();
        }
    }
    if name.is_empty() {
        name = meta.map(from_item).unwrap_or_default();
    }
    name
}

/// Creator-Pfad eines Videos (`/pb/onlyfans-creators/…`), falls bekannt.
fn upath_of(key: &str, meta: Option<&Value>) -> String {
    let from_item = |v: &Value| v["upath"].as_str().unwrap_or("").to_string();
    let mut p = cached_item(key).map(|v| from_item(&v)).unwrap_or_default();
    if p.is_empty() {
        if let Some(v) = VIDEOS.lock().ok().and_then(|g| g.as_ref().and_then(|m| m.get(key.trim_start_matches("pb:")).cloned())) {
            p = v.1["uploader"]["path"].as_str().unwrap_or("").to_string();
        }
    }
    if p.is_empty() {
        p = meta.map(from_item).unwrap_or_default();
    }
    if valid_user_path(&p) {
        p
    } else {
        String::new()
    }
}

/// Zustand eines Videos: Like, Favorit und ob dem Creator gefolgt wird.
pub fn state(key: &str, meta: Option<&Value>) -> Result<Value, String> {
    let c = db()?.lock().map_err(|e| e.to_string())?;
    let up: bool = c.query_row("SELECT 1 FROM likes WHERE key = ?1", [key], |_| Ok(true)).optional().map_err(|e| e.to_string())?.is_some();
    let fav: bool = c.query_row("SELECT 1 FROM favs WHERE key = ?1", [key], |_| Ok(true)).optional().map_err(|e| e.to_string())?.is_some();
    let who = uploader_of(key, meta);
    let sub: bool = !who.is_empty() && c.query_row("SELECT 1 FROM subs WHERE name = ?1", [norm(&who)], |_| Ok(true)).optional().map_err(|e| e.to_string())?.is_some();
    Ok(json!({ "ok": true, "up": up as u8, "down": 0, "fav": fav as u8, "sub": sub as u8, "sub_name": who }))
}

/// Like, Favorit oder Abo umschalten.
pub fn toggle(key: &str, action: &str, meta: Option<&Value>) -> Result<Value, String> {
    {
        let c = db()?.lock().map_err(|e| e.to_string())?;
        match action {
            "up" => {
                let n = c.execute("DELETE FROM likes WHERE key = ?1", [key]).map_err(|e| e.to_string())?;
                if n == 0 {
                    c.execute("INSERT INTO likes (key, ts) VALUES (?1, ?2)", params![key, now()]).map_err(|e| e.to_string())?;
                }
            }
            "fav" => {
                let n = c.execute("DELETE FROM favs WHERE key = ?1", [key]).map_err(|e| e.to_string())?;
                if n == 0 {
                    let mut item = cached_item(key).or_else(|| meta.cloned()).unwrap_or_else(|| json!({ "title": key }));
                    if let Some(o) = item.as_object_mut() {
                        o.insert("key".into(), json!(key));
                        o.insert("src".into(), json!("pb"));
                    }
                    let who = uploader_of(key, meta);
                    if !who.is_empty() {
                        item["uploader"] = json!(who);
                    }
                    c.execute("INSERT INTO favs (key, item, ts) VALUES (?1, ?2, ?3)", params![key, item.to_string(), now()]).map_err(|e| e.to_string())?;
                }
            }
            "sub" => {
                let who = uploader_of(key, meta);
                if who.is_empty() {
                    return Err("Creator dieses Videos ist nicht bekannt".into());
                }
                let n = c.execute("DELETE FROM subs WHERE name = ?1", [norm(&who)]).map_err(|e| e.to_string())?;
                if n == 0 {
                    let path = upath_of(key, meta);
                    c.execute("INSERT INTO subs (name, label, ts, path) VALUES (?1, ?2, ?3, ?4)", params![norm(&who), who, now(), path]).map_err(|e| e.to_string())?;
                }
            }
            _ => return Err("Unbekannte Aktion".into()),
        }
    }
    state(key, meta)
}

/// Eigene Favoriten, neueste zuerst, 32 je Seite.
pub fn favs(page: u32) -> Result<Vec<Value>, String> {
    let c = db()?.lock().map_err(|e| e.to_string())?;
    let mut st = c.prepare("SELECT item FROM favs ORDER BY ts DESC LIMIT 32 OFFSET ?1").map_err(|e| e.to_string())?;
    let rows = st.query_map([(page.max(1) as i64 - 1) * 32], |r| r.get::<_, String>(0)).map_err(|e| e.to_string())?;
    Ok(rows.filter_map(|r| r.ok()).filter_map(|s| serde_json::from_str::<Value>(&s).ok()).collect())
}

/// Videos der Creator, denen gefolgt wird: von der Creator-Seite, sonst Suche nach dem Namen (nur Treffer dieses Creators).
pub async fn subs_feed(page: u32) -> Result<Vec<Value>, browser::BrowserError> {
    let names: Vec<(String, String)> = {
        let Ok(db) = db() else { return Ok(vec![]) };
        let Ok(c) = db.lock() else { return Ok(vec![]) };
        let Ok(mut st) = c.prepare("SELECT label, path FROM subs ORDER BY ts DESC LIMIT 8") else { return Ok(vec![]) };
        let Ok(rows) = st.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))) else { return Ok(vec![]) };
        rows.filter_map(|r| r.ok()).collect()
    };
    let mut tasks = Vec::new();
    for (n, path) in names {
        tasks.push(tokio::spawn(async move {
            if valid_user_path(&path) {
                return user(&path, page).await.map(|(items, _)| items).unwrap_or_default();
            }
            let items = list(page, Some(&n)).await.unwrap_or_default();
            let want = norm(&n);
            items.into_iter().filter(|it| norm(it["uploader"].as_str().unwrap_or("")) == want).collect::<Vec<_>>()
        }));
    }
    let mut lists = Vec::new();
    for t in tasks {
        lists.push(t.await.unwrap_or_default());
    }
    // Abwechselnd mischen, ohne doppelte Videos
    let mut out: Vec<Value> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let longest = lists.iter().map(Vec::len).max().unwrap_or(0);
    for i in 0..longest {
        for l in &lists {
            if let Some(it) = l.get(i) {
                if seen.insert(it["key"].as_str().unwrap_or("").to_string()) {
                    out.push(it.clone());
                }
            }
        }
    }
    Ok(out)
}
