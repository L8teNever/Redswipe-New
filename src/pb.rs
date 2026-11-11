//! PimpBunny-Anbindung: Listen und Suche über den Server-Browser, Videos als MP4.

use serde_json::Value;

use std::{
    collections::HashMap,
    sync::Mutex,
    time::{Duration, Instant},
};

use crate::browser;

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

    if let Some((t, v)) = CACHE.lock().ok().and_then(|g| g.as_ref().and_then(|m| m.get(&url).cloned())) {
        if t.elapsed() < Duration::from_secs(300) {
            return Ok(v);
        }
    }

    let js = r#"(async()=>{
        if (!document.cookie.includes('is_age_ok')) {
            document.cookie = 'is_age_ok=1; path=/; max-age=31536000';
            const btn = document.querySelector('[class*="age-verification"] a, [class*="age-verification"] button');
            if (btn) btn.click();
        }
        const extract = () => {
        const list = [];
        document.querySelectorAll('a[href*="/videos/"]').forEach(a => {
            const href = a.getAttribute('href');
            if (!href || href === '/videos/' || href.includes('/search/') || href.includes('/categories/')) return;
            const m = href.match(/\/videos?\/([^/]+)/);
            if (!m) return;
            const img = a.querySelector('img');
            if (!img) return;
            const card = a.closest('.item, [class*="item"], [class*="video"], div') || a;
            const title = (a.getAttribute('title') || img.getAttribute('alt') || a.innerText).trim();
            const dur = (card.querySelector('[class*="duration"], [class*="time"], time') || {}).innerText || '';
            const uploader = (card.querySelector('[class*="model"], [class*="author"], a[href*="/models/"]') || {}).innerText || '';
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
        for (let i = 0; i < 40 && !items.length; i++) { await new Promise(r => setTimeout(r, 500)); items = extract(); }
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
        return { ok: true, items, title: document.title };
    })()"#;

    let v = browser::eval_in_page(&url, js).await?;
    let items = v["items"].as_array().cloned().unwrap_or_default();
    if !items.is_empty() {
        if let Ok(mut g) = CACHE.lock() {
            let m = g.get_or_insert_with(HashMap::new);
            if m.len() > 40 {
                m.clear();
            }
            m.insert(url, (Instant::now(), items.clone()));
        }
    }
    Ok(items)
}

/// Löst einen `get_file`-Link (Vorschau-Clip) auf den Speicher-Server auf, 20 Minuten zwischengespeichert.
pub async fn resolve_file(url: &str) -> Result<String, browser::BrowserError> {
    static FILES: Mutex<Option<HashMap<String, (Instant, String)>>> = Mutex::new(None);
    if let Some((t, v)) = FILES.lock().ok().and_then(|g| g.as_ref().and_then(|m| m.get(url).cloned())) {
        if t.elapsed() < Duration::from_secs(1200) {
            return Ok(v);
        }
    }
    let v = browser::resolve_redirect(url).await?;
    if let Ok(mut g) = FILES.lock() {
        let m = g.get_or_insert_with(HashMap::new);
        if m.len() > 500 {
            m.clear();
        }
        m.insert(url.to_string(), (Instant::now(), v.clone()));
    }
    Ok(v)
}
