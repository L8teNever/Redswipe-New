//! Pornhub als eigene Seite (/ph.html): Startseite, Abos und Suche über den eingeloggten Browser
//! auf dem Server, Videos als HLS-Stream über den Server durchgereicht (gleiche IP wie der Browser,
//! weil die Stream-Adressen an die IP gebunden sind).

use std::time::Duration;

use axum::{
    Json,
    body::Body,
    extract::Query,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
};
use reqwest::Url;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{browser, err};

const SITE: &str = "https://www.pornhub.com";

#[derive(Deserialize)]
pub struct ListQuery {
    pub kind: Option<String>,
    pub q: Option<String>,
    pub page: Option<u32>,
    /// Uploader-Pfad für kind=user, z. B. `/model/name`
    pub path: Option<String>,
    /// Sortierung: new, views, top, long (leer = Pornhubs Standard)
    pub sort: Option<String>,
    /// Kategorien für kind=home, mit Komma getrennt, je `kurzname` oder `kurzname~id` (z. B. `amateur~3,milf~29`)
    pub cats: Option<String>,
    /// Quelle: ph (Standard), pb (PimpBunny) oder all/both
    pub src: Option<String>,
}

#[derive(Deserialize)]
pub struct ActBody {
    key: String,
    action: String,
    /// Kachel des Videos (nur PimpBunny, falls der Server sie nicht mehr im Zwischenspeicher hat)
    meta: Option<Value>,
}

#[derive(Deserialize)]
pub struct VideoQuery {
    key: Option<String>,
}

#[derive(Deserialize)]
pub struct MediaQuery {
    u: Option<String>,
    /// Gesetzt = ganzes Video laden (langer Timeout); nicht leer = als Datei mit diesem Namen herunterladen
    dl: Option<String>,
}

fn valid_key(s: &str) -> bool {
    if let Some(rest) = s.strip_prefix("pb:") {
        !rest.is_empty() && rest.len() <= 120 && rest.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    } else {
        !s.is_empty() && s.len() <= 40 && s.chars().all(|c| c.is_ascii_alphanumeric())
    }
}

/// Uploader-Pfad wie `/model/name`, `/pornstar/name`, `/channels/name` oder `/users/name`.
fn valid_path(p: &str) -> bool {
    let Some(rest) = ["/model/", "/pornstar/", "/channels/", "/users/"].iter().find_map(|pre| p.strip_prefix(pre)) else {
        return false;
    };
    !rest.is_empty() && rest.len() <= 80 && rest.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

fn browser_failed(e: browser::BrowserError) -> Response {
    match e {
        browser::BrowserError::NeedsLogin(m) | browser::BrowserError::Unavailable(m) | browser::BrowserError::Failed(m) => {
            err(StatusCode::BAD_GATEWAY, m)
        }
    }
}

/// Liest die Video-Kacheln der geöffneten Seite aus (läuft im Browser).
/// Kacheln aus einem Dokument lesen (`doc` = Seite selbst oder per DOMParser geladene Folgeseite).
const PARSE_JS: &str = r#"const parse=(doc)=>{const out=[],seen=new Set();
const items=doc.querySelectorAll('li.pcVideoListItem[data-video-vkey], li[data-video-vkey]');
for(const li of items){const k=li.getAttribute('data-video-vkey');if(!k||seen.has(k))continue;
if(li.closest('#hottestMenuSection, [id*="ecommend" i], [class*="ecommend" i], #relatedVideosCenter, .relatedVideos, [id*="elated" i]'))continue;
const a=li.querySelector('a[href*="viewkey="]');if(!a)continue;const img=li.querySelector('img');
const ta=li.querySelector('a.thumbnailTitle, .title a[href*="viewkey="]');
const title=((ta&&(ta.getAttribute('title')||ta.textContent))||a.getAttribute('data-title')||(img&&(img.getAttribute('alt')||img.getAttribute('data-title')))||a.getAttribute('title')||'').trim();
const watched=!!li.querySelector('[class*="watched" i]')||/\bwatched\b/i.test(a.textContent||'');
const thumb=img&&(img.getAttribute('data-mediumthumb')||img.getAttribute('data-thumb_url')||img.getAttribute('data-src')||img.getAttribute('src'))||'';
if(!thumb||thumb.startsWith('data:'))continue;
const dur=li.querySelector('.duration');const up=li.querySelector('.usernameWrap a, .usernameWrapper a, .videoUploaderBlock a');
let upath='';try{upath=up?new URL(up.getAttribute('href')||'',location.href).pathname.replace(/\/$/,''):''}catch{}
const views=li.querySelector('.views var');
seen.add(k);out.push({key:k,title,thumb,preview:img&&img.getAttribute('data-mediabook')||'',duration:dur?dur.textContent.trim():'',uploader:up?up.textContent.trim():'',upath,watched,views:views?views.textContent.trim():''});}
return out};"#;

/// Profilbild des Uploaders aus einem Dokument lesen (leer, wenn keines gefunden wird).
const AVA_JS: &str = r#"const avaOf=(doc)=>{const c=['img#getAvatar','.avatarWrapper img','.userAvatar img','.profileHeader img.avatar','.topProfileHeader img','#profileHeader img','img[class*="avatar" i]'];
for(const q of c){const i=doc.querySelector(q);const u=i&&(i.getAttribute('data-src')||i.getAttribute('src'));if(u&&/^(https?:)?\/\//.test(u)&&!u.startsWith('data:'))return u.startsWith('//')?'https:'+u:u}
const m=doc.querySelector('meta[property="og:image"]');const u=m&&m.getAttribute('content');return u&&/phncdn\.com/.test(u)?u:''};
const statsOf=(doc)=>{const o=[];const t=e=>e?e.textContent.replace(/\s+/g,' ').trim():'';
for(const b of doc.querySelectorAll('.infoBoxes .infoBox:not(.rankDetails)')){const v=t(b.querySelector('.big')),l=t(b.querySelector('.title'));if(v&&l)o.push({l,v,f:(b.getAttribute('data-title')||'').split(':').pop().trim()})}
const sc=t(doc.querySelector('.showingCounter'));const m=sc.match(/of\s+([\d,.]+)/i);if(m)o.push({l:'Videos',v:m[1],f:m[1]});return o};"#;

/// Liest die Video-Kacheln der geöffneten Seite aus (läuft im Browser, nach PARSE_JS).
const LIST_TAIL_JS: &str = r#"(()=>{const out=parse(document);
const user=document.querySelector('#headerUserInfo, .userAvatar, [data-login-user], .profileUserName');
const login=!!(document.querySelector('a[href*="/logout"]')||user||document.cookie.includes('il='));
const h1=document.querySelector('h1');
const nums=[...document.querySelectorAll('.pagination3 a, .pagination3 li, .page_number a')].map(e=>parseInt((e.textContent||'').trim(),10)).filter(n=>n>0);
const cur=parseInt(new URL(location.href).searchParams.get('page')||'1',10)||1;
const lastPg=nums.length?Math.max(...nums):0;
const notFound=/not found/i.test(document.title)||/page not found|^error/i.test(h1?h1.textContent.trim():'');
return {ok:true,items:notFound?[]:out,notFound,login,url:location.href,curPage:cur,lastPage:lastPg,name:h1?h1.textContent.trim().slice(0,80):'',avatar:avaOf(document),stats:statsOf(document)}})()"#;

/// Alle Seiten eines Uploaders einsammeln (für die Suche in seinen Videos): Folgeseiten parallel
/// im eingeloggten Tab nachladen, höchstens 30 Seiten bzw. ~28 s.
const CRAWL_TAIL_JS: &str = r#"(async()=>{const out=[],seen=new Set();
const add=a=>{let n=0;for(const it of a){if(!seen.has(it.key)){seen.add(it.key);out.push(it);n++}}return n};
add(parse(document));
const nums=[...document.querySelectorAll('.pagination3 a, .pagination3 li, .page_number a')].map(e=>parseInt((e.textContent||'').trim(),10)).filter(n=>n>0);
const last=Math.min(30,nums.length?Math.max(...nums):30);
const u=p=>{const x=new URL(location.href);x.searchParams.set('page',p);return x.toString()};
const t0=Date.now();let partial=false;
for(let p=2;p<=last;p+=6){if(Date.now()-t0>26000){partial=true;break}
const batch=[];for(let i=p;i<Math.min(p+6,last+1);i++)batch.push(fetch(u(i),{credentials:'include'}).then(r=>r.ok?r.text():'').then(h=>{if(!h)return[];const d=new DOMParser().parseFromString(h,'text/html');
if(/not found/i.test(d.title)||/error/i.test((d.querySelector('h1')||{}).textContent||''))return[];return parse(d)}).catch(()=>[]));
const res=await Promise.all(batch);let got=0;for(const r of res)got+=add(r);if(!got)break;}
const h1=document.querySelector('h1');
return {ok:true,items:out,partial,name:h1?h1.textContent.trim().slice(0,80):'',avatar:avaOf(document),stats:statsOf(document)}})()"#;

/// Personen-Suche (Pornstars und Models) auslesen.
const PEOPLE_JS: &str = r#"(()=>{const out=[],seen=new Set();
for(const li of document.querySelectorAll('#pornstarsSearchResult > li')){const a=li.querySelector('a[href^="/pornstar/"], a[href^="/model/"], a[href^="/channels/"], a[href^="/users/"]');if(!a)continue;
const path=(a.getAttribute('href')||'').replace(/\/$/,'');if(seen.has(path))continue;seen.add(path);
const img=li.querySelector('img');const t=li.querySelector('a.title');
out.push({path,name:((t&&t.textContent)||(img&&img.getAttribute('alt'))||'').trim(),img:img&&(img.getAttribute('data-thumb_url')||img.getAttribute('src'))||'',
videos:((li.querySelector('.videosNumber')||{}).textContent||'').trim(),views:((li.querySelector('.pstarViews')||{}).textContent||'').trim()});}
return {ok:true,items:out}})()"#;

/// Kategorienliste von /categories.
const CATS_JS: &str = r#"(()=>{const out=[],seen=new Set();
for(const li of document.querySelectorAll('ul.categoriesListSection > li')){const a=li.querySelector('a[href]');if(!a)continue;
const href=a.getAttribute('href')||'';const name=((li.querySelector('strong')||{}).textContent||a.getAttribute('alt')||'').trim();if(!name)continue;
let slug=(href.match(/^\/categories\/([\w-]+)/)||[])[1]||name.toLowerCase().replace(/\(.*?\)/g,'').replace(/&/g,' ').trim().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'');
if(!slug||seen.has(slug))continue;seen.add(slug);const img=li.querySelector('img');
out.push({slug,id:(href.match(/[?&]c=(\d+)/)||[])[1]||'',name,img:img&&(img.getAttribute('data-image')||img.getAttribute('src'))||'',count:((li.querySelector('var')||{}).textContent||'').trim()});}
return {ok:true,items:out}})()"#;

/// Eigener Favoriten-Link aus dem Benutzermenü (nur eingeloggt vorhanden).
const ME_JS: &str = r#"(()=>{const a=[...document.querySelectorAll('a[href*="/users/"]')].map(a=>a.getAttribute('href')||'')
.map(h=>{try{return new URL(h,location.href).pathname}catch{return ''}});
const fav=a.find(h=>/^\/users\/[^/]+\/videos\/favorites\/?$/.test(h));
const login=!document.querySelector('.js_videoSubscribeButton.loggedOut, .loggedOut .js_gtm-subscribeEvent')&&!!(document.querySelector('a[href*="/logout"]')||fav);
return {ok:true,fav:fav||'',login}})()"#;

/// Zustand (und optional eine Aktion) auf der Videoseite im eingeloggten Browser.
/// `__ACT__` wird durch "", "up", "down", "fav" oder "sub" ersetzt.
const STATE_JS: &str = r#"(async()=>{const act='__ACT__';const $=s=>document.querySelector(s);
const el={up:$('.js-voteUp'),down:$('.js-voteDown'),fav:$('.js-favoriteBtn'),sub:$('.video-detailed-info .js_videoSubscribeButton, .js_videoSubscribeButton')};
const on=e=>!!e&&/(^|\s)(active|selected|voted|favorited|isFavorite|subscribed)(\s|$)/i.test(e.className);
const subOn=e=>{const b=e&&e.querySelector('[data-subscribed]');return b?b.getAttribute('data-subscribed')==='1':on(e)};
const loggedOut=()=>!!($('.js_videoSubscribeButton.loggedOut')||/log ?in|sign up/i.test((el.fav&&el.fav.getAttribute('data-title'))||''));
const read=()=>({ok:true,login:!loggedOut(),up:on(el.up),down:on(el.down),fav:on(el.fav)||/remove|entfernen/i.test((el.fav&&el.fav.getAttribute('data-title'))||''),sub:subOn(el.sub),
likes:+(($('.votesUp')||{}).dataset?.rating||0),dislikes:+(($('.votesDown')||{}).dataset?.rating||0)});
if(!act)return read();
if(loggedOut())return{...read(),ok:false,error:'Der Server-Browser ist nicht bei Pornhub eingeloggt.'};
const t=el[act];if(!t)return{...read(),ok:false,error:'Knopf auf der Seite nicht gefunden'};
const before=JSON.stringify(read());(t.querySelector('button')||t).click();
for(let i=0;i<20;i++){await new Promise(r=>setTimeout(r,250));if(JSON.stringify(read())!==before)break;}
return read()})()"#;

/// Liest Titel, Uploader und die HLS-Varianten der Videoseite aus (läuft im Browser).
const VIDEO_JS: &str = r#"(()=>{const k=Object.keys(window).find(x=>/^flashvars_\d+$/.test(x));const f=k&&window[k];
if(!f)return{ok:false,error:'Video nicht gefunden oder gesperrt'};
const v=(f.mediaDefinitions||[]).filter(m=>m.format==='hls'&&m.videoUrl).map(m=>({q:parseInt(m.quality)||0,url:m.videoUrl})).sort((a,b)=>b.q-a.q);
const up=document.querySelector('.video-detailed-info .usernameBadgesWrapper a, .userInfo .usernameWrap a, .video-info-row .usernameWrap a');
return{ok:true,title:f.video_title||document.title,duration:f.video_duration||0,thumb:f.image_url||'',uploader:up?up.textContent.trim():'',sources:v}})()"#;

/// Alle Seiten einer Liste einsammeln, 30 Minuten gemerkt.
async fn crawl_cached(url: &str) -> Result<Value, Response> {
    use std::collections::HashMap;
    use std::time::Instant;
    static CACHE: std::sync::Mutex<Option<HashMap<String, (Instant, Value)>>> = std::sync::Mutex::new(None);
    // Ohne Seitenzahl als Schlüssel (gleiche Liste für alle Suchen auf dieser Sortierung)
    let base = url.replace("?page=1&", "?").replace("?page=1", "");
    let cached = CACHE
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .get(&base)
        .filter(|(t, _)| t.elapsed() < Duration::from_secs(1800))
        .map(|(_, v)| v.clone());
    if let Some(v) = cached {
        return Ok(v);
    }
    match browser::eval_in_page(url, &format!("{PARSE_JS}{AVA_JS}{CRAWL_TAIL_JS}")).await {
        Ok(v) if v["ok"].as_bool() == Some(true) => {
            let mut c = CACHE.lock().unwrap();
            let m = c.get_or_insert_with(HashMap::new);
            if m.len() > 80 {
                m.clear();
            }
            m.insert(base, (Instant::now(), v.clone()));
            Ok(v)
        }
        Ok(v) => Err(err(StatusCode::BAD_GATEWAY, v["error"].as_str().unwrap_or("Seite nicht lesbar").to_string())),
        Err(e) => Err(browser_failed(e)),
    }
}

/// Suche und/oder Kategorien innerhalb der Videos eines Uploaders. Pornhub hat dafür keinen Filter:
/// Titel werden lokal gefiltert, Kategorien über die Pornhub-Suche nach dem Namen mit Kategorie-Filter
/// (nur Treffer dieses Uploaders), und alles mit der vollständigen Uploader-Liste abgeglichen.
async fn search_user(url: &str, q: &str, page: u32, path: &str, cats: &[(&str, &str)]) -> Response {
    if q.len() > 100 {
        return err(StatusCode::BAD_REQUEST, "Suchbegriff zu lang");
    }
    if page > 1 {
        return Json(json!({ "items": [], "page": page })).into_response();
    }
    let v = match crawl_cached(url).await {
        Ok(v) => v,
        Err(r) => return r,
    };
    let mut partial = v["partial"].as_bool() == Some(true);
    let last = |x: &str| x.rsplit('/').next().unwrap_or("").to_lowercase();
    let who = last(path);
    let mut keep: Option<std::collections::HashSet<String>> = None;
    let name = v["name"].as_str().unwrap_or("").trim().to_string();
    for (_, id) in cats.iter().filter(|c| !c.1.is_empty()) {
        if name.is_empty() {
            break;
        }
        let surl = format!("{SITE}/video/search?search={}&filter_category={id}&page=1", urlencoding::encode(&name));
        let r = match crawl_cached(&surl).await {
            Ok(r) => r,
            Err(r) => return r,
        };
        partial |= r["partial"].as_bool() == Some(true);
        let set: std::collections::HashSet<String> = r["items"]
            .as_array()
            .map(|a| a.iter().filter(|it| it["upath"].as_str().is_some_and(|u| last(u) == who)).filter_map(|it| it["key"].as_str().map(str::to_string)).collect())
            .unwrap_or_default();
        keep = Some(match keep {
            Some(k) => k.intersection(&set).cloned().collect(),
            None => set,
        });
    }
    let words: Vec<String> = q.to_lowercase().split_whitespace().map(str::to_string).collect();
    let all = v["items"].as_array().cloned().unwrap_or_default();
    let total = all.len();
    let items: Vec<Value> = all
        .into_iter()
        .filter(|it| {
            let t = it["title"].as_str().unwrap_or("").to_lowercase();
            words.iter().all(|w| t.contains(w.as_str())) && keep.as_ref().is_none_or(|k| it["key"].as_str().is_some_and(|x| k.contains(x)))
        })
        .collect();
    Json(json!({ "items": items, "page": page, "name": v["name"].clone(), "avatar": v["avatar"].clone(), "stats": v["stats"].clone(), "searched": total, "partial": partial })).into_response()
}

/// Personen suchen (Pornstars und Models).
pub async fn people(Query(p): Query<ListQuery>) -> Response {
    if !browser::configured() {
        return err(StatusCode::SERVICE_UNAVAILABLE, "Der Browser auf dem Server ist nicht eingerichtet.");
    }
    let page = p.page.unwrap_or(1).clamp(1, 100);
    let q = p.q.unwrap_or_default();
    let q = q.trim();
    if q.is_empty() || q.len() > 100 {
        return err(StatusCode::BAD_REQUEST, "Suchbegriff fehlt");
    }
    let url = format!("{SITE}/pornstars/search?search={}&page={page}", urlencoding::encode(q));
    match browser::eval_in_page(&url, PEOPLE_JS).await {
        Ok(v) if v["ok"].as_bool() == Some(true) => Json(json!({ "items": v["items"].clone(), "page": page })).into_response(),
        Ok(v) => err(StatusCode::BAD_GATEWAY, v["error"].as_str().unwrap_or("Seite nicht lesbar").to_string()),
        Err(e) => browser_failed(e),
    }
}

/// Alle Kategorien (einmal geladen, dann gemerkt).
pub async fn cats() -> Response {
    static CACHE: std::sync::Mutex<Option<Value>> = std::sync::Mutex::new(None);
    if let Some(v) = CACHE.lock().unwrap().clone() {
        return Json(v).into_response();
    }
    if !browser::configured() {
        return err(StatusCode::SERVICE_UNAVAILABLE, "Der Browser auf dem Server ist nicht eingerichtet.");
    }
    match browser::eval_in_page(&format!("{SITE}/categories"), CATS_JS).await {
        Ok(v) if v["items"].as_array().is_some_and(|a| a.len() > 10) => {
            let out = json!({ "items": v["items"].clone() });
            *CACHE.lock().unwrap() = Some(out.clone());
            Json(out).into_response()
        }
        Ok(_) => err(StatusCode::BAD_GATEWAY, "Kategorien nicht lesbar"),
        Err(e) => browser_failed(e),
    }
}

pub async fn list(Query(p): Query<ListQuery>) -> Response {
    if !browser::configured() {
        return err(StatusCode::SERVICE_UNAVAILABLE, "Der Browser auf dem Server ist nicht eingerichtet.");
    }
    let page = p.page.unwrap_or(1).clamp(1, 200);
    let src = p.src.as_deref().unwrap_or("ph");
    // Creator-Seite bei PimpBunny
    if p.kind.as_deref() == Some("user") && p.path.as_deref().is_some_and(|x| x.starts_with("/pb/")) {
        let path = p.path.as_deref().unwrap_or("");
        if !crate::pb::valid_user_path(path) {
            return err(StatusCode::BAD_REQUEST, "Ungültiger Creator");
        }
        return match crate::pb::user(path, page, p.sort.as_deref()).await {
            Ok((items, info)) => {
                let mut out = json!({ "items": &items, "page": page, "login": true });
                // Hinter der letzten Seite liefert PimpBunny „Page not found“ als Überschrift
                for k in ["name", "avatar", "stats"] {
                    if !items.is_empty() && !info[k].is_null() && info[k] != "" {
                        out[k] = info[k].clone();
                    }
                }
                Json(out).into_response()
            }
            Err(e) => browser_failed(e),
        };
    }
    if src == "pb" && matches!(p.kind.as_deref(), Some("favs" | "subs")) {
        let r = if p.kind.as_deref() == Some("favs") {
            crate::pb::favs(page).map_err(browser::BrowserError::Failed)
        } else {
            crate::pb::subs_feed(page).await
        };
        return match r {
            Ok(items) => Json(json!({ "items": items, "page": page, "login": true })).into_response(),
            Err(e) => browser_failed(e),
        };
    }
    if src == "pb" {
        return match crate::pb::list(page, p.q.as_deref()).await {
            Ok(items) => Json(json!({ "items": items, "page": page, "login": true })).into_response(),
            Err(e) => browser_failed(e),
        };
    }
    let is_both = src == "all" || src == "both";
    let subs = p.kind.as_deref() == Some("subs");
    let channel = p.path.as_deref().is_some_and(|x| x.starts_with("/channels/"));
    // Pornhubs `o=`-Werte je Liste (Kanäle nutzen eigene Kürzel und kennen „am längsten“ nicht)
    let o = match (p.kind.as_deref().unwrap_or("home"), p.sort.as_deref().unwrap_or("")) {
        ("home", "new") => Some("cm"),
        ("search", "new") => Some("mr"),
        ("user", "views") if channel => Some("vi"),
        ("user", "top") if channel => Some("ra"),
        ("user", "long") if channel => None,
        ("home" | "search" | "user", "views") => Some("mv"),
        ("home" | "search" | "user", "top") => Some("tr"),
        ("home" | "search" | "user", "long") => Some("lg"),
        _ => None,
    };
    let os = o.map(|o| format!("&o={o}")).unwrap_or_default();
    let cats: Vec<(&str, &str)> = p
        .cats
        .as_deref()
        .unwrap_or("")
        .split(',')
        .map(|c| c.trim().split_once('~').unwrap_or((c.trim(), "")))
        .filter(|(c, id)| {
            !c.is_empty() && c.len() <= 40 && c.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '-') && id.len() <= 6 && id.chars().all(|ch| ch.is_ascii_digit())
        })
        .take(6)
        .collect();
    let url = match p.kind.as_deref().unwrap_or("home") {
        // Eine Kategorie: über ihre Nummer (oder ihre eigene Seite), mehrere: kombiniert über die Kurznamen
        "home" if cats.len() == 1 && !cats[0].1.is_empty() => format!("{SITE}/video?c={}&page={page}{os}", cats[0].1),
        "home" if cats.len() == 1 => format!("{SITE}/categories/{}?page={page}{os}", cats[0].0),
        "home" if !cats.is_empty() => {
            format!("{SITE}/video/incategories/{}?page={page}{os}", cats.iter().map(|c| c.0).collect::<Vec<_>>().join("/"))
        }
        "home" if page == 1 && o.is_none() => format!("{SITE}/"),
        "home" => format!("{SITE}/video?page={page}{os}"),
        "subs" => format!("{SITE}/subscriptions?page={page}"),
        "search" => {
            let q = p.q.clone().unwrap_or_default();
            let q = q.trim();
            if q.is_empty() || q.len() > 100 {
                return err(StatusCode::BAD_REQUEST, "Suchbegriff fehlt");
            }
            format!("{SITE}/video/search?search={}&page={page}{os}", urlencoding::encode(q))
        }
        "favs" => match my_favs().await {
            Ok(Some(path)) => format!("{SITE}{path}?page={page}"),
            Ok(None) if is_both => {
                let items = crate::pb::favs(page).unwrap_or_default();
                return Json(json!({ "items": items, "login": !items.is_empty(), "page": page })).into_response();
            }
            Ok(None) => return Json(json!({ "items": [], "login": false, "page": page })).into_response(),
            Err(e) => return browser_failed(e),
        },
        "user" => {
            let path = p.path.clone().unwrap_or_default();
            if !valid_path(&path) {
                return err(StatusCode::BAD_REQUEST, "Ungültiger Uploader");
            }
            format!("{SITE}{path}/videos?page={page}{os}")
        }
        _ => return err(StatusCode::BAD_REQUEST, "Unbekannte Liste"),
    };
    if p.kind.as_deref() == Some("user") {
        let q = p.q.as_deref().map(str::trim).unwrap_or("");
        if !q.is_empty() || !cats.is_empty() {
            return search_user(&url, q, page, p.path.as_deref().unwrap_or(""), &cats).await;
        }
    }
    match browser::eval_in_page(&url, &format!("{PARSE_JS}{AVA_JS}{LIST_TAIL_JS}")).await {
        Ok(v) if v["ok"].as_bool() == Some(true) => {
            // Ohne Login leitet Pornhub die Abo-Seite auf eine andere Liste um: dann nichts zeigen
            let landed = v["url"].as_str().unwrap_or("");
            let lost = subs && !landed.contains("subscriptions");
            // Hinter der letzten Favoriten-Seite nichts nachladen (Pornhub leitet um oder zeigt Empfehlungen)
            let cur = v["curPage"].as_u64().unwrap_or(1);
            let last = v["lastPage"].as_u64().unwrap_or(0);
            let past = p.kind.as_deref() == Some("favs") && page > 1 && (cur < page as u64 || (last > 0 && page as u64 > last));
            let lost = lost || past;
            let mut ph_items = if lost { vec![] } else { v["items"].as_array().cloned().unwrap_or_default() };
            for it in &mut ph_items {
                if let Some(obj) = it.as_object_mut() {
                    obj.insert("src".to_string(), json!("ph"));
                }
            }
            let kind_s = p.kind.as_deref().unwrap_or("home");
            let mut pb_found = false;
            if is_both && matches!(kind_s, "home" | "search" | "favs" | "subs") {
                let pb_res = match kind_s {
                    "favs" => crate::pb::favs(page).map_err(browser::BrowserError::Failed),
                    "subs" => crate::pb::subs_feed(page).await,
                    _ => crate::pb::list(page, p.q.as_deref()).await,
                };
                if let Ok(pb_items) = pb_res {
                    pb_found = !pb_items.is_empty();
                    let mut combined = Vec::with_capacity(ph_items.len() + pb_items.len());
                    let mut ph_iter = ph_items.into_iter();
                    let mut pb_iter = pb_items.into_iter();
                    loop {
                        match (ph_iter.next(), pb_iter.next()) {
                            (Some(a), Some(b)) => { combined.push(a); combined.push(b); }
                            (Some(a), None) => { combined.push(a); combined.extend(ph_iter); break; }
                            (None, Some(b)) => { combined.push(b); combined.extend(pb_iter); break; }
                            (None, None) => break,
                        }
                    }
                    ph_items = combined;
                }
            }
            Json(json!({
            "items": ph_items,
            "login": (v["login"].as_bool() == Some(true) && !lost) || pb_found,
            "page": page,
            "name": v["name"].clone(),
            "avatar": v["avatar"].clone(),
            "stats": v["stats"].clone(),
        }))
        .into_response()
        }
        Ok(v) => err(StatusCode::BAD_GATEWAY, v["error"].as_str().unwrap_or("Seite nicht lesbar").to_string()),
        Err(e) => browser_failed(e),
    }
}

/// Video-Infos ohne Browser, direkt vom Server (ohne VPN): Die MP4-Adressen von get_media sind an die
/// IP gebunden, die sie abholt. Seite und Stream laufen deshalb beide über dieselbe direkte Verbindung.
pub async fn video(Query(p): Query<VideoQuery>) -> Response {
    let Some(key) = p.key.filter(|k| valid_key(k)) else {
        return err(StatusCode::BAD_REQUEST, "Ungültiger Video-Schlüssel");
    };
    if let Some(pb_key) = key.strip_prefix("pb:") {
        return match crate::pb::video(pb_key).await {
            Ok(v) => Json(v).into_response(),
            Err(e) => browser_failed(e),
        };
    }
    let page_url = format!("{SITE}/view_video.php?viewkey={key}");
    let resp = match direct().get(&page_url).send().await {
        Ok(r) => r,
        Err(e) => return err(StatusCode::BAD_GATEWAY, format!("Pornhub nicht erreichbar: {e}")),
    };
    if !resp.status().is_success() {
        return err(StatusCode::BAD_GATEWAY, format!("Pornhub antwortet mit HTTP {}", resp.status().as_u16()));
    }
    let cookies = resp
        .headers()
        .get_all(header::SET_COOKIE)
        .iter()
        .filter_map(|v| v.to_str().ok()?.split(';').next().map(str::to_string))
        .collect::<Vec<_>>()
        .join("; ");
    let Ok(html) = resp.text().await else {
        return err(StatusCode::BAD_GATEWAY, "Videoseite nicht lesbar");
    };
    let Some(f) = flashvars(&html) else {
        return err(StatusCode::BAD_GATEWAY, "Video nicht gefunden, gesperrt oder nur mit Login");
    };
    let Some(gm) = f["mediaDefinitions"]
        .as_array()
        .and_then(|a| a.iter().find(|m| m["format"] == "mp4"))
        .and_then(|m| m["videoUrl"].as_str())
        .map(str::to_string)
    else {
        return err(StatusCode::BAD_GATEWAY, "Kein Stream gefunden");
    };
    let mut req = direct().get(&gm).header(header::REFERER, &page_url);
    if !cookies.is_empty() {
        req = req.header(header::COOKIE, cookies);
    }
    let list: Value = match req.send().await {
        Ok(r) if r.status().is_success() => r.json().await.unwrap_or(Value::Null),
        Ok(r) => return err(StatusCode::BAD_GATEWAY, format!("Stream-Liste liefert HTTP {}", r.status().as_u16())),
        Err(e) => return err(StatusCode::BAD_GATEWAY, format!("Stream-Liste nicht ladbar: {e}")),
    };
    let mut sources: Vec<Value> = list
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|m| {
                    let url = m["videoUrl"].as_str().filter(|u| !u.is_empty())?;
                    let q = m["quality"].as_str().and_then(|q| q.parse::<u64>().ok()).or_else(|| m["quality"].as_u64()).unwrap_or(0);
                    Some(json!({ "q": q, "url": url }))
                })
                .collect()
        })
        .unwrap_or_default();
    sources.sort_by_key(|s| std::cmp::Reverse(s["q"].as_u64().unwrap_or(0)));
    if sources.is_empty() {
        return err(StatusCode::BAD_GATEWAY, "Kein abspielbarer Stream gefunden");
    }
    let uploader = uploader(&html);
    let duration = f["video_duration"].as_u64().or_else(|| f["video_duration"].as_str().and_then(|d| d.parse().ok())).unwrap_or(0);
    Json(json!({
        "title": f["video_title"].as_str().unwrap_or(""),
        "duration": duration,
        "thumb": f["image_url"].as_str().unwrap_or(""),
        "uploader": uploader.map(|(path, name)| json!({ "path": path, "name": name })),
        "likes": rating(&html, "votesUp"),
        "dislikes": rating(&html, "votesDown"),
        "sources": sources,
    }))
    .into_response()
}

/// Uploader unter dem Video: erster Profil-Link nach `video-detailed-info`.
fn uploader(html: &str) -> Option<(String, String)> {
    let start = html.find("video-detailed-info")?;
    let area = &html[start..html.len().min(start + 20_000)];
    let mut rest = area;
    let mut found: Option<String> = None;
    while let Some(i) = rest.find("href=\"/") {
        let tail = &rest[i + 6..];
        let end = tail.find('"')?;
        let path = &tail[..end];
        // Erster Profil-Link ist oft nur das Avatarbild: weitersuchen, bis derselbe Link mit Text kommt
        if valid_path(path) && found.as_deref().is_none_or(|f| f == path) {
            found = Some(path.to_string());
            let after = &tail[end..];
            let name = after.find('>').and_then(|g| {
                let t = &after[g + 1..];
                t.find('<').map(|l| t[..l].trim().to_string())
            });
            if let Some(n) = name.filter(|n| !n.is_empty()) {
                return Some((path.to_string(), html_unescape(&n)));
            }
        }
        rest = &tail[end..];
    }
    let path = found?;
    let name = path.rsplit('/').next().unwrap_or("").to_string();
    Some((path, name))
}

fn html_unescape(s: &str) -> String {
    s.replace("&amp;", "&").replace("&#039;", "'").replace("&quot;", "\"").replace("&lt;", "<").replace("&gt;", ">")
}

/// `class="votesUp" data-rating="1272"` → 1272
fn rating(html: &str, class: &str) -> u64 {
    let Some(i) = html.find(&format!("class=\"{class}\" data-rating=\"")) else { return 0 };
    let t = &html[i + class.len() + 22..];
    t.split('"').next().and_then(|n| n.parse().ok()).unwrap_or(0)
}

/// Pfad der eigenen Favoriten (`/users/<ich>/videos/favorites`), einmal gefunden wird er gemerkt.
async fn my_favs() -> Result<Option<String>, browser::BrowserError> {
    static FAV: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
    if let Some(p) = FAV.lock().unwrap().clone() {
        return Ok(Some(p));
    }
    let v = browser::eval_in_page(&format!("{SITE}/"), ME_JS).await?;
    let fav = v["fav"].as_str().filter(|f| !f.is_empty()).map(str::to_string);
    if let Some(f) = &fav {
        *FAV.lock().unwrap() = Some(f.clone());
    }
    Ok(fav)
}

/// Like-/Favoriten-/Abo-Zustand des eingeloggten Kontos für ein Video.
pub async fn state(Query(p): Query<VideoQuery>) -> Response {
    let Some(key) = p.key.filter(|k| valid_key(k)) else {
        return err(StatusCode::BAD_REQUEST, "Ungültiger Video-Schlüssel");
    };
    if key.starts_with("pb:") {
        return match crate::pb::state(&key, None) {
            Ok(v) => Json(v).into_response(),
            Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
        };
    }
    run_state(&key, "").await
}

/// Liken, Disliken, Favorit oder Abo umschalten: klickt den echten Knopf im eingeloggten Browser.
pub async fn act(Json(b): Json<ActBody>) -> Response {
    if !valid_key(&b.key) || !matches!(b.action.as_str(), "up" | "down" | "fav" | "sub") {
        return err(StatusCode::BAD_REQUEST, "Ungültige Aktion");
    }
    if b.key.starts_with("pb:") {
        if b.action == "down" {
            return err(StatusCode::BAD_REQUEST, "Dislike gibt es bei PimpBunny nicht.");
        }
        return match crate::pb::toggle(&b.key, &b.action, b.meta.as_ref()) {
            Ok(v) => Json(v).into_response(),
            Err(e) => err(StatusCode::BAD_REQUEST, e),
        };
    }
    run_state(&b.key, &b.action).await
}

async fn run_state(key: &str, action: &str) -> Response {
    if !browser::configured() {
        return err(StatusCode::SERVICE_UNAVAILABLE, "Der Browser auf dem Server ist nicht eingerichtet.");
    }
    let url = format!("{SITE}/view_video.php?viewkey={key}");
    match browser::eval_in_page(&url, &STATE_JS.replace("__ACT__", action)).await {
        Ok(v) if v["ok"].as_bool() == Some(true) => Json(v).into_response(),
        Ok(v) => err(
            if v["login"].as_bool() == Some(false) { StatusCode::UNAUTHORIZED } else { StatusCode::BAD_GATEWAY },
            v["error"].as_str().unwrap_or("Seite nicht lesbar").to_string(),
        ),
        Err(e) => browser_failed(e),
    }
}

/// Das Objekt hinter `var flashvars_<id> = {…};` aus dem HTML lesen.
fn flashvars(html: &str) -> Option<Value> {
    let i = html.find("var flashvars_")?;
    let rest = &html[i..];
    let j = rest.find('{')?;
    serde_json::Deserializer::from_str(&rest[j..]).into_iter::<Value>().next()?.ok()
}

const UA: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36";

/// Direkte Verbindung ohne VPN-Proxy (feste Server-IP, passend zu den IP-gebundenen Stream-Adressen).
fn direct() -> &'static reqwest::Client {
    static C: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    C.get_or_init(|| {
        reqwest::Client::builder()
            .no_proxy()
            .user_agent(UA)
            .connect_timeout(Duration::from_secs(15))
            .build()
            .expect("HTTP-Client")
    })
}

fn allowed(u: &Url) -> bool {
    u.scheme() == "https"
        && u.host_str().is_some_and(|h| {
            h == "phncdn.com"
                || h.ends_with(".phncdn.com")
                || h == "pimpbunny.com"
                || h.ends_with(".pimpbunny.com")
        })
}

fn proxied(u: &Url) -> String {
    format!("/api/ph/media?u={}", urlencoding::encode(u.as_str()))
}

/// Playlist-Zeilen auf den Server umschreiben (Segmente, Unter-Playlists und URI="…"-Attribute).
fn rewrite_m3u8(text: &str, base: &Url) -> String {
    let mut out = String::with_capacity(text.len() + 256);
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() {
            out.push('\n');
            continue;
        }
        if t.starts_with('#') {
            if let Some(i) = t.find("URI=\"") {
                let rest = &t[i + 5..];
                if let Some(j) = rest.find('"') {
                    if let Ok(abs) = base.join(&rest[..j]) {
                        out.push_str(&t[..i + 5]);
                        out.push_str(&proxied(&abs));
                        out.push_str(&rest[j..]);
                        out.push('\n');
                        continue;
                    }
                }
            }
            out.push_str(t);
        } else if let Ok(abs) = base.join(t) {
            out.push_str(&proxied(&abs));
        } else {
            out.push_str(t);
        }
        out.push('\n');
    }
    out
}

/// Vorschaubilder, HLS-Teile und MP4-Dateien durchreichen.
pub async fn media(headers: HeaderMap, Query(q): Query<MediaQuery>) -> Response {
    let Some(url) = q.u.as_deref().and_then(|u| Url::parse(u).ok()) else {
        return err(StatusCode::BAD_REQUEST, "Ungültige URL");
    };
    if !allowed(&url) {
        return err(StatusCode::FORBIDDEN, "Host nicht erlaubt");
    }
    // PimpBunny: `get_file`-Links (Vorschau-Clips) sind für den Server gesperrt, der Browser löst sie auf
    let url = if url.host_str() == Some("pimpbunny.com") && url.path().starts_with("/get_file/") {
        match crate::pb::resolve_file(url.as_str()).await.ok().and_then(|u| Url::parse(&u).ok()).filter(allowed) {
            Some(u) => u,
            None => return err(StatusCode::BAD_GATEWAY, "Vorschau nicht ladbar"),
        }
    } else {
        url
    };
    let host = url.host_str().unwrap_or_default();
    let is_pb = host.ends_with("pimpbunny.com");
    let ref_url = if is_pb {
        "https://pimpbunny.com/"
    } else {
        "https://www.pornhub.com/"
    };
    let origin = if is_pb {
        "https://pimpbunny.com"
    } else {
        SITE
    };
    let mut req = direct()
        .get(url.clone())
        .header(header::REFERER, ref_url)
        .header(header::ORIGIN, origin)
        .timeout(Duration::from_secs(if q.dl.is_some() { 3 * 3600 } else { 60 }));
    if let Some(range) = headers.get(header::RANGE) {
        req = req.header(header::RANGE, range.as_bytes());
    }
    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => return err(StatusCode::BAD_GATEWAY, format!("Medium nicht ladbar: {e}")),
    };
    let status = resp.status();
    if status != reqwest::StatusCode::OK && status != reqwest::StatusCode::PARTIAL_CONTENT {
        return err(StatusCode::BAD_GATEWAY, format!("Medium liefert HTTP {}", status.as_u16()));
    }
    let ctype = resp.headers().get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("").to_ascii_lowercase();
    let is_list = url.path().ends_with(".m3u8") || ctype.contains("mpegurl");
    if is_list {
        let base = resp.url().clone();
        let text = match resp.text().await {
            Ok(t) => t,
            Err(e) => return err(StatusCode::BAD_GATEWAY, format!("Playlist nicht lesbar: {e}")),
        };
        return Response::builder()
            .header(header::CONTENT_TYPE, "application/vnd.apple.mpegurl")
            .header(header::CACHE_CONTROL, "no-store")
            .body(Body::from(rewrite_m3u8(&text, &base)))
            .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response());
    }
    let mut b = Response::builder()
        .status(StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::OK))
        .header(header::CACHE_CONTROL, "private, max-age=3600");
    if let Some(name) = q.dl.as_deref().map(str::trim).filter(|n| !n.is_empty()) {
        let name: String = name.chars().filter(|c| !c.is_control() && !"\\/:*?\"<>|".contains(*c)).take(150).collect();
        let ascii: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || " .-_()".contains(c) { c } else { '_' }).collect();
        let enc: String = urlencoding::encode(&name).into_owned();
        b = b.header(header::CONTENT_DISPOSITION, format!("attachment; filename=\"{ascii}.mp4\"; filename*=UTF-8''{enc}.mp4"));
    }
    for name in [header::CONTENT_TYPE, header::CONTENT_LENGTH, header::CONTENT_RANGE, header::ACCEPT_RANGES] {
        if let Some(v) = resp.headers().get(&name) {
            b = b.header(name, v.as_bytes());
        }
    }
    b.body(Body::from_stream(resp.bytes_stream()))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}
