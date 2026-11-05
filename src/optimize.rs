//! Medien für die Anzeige verkleinern: Bilder → WebP, Videos → H.264/AAC (Ton eingebaut).
//! Das Ergebnis liegt im Cache-Ordner; schlägt etwas fehl, gibt es das Original.

use std::{
    collections::HashMap,
    env,
    hash::{DefaultHasher, Hash, Hasher},
    path::PathBuf,
    sync::{Arc, OnceLock},
    time::{Duration, SystemTime},
};

use axum::{
    body::Body,
    extract::{Query, Request, State},
    http::{StatusCode, header},
    response::{IntoResponse, Redirect, Response},
};
use futures_util::StreamExt;
use reqwest::Url;
use serde::Deserialize;
use tokio::{
    io::AsyncWriteExt,
    process::Command,
    sync::{Mutex, Semaphore},
    time::timeout,
};
use tower::ServiceExt;
use tower_http::services::ServeFile;

use crate::{AppState, err, media_allowed};

const MAX_DOWNLOAD: u64 = 200 * 1024 * 1024;
const MAX_CACHE_BYTES: u64 = 3 * 1024 * 1024 * 1024;
const MAX_AGE: Duration = Duration::from_secs(3 * 24 * 3600);

#[derive(Deserialize)]
pub struct OptQuery {
    k: Option<String>,
    url: Option<String>,
    /// Kandidaten für die Tonspur, mit "|" getrennt.
    a: Option<String>,
}

pub fn enabled() -> bool {
    env::var("OPTIMIZE").map(|v| v != "0").unwrap_or(true)
}

/// Adresse, über die das Frontend das optimierte Medium holt.
pub fn opt_url(kind: &str, raw: &str, audio: &[String]) -> String {
    let mut s = format!("/api/opt?k={kind}&url={}", urlencoding::encode(raw));
    if !audio.is_empty() {
        let joined = audio.iter().map(|a| a.replace("&amp;", "&")).collect::<Vec<_>>().join("|");
        s.push_str(&format!("&a={}", urlencoding::encode(&joined)));
    }
    s
}

pub(crate) fn cache_dir() -> PathBuf {
    PathBuf::from(env::var("CACHE_DIR").unwrap_or_else(|_| "/tmp/redswipe-cache".into()))
}

fn permits() -> &'static Semaphore {
    static S: OnceLock<Semaphore> = OnceLock::new();
    S.get_or_init(|| {
        let n = env::var("OPTIMIZE_JOBS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
        Semaphore::new(n)
    })
}

fn locks() -> &'static Mutex<HashMap<String, Arc<Mutex<()>>>> {
    static L: OnceLock<Mutex<HashMap<String, Arc<Mutex<()>>>>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(HashMap::new()))
}

fn key(kind: &str, url: &str, audio: &str) -> String {
    let mut a = DefaultHasher::new();
    (kind, url, audio).hash(&mut a);
    let mut b = DefaultHasher::new();
    (audio, url, kind, 7u8).hash(&mut b);
    format!("{:016x}{:016x}", a.finish(), b.finish())
}

fn original(url: &str) -> Response {
    Redirect::temporary(&format!("/api/media?url={}", urlencoding::encode(url))).into_response()
}

pub fn is_cached(kind: &str, raw: &str, audio: &[String]) -> bool {
    let joined = audio.iter().map(|a| a.replace("&amp;", "&")).collect::<Vec<_>>().join("|");
    let k = key(kind, raw, &joined);
    let ext = if kind == "image" { "webp" } else { "mp4" };
    cache_dir().join(format!("{k}.{ext}")).exists()
}

pub fn prewarm(st: &AppState, kind: &str, raw: &str, audio: &[String]) {
    if !enabled() {
        return;
    }
    let Ok(url) = Url::parse(raw) else { return };
    if !media_allowed(&url) { return };
    let joined = audio.iter().map(|a| a.replace("&amp;", "&")).collect::<Vec<_>>().join("|");
    let k = key(kind, raw, &joined);
    let dir = cache_dir();
    let ext = if kind == "image" { "webp" } else { "mp4" };
    let out = dir.join(format!("{k}.{ext}"));
    let skip = dir.join(format!("{k}.skip"));
    if out.exists() || skip.exists() {
        return;
    }
    let audio_urls: Vec<Url> = audio
        .iter()
        .filter_map(|s| Url::parse(s).ok())
        .filter(media_allowed)
        .take(5)
        .collect();
    let st = st.clone();
    let kind_owned = kind.to_string();
    tokio::spawn(async move {
        let lock = {
            let mut m = locks().lock().await;
            m.entry(k.clone()).or_default().clone()
        };
        let _g = lock.lock().await;
        if !out.exists() && !skip.exists() {
            let res = timeout(Duration::from_secs(300), run_job(&st, &kind_owned, &url, &audio_urls, &dir, &k, &out)).await;
            match res {
                Ok(Ok(())) => {},
                Ok(Err(e)) => {
                    eprintln!("Prewarm fehlgeschlagen ({kind_owned} {url}): {e}");
                    let _ = tokio::fs::write(&skip, b"").await;
                }
                Err(_) => {
                    eprintln!("Prewarm Timeout ({kind_owned} {url})");
                    let _ = tokio::fs::write(&skip, b"").await;
                }
            }
        }
        locks().lock().await.remove(&k);
    });
}

pub async fn handler(State(st): State<AppState>, Query(q): Query<OptQuery>, req: Request<Body>) -> Response {
    let (Some(kind), Some(raw)) = (q.k.as_deref(), q.url.as_deref()) else {
        return err(StatusCode::BAD_REQUEST, "Parameter fehlen");
    };
    if kind != "image" && kind != "video" {
        return err(StatusCode::BAD_REQUEST, "Unbekannte Art");
    }
    let Ok(url) = Url::parse(raw) else {
        return err(StatusCode::BAD_REQUEST, "Ungültige URL");
    };
    if !media_allowed(&url) {
        return err(StatusCode::FORBIDDEN, "Host nicht erlaubt");
    }
    let audio: Vec<Url> = q
        .a
        .as_deref()
        .unwrap_or("")
        .split('|')
        .filter(|s| !s.is_empty())
        .filter_map(|s| Url::parse(s).ok())
        .filter(media_allowed)
        .take(5)
        .collect();

    // GIFs bleiben unverändert (sonst würde die Animation verloren gehen).
    let is_gif = url.path().to_lowercase().ends_with(".gif");
    if !enabled() || (kind == "image" && is_gif) {
        return original(raw);
    }

    let k = key(kind, raw, q.a.as_deref().unwrap_or(""));
    let dir = cache_dir();
    let ext = if kind == "image" { "webp" } else { "mp4" };
    let out = dir.join(format!("{k}.{ext}"));
    let skip = dir.join(format!("{k}.skip"));

    // Wenn bereits optimiert auf Platte liegt: sofort ausliefern
    if out.exists() {
        return match ServeFile::new(&out).oneshot(req).await {
            Ok(mut r) => {
                r.headers_mut().insert(header::CACHE_CONTROL, "public, max-age=86400".parse().unwrap());
                r.map(Body::new)
            }
            Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        };
    }

    // Videos: Nie den Client sekundenlang blockieren! Sofort Original streamen und im Hintergrund vorbereiten.
    if kind == "video" {
        let audio_strings: Vec<String> = audio.iter().map(|u| u.as_str().to_string()).collect();
        prewarm(&st, kind, raw, &audio_strings);
        return original(raw);
    }

    if !skip.exists() {
        let lock = {
            let mut m = locks().lock().await;
            m.entry(k.clone()).or_default().clone()
        };
        let _g = lock.lock().await;
        if !out.exists() && !skip.exists() {
            let res = match timeout(Duration::from_secs(10), run_job(&st, kind, &url, &audio, &dir, &k, &out)).await {
                Ok(r) => r,
                Err(_) => Err("Zeitüberschreitung".to_string()),
            };
            if let Err(e) = res {
                eprintln!("Optimieren fehlgeschlagen ({kind} {raw}): {e}");
                let _ = tokio::fs::write(&skip, b"").await;
            }
        }
        locks().lock().await.remove(&k);
    }

    if out.exists() {
        return match ServeFile::new(&out).oneshot(req).await {
            Ok(mut r) => {
                r.headers_mut().insert(header::CACHE_CONTROL, "public, max-age=86400".parse().unwrap());
                r.map(Body::new)
            }
            Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        };
    }
    original(raw)
}

pub(crate) async fn download(st: &AppState, url: &Url, to: &PathBuf) -> Result<(), String> {
    let resp = st
        .client
        .get(url.clone())
        .timeout(Duration::from_secs(120))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status().as_u16()));
    }
    let mut file = tokio::fs::File::create(to).await.map_err(|e| e.to_string())?;
    let mut size = 0u64;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| e.to_string())?;
        size += chunk.len() as u64;
        if size > MAX_DOWNLOAD {
            return Err("Datei zu groß".into());
        }
        file.write_all(&chunk).await.map_err(|e| e.to_string())?;
    }
    file.flush().await.map_err(|e| e.to_string())
}

async fn run_job(
    st: &AppState,
    kind: &str,
    url: &Url,
    audio: &[Url],
    dir: &PathBuf,
    k: &str,
    out: &PathBuf,
) -> Result<(), String> {
    tokio::fs::create_dir_all(dir).await.map_err(|e| e.to_string())?;
    let _permit = permits().acquire().await.map_err(|e| e.to_string())?;
    let src = dir.join(format!("{k}.src"));
    let asrc = dir.join(format!("{k}.asrc"));
    let part = dir.join(format!("{k}.part.{}", if kind == "image" { "webp" } else { "mp4" }));
    let result = async {
        download(st, url, &src).await?;
        let mut have_audio = false;
        if kind == "video" {
            for a in audio {
                if download(st, a, &asrc).await.is_ok() {
                    have_audio = true;
                    break;
                }
            }
        }
        let mut cmd = Command::new("ffmpeg");
        cmd.kill_on_drop(true).args(["-y", "-v", "error", "-nostdin", "-i"]).arg(&src);
        if kind == "image" {
            cmd.args(["-vf", "scale='min(1440,iw)':-2", "-c:v", "libwebp", "-quality", "80", "-frames:v", "1"]);
        } else {
            if have_audio {
                cmd.arg("-i").arg(&asrc).args(["-map", "0:v:0", "-map", "1:a:0", "-shortest"]);
            } else {
                cmd.args(["-map", "0:v:0", "-map", "0:a:0?"]);
            }
            cmd.args([
                "-vf",
                "scale='if(gt(iw,ih),min(1280,iw),min(720,iw))':-2",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "28",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                "-movflags",
                "+faststart",
            ]);
        }
        cmd.arg(&part);
        let o = cmd.output().await.map_err(|e| format!("ffmpeg nicht startbar: {e}"))?;
        if !o.status.success() {
            return Err(format!("ffmpeg: {}", String::from_utf8_lossy(&o.stderr).trim()));
        }
        let new = tokio::fs::metadata(&part).await.map_err(|e| e.to_string())?.len();
        let old = tokio::fs::metadata(&src).await.map_err(|e| e.to_string())?.len();
        // Bilder, die nicht kleiner werden, bleiben original.
        if kind == "image" && new >= old {
            return Err("Ergebnis nicht kleiner".into());
        }
        tokio::fs::rename(&part, out).await.map_err(|e| e.to_string())
    }
    .await;
    for f in [&src, &asrc, &part] {
        let _ = tokio::fs::remove_file(f).await;
    }
    result
}

/// Räumt den Cache stündlich auf: alte Dateien weg, dann ältestes zuerst bis unter die Größengrenze.
pub async fn cleanup_loop() {
    loop {
        tokio::time::sleep(Duration::from_secs(3600)).await;
        let Ok(mut rd) = tokio::fs::read_dir(cache_dir()).await else { continue };
        let mut files = Vec::new();
        while let Ok(Some(e)) = rd.next_entry().await {
            if let Ok(md) = e.metadata().await {
                if md.is_file() {
                    files.push((md.modified().unwrap_or(SystemTime::now()), md.len(), e.path()));
                }
            }
        }
        files.sort_by_key(|f| f.0);
        let mut total: u64 = files.iter().map(|f| f.1).sum();
        for (mtime, len, path) in files {
            let old = mtime.elapsed().map(|a| a > MAX_AGE).unwrap_or(false);
            if old || total > MAX_CACHE_BYTES {
                if tokio::fs::remove_file(&path).await.is_ok() {
                    total = total.saturating_sub(len);
                }
            }
        }
    }
}
