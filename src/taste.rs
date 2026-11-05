//! Geschmacks-Algorithmus: bewertet jeden Beitrag mit 0-100 % ("passt zu dir") und lernt aus deinem Verhalten.
//!
//! Merkmale (Features) kommen aus den Scan-Ergebnissen (Haarfarbe, Körperbau, Brust-Größe und -Form, Hände an der
//! Brust, erkannte Körperteile, Anzahl Personen, Alter) und aus dem Text des Beitrags (Titel, Flair, Beschreibung,
//! Hashtags, Subreddit, Autor). Das Modell ist eine logistische Regression über diese Merkmale. Es startet mit deinen
//! genannten Vorlieben als Vorgabe und lernt per Online-Gradientenverfahren aus Votes (stark) und aus dem
//! Anschauverhalten (weitergewischt ohne Vote = leicht nein, schnell weg etwas deutlicher). Die Vorgabe wird dabei
//! nicht vergessen, sondern zieht die Gewichte sanft zurück, solange die Daten nichts anderes sagen.

use std::{
    collections::{HashMap, HashSet},
    env, fs,
    path::PathBuf,
    sync::{Mutex, OnceLock},
};

use axum::{Json, extract::DefaultBodyLimit, http::StatusCode, response::{IntoResponse, Response}};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{err, scan};

/// Temperatur: teilt die Summe der Gewichte, damit die Prozentwerte nicht sofort bei 0 und 100 kleben.
const TEMP: f64 = 2.0;
const MAX_TOKENS: usize = 24;
const KEEP_IDS: usize = 800;

#[derive(Serialize, Deserialize, Default, Clone)]
struct Model {
    version: u32,
    bias: f64,
    w: HashMap<String, f64>,
    prior: HashMap<String, f64>,
    n: u64,
    pos: u64,
    neg: u64,
    /// Beiträge, aus denen schon gelernt wurde ("<id>:<art>"), damit nichts doppelt zählt.
    seen: Vec<String>,
}

fn path() -> PathBuf {
    PathBuf::from(env::var("VPN_DIR").unwrap_or_else(|_| "/vpn".into())).join("taste.json")
}

/// Deine genannten Vorlieben als Startgewichte.
fn default_prior() -> HashMap<String, f64> {
    let p: &[(&str, f64)] = &[
        ("hair:blond", 1.2),
        ("hair:platinblond", 1.1),
        ("hair:dunkelblond", 0.7),
        // Stetige Rohwerte (0..1) statt fertiger Stufen: Körperbau-Index, Brustgröße (Cup-Differenz), Hängewert, Haarhelligkeit
        ("body:idx", -2.2),
        ("bra:size", 1.6),
        ("bra:sag", -2.0),
        ("hair:lum", 0.8),
        ("touch:chest", 1.2),
        ("touch:body", 0.5),
        ("det:FEMALE_BREAST_EXPOSED", 0.4),
        ("det:FACE_FEMALE", 0.3),
        ("det:MALE_GENITALIA_EXPOSED", -2.0),
        ("det:MALE_BREAST_EXPOSED", -0.8),
        ("persons:2+", -0.2),
        ("w:blonde", 0.6),
        ("w:blond", 0.6),
    ];
    p.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}

fn fresh() -> Model {
    let prior = default_prior();
    Model { version: 2, bias: -0.4, w: prior.clone(), prior, ..Default::default() }
}

fn model() -> &'static Mutex<Model> {
    static M: OnceLock<Mutex<Model>> = OnceLock::new();
    M.get_or_init(|| {
        let mut m = fs::read_to_string(path()).ok().and_then(|t| serde_json::from_str::<Model>(&t).ok()).unwrap_or_else(fresh);
        if m.version < 2 {
            // Umstellung auf stetige Rohwerte: alte Stufen-Merkmale entfernen, neue Startwerte übernehmen
            let old = |k: &String| k.starts_with("sag:") || k.starts_with("cup:") || k.starts_with("body:");
            m.w.retain(|k, _| !old(k));
            m.prior.retain(|k, _| !old(k));
            for (k, v) in default_prior() {
                if k.starts_with("body:") || k.starts_with("bra:") || k == "hair:lum" {
                    m.prior.insert(k.clone(), v);
                    m.w.insert(k, v);
                }
            }
            m.version = 2;
        }
        Mutex::new(m)
    })
}

fn save(m: &Model) {
    let p = path();
    let tmp = p.with_extension("tmp");
    if let Ok(t) = serde_json::to_string(m) {
        if fs::write(&tmp, t).is_ok() {
            let _ = fs::rename(&tmp, &p);
        }
    }
}

// ---------- Eingaben ----------

#[derive(Deserialize, Default)]
pub struct InMedia {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    src: String,
}

#[derive(Deserialize, Default)]
pub struct InPost {
    #[serde(default)]
    id: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    subreddit: String,
    #[serde(default)]
    author: String,
    #[serde(default)]
    flair: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    media: Vec<InMedia>,
}

pub type Feats = HashMap<String, f64>;

// ---------- Merkmale aus dem Scan ----------

fn num(v: &Value) -> Option<f64> {
    v.as_f64()
}

fn age_group(a: f64) -> &'static str {
    match a as i64 {
        ..=19 => "unter20",
        20..=24 => "20-24",
        25..=29 => "25-29",
        30..=39 => "30-39",
        _ => "40+",
    }
}

/// Eine Person (Erscheinungs-Eintrag) in Merkmale umsetzen.
fn person_feats(a: &Value, out: &mut Vec<String>) {
    for (key, pre) in [("hair", "hair"), ("skin", "skin")] {
        if let Some(s) = a[key].as_str() {
            out.push(format!("{pre}:{s}"));
        }
    }
    if let Some(age) = num(&a["age"]) {
        out.push(format!("age:{}", age_group(age)));
    }
}

/// Stetige Merkmale einer Person direkt aus den Messwerten (nicht aus den fertigen Stufen). Alle in 0..1.
fn person_cont(a: &Value, out: &mut Vec<(String, f64)>) {
    if let Some(idx) = num(&a["shape"]["idx"]) {
        out.push(("body:idx".into(), ((idx - 0.5) / 0.7).clamp(0.0, 1.0)));
    }
    if let Some(d) = num(&a["bra"]["diff_cm"]) {
        out.push(("bra:size".into(), ((d - 6.0) / 24.0).clamp(0.0, 1.0)));
    }
    if let Some(s) = num(&a["bra"]["sag"]) {
        out.push(("bra:sag".into(), s.clamp(0.0, 1.0)));
    }
    if let Some(c) = a["hair_rgb"].as_array() {
        let ch: Vec<f64> = c.iter().filter_map(|x| x.as_f64()).collect();
        if ch.len() == 3 {
            out.push(("hair:lum".into(), ((0.299 * ch[0] + 0.587 * ch[1] + 0.114 * ch[2]) / 255.0).clamp(0.0, 1.0)));
        }
    }
}

fn pt(p: &Value, i: usize) -> Option<(f64, f64, f64)> {
    let a = p.get(i)?;
    Some((a.get(0)?.as_f64()?, a.get(1)?.as_f64()?, a.get(2)?.as_f64()?))
}

/// Liegt die Hand auf der Brust bzw. am Oberkörper? Rechnet in Pixeln (Seitenverhältnis wird berücksichtigt).
/// Brust = Bereich zwischen Schultern und knapp unter der Brust, plus erkannte Brust-Boxen (leicht vergrößert).
fn touch(pose: &Value, hands: &[&Value], breasts: &[[f64; 4]], w: f64, h: f64) -> (bool, bool) {
    let (Some(ls), Some(rs)) = (pt(pose, 11), pt(pose, 12)) else { return (false, false) };
    if ls.2 < 0.5 || rs.2 < 0.5 {
        return (false, false);
    }
    let shw = ((ls.0 - rs.0) * w).hypot((ls.1 - rs.1) * h);
    if shw < 20.0 {
        return (false, false);
    }
    let hips: Vec<(f64, f64, f64)> = [23, 24].iter().filter_map(|&i| pt(pose, i)).filter(|p| p.2 >= 0.4).collect();
    let sy = (ls.1 + rs.1) / 2.0 * h;
    let tl = if hips.len() == 2 {
        let hy = (hips[0].1 + hips[1].1) / 2.0 * h;
        (hy - sy).abs().max(shw * 0.8)
    } else {
        shw * 1.25
    };
    let x0 = ls.0.min(rs.0) * w - 0.12 * shw;
    let x1 = ls.0.max(rs.0) * w + 0.12 * shw;
    let (chest_y0, chest_y1) = (sy - 0.15 * shw, sy + 0.55 * tl);
    let body_y1 = sy + 1.05 * tl;
    let (mut chest, mut body) = (false, false);
    for hand in hands {
        let Some(l) = hand.as_array() else { continue };
        if l.len() < 21 {
            continue;
        }
        let at = |i: usize| -> Option<(f64, f64)> { Some((l[i].get(0)?.as_f64()? * w, l[i].get(1)?.as_f64()? * h)) };
        let palm: Vec<(f64, f64)> = [0, 5, 9, 13, 17].iter().filter_map(|&i| at(i)).collect();
        if palm.len() < 5 {
            continue;
        }
        let cx = palm.iter().map(|p| p.0).sum::<f64>() / 5.0;
        let cy = palm.iter().map(|p| p.1).sum::<f64>() / 5.0;
        let tips: Vec<(f64, f64)> = [4, 8, 12, 16, 20].iter().filter_map(|&i| at(i)).collect();
        let pts: Vec<(f64, f64)> = std::iter::once((cx, cy)).chain(tips).collect();
        let in_box = |p: &(f64, f64), b: &[f64; 4]| {
            let (bx, by, bw, bh) = (b[0] * w, b[1] * h, b[2] * w, b[3] * h);
            p.0 >= bx - 0.1 * bw && p.0 <= bx + 1.1 * bw && p.1 >= by - 0.1 * bh && p.1 <= by + 1.1 * bh
        };
        if (cx >= x0 && cx <= x1 && cy >= chest_y0 && cy <= chest_y1)
            || breasts.iter().any(|b| pts.iter().any(|p| in_box(p, b)))
        {
            chest = true;
        }
        if cx >= x0 && cx <= x1 && cy >= chest_y0 && cy <= body_y1 {
            body = true;
        }
    }
    (chest, body)
}

fn is_breast(label: &str) -> bool {
    label == "FEMALE_BREAST_EXPOSED" || label == "FEMALE_BREAST_COVERED"
}

/// Merkmale aus einem Scan-Ergebnis (Bild oder Video). Werte liegen in 0..1.
fn scan_feats(v: &Value, out: &mut Feats) {
    let w = num(&v["width"]).unwrap_or(1.0).max(1.0);
    let h = num(&v["height"]).unwrap_or(1.0).max(1.0);
    let video = v["kind"].as_str() == Some("video");

    // Personen-Merkmale: Mittelwert über alle Beobachtungen (Personen bzw. Videoframes)
    let mut obs: Vec<Vec<String>> = Vec::new();
    let mut cont: Vec<(String, f64)> = Vec::new();
    if video {
        for fr in v["appearance_timeline"].as_array().into_iter().flatten() {
            for a in fr.get(1).and_then(|x| x.as_array()).into_iter().flatten() {
                let mut f = Vec::new();
                person_feats(a, &mut f);
                person_cont(a, &mut cont);
                if !f.is_empty() {
                    obs.push(f);
                }
            }
        }
    } else {
        for a in v["appearance"].as_array().into_iter().flatten() {
            let mut f = Vec::new();
            person_feats(a, &mut f);
            person_cont(a, &mut cont);
            if !f.is_empty() {
                obs.push(f);
            }
        }
    }
    // Stetige Werte: Mittelwert je Merkmal über alle Beobachtungen
    let mut agg: HashMap<String, (f64, f64)> = HashMap::new();
    for (k, v) in cont {
        let e = agg.entry(k).or_insert((0.0, 0.0));
        e.0 += v;
        e.1 += 1.0;
    }
    for (k, (sum, n)) in agg {
        out.insert(k, sum / n);
    }
    if !obs.is_empty() {
        let n = obs.len() as f64;
        for f in obs.iter().flatten() {
            *out.entry(f.clone()).or_insert(0.0) += 1.0 / n;
        }
    }

    // Erkannte Körperteile
    for d in v["detections"].as_array().into_iter().flatten() {
        let (Some(l), Some(sc)) = (d["label"].as_str(), num(&d["score"])) else { continue };
        if sc < 0.4 {
            continue;
        }
        let val = if video { num(&d["share"]).unwrap_or(0.0).clamp(0.0, 1.0) } else { 1.0 };
        if val >= 0.05 {
            let e = out.entry(format!("det:{l}")).or_insert(0.0);
            *e = e.max(val);
        }
    }

    // Anzahl Personen
    let persons = num(&v["persons"]).or_else(|| num(&v["persons_max"])).or_else(|| v["poses"].as_array().map(|a| a.len() as f64));
    if let Some(n) = persons {
        let k = if n >= 2.0 { "2+" } else if n >= 1.0 { "1" } else { "0" };
        out.insert(format!("persons:{k}"), 1.0);
    }

    // Hände an der Brust / am Körper
    let (mut chest_n, mut body_n, mut frames) = (0.0, 0.0, 0.0);
    if video {
        let mut hands_at: HashMap<i64, &Value> = HashMap::new();
        for fr in v["hand_timeline"].as_array().into_iter().flatten() {
            if let (Some(t), Some(hs)) = (num(&fr[0]), fr.get(1)) {
                hands_at.insert((t * 1000.0).round() as i64, hs);
            }
        }
        let names: Vec<&str> = v["names"].as_array().into_iter().flatten().filter_map(|x| x.as_str()).collect();
        let mut rows_at: HashMap<i64, &Value> = HashMap::new();
        for fr in v["timeline"].as_array().into_iter().flatten() {
            if let (Some(t), Some(rows)) = (num(&fr[0]), fr.get(1)) {
                rows_at.insert((t * 1000.0).round() as i64, rows);
            }
        }
        for fr in v["pose_timeline"].as_array().into_iter().flatten() {
            let (Some(t), Some(people)) = (num(&fr[0]), fr.get(1).and_then(|x| x.as_array())) else { continue };
            frames += 1.0;
            let key = (t * 1000.0).round() as i64;
            let Some(hs) = hands_at.get(&key).and_then(|x| x.as_array()) else { continue };
            let hands: Vec<&Value> = hs.iter().collect();
            let breasts: Vec<[f64; 4]> = rows_at
                .get(&key)
                .and_then(|r| r.as_array())
                .into_iter()
                .flatten()
                .filter_map(|r| {
                    let a = r.as_array()?;
                    let name = names.get(a.first()?.as_u64()? as usize)?;
                    if !is_breast(name) {
                        return None;
                    }
                    Some([a.get(2)?.as_f64()?, a.get(3)?.as_f64()?, a.get(4)?.as_f64()?, a.get(5)?.as_f64()?])
                })
                .collect();
            let (mut c, mut b) = (false, false);
            for p in people {
                let (c1, b1) = touch(p, &hands, &breasts, w, h);
                c |= c1;
                b |= b1;
            }
            chest_n += c as u8 as f64;
            body_n += b as u8 as f64;
        }
        if frames >= 3.0 {
            // Kurze Berührungen zählen schon: 1/3 der Frames reichen für den vollen Wert
            out.insert("touch:chest".into(), (chest_n / frames * 3.0).min(1.0));
            out.insert("touch:body".into(), (body_n / frames * 3.0).min(1.0));
        }
    } else if let (Some(poses), Some(hs)) = (v["poses"].as_array(), v["hands"].as_array()) {
        let hands: Vec<&Value> = hs.iter().collect();
        let breasts: Vec<[f64; 4]> = v["detections"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|d| d["label"].as_str().is_some_and(is_breast))
            .filter_map(|d| {
                let b = d["box"].as_array()?;
                Some([b.first()?.as_f64()?, b.get(1)?.as_f64()?, b.get(2)?.as_f64()?, b.get(3)?.as_f64()?])
            })
            .collect();
        let (mut c, mut b) = (false, false);
        for p in poses {
            let (c1, b1) = touch(p, &hands, &breasts, w, h);
            c |= c1;
            b |= b1;
        }
        out.insert("touch:chest".into(), c as u8 as f64);
        out.insert("touch:body".into(), b as u8 as f64);
    }
}

// ---------- Merkmale aus dem Text ----------

const STOP: &[&str] = &[
    "the", "and", "for", "with", "you", "your", "are", "this", "that", "from", "have", "has", "was", "but", "not", "all",
    "any", "can", "just", "like", "out", "get", "got", "her", "his", "she", "its", "our", "who", "how", "one", "too", "per",
    "mal", "und", "der", "die", "das", "ein", "eine", "ich", "mit", "von", "für", "auf", "den", "dem", "des", "bin", "ist",
];

fn text_feats(p: &InPost, out: &mut Feats) {
    let all = format!("{} {} {}", p.title, p.flair, p.text.chars().take(400).collect::<String>()).to_lowercase();
    let mut seen = HashSet::new();
    let mut toks: Vec<String> = Vec::new();
    for raw in all.split(|c: char| !c.is_alphanumeric()) {
        if raw.is_empty() {
            continue;
        }
        // Alter-Angaben wie "24f" oder "f24"
        let digits: String = raw.chars().filter(|c| c.is_ascii_digit()).collect();
        if (raw.ends_with('f') || raw.starts_with('f')) && (2..=2).contains(&digits.len()) && raw.len() <= 4 {
            if let Ok(a) = digits.parse::<f64>() {
                out.insert(format!("tage:{}", age_group(a)), 1.0);
            }
            continue;
        }
        if raw.chars().count() >= 3 && !raw.chars().all(|c| c.is_ascii_digit()) && !STOP.contains(&raw) && seen.insert(raw.to_string()) {
            toks.push(raw.to_string());
        }
    }
    toks.truncate(MAX_TOKENS);
    if !toks.is_empty() {
        let v = 1.0 / (toks.len() as f64).sqrt();
        for t in toks {
            out.insert(format!("w:{t}"), v);
        }
    }
    if !p.subreddit.is_empty() {
        out.insert(format!("sr:{}", p.subreddit.to_lowercase()), 1.0);
    }
    if !p.author.is_empty() && p.author != "[deleted]" {
        out.insert(format!("au:{}", p.author.to_lowercase()), 0.7);
    }
}

/// Alle Merkmale eines Beitrags. Zweiter Wert: liegt mindestens ein fertiger Scan vor?
fn features(p: &InPost) -> (Feats, bool) {
    let mut out = Feats::new();
    text_feats(p, &mut out);
    let mut scans: Vec<Feats> = Vec::new();
    let mut scannable = 0;
    for m in &p.media {
        if m.src.is_empty() || (m.kind != "image" && m.kind != "video") {
            continue;
        }
        scannable += 1;
        let k = scan::key(&m.kind, &m.src);
        let Ok(t) = fs::read_to_string(scan::result_path(&k)) else { continue };
        let Ok(v) = serde_json::from_str::<Value>(&t) else { continue };
        if v["status"].as_str() != Some("done") {
            continue;
        }
        let mut f = Feats::new();
        scan_feats(&v, &mut f);
        scans.push(f);
    }
    let ready = scannable == 0 || !scans.is_empty();
    if !scans.is_empty() {
        let n = scans.len() as f64;
        for f in scans {
            for (k, v) in f {
                *out.entry(k).or_insert(0.0) += v / n;
            }
        }
    }
    (out, ready)
}

/// Merkmale eines Beitrags im Feed-Format (für den Vorrat; wird einmal berechnet und gemerkt).
pub fn features_value(post: &Value) -> Feats {
    let p: InPost = serde_json::from_value(post.clone()).unwrap_or_default();
    features(&p).0
}

/// Bewertung vieler Merkmals-Sätze auf einmal: (Prozent, Gründe).
pub fn score_feats(list: &[&Feats]) -> Vec<(f64, Vec<Value>)> {
    let m = model().lock().unwrap_or_else(|e| e.into_inner()).clone();
    list.iter().map(|f| ((sigmoid(z_of(&m, f) / TEMP) * 100.0).round(), explain(&m, f))).collect()
}

// ---------- Modell ----------

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn z_of(m: &Model, f: &Feats) -> f64 {
    m.bias + f.iter().map(|(k, v)| m.w.get(k).copied().unwrap_or(0.0) * v).sum::<f64>()
}

/// Lesbarer Name eines Merkmals.
fn nice(k: &str) -> String {
    let (g, v) = k.split_once(':').unwrap_or(("", k));
    match g {
        "hair" => if v == "lum" { "helle Haare".into() } else { format!("Haar {v}") },
        "skin" => format!("Haut {v}"),
        "age" => format!("Alter {v}"),
        "tage" => format!("Alter im Titel {v}"),
        "body" => "Körperfülle".into(),
        "bra" => if v == "size" { "Brustgröße".into() } else { "Brust hängend".into() },
        "touch" => if v == "chest" { "Hände an der Brust".into() } else { "Hände am Körper".into() },
        "det" => format!("{}", v.to_lowercase().replace('_', " ")),
        "persons" => format!("Personen {v}"),
        "w" => format!("„{v}“"),
        "sr" => format!("r/{v}"),
        "au" => format!("u/{v}"),
        _ => k.to_string(),
    }
}

fn explain(m: &Model, f: &Feats) -> Vec<Value> {
    let mut c: Vec<(String, f64)> = f
        .iter()
        .filter_map(|(k, v)| {
            let x = m.w.get(k).copied().unwrap_or(0.0) * v;
            (x.abs() >= 0.15).then(|| (k.clone(), x))
        })
        .collect();
    c.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
    c.truncate(4);
    c.into_iter().map(|(k, x)| json!({ "t": nice(&k), "up": x > 0.0 })).collect()
}

fn learn(m: &mut Model, f: &Feats, y: f64, weight: f64) {
    let p = sigmoid(z_of(m, f) / TEMP);
    let lr = (0.9 / (1.0 + m.n as f64 / 150.0).sqrt()).max(0.12);
    let g = (p - y) * weight / TEMP;
    for (k, v) in f {
        let prior = m.prior.get(k).copied().unwrap_or(0.0);
        let w = m.w.entry(k.clone()).or_insert(prior);
        // Gradient + sanfter Zug zur Vorgabe (Regularisierung)
        *w -= lr * (g * v + 0.01 * (*w - prior));
        *w = w.clamp(-6.0, 6.0);
    }
    m.bias -= lr * g * 0.4;
    m.bias = m.bias.clamp(-3.0, 3.0);
    m.n += 1;
    if y >= 0.5 {
        m.pos += 1;
    } else {
        m.neg += 1;
    }
}

// ---------- API ----------

#[derive(Deserialize)]
pub struct ScoreReq {
    posts: Vec<InPost>,
}

/// Bewertung mehrerer Beiträge: [{id, pct, ready, why:[{t, up}]}]
pub async fn score(Json(r): Json<ScoreReq>) -> Response {
    let out = tokio::task::spawn_blocking(move || {
        let m = model().lock().unwrap_or_else(|e| e.into_inner()).clone();
        r.posts
            .iter()
            .take(12)
            .map(|p| {
                let (f, ready) = features(p);
                let pct = (sigmoid(z_of(&m, &f) / TEMP) * 100.0).round();
                json!({ "id": p.id, "pct": pct, "ready": ready, "why": explain(&m, &f) })
            })
            .collect::<Vec<_>>()
    })
    .await
    .unwrap_or_default();
    Json(json!({ "scores": out })).into_response()
}

#[derive(Deserialize)]
pub struct EventReq {
    post: InPost,
    /// "vote" (up/down gesetzt) oder "view" (Beitrag verlassen, ohne Vote).
    kind: String,
    #[serde(default)]
    vote: String,
    #[serde(default)]
    dwell_ms: f64,
    /// Länge des Videos in ms (0 bei Bildern); wird mitgeschickt, zählt aktuell nicht.
    #[serde(default)]
    #[allow(dead_code)]
    media_ms: f64,
}

/// Aus einem Ereignis lernen. Antwort: {learned: bool}
pub async fn event(Json(e): Json<EventReq>) -> Response {
    if e.post.id.is_empty() || e.post.id.len() > 64 {
        return err(StatusCode::BAD_REQUEST, "Beitrag fehlt");
    }
    // (Ziel, Gewicht) bestimmen
    let dwell = e.dwell_ms.clamp(0.0, 3_600_000.0);
    let target: Option<(f64, f64)> = match (e.kind.as_str(), e.vote.as_str()) {
        // Like: je länger angeschaut, desto stärker. Dislike: schnell weg = deutlich.
        ("vote", "up") => Some((1.0, 1.0 + 0.5 * (dwell / 30_000.0).min(1.0))),
        ("vote", "down") => Some((0.0, if dwell < 3000.0 { 1.5 } else { 1.0 })),
        // Weitergewischt ohne Vote: leicht negativ (schwächer als ein Dislike), schnell weg etwas deutlicher.
        ("view", _) => Some((0.0, if dwell < 1500.0 { 0.45 } else { 0.3 })),
        _ => None,
    };
    let Some((y, wgt)) = target else { return Json(json!({ "learned": false })).into_response() };
    let id_key = format!("{}:{}", e.post.id, if e.kind == "vote" { e.vote.as_str() } else { "view" });
    let learned = tokio::task::spawn_blocking(move || {
        let (f, _) = features(&e.post);
        let mut m = model().lock().unwrap_or_else(|x| x.into_inner());
        if m.seen.contains(&id_key) {
            return false;
        }
        m.seen.push(id_key);
        if m.seen.len() > KEEP_IDS {
            let cut = m.seen.len() - KEEP_IDS;
            m.seen.drain(..cut);
        }
        learn(&mut m, &f, y, wgt);
        save(&m);
        true
    })
    .await
    .unwrap_or(false);
    Json(json!({ "learned": learned })).into_response()
}

/// Was hat das Modell gelernt? Die stärksten Vorlieben und Abneigungen.
pub async fn stats() -> Response {
    let m = model().lock().unwrap_or_else(|e| e.into_inner()).clone();
    let mut all: Vec<(&String, &f64)> = m.w.iter().filter(|(k, v)| !k.starts_with("w:") || v.abs() >= 0.3).collect();
    all.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top: Vec<Value> = all.iter().filter(|(_, v)| **v > 0.1).take(10).map(|(k, v)| json!({ "t": nice(k), "w": (**v * 100.0).round() / 100.0 })).collect();
    let low: Vec<Value> = all.iter().rev().filter(|(_, v)| **v < -0.1).take(10).map(|(k, v)| json!({ "t": nice(k), "w": (**v * 100.0).round() / 100.0 })).collect();
    Json(json!({ "n": m.n, "likes": m.pos, "dislikes": m.neg, "top": top, "low": low })).into_response()
}

/// Alles Gelernte verwerfen (zurück zu deinen Start-Vorlieben).
pub async fn reset() -> Response {
    let m = fresh();
    save(&m);
    *model().lock().unwrap_or_else(|e| e.into_inner()) = m;
    Json(json!({ "ok": true })).into_response()
}

pub fn limit() -> DefaultBodyLimit {
    DefaultBodyLimit::max(64 * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(items: &[(&str, f64)]) -> Feats {
        items.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn prior_prefers_blond_straff_touch() {
        let m = fresh();
        let good = f(&[("hair:blond", 1.0), ("bra:sag", 0.0), ("touch:chest", 1.0), ("bra:size", 1.0)]);
        let bad = f(&[("hair:schwarz", 1.0), ("bra:sag", 1.0), ("body:idx", 1.0)]);
        assert!(sigmoid(z_of(&m, &good) / TEMP) > 0.85);
        assert!(sigmoid(z_of(&m, &bad) / TEMP) < 0.2);
    }

    #[test]
    fn learns_from_dislikes_and_keeps_prior_pull() {
        let mut m = fresh();
        let x = f(&[("hair:blond", 1.0), ("au:foo", 0.7)]);
        let before = sigmoid(z_of(&m, &x) / TEMP);
        for _ in 0..8 {
            learn(&mut m, &x, 0.0, 1.0);
        }
        let after = sigmoid(z_of(&m, &x) / TEMP);
        assert!(after < before - 0.2, "{before} -> {after}");
        // Das Autor-Merkmal trägt den Hauptteil, blond fällt nicht ins Bodenlose
        assert!(m.w["au:foo"] < 0.0);
    }

    #[test]
    fn text_features_parse_age_and_tokens() {
        let p = InPost { title: "Blonde and bubbly (24F) #natural".into(), subreddit: "Boobs".into(), ..Default::default() };
        let mut o = Feats::new();
        text_feats(&p, &mut o);
        assert!(o.contains_key("w:blonde"));
        assert!(o.contains_key("w:natural"));
        assert!(o.contains_key("tage:20-24"));
        assert!(o.contains_key("sr:boobs"));
        assert!(!o.contains_key("w:24f"));
    }

    #[test]
    fn touch_detects_hand_on_chest() {
        // Schultern bei y=0.30, Hüften bei y=0.60, Hand mittig auf Brusthöhe (y=0.38)
        let mut pose = vec![json!([0.0, 0.0, 0.0]); 33];
        pose[11] = json!([0.35, 0.30, 0.9]);
        pose[12] = json!([0.65, 0.30, 0.9]);
        pose[23] = json!([0.40, 0.60, 0.9]);
        pose[24] = json!([0.60, 0.60, 0.9]);
        let pose = Value::Array(pose);
        let mk = |cx: f64, cy: f64| -> Value {
            Value::Array((0..21).map(|_| json!([cx, cy, 1.0])).collect())
        };
        let on = mk(0.5, 0.38);
        let off = mk(0.9, 0.9);
        assert_eq!(touch(&pose, &[&on], &[], 1000.0, 1000.0), (true, true));
        assert_eq!(touch(&pose, &[&off], &[], 1000.0, 1000.0), (false, false));
    }
}
