//! Rust-Algorithmus zur präzisen Erkennung, Zuordnung und Paarung von Brustwarzen / Brüsten.
//!
//! Kombiniert NudeNet-Erkennungen (FEMALE_BREAST_EXPOSED, FEMALE_BREAST_COVERED, MALE_BREAST_EXPOSED)
//! mit MediaPipe-Pose-Landmarks (33 Körperpunkte) zu anatomisch exakten Brustwarzen-Punkten.
//!
//! Regeln:
//! 1. Als Sicherung: Ein Punkt wird NUR gesetzt, wenn NudeNet an der Stelle eine Brust erkannt hat.
//! 2. Unbedeckte Brüste (FEMALE_BREAST_EXPOSED) werden gegenüber bedeckten (FEMALE_BREAST_COVERED) bevorzugt
//!    und mit höherer Konfidenz bewertet (bedeckte werden aber nicht ausgeschlossen).
//! 3. Bei mehreren Personen werden Brüste anhand anatomischer Torso-Vektoren eindeutig der passenden
//!    Person und Körperseite (links vs. rechts) zugeordnet.
//! 4. Die beiden Brustwarzen einer Person werden explizit als zusammengehöriges Paar verknüpft.
//! 5. Bei Nahaufnahmen ohne vollständiges Posen-Skelett werden eng beieinander liegende Brüste
//!    automatisch als Paar gruppiert.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BreastKind {
    ExposedFemale,
    CoveredFemale,
    ExposedMale,
}

impl BreastKind {
    pub fn is_exposed(&self) -> bool {
        matches!(self, BreastKind::ExposedFemale | BreastKind::ExposedMale)
    }

    pub fn priority_weight(&self) -> f64 {
        match self {
            BreastKind::ExposedFemale => 1.5,
            BreastKind::ExposedMale => 1.2,
            BreastKind::CoveredFemale => 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BreastBox {
    pub kind: BreastKind,
    pub score: f64,
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
    pub cx: f64,
    pub cy: f64,
    /// Per Bildanalyse gefundene Brustwarze (x, y, Konfidenz), falls vorhanden.
    pub nip: Option<(f64, f64, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NipplePoint {
    pub x: f64,
    pub y: f64,
    pub conf: f64,
    pub exposed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NipplePair {
    pub person_id: Option<usize>,
    pub left: Option<NipplePoint>,
    pub right: Option<NipplePoint>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Side {
    Left,
    Right,
}

/// Verarbeitet das Scan-Ergebnis direkt in Rust (für Einzelbilder und Videos).
pub fn process_scan_result(v: &mut Value) {
    let kind = v.get("kind").and_then(|k| k.as_str()).unwrap_or("");
    if kind == "image" {
        process_image(v);
    } else if kind == "video" {
        process_video(v);
    }
}

fn parse_breast_box(label: &str, score: f64, box_arr: &[Value], nip: Option<(f64, f64, f64)>) -> Option<BreastBox> {
    let kind = match label {
        "FEMALE_BREAST_EXPOSED" => BreastKind::ExposedFemale,
        "FEMALE_BREAST_COVERED" => BreastKind::CoveredFemale,
        "MALE_BREAST_EXPOSED" => BreastKind::ExposedMale,
        _ => return None,
    };
    if box_arr.len() < 4 {
        return None;
    }
    let x = box_arr[0].as_f64()?;
    let y = box_arr[1].as_f64()?;
    let w = box_arr[2].as_f64()?;
    let h = box_arr[3].as_f64()?;
    let cx = x + w * 0.5;
    let cy = y + h * 0.5;
    Some(BreastBox { kind, score, x, y, w, h, cx, cy, nip })
}

/// Analysiert ein Einzelbild.
fn process_image(v: &mut Value) {
    let mut boxes = Vec::new();
    if let Some(dets) = v.get("detections").and_then(|d| d.as_array()) {
        for d in dets {
            let label = d.get("label").and_then(|l| l.as_str()).unwrap_or("");
            let score = d.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0);
            if let Some(b) = d.get("box").and_then(|b| b.as_array()) {
                let nip = d.get("nip").and_then(|n| n.as_array()).and_then(|n| Some((n.first()?.as_f64()?, n.get(1)?.as_f64()?, n.get(2)?.as_f64()?)));
                if let Some(bb) = parse_breast_box(label, score, b, nip) {
                    boxes.push(bb);
                }
            }
        }
    }

    let mut poses: Vec<Vec<[f64; 3]>> = Vec::new();
    if let Some(p_arr) = v.get("poses").and_then(|p| p.as_array()) {
        for person in p_arr {
            if let Some(lm_arr) = person.as_array() {
                let mut lm = Vec::with_capacity(33);
                for pt in lm_arr {
                    if let Some(coords) = pt.as_array() {
                        if coords.len() >= 3 {
                            lm.push([
                                coords[0].as_f64().unwrap_or(0.0),
                                coords[1].as_f64().unwrap_or(0.0),
                                coords[2].as_f64().unwrap_or(0.0),
                            ]);
                        }
                    }
                }
                if lm.len() == 33 {
                    poses.push(lm);
                }
            }
        }
    }

    let pairs = reconcile_frame(&mut poses, &boxes);

    // Aktualisierte Posen mit zusätzlichen Landmarks 33 (links) und 34 (rechts) zurückschreiben
    let mut updated_poses = Vec::with_capacity(poses.len());
    for person in &poses {
        let serialized_person: Vec<Value> = person
            .iter()
            .map(|pt| json!([round(pt[0], 4), round(pt[1], 4), round(pt[2], 3)]))
            .collect();
        updated_poses.push(Value::Array(serialized_person));
    }
    v["poses"] = Value::Array(updated_poses);
    v["nipples"] = serde_json::to_value(&pairs).unwrap_or(Value::Array(Vec::new()));
}

/// Analysiert ein Video über alle Frames hinweg.
fn process_video(v: &mut Value) {
    let names: Vec<String> = v
        .get("names")
        .and_then(|n| n.as_array())
        .map(|arr| arr.iter().filter_map(|s| s.as_str().map(String::from)).collect())
        .unwrap_or_default();

    // Detections nach Zeitstempel aufbereiten: t -> Vec<BreastBox>
    let mut time_boxes: Vec<(f64, Vec<BreastBox>)> = Vec::new();
    if let Some(tl) = v.get("timeline").and_then(|t| t.as_array()) {
        for row in tl {
            if let Some(row_arr) = row.as_array() {
                if row_arr.len() >= 2 {
                    let t = row_arr[0].as_f64().unwrap_or(0.0);
                    let mut b_list = Vec::new();
                    if let Some(items) = row_arr[1].as_array() {
                        for item in items {
                            if let Some(vals) = item.as_array() {
                                if vals.len() >= 6 {
                                    let label_idx = vals[0].as_u64().unwrap_or(0) as usize;
                                    let score = vals[1].as_f64().unwrap_or(0.0);
                                    let box_slice = &vals[2..6];
                                    if let Some(label) = names.get(label_idx) {
                                        let nip = if vals.len() >= 9 {
                                            Some((vals[6].as_f64().unwrap_or(0.0), vals[7].as_f64().unwrap_or(0.0), vals[8].as_f64().unwrap_or(0.0)))
                                        } else {
                                            None
                                        };
                                        if let Some(bb) = parse_breast_box(label, score, box_slice, nip) {
                                            b_list.push(bb);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if !b_list.is_empty() {
                        time_boxes.push((t, b_list));
                    }
                }
            }
        }
    }

    let mut nipple_tl: Vec<Value> = Vec::new();

    if let Some(ptl) = v.get_mut("pose_timeline").and_then(|p| p.as_array_mut()) {
        for frame in ptl {
            if let Some(frame_arr) = frame.as_array_mut() {
                if frame_arr.len() >= 2 {
                    let t = frame_arr[0].as_f64().unwrap_or(0.0);

                    // Finde die passenden NudeNet-Treffer für diesen Zeitstempel (Toleranz: ~0.15s)
                    let matched_boxes = find_closest_boxes(&time_boxes, t, 0.15);

                    let mut poses = Vec::new();
                    if let Some(people) = frame_arr[1].as_array() {
                        for person in people {
                            if let Some(pts) = person.as_array() {
                                let mut lm = Vec::with_capacity(33);
                                for pt in pts {
                                    if let Some(coords) = pt.as_array() {
                                        if coords.len() >= 3 {
                                            lm.push([
                                                coords[0].as_f64().unwrap_or(0.0),
                                                coords[1].as_f64().unwrap_or(0.0),
                                                coords[2].as_f64().unwrap_or(0.0),
                                            ]);
                                        }
                                    }
                                }
                                if lm.len() == 33 {
                                    poses.push(lm);
                                }
                            }
                        }
                    }

                    let pairs = reconcile_frame(&mut poses, &matched_boxes);

                    // Aktualisierte Posen mit 33/34 zurückschreiben
                    let updated_people: Vec<Value> = poses
                        .iter()
                        .map(|person| {
                            let pts: Vec<Value> = person
                                .iter()
                                .map(|pt| json!([round(pt[0], 4), round(pt[1], 4), round(pt[2], 3)]))
                                .collect();
                            Value::Array(pts)
                        })
                        .collect();
                    frame_arr[1] = Value::Array(updated_people);

                    if !pairs.is_empty() {
                        nipple_tl.push(json!([round(t, 3), pairs]));
                    }
                }
            }
        }
    }

    v["nipple_timeline"] = Value::Array(nipple_tl);
}

fn find_closest_boxes(time_boxes: &[(f64, Vec<BreastBox>)], t: f64, tol: f64) -> Vec<BreastBox> {
    let mut best_diff = tol;
    let mut best_idx = None;
    for (i, (tb, _)) in time_boxes.iter().enumerate() {
        let diff = (tb - t).abs();
        if diff < best_diff {
            best_diff = diff;
            best_idx = Some(i);
        }
    }
    match best_idx {
        Some(idx) => time_boxes[idx].1.clone(),
        None => Vec::new(),
    }
}

/// Kernalgorithmus für einen Frame:
/// Verknüpft NudeNet-Brust-Boxen mit Posen-Torsi, weist sie Person und Seite zu
/// und hängt Landmarks 33 (links) und 34 (rechts) an jede Pose an.
pub fn reconcile_frame(poses: &mut [Vec<[f64; 3]>], boxes: &[BreastBox]) -> Vec<NipplePair> {
    let mut results: Vec<NipplePair> = Vec::new();

    if poses.is_empty() {
        // Keine Posen vorhanden (z. B. reine Nahaufnahme von Brüsten):
        // Gruppiere eng beieinander liegende Boxen als Paare
        let pairs = group_isolated_breast_pairs(boxes);
        return pairs;
    }

    struct TorsoBox {
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
    }

    // Für jede Person Posen-Torsi analysieren
    struct PersonTorso {
        id: usize,
        anchor_left: (f64, f64),
        anchor_right: (f64, f64),
        torso_box: TorsoBox,
        scale: f64,
        spine_axis: ((f64, f64), (f64, f64)), // mid_shoulder -> mid_hip
    }

    let mut torsos = Vec::with_capacity(poses.len());
    for (i, pose) in poses.iter().enumerate() {
        if pose.len() < 33 {
            continue;
        }
        let sl = (pose[11][0], pose[11][1]);
        let sr = (pose[12][0], pose[12][1]);
        let hl = (pose[23][0], pose[23][1]);
        let hr = (pose[24][0], pose[24][1]);

        let vis_sl = pose[11][2];
        let vis_sr = pose[12][2];
        let vis_hl = pose[23][2];
        let vis_hr = pose[24][2];

        // Schultern müssen hinreichend erkennbar sein
        if vis_sl < 0.25 && vis_sr < 0.25 {
            continue;
        }

        let mid_s = ((sl.0 + sr.0) * 0.5, (sl.1 + sr.1) * 0.5);
        let s_dx = sl.0 - sr.0;
        let s_dy = sl.1 - sr.1;
        let s_len = (s_dx * s_dx + s_dy * s_dy).sqrt().max(0.04);

        // Vektor nach unten (Spine)
        let (down_dx, down_dy, t_len) = if vis_hl >= 0.25 || vis_hr >= 0.25 {
            let mid_h = ((hl.0 + hr.0) * 0.5, (hl.1 + hr.1) * 0.5);
            let d_dx = mid_h.0 - mid_s.0;
            let d_dy = mid_h.1 - mid_s.1;
            let len = (d_dx * d_dx + d_dy * d_dy).sqrt().max(0.05);
            (d_dx / len, d_dy / len, len)
        } else {
            // Falls Hüften verdeckt sind: Senkrechte zum Schultervektor nach unten
            let d_dx = -s_dy / s_len;
            let d_dy = s_dx / s_len;
            (d_dx, d_dy, s_len * 1.3)
        };

        // Anatomische Anker: ca. 35 % der Torsolänge nach unten,
        // und seitlich jeweils ~30 % der halben Schulterbreite vom Zentrum
        let down_step = t_len * 0.35;
        let lat_step = s_len * 0.22;

        let center_chest = (mid_s.0 + down_dx * down_step, mid_s.1 + down_dy * down_step);
        let s_unit_x = s_dx / s_len;
        let s_unit_y = s_dy / s_len;

        // Links = Richtung Schulter 11, Rechts = Richtung Schulter 12
        let anchor_left = (center_chest.0 + s_unit_x * lat_step, center_chest.1 + s_unit_y * lat_step);
        let anchor_right = (center_chest.0 - s_unit_x * lat_step, center_chest.1 - s_unit_y * lat_step);

        let min_x = sl.0.min(sr.0).min(hl.0).min(hr.0) - s_len * 0.35;
        let max_x = sl.0.max(sr.0).max(hl.0).max(hr.0) + s_len * 0.35;
        let min_y = sl.1.min(sr.1) - s_len * 0.2;
        let max_y = hl.1.max(hr.1).max(center_chest.1 + down_step * 1.2) + s_len * 0.3;

        torsos.push(PersonTorso {
            id: i,
            anchor_left,
            anchor_right,
            torso_box: TorsoBox { min_x, max_x, min_y, max_y },
            scale: s_len,
            spine_axis: (mid_s, (mid_s.0 + down_dx * t_len, mid_s.1 + down_dy * t_len)),
        });
    }

    if torsos.is_empty() {
        return group_isolated_breast_pairs(boxes);
    }

    // Bipartites Matching: (Person, Side) <-> NudeNet BreastBox
    struct MatchCandidate {
        person_idx: usize,
        side: Side,
        box_idx: usize,
        score: f64,
    }

    let mut candidates: Vec<MatchCandidate> = Vec::new();

    for (p_idx, torso) in torsos.iter().enumerate() {
        for (b_idx, b) in boxes.iter().enumerate() {
            // Liegt die Box grob im Torso-Bereich?
            let in_bb = b.cx >= torso.torso_box.min_x
                && b.cx <= torso.torso_box.max_x
                && b.cy >= torso.torso_box.min_y
                && b.cy <= torso.torso_box.max_y;
            if !in_bb {
                continue;
            }

            // Distanz zu linkem und rechtem Anker
            let d_left = ((b.cx - torso.anchor_left.0).powi(2) + (b.cy - torso.anchor_left.1).powi(2)).sqrt() / torso.scale;
            let d_right = ((b.cx - torso.anchor_right.0).powi(2) + (b.cy - torso.anchor_right.1).powi(2)).sqrt() / torso.scale;

            // Seitentrennung anhand der Spine-Achse:
            // Kreuzprodukt (B - MidS) x SpineDir
            let mid_s = torso.spine_axis.0;
            let spine_end = torso.spine_axis.1;
            let spine_dx = spine_end.0 - mid_s.0;
            let spine_dy = spine_end.1 - mid_s.1;
            let cross = (b.cx - mid_s.0) * spine_dy - (b.cy - mid_s.1) * spine_dx;

            // Unbedeckte Brüste werden durch ihren priority_weight bevorzugt
            let p_weight = b.kind.priority_weight();

            // Linke Seite Bewertung
            if d_left < 1.4 {
                let mut side_bonus = if cross < 0.0 { 0.25 } else { -0.25 };
                if d_left < d_right {
                    side_bonus += 0.2;
                }
                let score = (1.5 - d_left + side_bonus) * p_weight * b.score;
                if score > 0.3 {
                    candidates.push(MatchCandidate {
                        person_idx: p_idx,
                        side: Side::Left,
                        box_idx: b_idx,
                        score,
                    });
                }
            }

            // Rechte Seite Bewertung
            if d_right < 1.4 {
                let mut side_bonus = if cross > 0.0 { 0.25 } else { -0.25 };
                if d_right < d_left {
                    side_bonus += 0.2;
                }
                let score = (1.5 - d_right + side_bonus) * p_weight * b.score;
                if score > 0.3 {
                    candidates.push(MatchCandidate {
                        person_idx: p_idx,
                        side: Side::Right,
                        box_idx: b_idx,
                        score,
                    });
                }
            }
        }
    }

    // Nach bestem Score sortieren (Greedy Matching)
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let mut used_boxes = vec![false; boxes.len()];
    let mut person_left: Vec<Option<NipplePoint>> = vec![None; poses.len()];
    let mut person_right: Vec<Option<NipplePoint>> = vec![None; poses.len()];

    for cand in candidates {
        if used_boxes[cand.box_idx] {
            continue;
        }
        let p_id = torsos[cand.person_idx].id;
        match cand.side {
            Side::Left => {
                if person_left[p_id].is_some() {
                    continue;
                }
                let b = &boxes[cand.box_idx];
                used_boxes[cand.box_idx] = true;
                person_left[p_id] = Some(calculate_nipple_point(b, torsos[cand.person_idx].anchor_left));
            }
            Side::Right => {
                if person_right[p_id].is_some() {
                    continue;
                }
                let b = &boxes[cand.box_idx];
                used_boxes[cand.box_idx] = true;
                person_right[p_id] = Some(calculate_nipple_point(b, torsos[cand.person_idx].anchor_right));
            }
        }
    }

    // Landmarks 33 und 34 an die Posen anfügen
    for (i, pose) in poses.iter_mut().enumerate() {
        let left_pt = person_left[i].as_ref();
        let right_pt = person_right[i].as_ref();

        let l_coords = left_pt.map_or([0.0, 0.0, 0.0], |pt| [pt.x, pt.y, pt.conf]);
        let r_coords = right_pt.map_or([0.0, 0.0, 0.0], |pt| [pt.x, pt.y, pt.conf]);

        if pose.len() == 33 {
            pose.push(l_coords);
            pose.push(r_coords);
        } else if pose.len() >= 35 {
            pose[33] = l_coords;
            pose[34] = r_coords;
        }

        if left_pt.is_some() || right_pt.is_some() {
            results.push(NipplePair {
                person_id: Some(i),
                left: left_pt.cloned(),
                right: right_pt.cloned(),
            });
        }
    }

    // Gibt es unbenutzte Boxen (z. B. eine zweite Person ohne volles Skelett im Hintergrund)?
    let remaining_boxes: Vec<BreastBox> = boxes
        .iter()
        .enumerate()
        .filter(|(idx, _)| !used_boxes[*idx])
        .map(|(_, b)| b.clone())
        .collect();

    if !remaining_boxes.is_empty() {
        let isolated_pairs = group_isolated_breast_pairs(&remaining_boxes);
        results.extend(isolated_pairs);
    }

    results
}

/// Berechnet die genaue Brustwarzen-Position aus der NudeNet-Box und dem anatomischen Anker.
fn calculate_nipple_point(b: &BreastBox, anchor: (f64, f64)) -> NipplePoint {
    // Gemessene Position aus der Bildanalyse hat Vorrang vor der Schätzung.
    if let (Some((nx, ny, nc)), true) = (b.nip, b.kind.is_exposed()) {
        return NipplePoint { x: round(nx, 4), y: round(ny, 4), conf: round((nc * 0.5 + b.score * 0.5).clamp(0.5, 0.99), 3), exposed: true };
    }
    let (cx, cy) = if b.kind.is_exposed() {
        // Bei unbedeckter Brust: Das Zentrum der NudeNet-Box ist exakt die Brustwarze/Areola.
        // Feine anatomische Zentrierung: 85 % Box-Zentrum, 15 % Anker-Richtung.
        let x = b.cx * 0.85 + anchor.0 * 0.15;
        let y = b.cy * 0.85 + anchor.1 * 0.15;
        // Begrenzung innerhalb der Box
        (
            x.clamp(b.x + b.w * 0.15, b.x + b.w * 0.85),
            y.clamp(b.y + b.h * 0.15, b.y + b.h * 0.85),
        )
    } else {
        // Bei bedeckter Brust: Durch Schwerkraft/BH liegt der Nippel knapp unter der Boxmitte.
        let x = b.cx * 0.75 + anchor.0 * 0.25;
        let y = (b.y + b.h * 0.52) * 0.75 + anchor.1 * 0.25;
        (
            x.clamp(b.x + b.w * 0.2, b.x + b.w * 0.8),
            y.clamp(b.y + b.h * 0.2, b.y + b.h * 0.8),
        )
    };

    let conf = if b.kind.is_exposed() {
        (b.score * 0.95).clamp(0.6, 0.99)
    } else {
        (b.score * 0.70).clamp(0.4, 0.85)
    };

    NipplePoint {
        x: round(cx, 4),
        y: round(cy, 4),
        conf: round(conf, 3),
        exposed: b.kind.is_exposed(),
    }
}

/// Gruppiert isolierte Brüste (z. B. bei Nahaufnahmen ohne Gesicht/Körper) als Paare.
fn group_isolated_breast_pairs(boxes: &[BreastBox]) -> Vec<NipplePair> {
    let mut pairs = Vec::new();
    let mut used = vec![false; boxes.len()];

    for i in 0..boxes.len() {
        if used[i] {
            continue;
        }
        let b1 = &boxes[i];
        let mut best_j = None;
        let mut min_dist = 0.45; // Max. 45 % Bildbreite Abstand für ein Paar

        for j in (i + 1)..boxes.len() {
            if used[j] {
                continue;
            }
            let b2 = &boxes[j];
            let dx = (b1.cx - b2.cx).abs();
            let dy = (b1.cy - b2.cy).abs();
            // Brüste eines Paares liegen horizontal nebeneinander (dy ist deutlich kleiner als dx)
            if dx < min_dist && dy < 0.15 {
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < min_dist {
                    min_dist = dist;
                    best_j = Some(j);
                }
            }
        }

        if let Some(j) = best_j {
            used[i] = true;
            used[j] = true;
            let b2 = &boxes[j];
            // In Bildkoordinaten: kleineres x ist links im Bild (anatomisch rechte Brust bei Draufsicht),
            // größeres x ist rechts im Bild.
            let (left_box, right_box) = if b1.cx > b2.cx { (b1, b2) } else { (b2, b1) };

            pairs.push(NipplePair {
                person_id: None,
                left: Some(NipplePoint {
                    x: round(left_box.cx, 4),
                    y: round(left_box.cy, 4),
                    conf: round(if left_box.kind.is_exposed() { 0.9 } else { 0.65 }, 3),
                    exposed: left_box.kind.is_exposed(),
                }),
                right: Some(NipplePoint {
                    x: round(right_box.cx, 4),
                    y: round(right_box.cy, 4),
                    conf: round(if right_box.kind.is_exposed() { 0.9 } else { 0.65 }, 3),
                    exposed: right_box.kind.is_exposed(),
                }),
            });
        } else {
            // Einzelne Brust ohne Partner
            used[i] = true;
            pairs.push(NipplePair {
                person_id: None,
                left: Some(NipplePoint {
                    x: round(b1.cx, 4),
                    y: round(b1.cy, 4),
                    conf: round(if b1.kind.is_exposed() { 0.88 } else { 0.6 }, 3),
                    exposed: b1.kind.is_exposed(),
                }),
                right: None,
            });
        }
    }

    pairs
}

fn round(val: f64, decimals: u32) -> f64 {
    let factor = 10f64.powi(decimals as i32);
    (val * factor).round() / factor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nipple_pairing_single_person() {
        let mut poses = vec![vec![[0.5, 0.5, 0.9]; 33]];
        // Schultern setzen
        poses[0][11] = [0.6, 0.3, 0.95]; // Links
        poses[0][12] = [0.4, 0.3, 0.95]; // Rechts
        // Hüften setzen
        poses[0][23] = [0.58, 0.7, 0.9];
        poses[0][24] = [0.42, 0.7, 0.9];

        let boxes = vec![
            BreastBox {
                kind: BreastKind::ExposedFemale,
                score: 0.92,
                x: 0.53,
                y: 0.40,
                w: 0.08,
                h: 0.08,
                cx: 0.57,
                cy: 0.44,
                nip: None,
            },
            BreastBox {
                kind: BreastKind::CoveredFemale,
                score: 0.80,
                x: 0.39,
                y: 0.40,
                w: 0.08,
                h: 0.08,
                cx: 0.43,
                cy: 0.44,
                nip: None,
            },
        ];

        let results = reconcile_frame(&mut poses, &boxes);
        assert_eq!(results.len(), 1);
        let pair = &results[0];
        assert_eq!(pair.person_id, Some(0));
        assert!(pair.left.is_some());
        assert!(pair.right.is_some());
        assert!(pair.left.as_ref().unwrap().exposed);
        assert!(!pair.right.as_ref().unwrap().exposed);

        // Posen haben jetzt 35 Landmarks
        assert_eq!(poses[0].len(), 35);
        assert!(poses[0][33][2] > 0.5); // Linker Nippel vorhanden
        assert!(poses[0][34][2] > 0.5); // Rechter Nippel vorhanden
    }

    #[test]
    fn test_preference_exposed_over_covered() {
        let mut poses = vec![vec![[0.5, 0.5, 0.9]; 33]];
        poses[0][11] = [0.6, 0.3, 0.95];
        poses[0][12] = [0.4, 0.3, 0.95];
        poses[0][23] = [0.58, 0.7, 0.9];
        poses[0][24] = [0.42, 0.7, 0.9];

        // Zwei überlappende Boxen für die linke Brust: eine bedeckt, eine unbedeckt
        let boxes = vec![
            BreastBox {
                kind: BreastKind::CoveredFemale,
                score: 0.85,
                x: 0.53,
                y: 0.40,
                w: 0.08,
                h: 0.08,
                cx: 0.57,
                cy: 0.44,
                nip: None,
            },
            BreastBox {
                kind: BreastKind::ExposedFemale,
                score: 0.88,
                x: 0.535,
                y: 0.405,
                w: 0.07,
                h: 0.07,
                cx: 0.57,
                cy: 0.44,
                nip: None,
            },
        ];

        let results = reconcile_frame(&mut poses, &boxes);
        assert_eq!(results.len(), 1);
        let pair = &results[0];
        assert!(pair.left.as_ref().unwrap().exposed);
    }

    #[test]
    fn test_multi_person_pairing() {
        let mut poses = vec![
            vec![[0.3, 0.5, 0.9]; 33],
            vec![[0.7, 0.5, 0.9]; 33],
        ];
        // Person 0 (links im Bild)
        poses[0][11] = [0.35, 0.3, 0.95];
        poses[0][12] = [0.25, 0.3, 0.95];
        poses[0][23] = [0.34, 0.7, 0.9];
        poses[0][24] = [0.26, 0.7, 0.9];

        // Person 1 (rechts im Bild)
        poses[1][11] = [0.75, 0.3, 0.95];
        poses[1][12] = [0.65, 0.3, 0.95];
        poses[1][23] = [0.74, 0.7, 0.9];
        poses[1][24] = [0.66, 0.7, 0.9];

        let boxes = vec![
            // Brüste für Person 0
            BreastBox {
                kind: BreastKind::ExposedFemale,
                score: 0.90,
                x: 0.30,
                y: 0.40,
                w: 0.06,
                h: 0.06,
                cx: 0.33,
                cy: 0.43,
                nip: None,
            },
            BreastBox {
                kind: BreastKind::ExposedFemale,
                score: 0.90,
                x: 0.24,
                y: 0.40,
                w: 0.06,
                h: 0.06,
                cx: 0.27,
                cy: 0.43,
                nip: None,
            },
            // Brüste für Person 1
            BreastBox {
                kind: BreastKind::CoveredFemale,
                score: 0.85,
                x: 0.70,
                y: 0.40,
                w: 0.06,
                h: 0.06,
                cx: 0.73,
                cy: 0.43,
                nip: None,
            },
            BreastBox {
                kind: BreastKind::CoveredFemale,
                score: 0.85,
                x: 0.64,
                y: 0.40,
                w: 0.06,
                h: 0.06,
                cx: 0.67,
                cy: 0.43,
                nip: None,
            },
        ];

        let results = reconcile_frame(&mut poses, &boxes);
        assert_eq!(results.len(), 2);

        let p0 = results.iter().find(|p| p.person_id == Some(0)).unwrap();
        assert!(p0.left.is_some());
        assert!(p0.right.is_some());
        assert!(p0.left.as_ref().unwrap().x > 0.30);
        assert!(p0.right.as_ref().unwrap().x < 0.30);

        let p1 = results.iter().find(|p| p.person_id == Some(1)).unwrap();
        assert!(p1.left.is_some());
        assert!(p1.right.is_some());
        assert!(p1.left.as_ref().unwrap().x > 0.70);
        assert!(p1.right.as_ref().unwrap().x < 0.70);
    }
}
