//! Voten und Folgen, wie im Addon: Die Anfragen laufen im eingeloggten Browser auf dem Server,
//! mit den echten Cookies. Das Frontend schickt nur "Beitrag X up" bzw. "Nutzer Y folgen".

use axum::{
    Json,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use serde::Deserialize;
use serde_json::json;

use super::{browser, err, vpn::authorized};

#[derive(Deserialize)]
pub struct VoteBody {
    id: String,
    /// "up", "down" oder "none"
    state: String,
}

#[derive(Deserialize)]
pub struct FollowBody {
    user: String,
    follow: bool,
}

fn valid_name(s: &str) -> bool {
    !s.is_empty() && s.len() <= 40 && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Der Ausdruck liest den csrf_token-Cookie der Seite, wie das Addon.
const CSRF: &str = r#"(document.cookie.match(/(?:^|;\s*)csrf_token=([^;]+)/)||[])[1]"#;

fn failed(e: browser::BrowserError) -> Response {
    match e {
        browser::BrowserError::NeedsLogin(m) => err(
            StatusCode::BAD_GATEWAY,
            format!("{m}. Bitte im Browser neu bei Reddit einloggen (https://redswipe-login.kulbarts.com)."),
        ),
        browser::BrowserError::Unavailable(m) | browser::BrowserError::Failed(m) => err(StatusCode::BAD_GATEWAY, m),
    }
}

async fn run(expr: String) -> Response {
    if !browser::configured() {
        return err(StatusCode::SERVICE_UNAVAILABLE, "Der Browser ist nicht eingerichtet.");
    }
    match browser::eval_in_page("https://www.reddit.com/robots.txt", &expr).await {
        Ok(v) if v["ok"].as_bool() == Some(true) => Json(json!({ "ok": true })).into_response(),
        Ok(v) => {
            let why = v["error"].as_str().unwrap_or("Reddit hat die Aktion abgelehnt");
            err(StatusCode::BAD_GATEWAY, why.to_string())
        }
        Err(e) => failed(e),
    }
}

pub async fn vote(headers: HeaderMap, Json(b): Json<VoteBody>) -> Response {
    if let Err(r) = authorized(&headers) {
        return r;
    }
    let state = match b.state.as_str() {
        "up" => "UP",
        "down" => "DOWN",
        "none" => "NONE",
        _ => return err(StatusCode::BAD_REQUEST, "Unbekannter Vote-Zustand"),
    };
    if !valid_name(&b.id) {
        return err(StatusCode::BAD_REQUEST, "Ungültige Beitrags-ID");
    }
    let post_id = format!("t3_{}", b.id);
    let expr = format!(
        r#"(async()=>{{const c={CSRF};if(!c)return{{ok:false,error:'Kein CSRF-Token – im Browser einloggen'}};
try{{const r=await fetch('https://www.reddit.com/svc/shreddit/graphql',{{method:'POST',headers:{{accept:'application/json','content-type':'application/json'}},credentials:'include',
body:JSON.stringify({{operation:'UpdatePostVoteState',variables:{{input:{{postId:{post_id:?},voteState:{state:?}}}}},csrf_token:c}})}});
if(!r.ok)return{{ok:false,error:'Reddit antwortet mit HTTP '+r.status}};
const d=await r.json();const u=d&&d.data&&d.data.updatePostVoteState;
return u&&u.ok?{{ok:true}}:{{ok:false,error:'Reddit hat den Vote abgelehnt'}}}}catch(e){{return{{ok:false,error:String(e)}}}}}})()"#
    );
    run(expr).await
}

pub async fn follow(headers: HeaderMap, Json(b): Json<FollowBody>) -> Response {
    if let Err(r) = authorized(&headers) {
        return r;
    }
    if !valid_name(&b.user) {
        return err(StatusCode::BAD_REQUEST, "Ungültiger Nutzername");
    }
    let action = if b.follow { "sub" } else { "unsub" };
    let sr = format!("u_{}", b.user);
    let expr = format!(
        r#"(async()=>{{const c={CSRF};if(!c)return{{ok:false,error:'Kein CSRF-Token – im Browser einloggen'}};
try{{const f=new URLSearchParams();f.append('action',{action:?});f.append('sr_name',{sr:?});f.append('csrf_token',c);
const r=await fetch('https://www.reddit.com/api/subscribe',{{method:'POST',headers:{{'content-type':'application/x-www-form-urlencoded','x-csrf-token':c}},credentials:'include',body:f.toString()}});
if(r.ok)return{{ok:true}};const t=(await r.text()).slice(0,160);return{{ok:false,error:'Reddit antwortet mit HTTP '+r.status+': '+t}}}}catch(e){{return{{ok:false,error:String(e)}}}}}})()"#
    );
    run(expr).await
}
