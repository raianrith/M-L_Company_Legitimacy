import os
import json
import tldextract
import requests
import streamlit as st
import whois
import dns.resolver
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from duckduckgo_search import DDGS
from openai import OpenAI

# ---------------------------
# Helpers
# ---------------------------
def normalize_url(url_or_domain: str):
    if not url_or_domain:
        return None
    s = url_or_domain.strip()
    if not s:
        return None
    if not s.startswith("http"):
        s = "https://" + s
    try:
        _ = requests.utils.urlparse(s)
        return s
    except Exception:
        return None

def domain_from_any(s: str):
    if not s:
        return None
    if "@" in s:
        s = s.split("@", 1)[1]
    ext = tldextract.extract(s)
    if not ext.registered_domain:
        return None
    return ext.registered_domain.lower()

def has_mx_records(domain: str) -> bool:
    try:
        answers = dns.resolver.resolve(domain, "MX")
        return len(answers) > 0
    except Exception:
        return False

def whois_age_days(domain: str):
    try:
        w = whois.whois(domain)
        created = w.creation_date
        if isinstance(created, list):
            created = created[0]
        if not created:
            return None
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - created).days
    except Exception:
        return None

def http_probe(url: str, timeout=10):
    out = {"status_code": None, "title": None, "content_length": None, "server": None}
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        out["status_code"] = r.status_code
        out["server"] = r.headers.get("server")
        text = r.text or ""
        out["content_length"] = len(text)
        soup = BeautifulSoup(text, "html.parser")
        title = soup.find("title")
        out["title"] = title.get_text(strip=True) if title else None
    except Exception:
        pass
    return out

def duckduckgo_snippets(query: str, max_results=6):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        clean = []
        for r in results:
            clean.append({"title": r.get("title"), "href": r.get("href"), "body": r.get("body")})
        return clean
    except Exception:
        return []

def build_feature_bundle(company_name, website, contact_name, contact_email, message):
    site_domain = domain_from_any(website or "")
    email_domain = domain_from_any(contact_email or "")
    url = normalize_url(website or (site_domain or ""))

    mx_ok = has_mx_records(email_domain) if email_domain else None
    age_days = whois_age_days(site_domain) if site_domain else None
    http_info = http_probe(url) if url else {}

    domain_mismatch = (site_domain and email_domain and site_domain != email_domain)

    q_base = company_name or site_domain or email_domain
    ddg_results = duckduckgo_snippets(q_base) if q_base else []

    features = {
        "input_company_name": company_name,
        "input_website": website,
        "input_contact_name": contact_name,
        "input_contact_email": contact_email,
        "input_message": message,
        "site_domain": site_domain,
        "email_domain": email_domain,
        "domain_mismatch": bool(domain_mismatch),
        "email_mx_exists": mx_ok,
        "domain_whois_age_days": age_days,
        "http_status_code": http_info.get("status_code"),
        "http_server_header": http_info.get("server"),
        "http_title": http_info.get("title"),
        "http_content_length": http_info.get("content_length"),
        "duckduckgo_results": ddg_results,
    }
    return features

# --------- FIXED: use Chat Completions JSON mode (not Responses API) ----------
def openai_verdict(features: dict, model: str):
    """
    Uses Chat Completions with response_format={'type':'json_object'}
    to return a structured verdict compatible across more openai versions.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var is not set.")

    client = OpenAI(api_key=api_key)

    system_msg = (
        "You are an anti-fraud analyst for B2B lead forms. "
        "Given OSINT features and search snippets, decide if the lead is legitimate. "
        "Prefer a conservative bias: if uncertain, label as 'Unclear – Needs Manual Review'. "
        "STRICTLY return compact JSON matching this schema: "
        "{"
        "\"trust_score\": number (0-100), "
        "\"label\": one of [\"Likely Legit\",\"Unclear – Needs Manual Review\",\"Likely Spam/Fake\"], "
        "\"reasons\": [string,...], "
        "\"recommended_actions\": [string,...], "
        "\"key_signals\": {"
        "\"domain_age_days\": number|null, "
        "\"has_mx\": boolean|null, "
        "\"domain_mismatch\": boolean|null, "
        "\"http_title\": string|null, "
        "\"http_status_code\": number|null, "
        "\"search_hits_count\": number"
        "}"
        "}"
    )

    user_msg = (
        "Features JSON:\n"
        f"{json.dumps(features, ensure_ascii=False)}\n\n"
        "Return ONLY the JSON object, no extra text."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except Exception:
        # Fallback: wrap minimal object if parsing fails
        return {
            "trust_score": 50,
            "label": "Unclear – Needs Manual Review",
            "reasons": ["Model returned non-JSON or unparsable JSON."],
            "recommended_actions": ["Manual review"],
            "key_signals": {
                "domain_age_days": features.get("domain_whois_age_days"),
                "has_mx": features.get("email_mx_exists"),
                "domain_mismatch": features.get("domain_mismatch"),
                "http_title": features.get("http_title"),
                "http_status_code": features.get("http_status_code"),
                "search_hits_count": len(features.get("duckduckgo_results") or []),
            },
        }

def badge_color(score):
    if score is None:
        return "#999999"
    if score >= 75:
        return "#16a34a"  # green
    if score >= 55:
        return "#ca8a04"  # amber
    return "#dc2626"      # red

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Lead Trust Screening (AI)", page_icon="✅", layout="wide")

st.title("🔎 Lead Trust Screening — AI Pre-Qualification")
st.caption("Screens lead form submissions with OSINT checks + OpenAI verdict.")

with st.sidebar:
    model = st.selectbox("OpenAI model", ["gpt-4.1-mini","gpt-4o-mini","o4-mini","gpt-4.1"], index=0)
    st.markdown("Set your `OPENAI_API_KEY` in Streamlit secrets or env vars.")

col1, col2 = st.columns([1,1])
with col1:
    company_name = st.text_input("Company Name*", placeholder="Acme Thermal Systems")
    website = st.text_input("Company Website", placeholder="acme-thermal.com")
    contact_name = st.text_input("Contact Name", placeholder="Jane Smith")

with col2:
    contact_email = st.text_input("Contact Email", placeholder="jane.smith@acme-thermal.com")
    message = st.text_area("Message (optional)", placeholder="We need an industrial oven...")

if st.button("Run Trust Check", type="primary"):
    with st.spinner("Collecting OSINT signals…"):
        feats = build_feature_bundle(company_name, website, contact_name, contact_email, message)

    st.subheader("Signals")
    st.json(feats)

    with st.spinner("Asking OpenAI for a verdict…"):
        # Ensure search_hits_count is always present for the model
        if "duckduckgo_results" in feats:
            search_hits = len(feats.get("duckduckgo_results") or [])
        else:
            search_hits = 0

        # Add a mirror field to help the model
        feats_for_model = dict(feats)
        feats_for_model.setdefault("search_hits_count", search_hits)

        verdict = openai_verdict(feats_for_model, model=model)

    st.subheader("AI Verdict")
    if not verdict:
        st.error("Could not parse AI response. Check API key & model.")
    else:
        score = verdict.get("trust_score")
        label = verdict.get("label") or "—"
        reasons = verdict.get("reasons") or []
        actions = verdict.get("recommended_actions") or []
        key_signals = verdict.get("key_signals") or {}
        color = badge_color(score)

        st.markdown(
            f"""
<div style="border:1px solid #e5e7eb; border-radius:12px; padding:16px; display:flex; align-items:center; gap:16px;">
  <div style="width:110px; text-align:center;">
    <div style="font-size:14px; color:#6b7280; margin-bottom:6px;">Trust Score</div>
    <div style="font-size:32px; font-weight:700;">{int(score) if isinstance(score,(int,float)) else '—'}</div>
  </div>
  <div style="flex:1;">
    <span style="background:{color}22; color:{color}; padding:6px 10px; border-radius:999px; font-weight:600;">{label}</span>
    <div style="margin-top:10px; color:#374151;">
      Reasons:
      <ul>
        {''.join(f'<li>{r}</li>' for r in reasons[:6])}
      </ul>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True
        )

        c1, c2 = st.columns([1,1])
        with c1:
            st.markdown("**Recommended next actions**")
            if actions:
                st.markdown("\n".join([f"- {a}" for a in actions]))
            else:
                st.write("—")
        with c2:
            st.markdown("**Key signals**")
            # Ensure search_hits_count present in key_signals for display
            ksig = {
                "domain_age_days": key_signals.get("domain_age_days", feats.get("domain_whois_age_days")),
                "has_mx": key_signals.get("has_mx", feats.get("email_mx_exists")),
                "domain_mismatch": key_signals.get("domain_mismatch", feats.get("domain_mismatch")),
                "http_title": key_signals.get("http_title", feats.get("http_title")),
                "http_status_code": key_signals.get("http_status_code", feats.get("http_status_code")),
                "search_hits_count": key_signals.get("search_hits_count", len(feats.get("duckduckgo_results") or [])),
            }
            st.json(ksig)

    st.caption("Note: This is a screening tool; uncertain cases are routed to Manual Review.")
