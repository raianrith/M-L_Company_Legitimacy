import os
import re
import json
import time
import socket
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
    out = {
        "status_code": None,
        "title": None,
        "content_length": None,
        "server": None
    }
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
            clean.append({
                "title": r.get("title"),
                "href": r.get("href"),
                "body": r.get("body")
            })
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


def openai_verdict(features: dict, model: str):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    schema = {
        "name": "lead_verdict",
        "schema": {
            "type": "object",
            "properties": {
                "trust_score": {"type": "number", "minimum": 0, "maximum": 100},
                "label": {"type": "string", "enum": ["Likely Legit", "Unclear – Needs Manual Review", "Likely Spam/Fake"]},
                "reasons": {"type": "array", "items": {"type": "string"}},
                "recommended_actions": {"type": "array", "items": {"type": "string"}},
                "key_signals": {
                    "type": "object",
                    "properties": {
                        "domain_age_days": {"type": "integer", "nullable": True},
                        "has_mx": {"type": "boolean", "nullable": True},
                        "domain_mismatch": {"type": "boolean", "nullable": True},
                        "http_title": {"type": "string", "nullable": True},
                        "http_status_code": {"type": "integer", "nullable": True},
                        "search_hits_count": {"type": "integer"}
                    },
                    "required": ["search_hits_count"]
                }
            },
            "required": ["trust_score", "label", "reasons", "recommended_actions", "key_signals"],
            "additionalProperties": False
        }
    }

    system_msg = (
        "You are an anti-fraud analyst for B2B lead forms. "
        "Given OSINT features and search snippets, decide if the lead is legitimate. "
        "Prefer conservative false-negative bias: if uncertain, mark 'Unclear – Needs Manual Review'."
    )

    user_msg = (
        "Return a JSON verdict following the provided schema. "
        f"\n\nFeatures JSON:\n{json.dumps(features, ensure_ascii=False)}"
    )

    resp = client.responses.create(
        model=model,
        input=[{"role":"system","content":system_msg},
               {"role":"user","content":user_msg}],
        response_format={"type":"json_schema","json_schema":schema},
        temperature=0.1
    )
    try:
        content = resp.output[0].content[0].text
        return json.loads(content)
    except Exception:
        return None


def badge_color(score):
    if score is None:
        return "#999999"
    if score >= 75:
        return "#16a34a"
    if score >= 55:
        return "#ca8a04"
    return "#dc2626"


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Lead Trust Screening (AI)", page_icon="✅", layout="wide")

st.title("🔎 Lead Trust Screening — AI Pre-Qualification")
st.caption("Screens lead form submissions with OSINT checks + OpenAI verdict.")

with st.sidebar:
    model = st.selectbox("OpenAI model", ["gpt-4.1-mini","gpt-4o-mini","o4-mini","gpt-4.1"], index=0)

col1, col2 = st.columns([1,1])
with col1:
    company_name = st.text_input("Company Name*", placeholder="Acme Thermal Systems")
    website = st.text_input("Company Website", placeholder="acme-thermal.com")
    contact_name = st.text_input("Contact Name", placeholder="Jane Smith")

with col2:
    contact_email = st.text_input("Contact Email", placeholder="jane.smith@acme-thermal.com")
    message = st.text_area("Message (optional)", placeholder="We need an industrial oven...")

submitted = st.button("Run Trust Check", type="primary")

if submitted:
    with st.spinner("Collecting OSINT signals…"):
        feats = build_feature_bundle(company_name, website, contact_name, contact_email, message)

    st.subheader("Signals")
    st.json(feats)

    with st.spinner("Asking OpenAI for a verdict…"):
        verdict = openai_verdict(feats, model=model)

    st.subheader("AI Verdict")
    if not verdict:
        st.error("Could not parse AI response. Check API key & model.")
    else:
        st.json(verdict)
