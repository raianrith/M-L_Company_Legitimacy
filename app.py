import os
import io
import json
import time
import math
import tldextract
import requests
import pandas as pd
import streamlit as st
import whois
import dns.resolver
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from duckduckgo_search import DDGS
from openai import OpenAI

# =========================
# Utility / Signals
# =========================
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
        "search_hits_count": len(ddg_results),
    }
    return features

# =========================
# OpenAI verdict (Chat Completions JSON mode for broad compat)
# =========================
def openai_verdict(features: dict, model: str, max_retries: int = 3, retry_backoff: float = 2.0):
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

    last_err = None
    for attempt in range(max_retries):
        try:
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
            return json.loads(content)
        except Exception as e:
            last_err = e
            # simple backoff for rate limits/transients
            time.sleep(retry_backoff * (2 ** attempt))

    # Fallback if parsing fails consistently
    return {
        "trust_score": 50,
        "label": "Unclear – Needs Manual Review",
        "reasons": [f"Fallback due to error: {str(last_err)[:180]}"],
        "recommended_actions": ["Manual review"],
        "key_signals": {
            "domain_age_days": features.get("domain_whois_age_days"),
            "has_mx": features.get("email_mx_exists"),
            "domain_mismatch": features.get("domain_mismatch"),
            "http_title": features.get("http_title"),
            "http_status_code": features.get("http_status_code"),
            "search_hits_count": features.get("search_hits_count", 0),
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

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Lead Trust Screening (AI)", page_icon="✅", layout="wide")
st.title("🔎 Lead Trust Screening — AI Pre-Qualification")
st.caption("Screen single leads or a CSV of leads using OSINT signals + OpenAI.")

with st.sidebar:
    model = st.selectbox(
        "OpenAI model",
        ["gpt-4.1-mini", "gpt-4o-mini", "o4-mini", "gpt-4.1"],
        index=0
    )
    st.info("Set `OPENAI_API_KEY` in Streamlit Secrets or environment.")

tab_single, tab_batch = st.tabs(["Single Lead", "Batch CSV"])

# -------------------------
# Single Lead tab
# -------------------------
with tab_single:
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
            verdict = openai_verdict(feats, model=model)

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
                st.json({
                    "domain_age_days": key_signals.get("domain_age_days"),
                    "has_mx": key_signals.get("has_mx"),
                    "domain_mismatch": key_signals.get("domain_mismatch"),
                    "http_title": key_signals.get("http_title"),
                    "http_status_code": key_signals.get("http_status_code"),
                    "search_hits_count": key_signals.get("search_hits_count"),
                })

# -------------------------
# Batch CSV tab
# -------------------------
with tab_batch:
    st.markdown("**Upload CSV** with columns (case-insensitive ok):")
    st.code("company_name, website, contact_name, contact_email, message", language="text")

    file = st.file_uploader("Choose a CSV file", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, encoding_errors="ignore")

        # Normalize expected column names (flexible, case-insensitive)
        rename_map = {}
        cols_lower = {c.lower(): c for c in df.columns}
        for want in ["company_name", "website", "contact_name", "contact_email", "message"]:
            if want in cols_lower:
                rename_map[cols_lower[want]] = want
            else:
                # Try a few aliases
                for alias in [want.replace("_", " "), want.replace("_", ""), want.title().replace("_", " ")]:
                    if alias.lower() in cols_lower:
                        rename_map[cols_lower[alias.lower()]] = want
                        break
        df = df.rename(columns=rename_map)

        missing = [c for c in ["company_name", "website", "contact_name", "contact_email"] if c not in df.columns]
        if missing:
            st.warning(f"Missing columns in CSV: {missing}. You can still proceed; blanks will be treated as unknown.")

        st.write("Preview:")
        st.dataframe(df.head(10))

        run = st.button("Run Batch Screening", type="primary")
        if run:
            results = []
            progress = st.progress(0)
            status = st.empty()

            total = len(df)
            for i, row in df.iterrows():
                company_name = row.get("company_name", None)
                website = row.get("website", None)
                contact_name = row.get("contact_name", None)
                contact_email = row.get("contact_email", None)
                message = row.get("message", None)

                status.info(f"Processing {i+1}/{total}: {company_name or website or contact_email or 'lead'}")

                try:
                    feats = build_feature_bundle(company_name, website, contact_name, contact_email, message)
                    verdict = openai_verdict(feats, model=model)
                except Exception as e:
                    verdict = {
                        "trust_score": None,
                        "label": "Error",
                        "reasons": [str(e)[:180]],
                        "recommended_actions": ["Retry later / check inputs"],
                        "key_signals": {
                            "domain_age_days": None,
                            "has_mx": None,
                            "domain_mismatch": None,
                            "http_title": None,
                            "http_status_code": None,
                            "search_hits_count": None,
                        },
                    }

                out_row = dict(row)
                out_row["trust_score"] = verdict.get("trust_score")
                out_row["label"] = verdict.get("label")
                out_row["reasons"] = "; ".join(verdict.get("reasons") or [])
                out_row["recommended_actions"] = "; ".join(verdict.get("recommended_actions") or [])
                ks = verdict.get("key_signals") or {}
                out_row["sig_domain_age_days"] = ks.get("domain_age_days")
                out_row["sig_has_mx"] = ks.get("has_mx")
                out_row["sig_domain_mismatch"] = ks.get("domain_mismatch")
                out_row["sig_http_title"] = ks.get("http_title")
                out_row["sig_http_status_code"] = ks.get("http_status_code")
                out_row["sig_search_hits_count"] = ks.get("search_hits_count")

                results.append(out_row)

                # Lightweight pacing to reduce rate-limit risk on free tiers
                time.sleep(0.3)

                progress.progress((i + 1) / total)

            status.success("Batch complete.")
            out_df = pd.DataFrame(results)

            st.subheader("Results")
            st.dataframe(out_df)

            # Download button
            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV with verdicts",
                data=csv_bytes,
                file_name="lead_screening_results.csv",
                mime="text/csv"
            )

# Footer
st.caption("Note: Uncertain cases are intentionally routed to 'Manual Review' to minimize false positives.")
