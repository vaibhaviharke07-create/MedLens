"""
MedLens - Scan. Understand. Stay Informed.
Hackathon prototype: informational assistant, NOT a medical diagnosis/treatment system.
"""

import html
import os
import re
from pathlib import Path

import pandas as pd
import pytesseract
 
import requests
import streamlit as st
from PIL import Image, ImageOps
from rapidfuzz import fuzz

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")
else:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

DATA_PATH = Path(__file__).parent / "data" / "cdsco_nsq.csv"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
LOW_CONF_THRESHOLD = 60

SAFETY_NOTICE = (
    "MedLens is an informational assistant and is not a substitute for a doctor or "
    "pharmacist. Do not start, stop, or change any medicine based only on this "
    "application. Always verify important medical information with a qualified "
    "healthcare professional."
)

# --------------------------------------------------------------------------- #
# Page + styling
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="MedLens", page_icon="🩺", layout="wide")

st.markdown(
    """
<style>
:root { --ink:#0f2a43; --muted:#5b6b7b; --brand:#0d7c8f; --line:#e3e9ef; --bg:#f6f9fb; }
.block-container { max-width: 1100px; padding-top: 1.5rem; }
html, body, [class*="css"] { font-family: 'Inter','Segoe UI',system-ui,sans-serif; }
.hero { background: linear-gradient(135deg,#0f2a43 0%,#0d7c8f 100%); color:#fff;
        padding:2.2rem 2rem; border-radius:20px; box-shadow:0 6px 24px rgba(15,42,67,.18); }
.hero h1 { margin:0; font-size:2.4rem; letter-spacing:-.5px; color:#fff; }
.hero .tag { font-size:1.15rem; opacity:.95; margin:.2rem 0 .8rem; font-weight:500; }
.hero p { margin:0; opacity:.88; max-width:760px; line-height:1.55; }
.flow { display:flex; flex-wrap:wrap; gap:.4rem; margin:1rem 0 .5rem; }
.flow span { background:#fff; border:1px solid var(--line); border-radius:999px;
             padding:.3rem .8rem; font-size:.82rem; color:var(--ink); }
.card { background:#fff; border:1px solid var(--line); border-radius:16px;
        padding:1.2rem 1.3rem; box-shadow:0 2px 10px rgba(15,42,67,.05); margin-bottom:1rem; }
.card h4 { margin:0 0 .6rem; color:var(--ink); }
.kv { display:grid; grid-template-columns: 170px 1fr; gap:.35rem .8rem; font-size:.95rem; }
.kv .k { color:var(--muted); } .kv .v { color:var(--ink); font-weight:500; word-break:break-word; }
.warn { border-left:5px solid #d9822b; background:#fff8ef; }
.info { border-left:5px solid #0d7c8f; background:#f0fafb; }
.neutral { border-left:5px solid #8a97a5; background:#f7f9fb; }
.notice { border:1.5px solid #c0392b; background:#fff5f4; color:#7a1f16;
          border-radius:14px; padding:1rem 1.2rem; font-weight:500; }
.section { font-size:.78rem; letter-spacing:.12em; color:var(--brand); font-weight:700;
           margin:1.6rem 0 .5rem; text-transform:uppercase; }
.small { color:var(--muted); font-size:.85rem; }
</style>
""",
    unsafe_allow_html=True,
)

esc = lambda s: html.escape(str(s)) if s not in (None, "") else "—"


def kv_card(title, rows, css="", note=None):
    body = "".join(f'<div class="k">{esc(k)}</div><div class="v">{esc(v)}</div>' for k, v in rows)
    note_html = f'<p class="small" style="margin-top:.8rem">{note}</p>' if note else ""
    st.markdown(
        f'<div class="card {css}"><h4>{title}</h4><div class="kv">{body}</div>{note_html}</div>',
        unsafe_allow_html=True,
    )


def section(label):
    st.markdown(f'<div class="section">{label}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #
def run_ocr(image: Image.Image):
    """Returns (text, mean_confidence or None). Raises if Tesseract is missing."""
    img = ImageOps.exif_transpose(image).convert("L")
    if max(img.size) < 1600:  # upscale small images for better OCR
        scale = 1600 / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    img = ImageOps.autocontrast(img)

    cfg = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(img, config=cfg)
    data = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
    confs = [float(c) for c, t in zip(data["conf"], data["text"]) if str(t).strip() and float(c) >= 0]
    mean_conf = sum(confs) / len(confs) if confs else None
    return text.strip(), mean_conf


# --------------------------------------------------------------------------- #
# Information extraction (regex, never guesses)
# --------------------------------------------------------------------------- #
BATCH_RE = re.compile(
    r"(?:batch\s*(?:no|number|num|#)?|b\s*\.\s*no|lot\s*(?:no|number|num|#)?)"
    r"\s*[.:#\-]*\s*([A-Z0-9][A-Z0-9\- ]{2,18})",
    re.IGNORECASE,
)
DATE_TOKEN = r"(\d{1,2}\s*[/\-.]\s*\d{2,4}|[A-Za-z]{3,9}[\s.\-]*\d{2,4})"
MFG_RE = re.compile(r"(?:mfg|mfd|manufactur\w*)\.?\s*(?:date|dt)?\s*[:.\-]*\s*" + DATE_TOKEN, re.I)
EXP_RE = re.compile(r"(?:exp|expiry|expires?)\.?\s*(?:date|dt)?\s*[:.\-]*\s*" + DATE_TOKEN, re.I)
STRENGTH_RE = re.compile(
    r"\b(\d+(?:\.\d+)?\s?(?:mg|mcg|µg|g|ml|iu|%)(?:\s?/\s?\d*\s?(?:ml|mg|g|tablet|5\s?ml))?)", re.I
)
FORMS = ["tablet", "tablets", "capsule", "capsules", "syrup", "suspension", "injection",
         "ointment", "cream", "gel", "drops", "lotion", "powder", "inhaler", "sachet", "solution"]
MFR_RE = re.compile(
    r"(?:manufactured|mfd|mfg|marketed|made)\s*(?:by|at)?\s*[:\-]?\s*([^\n]{3,80})", re.I
)
SKIP_WORDS = re.compile(
    r"batch|b\.?\s*no|lot|mfg|mfd|exp|date|price|mrp|m\.r\.p|store|keep|dosage|each|contains|"
    r"manufactured|marketed|schedule|rx|warning|children|license|lic\b", re.I)


def normalize_batch(raw: str) -> str:
    raw = raw.strip().upper()
    # cut at second whitespace-separated chunk that looks like a new label (e.g. "MFG")
    raw = re.split(r"\s+(?:MFG|MFD|EXP|DATE|MRP|PRICE)\b", raw)[0]
    return re.sub(r"\s+", "", raw).strip("-./")


def extract_batch(text: str):
    for m in BATCH_RE.finditer(text):
        cand = normalize_batch(m.group(1))
        if len(cand) >= 3 and re.search(r"\d", cand):
            return cand
    return None


def extract_info(text: str) -> dict:
    info = {k: None for k in
            ["product_name", "strength", "dosage_form", "manufacturer",
             "batch_no", "manufacturing_date", "expiry_date"]}
    info["batch_no"] = extract_batch(text)
    if m := MFG_RE.search(text):
        info["manufacturing_date"] = re.sub(r"\s+", "", m.group(1)).upper()
    if m := EXP_RE.search(text):
        info["expiry_date"] = re.sub(r"\s+", "", m.group(1)).upper()
    if m := STRENGTH_RE.search(text):
        info["strength"] = re.sub(r"\s+", " ", m.group(1)).strip()
    low = text.lower()
    for f in FORMS:
        if re.search(rf"\b{f}\b", low):
            info["dosage_form"] = f.capitalize()
            break
    if m := MFR_RE.search(text):
        info["manufacturer"] = m.group(1).strip(" .:-")
    # Product name: first clean alphabetic line (best-guess heuristic, flagged in UI)
    for line in text.splitlines():
        ln = line.strip()
        letters = re.sub(r"[^A-Za-z]", "", ln)
        if len(letters) >= 4 and not SKIP_WORDS.search(ln) and len(ln) <= 60:
            info["product_name"] = ln
            break
    return info


# --------------------------------------------------------------------------- #
# CDSCO dataset lookup
# --------------------------------------------------------------------------- #
@st.cache_data
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH, dtype=str).fillna("")


def key(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def cdsco_lookup(info: dict) -> dict:
    df = load_dataset()
    out = {"status": "none", "records": [], "dataset_size": len(df)}
    if df.empty:
        out["status"] = "dataset_missing"
        return out

    batch = info.get("batch_no")
    if batch:
        hits = df[df["batch_no"].map(key) == key(batch)]
        if not hits.empty:
            out["status"], out["records"] = "exact", hits.to_dict("records")
            return out

    # Conservative fuzzy matching: clearly labelled "possible"
    name, mfr = info.get("product_name"), info.get("manufacturer")
    if name:
        cands = []
        for _, r in df.iterrows():
            p = fuzz.token_set_ratio(name.lower(), r["product_name"].lower())
            m = fuzz.token_set_ratio(mfr.lower(), r["manufacturer"].lower()) if mfr else 0
            if (p >= 92 and m >= 85) or (p >= 95 and not mfr):
                cands.append((p, m, r.to_dict()))
        if cands:
            cands.sort(key=lambda x: -(x[0] + x[1]))
            out["status"] = "possible"
            out["records"] = [c[2] for c in cands[:3]]
    return out


# --------------------------------------------------------------------------- #
# Gemini (REST) with clean error handling
# --------------------------------------------------------------------------- #
GEMINI_ERRORS = {
    400: "AI service configuration needs to be checked.",
    401: "AI authentication or API access needs to be checked.",
    403: "AI authentication or API access needs to be checked.",
    429: "AI explanation is temporarily unavailable because the AI service has reached its current usage limit.",
    500: "The AI service is temporarily unavailable. Please try again later.",
    503: "The AI service is temporarily unavailable. Please try again later.",
}

SYSTEM_RULES = """You are MedLens, an informational medicine-label assistant.
STRICT RULES:
- Never diagnose, prescribe, or assume a medical condition.
- Never recommend starting, stopping, changing, replacing, or adjusting any medicine or dosage.
- Never calculate or alter dosage.
- Never invent information that is not in the extracted text. If something is missing or unclear, say it needs verification.
- Treat OCR text as possibly inaccurate; do not present uncertain OCR as fact.
- Use simple, plain language. Keep it concise.
Respond using exactly these Markdown headings:
### What is this medicine?
### What is it commonly used for?
### Important information
### What should I verify?
End by reminding the user to verify with a doctor or pharmacist."""


def gemini_explain(info: dict, ocr_text: str, low_conf: bool):
    """Returns (text or None, user_friendly_error or None). Never raises."""
    api_key = str(os.getenv("GEMINI_API_KEY", "")).replace("\ufeff", "").strip()

    prompt = (
        f"Extracted label fields (may be incomplete/inaccurate):\n{info}\n\n"
        f"Raw OCR text:\n\"\"\"\n{ocr_text[:3500]}\n\"\"\"\n\n"
        f"OCR clarity: {'LOW - flag uncertainty clearly' if low_conf else 'acceptable'}.\n"
        "Explain this medicine information to a non-expert."
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_RULES}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1000},
    }
    try:
        r = requests.post(url, json=body, headers={"x-goog-api-key": api_key,
                                                   "Content-Type": "application/json"}, timeout=40)
    except requests.exceptions.Timeout:
        return None, "The AI service timed out. Please try again later."
    except requests.exceptions.RequestException:
        return None, "Could not reach the AI service. Please check your connection and try again."

    if r.status_code != 200:
        return None, GEMINI_ERRORS.get(r.status_code, "AI explanation is temporarily unavailable.")
    try:
        parts = r.json()["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts).strip()
        return (text, None) if text else (None, "The AI service returned no explanation. Please try again.")
    except (KeyError, IndexError, ValueError):
        return None, "The AI service returned an unexpected response. Please try again later."


# --------------------------------------------------------------------------- #
# Prescription cross-check
# --------------------------------------------------------------------------- #
RX_LINE = re.compile(
    r"\b(?:tab|tablet|cap|capsule|syp|syrup|inj|injection|oint|drops)\.?\s+"
    r"([A-Za-z][A-Za-z0-9\-]{2,}(?:\s+[A-Za-z0-9\-]{2,})?)\s*(\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml))?",
    re.I,
)


def prescription_crosscheck(rx_text: str, rx_conf, info: dict) -> dict:
    meds = [(m.group(1).strip(), (m.group(2) or "").strip()) for m in RX_LINE.finditer(rx_text)]
    flags = []
    if not meds:
        return {"meds": [], "flags": ["Medicine information is unreadable or no medicine names were identified in the prescription."]}
    if rx_conf is not None and rx_conf < LOW_CONF_THRESHOLD:
        flags.append("Prescription image clarity is low; some names may be misread.")

    name, strength = info.get("product_name"), info.get("strength")
    if not name:
        flags.append("Scanned medicine name could not be read, so it cannot be compared.")
    else:
        best = max(meds, key=lambda m: fuzz.partial_ratio(m[0].lower(), name.lower()))
        score = fuzz.partial_ratio(best[0].lower(), name.lower())
        if score < 80:
            flags.append(f"Medicine name does not clearly match any prescription entry "
                         f"(closest: “{best[0]}”, similarity {score:.0f}%).")
        elif best[1] and strength and key(best[1]) != key(strength.split("/")[0]):
            flags.append(f"Strength appears different: prescription “{best[1]}” vs scanned “{strength}”.")
    if not info.get("batch_no"):
        flags.append("Batch information is missing from the scanned package.")
    return {"meds": meds, "flags": flags}


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.markdown(
    """
<div class="hero">
  <h1>🩺 MedLens</h1>
  <div class="tag">Scan. Understand. Stay Informed.</div>
  <p>An AI-powered medicine information and verification assistant that helps users understand
  medicine labels and check identified batch information against the CDSCO dataset included in MedLens.</p>
</div>
<div class="flow">
  <span>① Upload Image</span><span>② OCR Extraction</span><span>③ Medicine Info</span>
  <span>④ CDSCO Check</span><span>⑤ AI Explanation</span><span>⑥ Report</span><span>⑦ Human Verification</span>
</div>
""",
    unsafe_allow_html=True,
)

section("Step 1 · Upload")
st.markdown('<div class="card"><h4>Scan a Medicine</h4>'
            '<span class="small">Upload a clear photo of a medicine label, package, or prescription.</span></div>',
            unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    med_file = st.file_uploader("Medicine label / package", type=["png", "jpg", "jpeg", "webp"], key="med")
    if med_file:
        st.image(med_file, caption="Medicine image preview", use_container_width=True)
with c2:
    rx_file = st.file_uploader("Prescription (optional, for cross-check)",
                               type=["png", "jpg", "jpeg", "webp"], key="rx")
    if rx_file:
        st.image(rx_file, caption="Prescription preview", use_container_width=True)

if st.button("Analyze Medicine", type="primary", disabled=med_file is None):
    result = {}
    try:
        with st.spinner("Running OCR..."):
            text, conf = run_ocr(Image.open(med_file))
            result.update(text=text, conf=conf, info=extract_info(text))
            if rx_file:
                rx_text, rx_conf = run_ocr(Image.open(rx_file))
                result.update(rx_text=rx_text, rx_conf=rx_conf)
    except pytesseract.TesseractNotFoundError:
        st.error("Tesseract OCR is not installed or not on PATH. See the setup notes (set TESSERACT_CMD if needed).")
        st.stop()
    except Exception:
        st.error("The image could not be processed. Please try another clear image.")
        st.stop()

    low = (result["conf"] is None) or result["conf"] < LOW_CONF_THRESHOLD or not result["text"]
    result["low_conf"] = low
    result["cdsco"] = cdsco_lookup(result["info"])
    with st.spinner("Generating AI explanation..."):
        result["ai"], result["ai_err"] = gemini_explain(result["info"], result["text"], low)
    if rx_file:
        result["rx"] = prescription_crosscheck(result["rx_text"], result["rx_conf"], result["info"])
    st.session_state["result"] = result

res = st.session_state.get("result")
if res:
    info, cd, low = res["info"], res["cdsco"], res["low_conf"]

    # ---------------- Scan results ----------------
    section("Scan Results")
    if low:
        st.warning("Some information could not be read confidently. Please verify it from the original package.")
    left, right = st.columns(2)
    with left:
        kv_card("Medicine Information", [
            ("Product name", info["product_name"]), ("Strength", info["strength"]),
            ("Dosage form", info["dosage_form"]), ("Manufacturer", info["manufacturer"]),
            ("Batch number", info["batch_no"]), ("Manufacturing date", info["manufacturing_date"]),
            ("Expiry date", info["expiry_date"]),
        ], note="Fields marked “—” could not be read. Product name is a best guess from label text — please verify.")
    with right:
        st.markdown('<div class="card"><h4>Extracted Text (OCR)</h4></div>', unsafe_allow_html=True)
        with st.expander("View raw OCR text", expanded=False):
            st.code(res["text"] or "(No text detected)", language=None)
        conf = res["conf"]
        st.caption(f"OCR confidence: {conf:.0f}%" if conf is not None else "OCR confidence: unavailable")

    if info["batch_no"]:
        st.success(f"Detected Batch Number: {info['batch_no']}")
    else:
        st.info("Batch number could not be detected. Please verify the package manually.")

    # ---------------- CDSCO ----------------
    section("CDSCO Verification")
    if cd["status"] == "exact":
        r = cd["records"][0]
        kv_card("⚠️ Official Quality Alert Record Match", [
            ("Product", r["product_name"]), ("Batch Number", r["batch_no"]),
            ("Manufacturer", r["manufacturer"]), ("Manufacturing Date", r["manufacturing_date"]),
            ("Expiry Date", r["expiry_date"]), ("NSQ Result", r["nsq_result"]),
            ("Reporting Source", r["reporting_source"]), ("Reporting Lab", r["reporting_lab"]),
            ("Alert Month", r["alert_month"]),
        ], css="warn",
            note="This batch matches an official CDSCO quality-alert record included in the MedLens dataset. "
                 "Please verify the information with a pharmacist, manufacturer, or appropriate authority "
                 "before making any decision.")
    elif cd["status"] == "possible":
        st.warning("Possible match only (based on product/manufacturer similarity, not an exact batch match). "
                   "Please verify manually.")
        for r in cd["records"]:
            kv_card("Possible Match — Not Confirmed", [
                ("Product", r["product_name"]), ("Batch in record", r["batch_no"]),
                ("Manufacturer", r["manufacturer"]), ("NSQ Result", r["nsq_result"]),
                ("Alert Month", r["alert_month"]),
            ], css="neutral", note="The batch number on your package did not exactly match this record.")
    elif cd["status"] == "dataset_missing":
        st.error("CDSCO dataset file (data/cdsco_nsq.csv) was not found.")
    else:
        kv_card("No matching record found in the MedLens CDSCO dataset.", [
            ("Dataset records checked", cd["dataset_size"]),
            ("Batch checked", info["batch_no"] or "Not detected"),
        ], css="info",
            note="A no-match only means no matching record exists in the dataset available to this "
                 "application. It is not a confirmation of quality or authenticity.")

    # ---------------- AI ----------------
    section("AI Explanation")
    if res["ai"]:
        st.markdown(f'<div class="card info">{""}</div>', unsafe_allow_html=True)
        st.markdown(res["ai"])
    else:
        st.warning(res["ai_err"])
        st.caption("OCR, batch extraction, CDSCO verification and the report below still work without AI.")

    # ---------------- Prescription ----------------
    section("Prescription Cross-Check")
    if "rx" in res:
        rx = res["rx"]
        if rx["meds"]:
            st.markdown("**Medicines identified in prescription:** " +
                        ", ".join(f"{n} {s}".strip() for n, s in rx["meds"]))
        if rx["flags"]:
            st.warning("Possible discrepancy detected — please verify with your doctor or pharmacist.")
            for f in rx["flags"]:
                st.markdown(f"- {f}")
        else:
            st.info("No discrepancies detected by automated comparison. This does not replace "
                    "verification by your doctor or pharmacist.")
    else:
        st.caption("No prescription uploaded. Upload one above to enable cross-checking.")

    # ---------------- Report ----------------
    section("Verification Report")
    st.markdown("## MedLens Verification Report")
    r1, r2 = st.columns(2)
    with r1:
        kv_card("Scan Status", [
            ("Image received", "Yes"), ("OCR completed", "Yes"),
            ("OCR clarity", "Low — verify manually" if low else "Acceptable"),
        ])
        cd_label = {"exact": "Match found", "possible": "Possible match (unconfirmed)",
                    "none": "No matching record", "dataset_missing": "Dataset unavailable"}[cd["status"]]
        kv_card("CDSCO Check", [("Result", cd_label), ("Dataset", "Local file: cdsco_nsq.csv"),
                                ("Records in dataset", cd["dataset_size"])])
    with r2:
        kv_card("Medicine Information", [
            ("Product", info["product_name"]), ("Strength", info["strength"]),
            ("Dosage form", info["dosage_form"]), ("Manufacturer", info["manufacturer"]),
            ("Batch", info["batch_no"]), ("Mfg date", info["manufacturing_date"]),
            ("Expiry", info["expiry_date"]),
        ])
    todo = [f"{label} was not detected — check the package." for label, k in
            [("Product name", "product_name"), ("Strength", "strength"), ("Manufacturer", "manufacturer"),
             ("Batch number", "batch_no"), ("Manufacturing date", "manufacturing_date"),
             ("Expiry date", "expiry_date")] if not info[k]]
    if low:
        todo.insert(0, "OCR clarity was low — confirm all details against the physical package.")
    st.markdown("**Items requiring verification**")
    for t in todo or ["No unreadable fields, but please still verify everything on the physical package."]:
        st.markdown(f"- ☐ {t}")

    # ---------------- Checklist ----------------
    section("Human Verification")
    st.markdown("""
<div class="card"><h4>Before You Rely on This Information</h4>
<ul>
<li>Verify the medicine name on the physical package.</li>
<li>Verify the batch number.</li>
<li>Verify the expiry date.</li>
<li>Verify manufacturer details.</li>
<li>Confirm important information with a pharmacist or doctor.</li>
<li>Do not make medication changes based only on MedLens.</li>
</ul></div>""", unsafe_allow_html=True)

# ---------------- Always-visible safety notice ----------------
section("Medical Safety Notice")
st.markdown(f'<div class="notice">⚠️ {SAFETY_NOTICE}</div>', unsafe_allow_html=True)
st.caption("MedLens is a hackathon prototype. Demo dataset records are fictional.")