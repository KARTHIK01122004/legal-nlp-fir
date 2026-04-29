from pydub import AudioSegment

AudioSegment.converter = r"D:\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"
AudioSegment.ffmpeg = r"D:\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"D:\ffmpeg-8.1-essentials_build\bin\ffprobe.exe"


import datetime
import random
import json
import base64
import html
import os
import logging
import io
import speech_recognition

from pathlib import Path



os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import streamlit as st
import streamlit.components.v1 as components
from dateutil import parser as dateparser
from deep_translator import GoogleTranslator

try:
    from utils.speech_to_text import transcribe_audio
except ImportError:
    transcribe_audio = None

from tools.ipc_search_tool import match_top_ipc, format_ipc_for_fir
from utils.preprocessing import clean_text          # noqa: F401
from utils.legal_ai_writer import ai_legal_rewrite
from tools.ner_tool import extract_entities


# ╔══════════════════════════════════════════════════════════════════╗
# ║                        CONSTANTS                                ║
# ╚══════════════════════════════════════════════════════════════════╝

DATA_FILE = "generated_firs.json"
RECORDINGS_DIR = Path(__file__).parent / "recordings"

LANGUAGES = {
    "English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te",
    "Kannada": "kn", "Malayalam": "ml", "Bengali": "bn", "Marathi": "mr",
    "Gujarati": "gu", "Punjabi": "pa", "Odia": "or", "Assamese": "as",
}

INCIDENT_OPTIONS = [
    "Theft / Robbery", "Assault", "Sexual Offence", "Murder / Attempt to Murder",
    "Cyber Crime / Fraud", "Criminal Intimidation", "Domestic Violence",
    "Missing Person / Kidnapping", "Property Damage / Trespass", "Harassment", "Other",
]


# ╔══════════════════════════════════════════════════════════════════╗
# ║                     PAGE CONFIG                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

st.set_page_config(page_title="AI FIR System", layout="wide", page_icon="⚖️")


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   DARK MODE PRO CSS                              ║
# ╚══════════════════════════════════════════════════════════════════╝

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=DM+Serif+Display&display=swap');

:root {
    --bg-deep:      #0a0d12;
    --bg-base:      #0f1319;
    --bg-surface:   #151c26;
    --bg-elevated:  #1c2535;
    --bg-hover:     #222e42;

    --teal:         #00e5c3;
    --teal-dim:     #00b89d;
    --teal-glow:    rgba(0,229,195,0.12);
    --teal-border:  rgba(0,229,195,0.25);

    --cyan:         #38bdf8;
    --amber:        #f59e0b;
    --red-soft:     #f87171;
    --green-soft:   #4ade80;

    --text-primary:   #e8edf5;
    --text-secondary: #8b9ab5;
    --text-muted:     #4d5e78;
    --text-accent:    #00e5c3;

    --border-subtle:  rgba(255,255,255,0.06);
    --border-mid:     rgba(255,255,255,0.10);
    --border-accent:  rgba(0,229,195,0.30);

    --radius-sm:  6px;
    --radius-md:  10px;
    --radius-lg:  14px;
    --radius-xl:  18px;

    --font-body:    'Space Grotesk', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
    --font-display: 'DM Serif Display', serif;

    --shadow-card: 0 1px 3px rgba(0,0,0,0.4), 0 4px 16px rgba(0,0,0,0.3);
    --shadow-glow: 0 0 24px rgba(0,229,195,0.12);
}

html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background: var(--bg-base) !important;
    color: var(--text-primary) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1100px !important; }

.hero {
    background: linear-gradient(135deg, #0d1825 0%, #0f1f35 50%, #101a2c 100%);
    border: 1px solid var(--border-accent);
    border-radius: var(--radius-xl);
    padding: 2.4rem 2.8rem 2.2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -1px; left: -1px; right: -1px;
    height: 2px;
    background: linear-gradient(90deg, transparent 0%, var(--teal) 40%, var(--cyan) 70%, transparent 100%);
}
.hero::after {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 80% at 85% 50%, rgba(0,229,195,0.06) 0%, transparent 70%),
        radial-gradient(ellipse 40% 60% at 10% 80%, rgba(56,189,248,0.04) 0%, transparent 60%);
    pointer-events: none;
}
.hero-inner { position: relative; z-index: 1; }
.hero-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--teal-glow); border: 1px solid var(--teal-border);
    color: var(--teal); border-radius: 20px; padding: 0.22rem 0.85rem;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 1.2px;
    text-transform: uppercase; margin-bottom: 0.85rem;
}
.hero h1 {
    font-family: var(--font-display) !important; color: var(--text-primary) !important;
    font-size: 2.3rem; margin: 0 0 0.4rem; line-height: 1.15;
}
.hero h1 span { color: var(--teal); }
.hero-subtitle { color: var(--text-secondary); font-size: 0.88rem; letter-spacing: 0.3px; line-height: 1.6; }
.hero-subtitle strong { color: var(--teal-dim); font-weight: 500; }

.card {
    background: var(--bg-surface); border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg); padding: 1.6rem 1.8rem 1.4rem;
    margin-bottom: 1.2rem; box-shadow: var(--shadow-card); position: relative; overflow: hidden;
}
.card::before {
    content: ''; position: absolute; top: 0; left: 1.5rem; right: 1.5rem; height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-mid), transparent);
}
.card-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.2rem; padding-bottom: 0.8rem; border-bottom: 1px solid var(--border-subtle); }
.card-step {
    width: 26px; height: 26px; border-radius: 50%; background: var(--teal-glow);
    border: 1px solid var(--teal-border); display: inline-flex; align-items: center;
    justify-content: center; font-size: 0.68rem; font-weight: 700; color: var(--teal);
    font-family: var(--font-mono); flex-shrink: 0;
}
.card-title { font-size: 0.92rem; font-weight: 600; color: var(--text-primary); letter-spacing: 0.2px; }

.transcript-box {
    background: var(--bg-elevated); border: 1px solid var(--border-mid);
    border-left: 3px solid var(--teal); border-radius: 0 var(--radius-md) var(--radius-md) 0;
    padding: 1rem 1.3rem; font-size: 0.88rem; line-height: 1.8; color: var(--text-secondary);
    font-style: italic; margin-bottom: 0.85rem; position: relative;
}
.transcript-box::before {
    content: '"'; position: absolute; top: -4px; left: 12px; font-size: 2.5rem;
    color: var(--teal); opacity: 0.2; font-family: var(--font-display); line-height: 1;
}

.ipc-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1rem; }
.ipc-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.25);
    border-radius: var(--radius-sm); padding: 0.3rem 0.85rem; font-size: 0.76rem;
    font-weight: 600; color: var(--cyan); font-family: var(--font-mono); letter-spacing: 0.3px;
}
.ipc-badge::before { content: '§'; opacity: 0.6; font-size: 0.9em; }

.fir-wrapper {
    background: #060a10; border: 1px solid var(--teal-border);
    border-radius: var(--radius-lg); overflow: hidden; box-shadow: var(--shadow-glow);
}
.fir-toolbar {
    background: var(--bg-elevated); border-bottom: 1px solid var(--border-subtle);
    padding: 0.7rem 1.2rem; display: flex; align-items: center; gap: 8px;
}
.fir-dot { width: 10px; height: 10px; border-radius: 50%; }
.fir-toolbar-title {
    font-size: 0.72rem; color: var(--text-muted); font-family: var(--font-mono);
    margin-left: auto; letter-spacing: 0.5px;
}
.fir-box {
    background: #060a10; color: #c8daf0; font-family: var(--font-mono);
    font-size: 0.78rem; line-height: 1.85; padding: 1.6rem 2rem;
    white-space: pre-wrap; word-break: break-word;
}

.tip {
    background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.2);
    border-left: 3px solid var(--amber); border-radius: 0 var(--radius-md) var(--radius-md) 0;
    padding: 0.7rem 1.1rem; font-size: 0.83rem; color: #d4a849; margin-bottom: 0.9rem; line-height: 1.6;
}

.section-divider {
    height: 1px; background: linear-gradient(to right, var(--teal-border), transparent 80%);
    border: none; margin: 1.8rem 0;
}

.stTextInput input, .stTextArea textarea {
    background: var(--bg-elevated) !important; border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius-md) !important; color: var(--text-primary) !important;
    font-family: var(--font-body) !important; font-size: 0.88rem !important;
    transition: border-color 0.18s, box-shadow 0.18s !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--teal-dim) !important; box-shadow: 0 0 0 3px rgba(0,229,195,0.1) !important;
}
.stTextInput label, .stTextArea label, .stSelectbox label, .stDateInput label,
.stRadio label, .stFileUploader label {
    color: var(--text-secondary) !important; font-size: 0.82rem !important;
    font-weight: 500 !important; letter-spacing: 0.3px !important;
}
.stRadio > div {
    background: var(--bg-elevated) !important; border-radius: var(--radius-md) !important;
    padding: 0.5rem 1rem !important; border: 1px solid var(--border-subtle) !important;
}
.stRadio [data-testid="stMarkdownContainer"] p { color: var(--text-primary) !important; font-size: 0.88rem !important; }
.stDateInput input {
    background: var(--bg-elevated) !important; border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius-md) !important; color: var(--text-primary) !important;
    font-family: var(--font-body) !important; font-size: 0.88rem !important;
}

.stButton > button {
    background: linear-gradient(135deg, #00c4a8 0%, #009985 100%) !important;
    color: #031a15 !important; border: none !important; border-radius: var(--radius-md) !important;
    font-weight: 700 !important; font-family: var(--font-body) !important; font-size: 0.88rem !important;
    padding: 0.6rem 2rem !important; letter-spacing: 0.4px !important;
    transition: opacity 0.18s, transform 0.12s !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }
.stDownloadButton > button {
    background: rgba(74,222,128,0.10) !important; color: var(--green-soft) !important;
    border: 1px solid rgba(74,222,128,0.28) !important; border-radius: var(--radius-md) !important;
    font-weight: 600 !important;
}
.stDownloadButton > button:hover { background: rgba(74,222,128,0.18) !important; }

[data-testid="stExpander"] {
    background: var(--bg-surface) !important; border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important; margin-bottom: 0.6rem !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-secondary) !important; font-size: 0.88rem !important;
    font-weight: 500 !important; padding: 0.75rem 1rem !important;
}
[data-testid="stExpander"] summary:hover { color: var(--teal) !important; }

[data-testid="stFileUploader"] {
    background: var(--bg-elevated) !important; border: 1.5px dashed var(--border-mid) !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--teal-dim) !important; }
[data-testid="stFileUploader"] * { color: var(--text-secondary) !important; }

[data-testid="stNotification"][data-type="success"] {
    background: rgba(74,222,128,0.08) !important; border: 1px solid rgba(74,222,128,0.25) !important;
    color: var(--green-soft) !important; border-radius: var(--radius-md) !important;
}
[data-testid="stNotification"][data-type="error"] {
    background: rgba(248,113,113,0.08) !important; border: 1px solid rgba(248,113,113,0.25) !important;
    color: var(--red-soft) !important; border-radius: var(--radius-md) !important;
}
[data-testid="stNotification"][data-type="warning"] {
    background: rgba(245,158,11,0.08) !important; border: 1px solid rgba(245,158,11,0.25) !important;
    color: var(--amber) !important; border-radius: var(--radius-md) !important;
}
[data-testid="stNotification"][data-type="info"] {
    background: rgba(56,189,248,0.08) !important; border: 1px solid rgba(56,189,248,0.25) !important;
    color: var(--cyan) !important; border-radius: var(--radius-md) !important;
}

[data-testid="stSidebar"] {
    background: var(--bg-deep) !important; border-right: 1px solid var(--border-subtle) !important;
}
[data-testdata="stSidebar"] * { color: var(--text-secondary) !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: var(--teal) !important; font-family: var(--font-display) !important;
}
[data-testid="stSidebar"] hr { border-color: var(--border-subtle) !important; }

.stSpinner > div { border-top-color: var(--teal) !important; }
h3 { color: var(--text-primary) !important; font-family: var(--font-display) !important; }
audio { border-radius: var(--radius-md) !important; width: 100% !important; }
[data-testid="column"] { padding: 0 0.4rem !important; }
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   AUDIO CONVERSION HELPER                       ║
# ╚══════════════════════════════════════════════════════════════════╝

def convert_audio_to_wav(audio_bytes: bytes, source_mime: str = "audio/webm") -> bytes:
    try:
        from pydub import AudioSegment
        fmt_map = {
            "audio/webm": "webm", "audio/ogg":  "ogg",  "audio/mp3":  "mp3",
            "audio/mpeg": "mp3",  "audio/mp4":  "mp4",  "audio/m4a":  "mp4",
            "audio/wav":  "wav",  "audio/x-wav":"wav",  "audio/flac": "flac",
        }
        fmt = fmt_map.get(source_mime.lower().split(";")[0].strip(), "webm")
        segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
        segment = segment.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        wav_buf = io.BytesIO()
        segment.export(wav_buf, format="wav")
        return wav_buf.getvalue()
    except ImportError:
        return audio_bytes
    except Exception:
        return audio_bytes


# ╔══════════════════════════════════════════════════════════════════╗
# ║                     UI — HTML COMPONENTS                        ║
# ╚══════════════════════════════════════════════════════════════════╝

def ui_hero():
    st.markdown("""
    <div class="hero">
      <div class="hero-inner">
        <div class="hero-pill"></div>
        <h1>⚖️ AI <span>FIR</span> Generating System</h1>
      </div>
    </div>
    """, unsafe_allow_html=True)


def ui_card_open(title: str, step: str = ""):
    badge = f'<div class="card-step">{html.escape(step)}</div>' if step else ""
    safe_title = html.escape(title)
    st.markdown(
        f"""<div class="card">
              <div class="card-header">
                {badge}
                <span class="card-title">{safe_title}</span>
              </div>""",
        unsafe_allow_html=True,
    )


def ui_card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def ui_transcript(text: str):
    safe = html.escape(text)
    st.markdown(f'<div class="transcript-box">{safe}</div>', unsafe_allow_html=True)


def ui_tip(text: str):
    st.markdown(f'<div class="tip">{text}</div>', unsafe_allow_html=True)


def ui_ipc_badges(matches: list):
    badges = "".join(
        f'<span class="ipc-badge">{html.escape(str(m["section"]))} — {html.escape(m["title"])}</span>'
        for m in matches
    )
    st.markdown(f'<div class="ipc-row">{badges}</div>', unsafe_allow_html=True)


def ui_fir_box(text: str):
    safe = html.escape(text)
    st.markdown(f"""
    <div class="fir-wrapper">
      <div class="fir-toolbar">
        <div class="fir-dot" style="background:#f87171;"></div>
        <div class="fir-dot" style="background:#f59e0b;"></div>
        <div class="fir-dot" style="background:#4ade80;"></div>
        <span class="fir-toolbar-title">FIRST INFORMATION REPORT · OFFICIAL DOCUMENT</span>
      </div>
      <div class="fir-box">{safe}</div>
    </div>
    """, unsafe_allow_html=True)


def ui_divider():
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


def ui_sidebar():
    st.sidebar.markdown('<h2 style="margin-top:0">⚙️ Settings</h2>', unsafe_allow_html=True)
    lang = st.sidebar.selectbox("Output Language", list(LANGUAGES.keys()))
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 How to use")
    st.sidebar.markdown("""
1. Describe the incident (voice or text)
2. Click **Extract Details** — fields auto-fill
3. Review Step 2 fields and edit if needed
4. Add evidence / witnesses (optional)
5. Click **Generate FIR**
    """)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<p style="font-size:0.73rem;color:#4d5e78;">AI Generated FIR</p>',
        unsafe_allow_html=True,
    )
    return lang, LANGUAGES[lang]


# ╔══════════════════════════════════════════════════════════════════╗
# ║                       LOGIC — HELPERS                           ║
# ╚══════════════════════════════════════════════════════════════════╝

def save_fir_record(record: dict, path: str = DATA_FILE):
    existing = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except (json.JSONDecodeError, ValueError):
            existing = []
    existing.append(record)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
    except OSError as e:
        st.error(f"⚠️ Could not save record: {e}")


def translate_to_english(text: str) -> str:
    if not text or not text.strip():
        return text
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text


def translate_output(text: str, target_lang: str) -> str:
    if target_lang == "en":
        return text
    if not text or not text.strip():
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except Exception:
        return text


def generate_fir_number() -> str:
    return f"FIR-{datetime.datetime.now().year}-{random.randint(1000, 9999)}"


def classify_intent(text: str) -> str:
    if not text:
        return "Other"
    t = text.lower()
    if any(w in t for w in ["rape", "sexual assault", "molest", "outrage"]):
        return "Sexual Offence"
    if any(w in t for w in ["murder", "killed", "stabbed", "dead body"]):
        return "Murder / Attempt to Murder"
    if any(w in t for w in ["stolen", "theft", "pickpocket", "snatch", "snatched", "looted", "robbed", "robbery"]):
        return "Theft / Robbery"
    if any(w in t for w in ["attack", "assault", "beat", "hit", "punch", "slap", "hurt"]):
        return "Assault"
    if any(w in t for w in ["fraud", "scam", "hack", "cheat", "fake", "upi", "online"]):
        return "Cyber Crime / Fraud"
    if any(w in t for w in ["threat", "blackmail", "intimidate", "warn"]):
        return "Criminal Intimidation"
    if any(w in t for w in ["domestic", "husband", "wife", "dowry", "in-laws", "cruelty"]):
        return "Domestic Violence"
    if any(w in t for w in ["missing", "lost", "kidnap", "abduct"]):
        return "Missing Person / Kidnapping"
    if any(w in t for w in ["damage", "vandal", "broke", "destroyed", "trespass"]):
        return "Property Damage / Trespass"
    return "Other"


def summarize_case(text: str) -> str:
    if not text:
        return "No description provided."
    return text[:200] + "..." if len(text) > 200 else text


def build_fir_report(fir_number, summary, narrative, ipc_block, evidence_block, incident_type) -> str:
    ss = st.session_state
    return f"""
╔══════════════════════════════════════════════════════════════════╗
║              FIRST INFORMATION REPORT (FIR)                    ║
║                  Under Section 154 Cr.P.C                      ║
╚══════════════════════════════════════════════════════════════════╝

  FIR Number       : {fir_number}
  Date Filed       : {datetime.date.today().strftime('%d %B %Y')}
  Time Recorded    : {datetime.datetime.now().strftime('%I:%M %p')}
  Police Station   : {ss.auto_location or 'Not specified'}

══════════════════════════════════════════════════════════════════
  SECTION 1 — COMPLAINANT DETAILS
══════════════════════════════════════════════════════════════════

  Name             : {ss.auto_name}
  Contact          : {ss.auto_contact or 'Not provided'}
  Incident Date    : {str(ss.auto_date)}
  Incident Location: {ss.auto_location or 'Not specified'}
  Incident Type    : {incident_type}

══════════════════════════════════════════════════════════════════
  SECTION 2 — CASE SUMMARY
══════════════════════════════════════════════════════════════════

  {summary}

══════════════════════════════════════════════════════════════════
  SECTION 3 — FORMAL COMPLAINT NARRATIVE
══════════════════════════════════════════════════════════════════

{narrative}

══════════════════════════════════════════════════════════════════
  SECTION 4 — APPLICABLE IPC SECTIONS
══════════════════════════════════════════════════════════════════

{ipc_block}

══════════════════════════════════════════════════════════════════
  SECTION 5 — EVIDENCE & WITNESSES
══════════════════════════════════════════════════════════════════

{evidence_block}

══════════════════════════════════════════════════════════════════
  SECTION 6 — DECLARATION
══════════════════════════════════════════════════════════════════

  I, {ss.auto_name}, hereby declare that the above
  information is true and correct to the best of my knowledge.

  Complainant Signature : ___________________________
  Name                  : {ss.auto_name}
  Date                  : {datetime.date.today().strftime('%d/%m/%Y')}

══════════════════════════════════════════════════════════════════
  * Computer-generated FIR. Requires officer verification & stamp.
══════════════════════════════════════════════════════════════════
""".strip()


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   LOGIC — STATE MUTATIONS                       ║
# ╚══════════════════════════════════════════════════════════════════╝

def apply_ner_fields(text: str):
    if not text or not text.strip():
        return
    try:
        entities = extract_entities(text)
    except Exception:
        return
    if entities.get("PERSON") and not st.session_state.auto_name:
        st.session_state.auto_name = entities["PERSON"][0]
    if entities.get("LOCATION") and not st.session_state.auto_location:
        st.session_state.auto_location = entities["LOCATION"][0]
    if entities.get("CONTACT") and not st.session_state.auto_contact:
        st.session_state.auto_contact = entities["CONTACT"][0]
    raw_date = (entities.get("DATE") or [None])[0]
    if raw_date and st.session_state.auto_date == datetime.date.today():
        try:
            parsed = dateparser.parse(str(raw_date))
            if parsed:
                st.session_state.auto_date = parsed.date()
        except Exception:
            pass


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    LOGIC — SESSION STATE                        ║
# ╚══════════════════════════════════════════════════════════════════╝

_defaults = {
    "description": "",
    "auto_name": "",
    "auto_location": "",
    "auto_contact": "",
    "auto_date": datetime.date.today(),
    "incident_type": "Other",
    "fir_output": None,
    "fir_number": "",
    "ipc_matches": [],
    "raw_transcript": "",
    "typed_complaint_saved": "",
    "photo_name": "",
    "photo_b64": "",
    "evidence_text": "",
    "witnesses_text": "",
    "editable_transcript": "",
    # Live mic state: idle | captured | transcribed
    "mic_phase": "idle",
    "captured_audio_bytes": None,
    "captured_audio_mime": "audio/wav",
}

for key, value in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def ensure_recordings_dir() -> Path:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return RECORDINGS_DIR


# ╔══════════════════════════════════════════════════════════════════╗
# ║               LIVE MICROPHONE — RENDER FUNCTION                 ║
# ║  Clean single-widget approach using st.audio_input only        ║
# ╚══════════════════════════════════════════════════════════════════╝

def render_live_mic():
    phase = st.session_state.mic_phase

    # ══════════════════════════════════════════════════════════════
    # PHASE 1 — idle: show only the st.audio_input widget
    # ══════════════════════════════════════════════════════════════
    if phase == "idle":
        st.markdown(
            '<p style="color:#8b9ab5;font-size:0.84rem;margin-bottom:12px;">'
            '🎙️ Click the <strong>microphone icon</strong> below to start recording. '
            'Click it again to stop. Your recording will appear automatically.</p>',
            unsafe_allow_html=True,
        )

        try:
            audio_value = st.audio_input(
                "🎤 Record your complaint",
                key="live_mic_widget",
            )
        except AttributeError:
            st.warning(
                "⚠️ Your Streamlit version does not support the built-in mic widget. "
                "Please upgrade (`pip install --upgrade streamlit`) or use the "
                "**Upload Audio File** option instead."
            )
            return

        if audio_value is not None:
            audio_bytes = audio_value.read()
            if audio_bytes:
                st.session_state.captured_audio_bytes = audio_bytes
                st.session_state.captured_audio_mime = "audio/wav"
                st.session_state.mic_phase = "captured"
                st.rerun()
            else:
                st.error("⚠️ Recorded audio appears empty. Please try again.")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2 — captured: playback + Transcribe button
    # ══════════════════════════════════════════════════════════════
    elif phase == "captured":
        audio_bytes = st.session_state.captured_audio_bytes
        audio_mime  = st.session_state.captured_audio_mime or "audio/wav"

        st.markdown(
            '<div style="background:rgba(0,229,195,0.12);border:1px solid rgba(0,229,195,0.35);'
            'border-radius:10px;padding:12px 16px;color:#00e5c3;font-size:0.88rem;'
            'font-weight:600;margin-bottom:14px;">'
            '✅ Recording captured! Play it back, then click <strong>Transcribe Audio</strong>.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("**🎧 Playback your recording:**")
        st.markdown(
            '<div style="background:rgba(0,229,195,0.05);border:1px solid rgba(0,229,195,0.18);'
            'border-radius:10px;padding:12px 14px;margin-bottom:16px;">',
            unsafe_allow_html=True,
        )
        if audio_bytes:
            st.audio(audio_bytes, format=audio_mime)
        else:
            st.warning("⚠️ Audio data missing. Please re-record.")
        st.markdown("</div>", unsafe_allow_html=True)

        col_transcribe, col_rerecord = st.columns([3, 1])

        with col_transcribe:
            transcribe_clicked = st.button(
                "📝 Transcribe Audio",
                key="btn_transcribe",
                use_container_width=True,
            )

        with col_rerecord:
            rerecord_clicked = st.button(
                "🔄 Re-record",
                key="btn_rerecord_captured",
                use_container_width=True,
            )

        if rerecord_clicked:
            st.session_state.mic_phase = "idle"
            st.session_state.captured_audio_bytes = None
            st.session_state.captured_audio_mime  = "audio/wav"
            st.session_state.raw_transcript        = ""
            st.session_state.editable_transcript   = ""
            st.rerun()

        if transcribe_clicked:
            if transcribe_audio is None:
                st.error("⚠️ `transcribe_audio` not found. Check utils/speech_to_text.py.")
            elif not audio_bytes:
                st.error("⚠️ No audio data. Please re-record.")
            else:
                with st.spinner("🔄 Converting and transcribing… this may take a moment."):
                    try:
                        wav_bytes = convert_audio_to_wav(audio_bytes, audio_mime)
                        raw = transcribe_audio(wav_bytes)
                    except Exception as exc:
                        raw = f"Transcription error: {exc}"

                error_kw = ["error", "unavailable", "failed", "unclear"]
                if any(kw in raw.lower() for kw in error_kw):
                    st.error(f"⚠️ Transcription issue: {raw}")
                else:
                    english = translate_to_english(raw)
                    st.session_state.raw_transcript      = english
                    st.session_state.editable_transcript = english
                    st.session_state.description         = english
                    st.session_state.mic_phase           = "transcribed"
                    st.rerun()

    # ══════════════════════════════════════════════════════════════
    # PHASE 3 — transcribed: editable text + Extract Details
    # ══════════════════════════════════════════════════════════════
    elif phase == "transcribed":
        st.markdown(
            '<div style="background:rgba(74,222,128,0.08);border:1px solid rgba(74,222,128,0.28);'
            'border-radius:10px;padding:12px 16px;color:#4ade80;font-size:0.88rem;'
            'font-weight:600;margin-bottom:14px;">'
            '✅ Transcription complete! Review and edit, then click '
            '<strong>Extract Details & Auto-fill Form</strong>.</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p style="color:#00e5c3;font-size:0.85rem;font-weight:600;margin-bottom:6px;">'
            '📜 Transcribed Text — edit before extracting:</p>',
            unsafe_allow_html=True,
        )
        edited = st.text_area(
            "Transcript:",
            value=st.session_state.editable_transcript,
            height=160,
            key="transcript_edit_transcribed",
            label_visibility="collapsed",
            placeholder="Your transcribed complaint appears here. Edit if needed.",
        )
        if edited != st.session_state.editable_transcript:
            st.session_state.editable_transcript = edited
            st.session_state.description = edited

        col_extract, col_new = st.columns([3, 1])

        with col_extract:
            extract_clicked = st.button(
                "🔍 Extract Details & Auto-fill Form",
                key="btn_extract_details",
                use_container_width=True,
            )

        with col_new:
            new_rec_clicked = st.button(
                "🗑️ New Recording",
                key="btn_new_recording",
                use_container_width=True,
            )

        if new_rec_clicked:
            for k in [
                "mic_phase", "captured_audio_bytes", "captured_audio_mime",
                "raw_transcript", "editable_transcript", "description",
            ]:
                st.session_state.pop(k, None)
            st.rerun()

        if extract_clicked:
            text = st.session_state.editable_transcript.strip()
            if not text:
                st.warning("⚠️ Transcript is empty. Please add some text first.")
            else:
                with st.spinner("🔍 Extracting name, date, location…"):
                    st.session_state.description   = text
                    st.session_state.incident_type = classify_intent(text)
                    apply_ner_fields(text)
                st.success("✅ Form fields auto-filled! Scroll down to review Step 2.")
                st.rerun()


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   EXPORT BUILDERS                               ║
# ╚══════════════════════════════════════════════════════════════════╝

def build_html_export(report_text: str, photo_b64: str = "", photo_name: str = "") -> str:
    safe_report = html.escape(report_text)
    photo_block = ""
    if photo_b64:
        safe_name = html.escape(photo_name or "evidence")
        photo_block = f"""
        <div class="section">
            <div class="section-title">EVIDENCE PHOTO</div>
            <img src="data:image/jpeg;base64,{photo_b64}"
                 alt="{safe_name}"
                 style="max-width:100%;border:1px solid #ccc;border-radius:4px;margin-top:8px;" />
            <div style="font-size:11px;color:#666;margin-top:4px;">{safe_name}</div>
        </div>"""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>First Information Report</title>
<style>
  body {{ font-family:'Courier New',monospace; background:#fff; color:#111;
          max-width:860px; margin:40px auto; padding:0 24px; font-size:13px; }}
  h1   {{ font-family:Georgia,serif; text-align:center; font-size:20px;
          border-bottom:2px solid #111; padding-bottom:10px; margin-bottom:24px; }}
  .section {{ margin-bottom:28px; }}
  .section-title {{ font-weight:bold; font-size:12px; letter-spacing:1px;
                    border-bottom:1px solid #aaa; padding-bottom:4px;
                    margin-bottom:10px; color:#333; }}
  pre  {{ white-space:pre-wrap; word-break:break-word; line-height:1.8;
          background:#f9f9f9; padding:16px; border-radius:4px;
          border:1px solid #ddd; font-size:12px; }}
  .footer {{ font-size:11px; color:#888; text-align:center;
             border-top:1px solid #ddd; padding-top:12px; margin-top:32px; }}
</style>
</head>
<body>
  <h1>⚖️ First Information Report</h1>
  <div class="section">
    <div class="section-title">OFFICIAL FIR DOCUMENT</div>
    <pre>{safe_report}</pre>
  </div>
  {photo_block}
  <div class="footer">Computer-generated FIR · Requires officer verification & stamp.</div>
</body>
</html>"""


def build_pdf_export(report_text: str, photo_b64: str = "", photo_name: str = "") -> bytes | None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Image as RLImage,
        )
        from reportlab.lib.enums import TA_CENTER

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm,
        )
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "FIRTitle", parent=styles["Heading1"],
            fontSize=16, alignment=TA_CENTER,
            spaceAfter=6, textColor=colors.HexColor("#0a1628"),
        )
        sub_style = ParagraphStyle(
            "FIRSub", parent=styles["Normal"],
            fontSize=9, alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"), spaceAfter=14,
        )
        body_style = ParagraphStyle(
            "FIRBody", parent=styles["Normal"],
            fontName="Courier", fontSize=8,
            leading=13, spaceAfter=2,
            textColor=colors.HexColor("#111111"),
        )
        caption_style = ParagraphStyle(
            "Caption", parent=styles["Normal"],
            fontSize=8, alignment=TA_CENTER,
            textColor=colors.HexColor("#666666"), spaceBefore=4,
        )
        footer_style = ParagraphStyle(
            "Footer", parent=styles["Normal"],
            fontSize=7, alignment=TA_CENTER,
            textColor=colors.HexColor("#999999"), spaceBefore=6,
        )

        story = []
        story.append(Paragraph("First Information Report", title_style))
        story.append(Paragraph("Under Section 154 Cr.P.C · Computer Generated", sub_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
        story.append(Spacer(1, 10))

        for line in report_text.splitlines():
            safe_line = (line
                         .replace("&", "&amp;")
                         .replace("<", "&lt;")
                         .replace(">", "&gt;"))
            story.append(Paragraph(safe_line or "&nbsp;", body_style))

        if photo_b64:
            story.append(Spacer(1, 16))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
            story.append(Spacer(1, 8))
            story.append(Paragraph("EVIDENCE PHOTO", ParagraphStyle(
                "EvidHdr", parent=styles["Normal"],
                fontSize=9, fontName="Helvetica-Bold",
                textColor=colors.HexColor("#333333"), spaceAfter=6,
            )))
            try:
                img_bytes = base64.b64decode(photo_b64)
                img_buf   = io.BytesIO(img_bytes)
                rl_img    = RLImage(img_buf, width=12*cm, height=9*cm, kind="proportional")
                story.append(rl_img)
                if photo_name:
                    story.append(Paragraph(html.escape(photo_name), caption_style))
            except Exception:
                story.append(Paragraph("(Evidence photo could not be embedded)", caption_style))

        story.append(Spacer(1, 16))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
        story.append(Paragraph(
            "Computer-generated FIR · Requires officer verification &amp; stamp.",
            footer_style,
        ))
        doc.build(story)
        return buf.getvalue()

    except ImportError:
        st.error("⚠️ PDF export requires reportlab — run: pip install reportlab")
        return None
    except Exception as e:
        st.error(f"⚠️ PDF build error: {e}")
        return None


def render_download_buttons(report_text: str, fir_num: str, photo_b64: str, photo_name: str, key_prefix: str):
    fmt = st.radio(
        "📥 Download format:",
        ["TXT", "HTML (with photo)", "PDF (with photo)"],
        horizontal=True,
        key=f"fmt_{key_prefix}",
    )
    if fmt == "TXT":
        if photo_b64:
            st.caption("⚠️ TXT cannot include the evidence photo. Choose HTML or PDF.")
        st.download_button(
            "⬇️ Download TXT", report_text,
            file_name=f"{fir_num}.txt", mime="text/plain",
            use_container_width=True, key=f"dl_{key_prefix}_txt",
        )
    elif fmt == "HTML (with photo)":
        html_out = build_html_export(report_text, photo_b64, photo_name)
        st.download_button(
            "⬇️ Download HTML", html_out,
            file_name=f"{fir_num}.html", mime="text/html",
            use_container_width=True, key=f"dl_{key_prefix}_html",
        )
    elif fmt == "PDF (with photo)":
        pdf_out = build_pdf_export(report_text, photo_b64, photo_name)
        if pdf_out:
            st.download_button(
                "⬇️ Download PDF", pdf_out,
                file_name=f"{fir_num}.pdf", mime="application/pdf",
                use_container_width=True, key=f"dl_{key_prefix}_pdf",
            )


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   RENDER — HEADER + SIDEBAR                     ║
# ╚══════════════════════════════════════════════════════════════════╝

ui_hero()
selected_language, output_lang_code = ui_sidebar()


# ╔══════════════════════════════════════════════════════════════════╗
# ║              RENDER — STEP 1: INCIDENT DESCRIPTION              ║
# ╚══════════════════════════════════════════════════════════════════╝

ui_card_open("Describe the Incident", "1")

input_mode = st.radio(
    "Input Method:",
    ["✍️ Type Complaint", "🎤 Voice Complaint"],
    horizontal=True,
    key="input_mode_radio",
)

if input_mode == "🎤 Voice Complaint":
    voice_option = st.radio(
        "Voice Input Method:",
        ["📁 Upload Audio File", "🎙️ Live Microphone"],
        horizontal=True,
        key="voice_option_radio",
    )

    # ── LIVE MICROPHONE ────────────────────────────────────────────
    if voice_option == "🎙️ Live Microphone":
        render_live_mic()

    # ── UPLOAD AUDIO FILE ──────────────────────────────────────────
    else:
        if transcribe_audio is None:
            st.warning("⚠️ Run: pip install SpeechRecognition pydub")
        else:
            ui_tip("Upload WAV, MP3, FLAC, OGG or M4A. Details will be auto-filled below.")
            audio_file = st.file_uploader(
                "Upload Audio File",
                type=["wav", "mp3", "flac", "ogg", "m4a", "webm"],
            )
            if audio_file is not None:
                try:
                    ext = audio_file.name.rsplit(".", 1)[-1].lower()
                    st.audio(audio_file, format=f"audio/{ext}")
                except Exception:
                    st.audio(audio_file)
                if st.button("📝 Transcribe & Auto-fill Form"):
                    audio_bytes = audio_file.read()
                    if not audio_bytes:
                        st.error("⚠️ Uploaded file appears empty.")
                    else:
                        with st.spinner("Converting and transcribing audio…"):
                            try:
                                ext_mime_map = {
                                    "wav": "audio/wav", "mp3": "audio/mp3",
                                    "flac": "audio/flac", "ogg": "audio/ogg",
                                    "m4a": "audio/mp4", "webm": "audio/webm",
                                }
                                src_mime = ext_mime_map.get(
                                    audio_file.name.rsplit(".", 1)[-1].lower(),
                                    audio_file.type or "audio/webm"
                                )
                                wav_bytes  = convert_audio_to_wav(audio_bytes, src_mime)
                                transcript = transcribe_audio(wav_bytes)
                            except Exception as exc:
                                transcript = f"Transcription error: {exc}"
                        _error_words = ["error", "unavailable", "failed", "unclear"]
                        if any(w in transcript.lower() for w in _error_words):
                            st.error(f"⚠️ {transcript}")
                        else:
                            english = translate_to_english(transcript)
                            st.session_state.raw_transcript      = english
                            st.session_state.editable_transcript = english
                            st.session_state.description         = english
                            st.session_state.incident_type       = classify_intent(english)
                            apply_ner_fields(english)
                            st.rerun()

# ── TRANSCRIPT DISPLAY (upload flow) ──────────────────────────────
if input_mode == "🎤 Voice Complaint" and st.session_state.get("raw_transcript"):
    if st.session_state.get("voice_option_radio") == "📁 Upload Audio File":
        st.markdown("**📜 Transcript (edit if needed):**")
        edited = st.text_area(
            "✏️ Edit before extracting details:",
            value=st.session_state.get("editable_transcript", st.session_state.raw_transcript),
            height=120,
            key="editable_transcript_area",
        )
        st.session_state.editable_transcript = edited
        st.session_state.description = edited
        col_ext, col_clr = st.columns([3, 1])
        with col_ext:
            if st.button("🔍 Extract Details & Auto-fill Form", key="extract_btn", use_container_width=True):
                if not edited.strip():
                    st.warning("⚠️ Transcript is empty.")
                else:
                    with st.spinner("Extracting…"):
                        st.session_state.description   = edited
                        st.session_state.incident_type = classify_intent(edited)
                        apply_ner_fields(edited)
                    st.success("✅ Details extracted!")
                    st.rerun()
        with col_clr:
            if st.button("🗑️ Clear", key="clear_transcript_btn", use_container_width=True):
                st.session_state.raw_transcript      = ""
                st.session_state.editable_transcript = ""
                st.session_state.description         = ""
                st.rerun()

# ── TYPE COMPLAINT FLOW ────────────────────────────────────────────
if input_mode == "✍️ Type Complaint":
    user_text = st.text_area(
        "Describe the incident in detail",
        height=160,
        value=st.session_state.typed_complaint_saved,
        placeholder="e.g. My name is Ramesh Kumar, on 1st April 2026 at MG Road, Bangalore…",
        key="typed_complaint",
    )
    if user_text and user_text.strip():
        st.session_state.typed_complaint_saved = user_text
        english = translate_to_english(user_text)
        st.session_state.description   = english
        st.session_state.incident_type = classify_intent(english)
        if st.button("🔍 Extract Details from Text", use_container_width=True):
            with st.spinner("Extracting…"):
                apply_ner_fields(english)
                st.session_state.description = english
            st.success("✅ Details extracted!")
            st.rerun()

ui_card_close()

st.markdown("")
st.divider()
st.markdown("")

# ╔══════════════════════════════════════════════════════════════════╗
# ║             RENDER — STEP 2: COMPLAINANT DETAILS                ║
# ╚══════════════════════════════════════════════════════════════════╝

ui_card_open("Complainant Details", "2")

col1, col2 = st.columns(2)
with col1:
    complainant = st.text_input("👤 Complainant Name *", key="auto_name")
    st.markdown("")
    location    = st.text_input("📍 Incident Location *", key="auto_location")
with col2:
    contact = st.text_input("📞 Contact Number", key="auto_contact")
    st.markdown("")
    date    = st.date_input("📅 Incident Date *", key="auto_date")

st.markdown("")
auto_incident = st.session_state.get("incident_type", "Other")
default_index = next(
    (i for i, o in enumerate(INCIDENT_OPTIONS)
     if auto_incident.replace("/", "").lower() in o.replace("/", "").lower()),
    len(INCIDENT_OPTIONS) - 1,
)
incident_type = st.selectbox("⚖️ Incident Type", INCIDENT_OPTIONS, index=default_index)

ui_card_close()

st.markdown("")
st.divider()
st.markdown("")


# ╔══════════════════════════════════════════════════════════════════╗
# ║               RENDER — STEP 3: EVIDENCE (optional)              ║
# ╚══════════════════════════════════════════════════════════════════╝

with st.expander("➕ Evidence & Witnesses (optional)"):
    evidence_input = st.text_area(
        "🧾 Evidence Details", height=80,
        value=st.session_state.evidence_text,
        placeholder="CCTV footage, bank statement, photos…",
        key="evidence_input",
    )
    st.session_state.evidence_text = evidence_input

    witnesses_input = st.text_area(
        "👥 Witnesses", height=80,
        value=st.session_state.witnesses_text,
        placeholder="Names and contact details of witnesses",
        key="witnesses_input",
    )
    st.session_state.witnesses_text = witnesses_input

    st.markdown("#### 📷 Evidence Photo")
    photo_file = st.file_uploader(
        "Upload Evidence Photo (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        key="evidence_photo",
    )
    if photo_file is not None:
        try:
            photo_bytes = photo_file.read()
            if photo_bytes:
                st.session_state.photo_name = photo_file.name
                st.session_state.photo_b64  = base64.b64encode(photo_bytes).decode("utf-8")
                st.image(photo_bytes, caption=photo_file.name, use_container_width=True)
                st.success(f"✅ Photo saved: {photo_file.name}")
            else:
                st.warning("⚠️ Uploaded photo appears empty.")
        except Exception as e:
            st.error(f"⚠️ Could not read photo: {e}")
    elif st.session_state.photo_b64:
        try:
            img_bytes = base64.b64decode(st.session_state.photo_b64)
            st.image(img_bytes, caption=st.session_state.photo_name, use_container_width=True)
        except Exception:
            st.session_state.photo_b64 = ""

st.markdown("")
ui_divider()
st.markdown("")


# ╔══════════════════════════════════════════════════════════════════╗
# ║                  LOGIC — GENERATE FIR                           ║
# ╚══════════════════════════════════════════════════════════════════╝

if st.button("🚀 Generate FIR", use_container_width=True):

    if not st.session_state.auto_name or not st.session_state.auto_name.strip():
        st.error("⚠️ Complainant name is required.")
        st.stop()
    if not st.session_state.description or not st.session_state.description.strip():
        st.error("⚠️ Incident description is required.")
        st.stop()

    with st.spinner("Generating FIR…"):
        try:
            ipc_matches = match_top_ipc(st.session_state.description, top_n=3)
        except Exception:
            ipc_matches = []

        ipc_block  = format_ipc_for_fir(ipc_matches) if ipc_matches else "  Manual IPC review required."
        fir_number = generate_fir_number()
        summary    = summarize_case(st.session_state.description)

        clean_desc = st.session_state.description.strip().strip('"').strip("'")
        try:
            narrative = ai_legal_rewrite(clean_desc, complainant=st.session_state.auto_name)
        except Exception as exc:
            narrative = f"Narrative generation failed: {exc}\n\n{clean_desc}"

        evidence_lines = []
        if st.session_state.evidence_text.strip():
            evidence_lines.append(f"  Evidence   : {st.session_state.evidence_text.strip()}")
        if st.session_state.witnesses_text.strip():
            evidence_lines.append(f"  Witnesses  : {st.session_state.witnesses_text.strip()}")
        if st.session_state.photo_name:
            evidence_lines.append(f"  Photo      : {st.session_state.photo_name} (attached)")
        evidence_block = "\n".join(evidence_lines) if evidence_lines else "  None provided."

        try:
            report = build_fir_report(
                fir_number, summary, narrative, ipc_block, evidence_block, incident_type
            )
        except Exception as exc:
            st.error(f"⚠️ Could not build FIR report: {exc}")
            st.stop()

        try:
            final_output = translate_output(report, output_lang_code)
        except Exception:
            final_output = report

        st.session_state.fir_output  = final_output
        st.session_state.fir_number  = fir_number
        st.session_state.ipc_matches = ipc_matches
        st.session_state.fir_record  = {
            "fir_number":    fir_number,
            "complainant":   st.session_state.auto_name,
            "contact":       st.session_state.auto_contact,
            "location":      st.session_state.auto_location,
            "incident_date": str(st.session_state.auto_date),
            "incident_type": incident_type,
            "ipc_sections":  ipc_matches,
            "report":        final_output,
            "photo_b64":     st.session_state.get("photo_b64", ""),
            "photo_name":    st.session_state.get("photo_name", ""),
            "timestamp":     datetime.datetime.now().isoformat(),
        }


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    RENDER — FIR OUTPUT                          ║
# ╚══════════════════════════════════════════════════════════════════╝

if st.session_state.get("fir_output"):

    ui_divider()
    st.markdown("### 📄 FIR Generated")

    st.markdown("### ⚖️ Matched IPC Sections")
    if st.session_state.ipc_matches:
        ui_ipc_badges(st.session_state.ipc_matches)
        for m in st.session_state.ipc_matches:
            with st.expander(f"Section {m['section']} IPC — {m['title']}"):
                st.write(f"**Description:** {m['description']}")
                st.write(f"**Punishment:** {m['punishment']}")
    else:
        st.info("No specific IPC sections matched automatically. Manual review required.")

    st.markdown("### 📋 Full FIR Document")
    ui_fir_box(st.session_state.fir_output)

    if st.session_state.photo_b64:
        st.markdown("### 📷 Attached Evidence Photo")
        try:
            st.image(
                base64.b64decode(st.session_state.photo_b64),
                caption=st.session_state.photo_name,
                use_container_width=True,
            )
        except Exception:
            st.warning("⚠️ Could not display attached photo.")

    col_dl, col_save = st.columns(2)
    with col_dl:
        render_download_buttons(
            report_text=st.session_state.fir_output,
            fir_num=st.session_state.fir_number,
            photo_b64=st.session_state.get("photo_b64", ""),
            photo_name=st.session_state.get("photo_name", ""),
            key_prefix="main",
        )
    with col_save:
        if st.button("💾 Save to Local Store", use_container_width=True):
            if st.session_state.get("fir_record"):
                save_fir_record(st.session_state.fir_record)
                st.success(f"✅ Saved to {DATA_FILE}")
            else:
                st.warning("⚠️ No FIR record to save.")


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   RENDER — SAVED RECORDS                        ║
# ╚══════════════════════════════════════════════════════════════════╝

with st.expander("📂 View Saved FIR Records"):
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        except (json.JSONDecodeError, ValueError):
            data = []
            st.warning("⚠️ Records file appears corrupted.")
        except OSError as e:
            data = []
            st.warning(f"⚠️ Could not read records file: {e}")

        if not data:
            st.info("No saved records yet.")
        else:
            for i, record in enumerate(reversed(data)):
                with st.expander(
                    f"🧾 {record.get('fir_number','?')} — "
                    f"{record.get('complainant','')} — "
                    f"{record.get('incident_date','')}"
                ):
                    report_text = record.get("report", "")
                    if report_text:
                        ui_fir_box(report_text)
                    else:
                        st.info("No report content saved.")

                    rec_pb64  = record.get("photo_b64", "")
                    rec_pname = record.get("photo_name", "")
                    if rec_pb64:
                        st.markdown("**📷 Evidence Photo:**")
                        try:
                            st.image(
                                base64.b64decode(rec_pb64),
                                caption=rec_pname or "Evidence",
                                use_container_width=True,
                            )
                        except Exception:
                            st.warning("⚠️ Could not display saved evidence photo.")

                    fir_num = record.get("fir_number", f"FIR-{i}")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if report_text:
                            render_download_buttons(
                                report_text=report_text,
                                fir_num=fir_num,
                                photo_b64=rec_pb64,
                                photo_name=rec_pname,
                                key_prefix=f"saved_{i}",
                            )
                    with c2:
                        if st.button("🗑 Delete", key=f"del_{i}", use_container_width=True):
                            target_num = record.get("fir_number")
                            data = [r for r in data if r.get("fir_number") != target_num]
                            try:
                                with open(DATA_FILE, "w", encoding="utf-8") as f:
                                    json.dump(data, f, indent=2, ensure_ascii=False)
                                st.success("Deleted ✅")
                            except OSError as e:
                                st.error(f"⚠️ Could not delete: {e}")
                            st.rerun()
                    with c3:
                        st.caption(f"Record #{len(data) - i}")
    else:
        st.info("No saved records yet.")