"""
utils/speech_to_text.py — Speech-to-Text Transcription
========================================================

PRIMARY engine  : OpenAI Whisper (local, free, no API key needed)
                  — handles Indian English, Hindi, Tamil, Telugu etc.
                    with far higher accuracy than Google STT free tier.
FALLBACK engine : Google Speech Recognition free tier
                  — used only when Whisper is not installed.

INSTALL (one-time):
    pip install openai-whisper
    # ffmpeg is already configured via pydub in your project

Model sizes and trade-offs:
    "tiny"   — fastest, least accurate  (~39 MB)
    "base"   — good balance             (~74 MB)   <- DEFAULT
    "small"  — better accuracy          (~244 MB)
    "medium" — best for Indian accents  (~769 MB)
    "large"  — most accurate, slow      (~1550 MB)

Change WHISPER_MODEL below to trade speed for accuracy.
On first run the model weights are downloaded automatically (~74 MB for base).
"""

from __future__ import annotations

import logging
import os
import tempfile
from io import BytesIO
from typing import Optional, Tuple

from pydub import AudioSegment
from pydub.silence import split_on_silence

logger = logging.getLogger(__name__)

# ── Whisper model to use ─────────────────────────────────────────────
WHISPER_MODEL = "tiny"

# ── Chunk size for Google STT fallback only ──────────────────────────
_CHUNK_MS       = 30_000
_OVERLAP_MS     =    500
_SILENCE_THRESH = -40
_MIN_SILENCE_MS =  500

SUPPORTED_LANGUAGES: dict[str, str] = {
    "en-US": "English (US)",
    "en-IN": "English (India)",
    "hi-IN": "Hindi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "bn-IN": "Bengali",
    "gu-IN": "Gujarati",
    "pa-IN": "Punjabi",
    "ur-PK": "Urdu",
}

# BCP-47 to Whisper language code mapping
_WHISPER_LANG_MAP: dict[str, Optional[str]] = {
    "en-US": "en",
    "en-IN": "en",
    "hi-IN": "hi",
    "ta-IN": "ta",
    "te-IN": "te",
    "kn-IN": "kn",
    "ml-IN": "ml",
    "bn-IN": "bn",
    "gu-IN": "gu",
    "pa-IN": "pa",
    "ur-PK": "ur",
}


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def transcribe_audio(
    audio_bytes: bytes,
    language: str = "en-IN",
    max_retries: int = 2,
    auto_detect: bool = False,
) -> Tuple[str, dict]:
    """
    Transcribe audio bytes to text.

    Tries Whisper first. Falls back to Google STT if Whisper is not
    installed or fails completely.

    Returns:
        (transcribed_text, metadata_dict)
        metadata keys: success, language, format, engine, retries_used,
                       chunk_count, chunk_errors, error
    """
    metadata: dict = {
        "success":      False,
        "language":     language,
        "format":       "unknown",
        "engine":       "unknown",
        "retries_used": 0,
        "chunk_count":  1,
        "chunk_errors": 0,
        "error":        None,
    }

    if not audio_bytes:
        metadata["error"] = "No audio data received"
        return "No audio data received.", metadata

    detected_format = _detect_format(audio_bytes)
    metadata["format"] = detected_format

    segment = _load_and_normalise(audio_bytes, detected_format)
    if segment is None:
        metadata["error"] = f"Could not load audio as '{detected_format}'"
        return f"Audio format '{detected_format}' is not supported.", metadata

    segment = _trim_silence(segment)
    if len(segment) == 0:
        metadata["error"] = "Audio contains only silence"
        return "No speech detected — the recording appears to be silent.", metadata

    # ── Try Whisper first ────────────────────────────────────────────
    try:
        import whisper  # noqa: PLC0415
        text, meta_update = _transcribe_whisper(segment, language, auto_detect)
        metadata.update(meta_update)
        if metadata["success"]:
            return text, metadata
        logger.warning("Whisper returned empty result, trying Google STT fallback")
    except ImportError:
        logger.info(
            "Whisper not installed — using Google STT (lower accuracy). "
            "Install with: pip install openai-whisper"
        )
    except Exception as exc:
        logger.warning("Whisper failed (%s), falling back to Google STT", exc)

    # ── Google STT fallback ──────────────────────────────────────────
    text, meta_update = _transcribe_google(segment, language, max_retries)
    metadata.update(meta_update)
    return text, metadata


# ──────────────────────────────────────────────────────────────────────
# Whisper engine
# ──────────────────────────────────────────────────────────────────────

def _transcribe_whisper(
    segment: AudioSegment,
    language: str,
    auto_detect: bool,
) -> Tuple[str, dict]:
    """Transcribe using local OpenAI Whisper."""
    import whisper  # noqa: PLC0415

    meta: dict = {
        "engine":       "whisper",
        "success":      False,
        "chunk_count":  1,
        "chunk_errors": 0,
        "error":        None,
    }

    tmp_path: Optional[str] = None
    try:
        wav_bytes = _segment_to_wav_bytes(segment)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name

        logger.info("Loading Whisper model '%s'…", WHISPER_MODEL)
        model = whisper.load_model(WHISPER_MODEL)

        whisper_lang: Optional[str] = None
        if not auto_detect:
            whisper_lang = _WHISPER_LANG_MAP.get(language)

        logger.info(
            "Whisper transcribing %.1f s [model=%s, lang=%s]",
            len(segment) / 1000,
            WHISPER_MODEL,
            whisper_lang or "auto",
        )

        result = model.transcribe(
            tmp_path,
            language=whisper_lang,
            task="transcribe",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            fp16=False,
            initial_prompt=(
                "This is a police complaint dictated in Indian English. "
                "The speaker mentions their full name, address, street name, "
                "area, city, date, time, incident details, and phone number. "
                "Transcribe numbers and phone numbers exactly as spoken."
            ),
        )

        text = result.get("text", "").strip()
        if not text:
            meta["error"] = "Whisper returned empty transcription"
            return "", meta

        logger.info("Whisper succeeded: %d chars transcribed", len(text))
        meta["success"] = True
        return text, meta

    except Exception as exc:
        meta["error"] = str(exc)
        logger.error("Whisper transcription error: %s", exc, exc_info=True)
        return "", meta

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ──────────────────────────────────────────────────────────────────────
# Google STT fallback engine (chunked)
# ──────────────────────────────────────────────────────────────────────

def _transcribe_google(
    segment: AudioSegment,
    language: str,
    max_retries: int,
) -> Tuple[str, dict]:
    """Chunked Google Speech Recognition fallback."""
    import speech_recognition as sr  # noqa: PLC0415

    meta: dict = {
        "engine":       "google_stt",
        "success":      False,
        "retries_used": 0,
        "chunk_count":  0,
        "chunk_errors": 0,
        "error":        None,
    }

    chunks = _split_into_chunks(segment)
    meta["chunk_count"] = len(chunks)
    logger.info(
        "Google STT: %.1f s in %d chunk(s) [lang=%s]",
        len(segment) / 1000, len(chunks), language,
    )

    recognizer     = sr.Recognizer()
    transcripts: list[str] = []
    total_retries  = 0

    for idx, chunk in enumerate(chunks):
        text, retries, error = _google_chunk(chunk, recognizer, language, max_retries)
        total_retries += retries
        if text:
            transcripts.append(text)
        else:
            meta["chunk_errors"] += 1
            logger.warning("Chunk %d/%d failed: %s", idx + 1, len(chunks), error)

    meta["retries_used"] = total_retries

    if not transcripts:
        meta["error"] = "Google STT: speech not recognisable in any chunk"
        return (
            "Could not understand the audio clearly. "
            "Please speak more distinctly and try again.",
            meta,
        )

    meta["success"] = True
    return " ".join(transcripts), meta


def _google_chunk(
    chunk: AudioSegment,
    recognizer,
    language: str,
    max_retries: int,
) -> Tuple[str, int, Optional[str]]:
    """Transcribe one chunk via Google STT."""
    import speech_recognition as sr  # noqa: PLC0415

    wav_bytes = _segment_to_wav_bytes(chunk)
    tmp_path: Optional[str] = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name

        for attempt in range(max_retries + 1):
            try:
                with sr.AudioFile(tmp_path) as source:
                    if attempt == 0:
                        recognizer.adjust_for_ambient_noise(source, duration=0.3)
                    audio_data = recognizer.record(source)

                text = recognizer.recognize_google(
                    audio_data, language=language, show_all=False
                )
                return text, attempt, None

            except sr.UnknownValueError:
                if attempt < max_retries:
                    continue
                return "", attempt, "Unrecognisable speech"

            except sr.RequestError as exc:
                if attempt < max_retries:
                    continue
                return "", attempt, f"Google API error: {exc}"

    except Exception as exc:
        return "", 0, str(exc)

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return "", 0, "Unknown failure"


# ──────────────────────────────────────────────────────────────────────
# Shared audio helpers
# ──────────────────────────────────────────────────────────────────────

def _detect_format(data: bytes) -> str:
    """Detect audio container format from magic bytes."""
    if len(data) < 12:
        return "wav"
    if data[:4] == b"OggS":
        return "ogg"
    if data[:4] == b"fLaC":
        return "flac"
    if data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "mp3"
    if data[4:8] == b"ftyp":
        return "m4a"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    logger.debug("Unknown audio header — defaulting to webm")
    return "webm"


def _load_and_normalise(audio_bytes: bytes, source_format: str) -> Optional[AudioSegment]:
    """Load audio bytes and normalise to 16-bit mono 16 kHz PCM."""
    formats_to_try = list(dict.fromkeys([source_format, "webm", "wav"]))
    for fmt in formats_to_try:
        try:
            audio = AudioSegment.from_file(BytesIO(audio_bytes), format=fmt)
            audio = (
                audio
                .set_channels(1)
                .set_frame_rate(16_000)
                .set_sample_width(2)
            )
            logger.debug("Loaded audio as '%s': %.1f s", fmt, len(audio) / 1000)
            return audio
        except Exception as exc:
            logger.debug("Could not load as '%s': %s", fmt, exc)
    return None


def _trim_silence(segment: AudioSegment) -> AudioSegment:
    """Strip leading and trailing silence."""
    try:
        parts = split_on_silence(
            segment,
            min_silence_len=_MIN_SILENCE_MS,
            silence_thresh=_SILENCE_THRESH,
            keep_silence=200,
        )
        if parts:
            return sum(parts[1:], parts[0])
    except Exception as exc:
        logger.debug("Silence trimming skipped: %s", exc)
    return segment


def _split_into_chunks(segment: AudioSegment) -> list[AudioSegment]:
    """Split into overlapping fixed-size chunks for Google STT."""
    duration_ms = len(segment)
    if duration_ms <= _CHUNK_MS:
        return [segment]
    chunks: list[AudioSegment] = []
    start = 0
    while start < duration_ms:
        end = min(start + _CHUNK_MS, duration_ms)
        chunks.append(segment[start:end])
        if end == duration_ms:
            break
        start = end - _OVERLAP_MS
    return chunks


def _segment_to_wav_bytes(segment: AudioSegment) -> bytes:
    """Export a pydub AudioSegment to in-memory WAV bytes."""
    buf = BytesIO()
    segment.export(buf, format="wav")
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────
# Public utilities
# ──────────────────────────────────────────────────────────────────────

def get_supported_languages() -> dict[str, str]:
    """Return a copy of the supported language code → name mapping."""
    return SUPPORTED_LANGUAGES.copy()


def estimate_audio_duration(audio_bytes: bytes) -> float:
    """Estimate audio duration in seconds. Returns 0.0 on failure."""
    try:
        fmt     = _detect_format(audio_bytes)
        segment = _load_and_normalise(audio_bytes, fmt)
        if segment:
            return len(segment) / 1000.0
    except Exception as exc:
        logger.warning("Could not estimate audio duration: %s", exc)
    return 0.0