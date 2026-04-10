# utils/speech_to_text.py

import os
import tempfile
import speech_recognition as sr


def transcribe_audio(audio_bytes: bytes, language: str = "en-IN") -> str:
    """
    Transcribes audio bytes to text using Google Speech Recognition.
    Accepts WAV audio bytes from Streamlit WebRTC.
    Returns clean transcript or an error message.
    """
    if not audio_bytes:
        return "No audio data received."

    recognizer = sr.Recognizer()

    # Save audio bytes to a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(audio_bytes)
        temp_file_path = temp_file.name

    try:
        # Load audio from the temporary file
        with sr.AudioFile(temp_file_path) as source:
            audio_data = recognizer.record(source)

        # Transcribe using Google API
        text = recognizer.recognize_google(audio_data, language=language)
        return text

    except sr.UnknownValueError:
        return "Could not understand audio clearly"
    except sr.RequestError:
        return "Speech service unavailable"
    except Exception as e:
        return f"Transcription error: {str(e)}"
    finally:
        # Ensure temporary file is deleted
        try:
            os.unlink(temp_file_path)
        except OSError:
            pass  # File may already be deleted or inaccessible


def _detect_format(data: bytes) -> str:
    """Detects audio format by reading file header bytes."""
    if data[:4] == b"RIFF":
        return "wav"
    if data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "mp3"
    if data[:4] == b"fLaC":
        return "flac"
    if data[:4] == b"OggS":
        return "ogg"
    if data[4:8] == b"ftyp" or data[:4] == b"\x00\x00\x00\x1c":
        return "m4a"
    return "wav"  # default fallback
