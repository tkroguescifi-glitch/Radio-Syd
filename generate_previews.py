#!/usr/bin/env python3
"""Generate voice preview audio clips."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import numpy as np
import soundfile as sf
from scipy import signal
from google.cloud import texttospeech

PREVIEW_TEXT = "This session is being recorded. State your designation for the record."
SAMPLE_RATE = 16000


def apply_robotic_effect(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply bandpass filter and compression for institutional/intercom sound."""
    # Bandpass filter (300Hz - 3400Hz) - telephone/intercom effect
    low_freq = 300
    high_freq = 3400
    nyquist = sample_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist

    # Design butterworth bandpass filter
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, audio_data)

    # Simple compression (reduce dynamic range)
    threshold = 0.3
    ratio = 3.0
    compressed = np.where(
        np.abs(filtered) > threshold,
        np.sign(filtered) * (threshold + (np.abs(filtered) - threshold) / ratio),
        filtered
    )

    # Normalize
    max_val = np.abs(compressed).max()
    if max_val > 0:
        compressed = compressed * 0.9 / max_val

    return compressed.astype(np.float32)

VOICES = {
    # Neural2
    "en-US-Neural2-D": "US Neural2 - Neutral (Default)",
    "en-US-Neural2-A": "US Neural2 - Deeper",
    "en-US-Neural2-I": "US Neural2 - Authoritative",
    "en-US-Neural2-J": "US Neural2 - Calm",
    "en-GB-Neural2-B": "British Neural2 - Formal",
    "en-GB-Neural2-D": "British Neural2 - Neutral",
    # Wavenet
    "en-US-Wavenet-A": "US Wavenet - A",
    "en-US-Wavenet-B": "US Wavenet - B",
    "en-US-Wavenet-D": "US Wavenet - D",
    "en-US-Wavenet-I": "US Wavenet - I",
    "en-US-Wavenet-J": "US Wavenet - J",
    "en-GB-Wavenet-B": "British Wavenet - B",
    "en-GB-Wavenet-D": "British Wavenet - D",
    "en-GB-Wavenet-O": "British Wavenet - O",
    # News
    "en-US-News-N": "US News - Male",
    "en-GB-News-J": "British News - J",
    "en-GB-News-K": "British News - K",
    "en-GB-News-L": "British News - L",
    "en-GB-News-M": "British News - M",
    # Afrikaans
    "af-ZA-Standard-A": "Afrikaans - Female",
}

# Special preview text for Afrikaans
PREVIEW_TEXT_AF = "Hierdie sessie word opgeneem. Noem jou aanwysing vir die rekord."

preview_dir = Path("audio/previews")
preview_dir.mkdir(parents=True, exist_ok=True)

client = texttospeech.TextToSpeechClient()

for voice_name, label in VOICES.items():
    print(f"Generating: {label}...")

    lang_code = "-".join(voice_name.split("-")[:2])

    # Use Afrikaans text for Afrikaans voice
    text = PREVIEW_TEXT_AF if voice_name.startswith("af-") else PREVIEW_TEXT

    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=lang_code,
        name=voice_name
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        speaking_rate=0.85,  # Slower for procedural tone
        pitch=-4.0  # Lower pitch
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    output_path = preview_dir / f"{voice_name}.wav"
    with open(output_path, "wb") as f:
        f.write(response.audio_content)

    # Apply robotic effect
    audio_data, sr = sf.read(output_path)
    processed = apply_robotic_effect(audio_data, sr)
    sf.write(output_path, processed, sr)

    print(f"  Saved: {output_path}")

print()
print("Done! All previews generated.")
