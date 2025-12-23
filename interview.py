#!/usr/bin/env python3
"""
Radio SYD - Interview Loop
Semi-live interview system with SYD as procedural interviewer.

Flow:
1. Claude generates SYD text
2. Google TTS renders audio
3. Audio plays
4. User responds (push-to-talk)
5. Whisper transcribes
6. Loop continues
"""

import os
import sys
import time
import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv

# Load environment (override=True to use .env over shell variables)
load_dotenv(override=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GOOGLE_TTS_VOICE = os.getenv("GOOGLE_TTS_VOICE", "en-US-Neural2-D")
GOOGLE_TTS_LANGUAGE = os.getenv("GOOGLE_TTS_LANGUAGE", "en-US")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))

# Paths
BASE_DIR = Path(__file__).parent
AUDIO_TTS_DIR = BASE_DIR / "audio" / "tts"
AUDIO_REC_DIR = BASE_DIR / "audio" / "recordings"
SCRIPTS_DIR = BASE_DIR / "scripts"
SYSTEM_PROMPT_PATH = BASE_DIR / "syd_system_prompt.txt"

# ============================================================================
# LAZY IMPORTS (heavy libraries loaded on demand)
# ============================================================================

_anthropic_client = None
_tts_client = None
_whisper_model = None


def get_anthropic_client():
    """Lazy load Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


def get_tts_client():
    """Lazy load Google TTS client."""
    global _tts_client
    if _tts_client is None:
        from google.cloud import texttospeech
        _tts_client = texttospeech.TextToSpeechClient()
    return _tts_client


def get_whisper_model():
    """Lazy load Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print(f"Loading Whisper model '{WHISPER_MODEL}'...")
        _whisper_model = whisper.load_model(WHISPER_MODEL)
    return _whisper_model


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def load_system_prompt() -> str:
    """Load SYD system prompt from file."""
    with open(SYSTEM_PROMPT_PATH, "r") as f:
        return f.read()


def generate_syd_response(conversation: list[dict], system_prompt: str) -> str:
    """Generate SYD's next line using Claude."""
    client = get_anthropic_client()

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=120,  # Punchy but room for mysterious asides
        system=system_prompt,
        messages=conversation
    )

    return response.content[0].text.strip()


def generate_syd_response_with_usage(conversation: list[dict], system_prompt: str) -> tuple[str, dict]:
    """Generate SYD's next line and return token usage."""
    client = get_anthropic_client()

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=120,
        system=system_prompt,
        messages=conversation
    )

    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens
    }

    return response.content[0].text.strip(), usage


def text_to_speech(text: str, output_path: Path) -> Path:
    """Convert text to speech using Google Cloud TTS."""
    from google.cloud import texttospeech

    client = get_tts_client()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=GOOGLE_TTS_LANGUAGE,
        name=GOOGLE_TTS_VOICE
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        speaking_rate=1.05,  # Quicker, more energetic
        pitch=-1.5  # Subtle depth
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    with open(output_path, "wb") as f:
        f.write(response.audio_content)

    return output_path


def play_audio(file_path: Path):
    """Play audio file through speakers."""
    data, samplerate = sf.read(file_path)
    sd.play(data, samplerate)
    sd.wait()


def record_audio(max_duration: float = 60.0) -> np.ndarray:
    """
    Record audio until user presses Enter.
    Returns numpy array of audio data.
    """
    print("\n  [Recording... Press ENTER when done]\n")

    recording = []
    is_recording = True

    def callback(indata, frames, time_info, status):
        if is_recording:
            recording.append(indata.copy())

    # Start recording in background
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32,
        callback=callback
    )

    with stream:
        input()  # Wait for Enter
        is_recording = False

    if recording:
        return np.concatenate(recording, axis=0)
    return np.array([])


def transcribe_audio(audio_data: np.ndarray) -> str:
    """Transcribe audio using Whisper."""
    model = get_whisper_model()

    # Ensure audio is float32 and normalized for Whisper
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # Ensure mono
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Normalize to [-1, 1] range if needed
    max_val = np.abs(audio_data).max()
    if max_val > 1.0:
        audio_data = audio_data / max_val

    # Pass numpy array directly to Whisper (avoids ffmpeg dependency)
    result = model.transcribe(audio_data, language="en")
    return result["text"].strip()


def save_session(session_id: str, transcript: list[dict], audio_files: list[Path]):
    """Save session transcript and metadata."""
    session_dir = SCRIPTS_DIR / session_id
    session_dir.mkdir(exist_ok=True)

    # Save transcript
    transcript_path = session_dir / "transcript.json"
    with open(transcript_path, "w") as f:
        json.dump(transcript, f, indent=2)

    # Save human-readable transcript
    readable_path = session_dir / "transcript.txt"
    with open(readable_path, "w") as f:
        f.write(f"SESSION: {session_id}\n")
        f.write(f"DATE: {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")

        for entry in transcript:
            speaker = entry["role"].upper()
            if speaker == "ASSISTANT":
                speaker = "SYD"
            elif speaker == "USER":
                speaker = "SUBJECT"
            f.write(f"{speaker}:\n{entry['content']}\n\n")

    print(f"\n  Session saved to: {session_dir}")


# ============================================================================
# INTERVIEW LOOP
# ============================================================================

def run_interview(topic: str = None):
    """
    Main interview loop.

    Args:
        topic: Optional topic/context for the review
    """
    # Generate session ID
    session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")

    print("\n" + "=" * 60)
    print("  RADIO SYD - INTERVIEW SESSION")
    print("=" * 60)
    print(f"  Session ID: {session_id}")
    print(f"  TTS Voice: {GOOGLE_TTS_VOICE}")
    print(f"  Whisper Model: {WHISPER_MODEL}")
    print("=" * 60)

    # Load system prompt
    system_prompt = load_system_prompt()

    # Add topic context if provided
    if topic:
        system_prompt += f"\n\nREVIEW CONTEXT:\nThis session concerns: {topic}"

    # Initialize conversation
    conversation = []
    transcript = []
    audio_files = []
    turn_count = 0

    print("\n  Initializing SYD...\n")

    # Get opening statement
    syd_text = generate_syd_response(conversation, system_prompt)

    try:
        while True:
            turn_count += 1

            # Display SYD's text
            print(f"  SYD: {syd_text}")

            # Generate and play TTS
            tts_filename = f"SYD_{turn_count:02d}.wav"
            tts_path = AUDIO_TTS_DIR / session_id
            tts_path.mkdir(parents=True, exist_ok=True)
            tts_file = tts_path / tts_filename

            text_to_speech(syd_text, tts_file)
            audio_files.append(tts_file)

            print("\n  [Playing SYD audio...]")
            play_audio(tts_file)

            # Add to conversation history
            conversation.append({"role": "assistant", "content": syd_text})
            transcript.append({"role": "assistant", "content": syd_text})

            # Check for session end
            if "concludes" in syd_text.lower() and "review" in syd_text.lower():
                print("\n  [Session concluded by SYD]")
                break

            # Record user response
            print("\n  Your response:")
            audio_data = record_audio()

            if len(audio_data) == 0:
                print("  [No audio recorded]")
                continue

            # Save recording
            rec_path = AUDIO_REC_DIR / session_id
            rec_path.mkdir(parents=True, exist_ok=True)
            rec_file = rec_path / f"SUBJECT_{turn_count:02d}.wav"
            sf.write(rec_file, audio_data, SAMPLE_RATE)
            audio_files.append(rec_file)

            # Transcribe
            print("  [Transcribing...]")
            user_text = transcribe_audio(audio_data)
            print(f"  YOU: {user_text}")

            if not user_text:
                print("  [No speech detected, continuing...]")
                continue

            # Add to conversation
            conversation.append({"role": "user", "content": user_text})
            transcript.append({"role": "user", "content": user_text})

            # Generate next SYD response
            print("\n  [SYD processing...]")
            syd_text = generate_syd_response(conversation, system_prompt)

    except KeyboardInterrupt:
        print("\n\n  [Session interrupted]")

    finally:
        # Save session
        save_session(session_id, transcript, audio_files)

    print("\n" + "=" * 60)
    print("  SESSION COMPLETE")
    print("=" * 60)
    print(f"\n  Audio files: {AUDIO_TTS_DIR / session_id}")
    print(f"  Recordings:  {AUDIO_REC_DIR / session_id}")
    print(f"  Transcript:  {SCRIPTS_DIR / session_id}")
    print()


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Radio SYD - Semi-live archival interview system"
    )
    parser.add_argument(
        "--topic", "-t",
        type=str,
        help="Topic or context for the review session"
    )
    parser.add_argument(
        "--check", "-c",
        action="store_true",
        help="Check configuration and dependencies"
    )

    args = parser.parse_args()

    if args.check:
        print("\nChecking configuration...\n")

        errors = []

        # Check API key
        if not ANTHROPIC_API_KEY:
            errors.append("ANTHROPIC_API_KEY not set in .env")
        else:
            print("  [OK] ANTHROPIC_API_KEY")

        # Check Google credentials
        if not GOOGLE_APPLICATION_CREDENTIALS:
            errors.append("GOOGLE_APPLICATION_CREDENTIALS not set in .env")
        elif not Path(GOOGLE_APPLICATION_CREDENTIALS).exists():
            errors.append(f"Google credentials file not found: {GOOGLE_APPLICATION_CREDENTIALS}")
        else:
            print("  [OK] GOOGLE_APPLICATION_CREDENTIALS")

        # Check system prompt
        if not SYSTEM_PROMPT_PATH.exists():
            errors.append(f"System prompt not found: {SYSTEM_PROMPT_PATH}")
        else:
            print("  [OK] System prompt file")

        # Check directories
        for d in [AUDIO_TTS_DIR, AUDIO_REC_DIR, SCRIPTS_DIR]:
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
            print(f"  [OK] Directory: {d}")

        # Test imports
        try:
            import anthropic
            print("  [OK] anthropic")
        except ImportError:
            errors.append("anthropic not installed (pip install anthropic)")

        try:
            from google.cloud import texttospeech
            print("  [OK] google-cloud-texttospeech")
        except ImportError:
            errors.append("google-cloud-texttospeech not installed")

        try:
            import whisper
            print("  [OK] whisper")
        except ImportError:
            errors.append("openai-whisper not installed")

        try:
            import sounddevice
            print("  [OK] sounddevice")
        except ImportError:
            errors.append("sounddevice not installed")

        print()

        if errors:
            print("ERRORS:")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)
        else:
            print("All checks passed. Ready to run.")
            sys.exit(0)

    # Run interview
    run_interview(topic=args.topic)


if __name__ == "__main__":
    main()
