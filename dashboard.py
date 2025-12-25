#!/usr/bin/env python3
"""
Radio SYD - Dashboard
Web interface for managing sessions and recordings.
"""

import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from flask import Flask, render_template, send_from_directory, abort, request, redirect, url_for, jsonify

# Import interview functions
from interview import (
    generate_syd_response,
    generate_syd_response_with_usage,
    transcribe_audio,
    load_system_prompt,
    save_session,
    SAMPLE_RATE,
    GOOGLE_TTS_LANGUAGE,
    get_tts_client
)

# Available SYD voices
SYD_VOICES = {
    # OpenAI TTS (Best quality)
    "openai:onyx": "OpenAI Onyx - Deep (Recommended)",
    "openai:echo": "OpenAI Echo - Neutral",
    "openai:fable": "OpenAI Fable - Expressive",
    "openai:alloy": "OpenAI Alloy - Balanced",
    "openai:nova": "OpenAI Nova - Warm",
    "openai:shimmer": "OpenAI Shimmer - Clear",
    # Google Neural2 (High quality, natural)
    "en-US-Neural2-D": "Google US Neural2 - Neutral",
    "en-US-Neural2-A": "Google US Neural2 - Deeper",
    "en-US-Neural2-I": "Google US Neural2 - Authoritative",
    "en-US-Neural2-J": "Google US Neural2 - Calm",
    "en-GB-Neural2-B": "Google British Neural2 - Formal",
    "en-GB-Neural2-D": "Google British Neural2 - Neutral",
    # Google Wavenet
    "en-US-Wavenet-D": "Google US Wavenet - D",
    "en-GB-Wavenet-B": "Google British Wavenet - B",
    # Google News (Broadcast)
    "en-US-News-N": "Google US News - Male",
    "en-GB-News-J": "Google British News - J",
}

DEFAULT_VOICE = os.getenv("SYD_VOICE", "openai:onyx")

# Available LLM models for SYD
SYD_MODELS = {
    # Claude models
    "claude-sonnet-4": "Claude Sonnet 4 - Fast, capable (Default)",
    "claude-opus-4": "Claude Opus 4 - Most nuanced, best for subtle cues",
    # OpenAI models
    "gpt-4o": "GPT-4o - OpenAI's flagship model",
    "gpt-4o-mini": "GPT-4o Mini - Fast and affordable",
}

DEFAULT_MODEL = os.getenv("SYD_MODEL", "claude-sonnet-4")

# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR = Path(__file__).parent
SCRIPTS_DIR = BASE_DIR / "scripts"
AUDIO_TTS_DIR = BASE_DIR / "audio" / "tts"
AUDIO_REC_DIR = BASE_DIR / "audio" / "recordings"
AUDIO_EPISODES_DIR = BASE_DIR / "audio" / "episodes"
AUDIO_SFX_DIR = BASE_DIR / "audio" / "sfx"
AUDIO_INTROS_DIR = BASE_DIR / "audio" / "intros"
PROFILES_DIR = BASE_DIR / "profiles"

app = Flask(__name__)

# ============================================================================
# HELPERS
# ============================================================================

def apply_robotic_effect(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply bandpass filter and compression for institutional/intercom sound."""
    from scipy import signal

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


def strip_stage_directions(text: str) -> str:
    """Remove stage directions (text between asterisks) from speech."""
    import re
    # Remove *anything between asterisks*
    cleaned = re.sub(r'\*[^*]+\*', '', text)
    # Remove (parenthetical directions)
    cleaned = re.sub(r'\([^)]+\)', '', cleaned)
    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


# Pronunciation fixes file path
PRONUNCIATION_FILE = BASE_DIR / "pronunciation.json"


def load_pronunciation_fixes() -> dict:
    """Load pronunciation fixes from JSON file."""
    if PRONUNCIATION_FILE.exists():
        with open(PRONUNCIATION_FILE) as f:
            return json.load(f)
    return {}


def save_pronunciation_fixes(fixes: dict):
    """Save pronunciation fixes to JSON file."""
    with open(PRONUNCIATION_FILE, "w") as f:
        json.dump(fixes, f, indent=2, ensure_ascii=False)


def fix_pronunciation(text: str) -> str:
    """Replace words with phonetic equivalents for better TTS pronunciation."""
    import re
    fixes = load_pronunciation_fixes()
    for word, phonetic in fixes.items():
        # Case-insensitive replacement
        text = re.sub(re.escape(word), phonetic, text, flags=re.IGNORECASE)
    return text


def text_to_speech_with_voice(text: str, output_path: Path, voice_name: str = None, robotic: bool = False) -> Path:
    """Convert text to speech using OpenAI or Google Cloud TTS."""
    # Strip any stage directions before TTS
    text = strip_stage_directions(text)
    # Fix pronunciation for non-English words
    text = fix_pronunciation(text)
    voice_name = voice_name or DEFAULT_VOICE

    # Check if using OpenAI voice
    if voice_name.startswith("openai:"):
        return _openai_tts(text, output_path, voice_name.split(":")[1])

    # Otherwise use Google Cloud TTS
    return _google_tts(text, output_path, voice_name, robotic)


def _openai_tts(text: str, output_path: Path, voice: str) -> Path:
    """Generate speech using OpenAI TTS."""
    from openai import OpenAI

    client = OpenAI()

    # Generate speech (returns MP3)
    response = client.audio.speech.create(
        model="tts-1-hd",
        voice=voice,
        speed=1.05,  # Subtle energy boost
        input=text
    )

    # Save as MP3 first, then convert to WAV for consistency
    mp3_path = output_path.with_suffix(".mp3")
    with open(mp3_path, "wb") as f:
        for chunk in response.iter_bytes():
            f.write(chunk)

    # Convert MP3 to WAV using ffmpeg
    subprocess.run([
        "/opt/homebrew/bin/ffmpeg", "-y", "-i", str(mp3_path),
        "-ar", str(SAMPLE_RATE), "-ac", "1",
        str(output_path)
    ], capture_output=True)
    mp3_path.unlink()  # Remove temp MP3

    return output_path


def _google_tts(text: str, output_path: Path, voice_name: str, robotic: bool = False) -> Path:
    """Generate speech using Google Cloud TTS."""
    from google.cloud import texttospeech

    client = get_tts_client()

    # Determine language code from voice name
    lang_code = "-".join(voice_name.split("-")[:2])

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=lang_code,
        name=voice_name
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        speaking_rate=0.92,  # Measured, deliberate
        pitch=-1.5  # Subtle depth
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    # Save raw audio first
    with open(output_path, "wb") as f:
        f.write(response.audio_content)

    # Apply robotic effect if enabled
    if robotic:
        audio_data, sr = sf.read(output_path)
        processed = apply_robotic_effect(audio_data, sr)
        sf.write(output_path, processed, sr)

    return output_path


def assemble_episode(session_id: str, gap_ms: int = 800, ambiance_volume: float = 0.0) -> Path:
    """
    Assemble session audio files into a single episode.

    Args:
        session_id: The session to assemble
        gap_ms: Silence gap between turns in milliseconds
        ambiance_volume: Volume level for server room ambiance (0.0-1.0)

    Returns:
        Path to the assembled episode file
    """
    tts_dir = AUDIO_TTS_DIR / session_id
    rec_dir = AUDIO_REC_DIR / session_id
    episodes_dir = AUDIO_EPISODES_DIR
    episodes_dir.mkdir(parents=True, exist_ok=True)

    # Get sorted audio files
    syd_files = sorted(tts_dir.glob("SYD_*.wav")) if tts_dir.exists() else []
    subject_files = sorted(rec_dir.glob("SUBJECT_*.wav")) if rec_dir.exists() else []

    if not syd_files:
        raise ValueError("No SYD audio files found")

    # Build list of audio segments
    segments = []
    gap_samples = int(SAMPLE_RATE * gap_ms / 1000)
    silence = np.zeros(gap_samples, dtype=np.float32)

    # Add intro at the beginning
    intro_file = AUDIO_INTROS_DIR / "syd_intro.wav"
    if intro_file.exists():
        intro_audio, intro_sr = sf.read(intro_file)
        if intro_sr != SAMPLE_RATE:
            from scipy import signal
            num_samples = int(len(intro_audio) * SAMPLE_RATE / intro_sr)
            intro_audio = signal.resample(intro_audio, num_samples)
        # Convert to mono if stereo
        if len(intro_audio.shape) > 1:
            intro_audio = intro_audio.mean(axis=1)
        segments.append(intro_audio.astype(np.float32))
        # Small gap after intro
        segments.append(np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32))  # 0.5s gap

    # Add phone ringing after intro
    phone_ring_file = AUDIO_SFX_DIR / "phone-ringing.wav"
    if phone_ring_file.exists():
        ring_audio, ring_sr = sf.read(phone_ring_file)
        if ring_sr != SAMPLE_RATE:
            from scipy import signal
            num_samples = int(len(ring_audio) * SAMPLE_RATE / ring_sr)
            ring_audio = signal.resample(ring_audio, num_samples)
        # Convert to mono if stereo
        if len(ring_audio.shape) > 1:
            ring_audio = ring_audio.mean(axis=1)
        segments.append(ring_audio.astype(np.float32))
        # Gap after phone ring before SYD speaks
        segments.append(np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.float32))  # 1s gap

    for i, syd_file in enumerate(syd_files):
        # Add SYD's audio
        syd_audio, sr = sf.read(syd_file)
        if sr != SAMPLE_RATE:
            # Resample if needed
            from scipy import signal
            num_samples = int(len(syd_audio) * SAMPLE_RATE / sr)
            syd_audio = signal.resample(syd_audio, num_samples)
        segments.append(syd_audio.astype(np.float32))

        # Add subject response if available
        if i < len(subject_files):
            segments.append(silence.copy())
            subject_audio, sr = sf.read(subject_files[i])
            if sr != SAMPLE_RATE:
                from scipy import signal
                num_samples = int(len(subject_audio) * SAMPLE_RATE / sr)
                subject_audio = signal.resample(subject_audio, num_samples)
            segments.append(subject_audio.astype(np.float32))

            # Add gap after subject (before next SYD line)
            if i < len(syd_files) - 1:
                segments.append(silence.copy())

    # Concatenate all segments
    episode_audio = np.concatenate(segments)

    # Mix in server room ambiance
    ambiance_file = AUDIO_SFX_DIR / "server-room-ambience.wav"
    if ambiance_file.exists() and ambiance_volume > 0:
        ambiance_audio, amb_sr = sf.read(ambiance_file)

        # Resample if needed
        if amb_sr != SAMPLE_RATE:
            from scipy import signal
            num_samples = int(len(ambiance_audio) * SAMPLE_RATE / amb_sr)
            ambiance_audio = signal.resample(ambiance_audio, num_samples)

        # Convert to mono if stereo
        if len(ambiance_audio.shape) > 1:
            ambiance_audio = ambiance_audio.mean(axis=1)

        ambiance_audio = ambiance_audio.astype(np.float32)

        # Loop ambiance to match episode length
        episode_len = len(episode_audio)
        ambiance_len = len(ambiance_audio)
        if ambiance_len < episode_len:
            # Tile/loop the ambiance
            repeats = (episode_len // ambiance_len) + 1
            ambiance_audio = np.tile(ambiance_audio, repeats)[:episode_len]
        else:
            ambiance_audio = ambiance_audio[:episode_len]

        # Mix ambiance at specified volume
        episode_audio = episode_audio + (ambiance_audio * ambiance_volume)

    # Normalize to prevent clipping
    max_val = np.abs(episode_audio).max()
    if max_val > 0.95:
        episode_audio = episode_audio * 0.95 / max_val

    # Export
    output_path = episodes_dir / f"{session_id}.wav"
    sf.write(output_path, episode_audio, SAMPLE_RATE)

    return output_path


def get_sessions():
    """Get all sessions with metadata."""
    sessions = []

    if not SCRIPTS_DIR.exists():
        return sessions

    for session_dir in sorted(SCRIPTS_DIR.iterdir(), reverse=True):
        if not session_dir.is_dir() or session_dir.name.startswith('.'):
            continue

        transcript_json = session_dir / "transcript.json"
        transcript_txt = session_dir / "transcript.txt"

        # Parse session ID for date
        # Format: session_YYYYMMDD_HHMMSS
        try:
            date_str = session_dir.name.replace("session_", "")
            date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
        except ValueError:
            date = None

        # Count turns
        turn_count = 0
        if transcript_json.exists():
            try:
                with open(transcript_json) as f:
                    transcript = json.load(f)
                    turn_count = len([t for t in transcript if t.get("role") == "user"])
            except:
                pass

        # Check for audio files
        tts_dir = AUDIO_TTS_DIR / session_dir.name
        rec_dir = AUDIO_REC_DIR / session_dir.name

        syd_clips = list(tts_dir.glob("*.wav")) if tts_dir.exists() else []
        subject_clips = list(rec_dir.glob("*.wav")) if rec_dir.exists() else []

        sessions.append({
            "id": session_dir.name,
            "date": date,
            "date_formatted": date.strftime("%Y-%m-%d %H:%M") if date else "Unknown",
            "turn_count": turn_count,
            "syd_clips": len(syd_clips),
            "subject_clips": len(subject_clips),
            "has_transcript": transcript_txt.exists(),
        })

    return sessions


def get_session_detail(session_id):
    """Get full session details including transcript and audio."""
    session_dir = SCRIPTS_DIR / session_id

    if not session_dir.exists():
        return None

    # Load transcript
    transcript = []
    transcript_json = session_dir / "transcript.json"
    if transcript_json.exists():
        with open(transcript_json) as f:
            transcript = json.load(f)

    # Get audio files
    tts_dir = AUDIO_TTS_DIR / session_id
    rec_dir = AUDIO_REC_DIR / session_id

    syd_clips = sorted(tts_dir.glob("*.wav")) if tts_dir.exists() else []
    subject_clips = sorted(rec_dir.glob("*.wav")) if rec_dir.exists() else []

    # Build timeline
    timeline = []
    syd_idx = 0
    subject_idx = 0

    for entry in transcript:
        if entry["role"] == "assistant":
            audio_file = syd_clips[syd_idx].name if syd_idx < len(syd_clips) else None
            timeline.append({
                "speaker": "SYD",
                "content": entry["content"],
                "audio": audio_file,
                "audio_type": "tts"
            })
            syd_idx += 1
        elif entry["role"] == "user":
            audio_file = subject_clips[subject_idx].name if subject_idx < len(subject_clips) else None
            timeline.append({
                "speaker": "SUBJECT",
                "content": entry["content"],
                "audio": audio_file,
                "audio_type": "recordings"
            })
            subject_idx += 1

    # Parse date
    try:
        date_str = session_id.replace("session_", "")
        date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
    except ValueError:
        date = None

    return {
        "id": session_id,
        "date": date,
        "date_formatted": date.strftime("%Y-%m-%d %H:%M") if date else "Unknown",
        "timeline": timeline,
        "syd_clips": [c.name for c in syd_clips],
        "subject_clips": [c.name for c in subject_clips],
    }


# ============================================================================
# ROUTES
# ============================================================================

@app.route("/")
def index():
    """Session list."""
    sessions = get_sessions()
    return render_template("index.html", sessions=sessions)


@app.route("/session/<session_id>")
def session_detail(session_id):
    """Single session view."""
    session = get_session_detail(session_id)
    if not session:
        abort(404)
    return render_template("session.html", session=session)


@app.route("/audio/tts/<session_id>/<filename>")
def serve_tts_audio(session_id, filename):
    """Serve SYD TTS audio files."""
    directory = AUDIO_TTS_DIR / session_id
    if not directory.exists():
        abort(404)
    return send_from_directory(directory, filename)


@app.route("/audio/recordings/<session_id>/<filename>")
def serve_recording_audio(session_id, filename):
    """Serve subject recording audio files."""
    directory = AUDIO_REC_DIR / session_id
    if not directory.exists():
        abort(404)
    return send_from_directory(directory, filename)


@app.route("/start", methods=["POST"])
def start_session():
    """Start a new interview session."""
    import subprocess

    topic = request.form.get("topic", "").strip()
    instructions = request.form.get("instructions", "").strip()

    # Build the command
    cmd_parts = [
        f"cd {BASE_DIR}",
        "source venv/bin/activate",
    ]

    interview_cmd = "python interview.py"
    if topic:
        # Combine topic and instructions
        full_topic = topic
        if instructions:
            full_topic += f". Additional focus: {instructions}"
        # Escape quotes for shell
        full_topic = full_topic.replace('"', '\\"')
        interview_cmd += f' --topic "{full_topic}"'

    cmd_parts.append(interview_cmd)
    full_cmd = " && ".join(cmd_parts)

    # Open Terminal with the command (macOS)
    apple_script = f'''
    tell application "Terminal"
        activate
        do script "{full_cmd}"
    end tell
    '''

    subprocess.run(["osascript", "-e", apple_script])

    return redirect(url_for("index"))


@app.route("/delete/<session_id>", methods=["POST"])
def delete_session(session_id):
    """Delete a session."""
    import shutil

    # Delete transcript
    session_dir = SCRIPTS_DIR / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)

    # Delete TTS audio
    tts_dir = AUDIO_TTS_DIR / session_id
    if tts_dir.exists():
        shutil.rmtree(tts_dir)

    # Delete recordings
    rec_dir = AUDIO_REC_DIR / session_id
    if rec_dir.exists():
        shutil.rmtree(rec_dir)

    return redirect(url_for("index"))


# ============================================================================
# LIVE INTERVIEW API
# ============================================================================

# In-memory session state (for active sessions)
_active_sessions = {}

# Global API usage tracking
_api_usage = {
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_requests": 0
}


def get_session_state_path(session_id: str) -> Path:
    """Get path to session state file."""
    return SCRIPTS_DIR / session_id / "state.json"


def save_session_state(session_id: str, state: dict):
    """Save session state to disk for recovery."""
    state_path = get_session_state_path(session_id)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def load_session_state(session_id: str) -> dict | None:
    """Load session state from disk."""
    state_path = get_session_state_path(session_id)
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return None


@app.route("/interview/live")
def interview_live():
    """Live interview page."""
    guest_name = request.args.get("guest_name", "")
    focus_work = request.args.get("focus_work", "")
    return render_template("interview_live.html", guest_name=guest_name, focus_work=focus_work)


@app.route("/mic-test")
def mic_test():
    """Microphone test page."""
    return render_template("mic_test.html")


@app.route("/api/session/create", methods=["POST"])
def api_create_session():
    """Create a new interview session and generate SYD's opening."""
    data = request.json or {}
    guest_name = data.get("guest_name", "").strip()
    context = data.get("context", "").strip()
    voice = data.get("voice", DEFAULT_VOICE)
    model = data.get("model", DEFAULT_MODEL)
    text_only = data.get("text_only", False)

    # Validate voice and model
    if voice not in SYD_VOICES:
        voice = DEFAULT_VOICE
    if model not in SYD_MODELS:
        model = DEFAULT_MODEL

    # Generate session ID
    session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")

    # Load and customize system prompt
    system_prompt = load_system_prompt()
    if context:
        system_prompt += f"\n\nINTERVIEW CONTEXT:\n{context}"

    # Generate SYD's opening (need initial user message to prompt Claude)
    conversation = [{"role": "user", "content": "Begin the interview."}]
    syd_text, usage = generate_syd_response_with_usage(conversation, system_prompt, model)

    # Track API usage
    _api_usage["total_input_tokens"] += usage["input_tokens"]
    _api_usage["total_output_tokens"] += usage["output_tokens"]
    _api_usage["total_requests"] += 1

    # Generate TTS with selected voice (unless text-only mode)
    audio_url = None
    if not text_only:
        tts_dir = AUDIO_TTS_DIR / session_id
        tts_dir.mkdir(parents=True, exist_ok=True)
        tts_file = tts_dir / "SYD_01.wav"
        text_to_speech_with_voice(syd_text, tts_file, voice)
        audio_url = f"/audio/tts/{session_id}/SYD_01.wav"

    # Store session state
    state = {
        "session_id": session_id,
        "system_prompt": system_prompt,
        "conversation": [{"role": "assistant", "content": syd_text}],
        "turn_count": 1,
        "is_complete": False,
        "voice": voice,
        "model": model,
        "text_only": text_only
    }
    _active_sessions[session_id] = state
    save_session_state(session_id, state)

    return jsonify({
        "session_id": session_id,
        "syd_text": syd_text,
        "audio_url": audio_url,
        "turn_count": 1,
        "voice": voice,
        "model": model,
        "text_only": text_only
    })


@app.route("/api/session/<session_id>/respond", methods=["POST"])
def api_session_respond(session_id):
    """Process user response and generate SYD's reply."""
    data = request.json or {}
    user_text = data.get("user_text", "").strip()

    if not user_text:
        return jsonify({"error": "No user text provided"}), 400

    # Get session state
    state = _active_sessions.get(session_id) or load_session_state(session_id)
    if not state:
        return jsonify({"error": "Session not found"}), 404

    # Add user response to conversation
    state["conversation"].append({"role": "user", "content": user_text})

    # Generate SYD's response with selected model
    model = state.get("model", DEFAULT_MODEL)
    syd_text, usage = generate_syd_response_with_usage(state["conversation"], state["system_prompt"], model)
    state["conversation"].append({"role": "assistant", "content": syd_text})
    state["turn_count"] += 1

    # Track API usage
    _api_usage["total_input_tokens"] += usage["input_tokens"]
    _api_usage["total_output_tokens"] += usage["output_tokens"]
    _api_usage["total_requests"] += 1

    # Check if session is complete (SYD concludes the review)
    is_complete = "concludes" in syd_text.lower() and "review" in syd_text.lower()
    state["is_complete"] = is_complete

    # Generate TTS with session's voice (unless text-only mode)
    audio_url = None
    text_only = state.get("text_only", False)
    if not text_only:
        voice = state.get("voice", DEFAULT_VOICE)
        tts_dir = AUDIO_TTS_DIR / session_id
        tts_dir.mkdir(parents=True, exist_ok=True)
        tts_file = tts_dir / f"SYD_{state['turn_count']:02d}.wav"
        text_to_speech_with_voice(syd_text, tts_file, voice)
        audio_url = f"/audio/tts/{session_id}/{tts_file.name}"

    # Save user recording reference (audio saved separately via /api/transcribe)
    rec_dir = AUDIO_REC_DIR / session_id
    rec_dir.mkdir(parents=True, exist_ok=True)

    # Update state
    _active_sessions[session_id] = state
    save_session_state(session_id, state)

    # If complete, also save the final transcript
    if is_complete:
        save_session(session_id, state["conversation"], [])

    return jsonify({
        "syd_text": syd_text,
        "audio_url": audio_url,
        "turn_count": state["turn_count"],
        "is_complete": is_complete,
        "model": model
    })


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    """Transcribe audio from browser."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    session_id = request.form.get("session_id", "")

    # Determine file extension from content type
    content_type = audio_file.content_type or "audio/webm"
    if "mp4" in content_type or "m4a" in content_type:
        suffix = ".mp4"
    elif "ogg" in content_type:
        suffix = ".ogg"
    else:
        suffix = ".webm"

    # Save to temp file for processing
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        temp_input = f.name
        audio_file.save(temp_input)

    try:
        # Convert to WAV using ffmpeg (if needed)
        temp_wav = temp_input.rsplit(".", 1)[0] + ".wav"

        # Try to read directly first (in case it's already WAV)
        try:
            audio_data, sr = sf.read(temp_input)
        except Exception as read_err:
            # Convert using ffmpeg (use full path for Homebrew install)
            import subprocess
            result = subprocess.run([
                "/opt/homebrew/bin/ffmpeg", "-y",
                "-f", "webm",  # Force webm input format
                "-i", temp_input,
                "-ar", str(SAMPLE_RATE),
                "-ac", "1",
                "-f", "wav",
                temp_wav
            ], capture_output=True)

            if result.returncode != 0:
                # Log the actual ffmpeg error
                print(f"ffmpeg error: {result.stderr.decode()}")
                # Return error as JSON
                Path(temp_input).unlink(missing_ok=True)
                return jsonify({"error": "Failed to process audio. Please try Chrome browser."}), 400

            audio_data, sr = sf.read(temp_wav)
            Path(temp_wav).unlink(missing_ok=True)

        # Ensure correct sample rate
        if sr != SAMPLE_RATE:
            # Resample using scipy
            from scipy import signal
            num_samples = int(len(audio_data) * SAMPLE_RATE / sr)
            audio_data = signal.resample(audio_data, num_samples)

        # Save recording if session_id provided
        if session_id:
            rec_dir = AUDIO_REC_DIR / session_id
            rec_dir.mkdir(parents=True, exist_ok=True)

            # Get turn count from state
            state = _active_sessions.get(session_id) or load_session_state(session_id)
            turn_num = (len(state["conversation"]) // 2) + 1 if state else 1

            rec_file = rec_dir / f"SUBJECT_{turn_num:02d}.wav"
            sf.write(rec_file, audio_data, SAMPLE_RATE)

        # Transcribe
        text = transcribe_audio(audio_data)

        return jsonify({"text": text})

    finally:
        Path(temp_input).unlink(missing_ok=True)


@app.route("/api/session/<session_id>/state", methods=["GET"])
def api_get_session_state(session_id):
    """Get session state for recovery."""
    state = _active_sessions.get(session_id) or load_session_state(session_id)

    if not state:
        return jsonify({"error": "Session not found"}), 404

    return jsonify({
        "session_id": state["session_id"],
        "conversation": state["conversation"],
        "turn_count": state["turn_count"],
        "is_complete": state.get("is_complete", False)
    })


@app.route("/api/session/<session_id>/end", methods=["POST"])
def api_end_session(session_id):
    """End and save a session."""
    state = _active_sessions.get(session_id) or load_session_state(session_id)

    if not state:
        return jsonify({"error": "Session not found"}), 404

    # Mark as complete
    state["is_complete"] = True
    save_session_state(session_id, state)

    # Save final transcript
    save_session(session_id, state["conversation"], [])

    # Clean up active session
    if session_id in _active_sessions:
        del _active_sessions[session_id]

    return jsonify({"success": True, "session_id": session_id})


@app.route("/api/session/<session_id>/assemble", methods=["POST"])
def api_assemble_session(session_id):
    """Assemble session audio into a single episode file."""
    data = request.json or {}
    gap_ms = data.get("gap_ms", 800)

    try:
        episode_path = assemble_episode(session_id, gap_ms)
        return jsonify({
            "success": True,
            "episode_url": f"/audio/episodes/{session_id}.wav",
            "episode_path": str(episode_path)
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Assembly failed: {str(e)}"}), 500


@app.route("/audio/episodes/<filename>")
def serve_episode(filename):
    """Serve assembled episode files."""
    if not AUDIO_EPISODES_DIR.exists():
        abort(404)
    return send_from_directory(AUDIO_EPISODES_DIR, filename)


@app.route("/api/voices", methods=["GET"])
def api_get_voices():
    """Get available SYD voices."""
    return jsonify({
        "voices": SYD_VOICES,
        "default": DEFAULT_VOICE
    })


@app.route("/api/models", methods=["GET"])
def api_get_models():
    """Get available LLM models for SYD."""
    return jsonify({
        "models": SYD_MODELS,
        "default": DEFAULT_MODEL
    })


@app.route("/api/profiles", methods=["GET"])
def api_get_profiles():
    """Get available interview profiles."""
    profiles = []
    if PROFILES_DIR.exists():
        for f in sorted(PROFILES_DIR.glob("*.json")):
            try:
                with open(f) as pf:
                    data = json.load(pf)
                    profiles.append({
                        "id": f.stem,
                        "name": data.get("name", f.stem),
                        "guest_name": data.get("guest_name", "")
                    })
            except:
                pass
    return jsonify({"profiles": profiles})


@app.route("/api/profiles/<profile_id>", methods=["GET"])
def api_get_profile(profile_id):
    """Get a specific profile."""
    profile_path = PROFILES_DIR / f"{profile_id}.json"
    if not profile_path.exists():
        return jsonify({"error": "Profile not found"}), 404

    with open(profile_path) as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/audio/previews/<filename>")
def serve_voice_preview(filename):
    """Serve voice preview audio files."""
    previews_dir = BASE_DIR / "audio" / "previews"
    if not previews_dir.exists():
        abort(404)
    return send_from_directory(previews_dir, filename)


@app.route("/audio/sfx/<filename>")
def serve_sfx(filename):
    """Serve sound effect files."""
    if not AUDIO_SFX_DIR.exists():
        abort(404)
    return send_from_directory(AUDIO_SFX_DIR, filename)


@app.route("/audio/intros/<filename>")
def serve_intro(filename):
    """Serve intro audio files."""
    if not AUDIO_INTROS_DIR.exists():
        abort(404)
    return send_from_directory(AUDIO_INTROS_DIR, filename)


@app.route("/api/sfx", methods=["GET"])
def api_list_sfx():
    """List available sound effects."""
    sfx = []
    if AUDIO_SFX_DIR.exists():
        for f in AUDIO_SFX_DIR.glob("*.wav"):
            sfx.append({
                "name": f.stem,
                "url": f"/audio/sfx/{f.name}"
            })
        for f in AUDIO_SFX_DIR.glob("*.mp3"):
            sfx.append({
                "name": f.stem,
                "url": f"/audio/sfx/{f.name}"
            })
    return jsonify({"sfx": sfx})


@app.route("/api/usage", methods=["GET"])
def api_get_usage():
    """Get API usage statistics."""
    # Calculate estimated cost (Claude Sonnet pricing: $3/MTok input, $15/MTok output)
    input_cost = (_api_usage["total_input_tokens"] / 1_000_000) * 3.0
    output_cost = (_api_usage["total_output_tokens"] / 1_000_000) * 15.0
    total_cost = input_cost + output_cost

    return jsonify({
        "input_tokens": _api_usage["total_input_tokens"],
        "output_tokens": _api_usage["total_output_tokens"],
        "total_tokens": _api_usage["total_input_tokens"] + _api_usage["total_output_tokens"],
        "requests": _api_usage["total_requests"],
        "estimated_cost_usd": round(total_cost, 4)
    })


# ============================================================================
# PRONUNCIATION MANAGEMENT
# ============================================================================

@app.route("/pronunciation")
def pronunciation_page():
    """Pronunciation management page."""
    return render_template("pronunciation.html")


@app.route("/api/pronunciation", methods=["GET"])
def api_get_pronunciation():
    """Get all pronunciation mappings."""
    fixes = load_pronunciation_fixes()
    return jsonify({"pronunciations": fixes})


@app.route("/api/pronunciation", methods=["POST"])
def api_add_pronunciation():
    """Add a new pronunciation mapping."""
    data = request.json or {}
    word = data.get("word", "").strip()
    phonetic = data.get("phonetic", "").strip()

    if not word or not phonetic:
        return jsonify({"error": "Both word and phonetic are required"}), 400

    fixes = load_pronunciation_fixes()
    fixes[word] = phonetic
    save_pronunciation_fixes(fixes)

    return jsonify({"success": True, "word": word, "phonetic": phonetic})


@app.route("/api/pronunciation/<path:word>", methods=["DELETE"])
def api_delete_pronunciation(word):
    """Delete a pronunciation mapping."""
    fixes = load_pronunciation_fixes()

    if word not in fixes:
        return jsonify({"error": "Word not found"}), 404

    del fixes[word]
    save_pronunciation_fixes(fixes)

    return jsonify({"success": True, "deleted": word})


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  RADIO SYD - DASHBOARD")
    print("=" * 50)
    print("  http://localhost:5001")
    print("=" * 50 + "\n")

    app.run(debug=True, port=5001)
