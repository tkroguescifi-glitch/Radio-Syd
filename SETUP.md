# Radio SYD - Setup Guide

## Prerequisites

- Python 3.10+
- Microphone
- Speakers/headphones
- Google Cloud account
- Anthropic API key

---

## 1. Install Dependencies

```bash
cd /Users/gerhardvanonselen/Development/Radio_Syd
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Note:** Whisper may require `ffmpeg`:
```bash
brew install ffmpeg
```

---

## 2. Configure API Keys

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` with your credentials:

### Anthropic API Key
Get from: https://console.anthropic.com/

```
ANTHROPIC_API_KEY=sk-ant-...
```

### Google Cloud TTS

1. Go to https://console.cloud.google.com/
2. Create a project (or use existing)
3. Enable "Cloud Text-to-Speech API"
4. Create a service account:
   - IAM & Admin → Service Accounts → Create
   - Grant role: "Cloud Text-to-Speech User"
   - Create key (JSON) → Download
5. Save the JSON file (e.g., `~/keys/google-tts.json`)
6. Add to `.env`:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/google-tts.json
   ```

---

## 3. Verify Setup

```bash
python interview.py --check
```

Expected output:
```
  [OK] ANTHROPIC_API_KEY
  [OK] GOOGLE_APPLICATION_CREDENTIALS
  [OK] System prompt file
  [OK] Directory: audio/tts
  [OK] Directory: audio/recordings
  [OK] Directory: scripts
  [OK] anthropic
  [OK] google-cloud-texttospeech
  [OK] whisper
  [OK] sounddevice

All checks passed. Ready to run.
```

---

## 4. Run an Interview

Basic run:
```bash
python interview.py
```

With topic context:
```bash
python interview.py --topic "classification of creative work"
```

---

## 5. During the Interview

1. SYD speaks first (TTS plays automatically)
2. After SYD finishes, you'll see: `[Recording... Press ENTER when done]`
3. Speak your response
4. Press ENTER when finished
5. Whisper transcribes your response
6. SYD responds to what you said
7. Loop continues until SYD closes the session

**To end early:** Press `Ctrl+C`

---

## 6. Output Files

After each session:

```
scripts/session_YYYYMMDD_HHMMSS/
├── transcript.json    # Machine-readable
└── transcript.txt     # Human-readable

audio/tts/session_YYYYMMDD_HHMMSS/
├── SYD_01.wav
├── SYD_02.wav
└── ...

audio/recordings/session_YYYYMMDD_HHMMSS/
├── SUBJECT_01.wav
├── SUBJECT_02.wav
└── ...
```

---

## 7. Voice Options

Edit `.env` to change SYD's voice:

| Voice | Character |
|-------|-----------|
| `en-US-Neural2-D` | Low, flat, male (default) |
| `en-US-Neural2-A` | Neutral male |
| `en-US-Neural2-J` | Warmer male |
| `en-GB-Neural2-B` | British institutional |
| `en-GB-Neural2-D` | British, lower |

Full list: https://cloud.google.com/text-to-speech/docs/voices

---

## 8. Whisper Model Options

Edit `.env` to change transcription quality/speed:

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `tiny` | 39M | Fastest | Lower |
| `base` | 74M | Fast | Good (default) |
| `small` | 244M | Medium | Better |
| `medium` | 769M | Slow | High |
| `large` | 1.5G | Slowest | Best |

---

## 9. Costs

| Service | Per Episode | Monthly (16 eps) |
|---------|-------------|------------------|
| Claude Sonnet | ~$0.02 | ~$0.32 |
| Google TTS | ~$0.01 | ~$0.16 |
| Whisper | Free | Free |
| **Total** | **~$0.03** | **~$0.50** |

Google TTS free tier: 1M characters/month (you'll never hit it).

---

## Troubleshooting

**No audio input detected:**
- Check microphone permissions in System Preferences
- Run: `python -c "import sounddevice; print(sounddevice.query_devices())"`

**Google TTS error:**
- Verify `GOOGLE_APPLICATION_CREDENTIALS` path
- Ensure TTS API is enabled in Google Cloud Console

**Whisper slow on first run:**
- Model downloads on first use (~150MB for base)
- Subsequent runs are faster

**Permission denied on audio:**
```bash
# macOS: Grant terminal microphone access
# System Preferences → Privacy & Security → Microphone → Terminal
```
