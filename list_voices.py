#!/usr/bin/env python3
"""List all available English TTS voices."""

from dotenv import load_dotenv
load_dotenv()

from google.cloud import texttospeech

client = texttospeech.TextToSpeechClient()
voices = client.list_voices(language_code="en")

# Group by language
by_lang = {}
for voice in voices.voices:
    lang = voice.language_codes[0]
    if lang not in by_lang:
        by_lang[lang] = []

    parts = voice.name.split("-")
    voice_type = parts[2] if len(parts) > 2 else "Standard"

    by_lang[lang].append({
        "name": voice.name,
        "gender": "Male" if voice.ssml_gender.name == "MALE" else "Female",
        "type": voice_type
    })

for lang in sorted(by_lang.keys()):
    print(f"\n=== {lang} ===")
    for v in sorted(by_lang[lang], key=lambda x: (x["type"], x["gender"], x["name"])):
        print(f"  {v['name']:30} {v['gender']:8} {v['type']}")
