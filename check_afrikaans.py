#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()

from google.cloud import texttospeech

client = texttospeech.TextToSpeechClient()

# Check for Afrikaans
voices = client.list_voices(language_code="af")
if voices.voices:
    print("=== Afrikaans (af) ===")
    for v in voices.voices:
        gender = "Male" if v.ssml_gender.name == "MALE" else "Female"
        print(f"  {v.name:30} {gender}")
else:
    print("No Afrikaans voices found")

# Also check South African English
print("\n=== South African English ===")
all_voices = client.list_voices()
for v in all_voices.voices:
    if "ZA" in v.name or any("en-ZA" in lc for lc in v.language_codes):
        gender = "Male" if v.ssml_gender.name == "MALE" else "Female"
        print(f"  {v.name:30} {gender}")
