# PROJECT PLAN
## SYD: ARCHIVAL INTERVIEW PODCAST

---

## 1. PROJECT GOAL

Create a recurring audio series presented as archival review sessions, in which a character named SYD conducts procedural interviews with an author.

The output must:
- sound like a real institutional record
- feel conversational without being live
- avoid AI assistant tropes
- be reproducible with minimal friction
- scale to multiple episodes

---

## 2. CORE DESIGN PRINCIPLES (NON-NEGOTIABLE)

- SYD never manages turn-taking
- SYD never improvises in voice
- All authority comes from procedure
- Silence is intentional
- No exposition inside the episode
- The listener infers context through form

**If any implementation violates these principles, it is incorrect.**

---

## 3. SYSTEM ARCHITECTURE (HIGH LEVEL)

The project is split into four layers, each with a single responsibility.

### Layer 1 — Script Generation (TEXT)
- **Tool:** Claude / ChatGPT (text mode)
- **Output:** SYD dialogue + session structure
- No audio, no timing, no conversation

### Layer 2 — Voice Rendering (TTS)
- **Tool:** TTS engine (ElevenLabs / OpenAI / OS voice)
- **Input:** individual SYD lines
- **Output:** discrete audio clips
- One line = one file

### Layer 3 — Human Performance
- **Tool:** DAW (Audacity / Reaper / GarageBand)
- **Input:** SYD audio clips
- **Output:** natural human responses
- Human controls pacing

### Layer 4 — Assembly & Publishing
- **Tool:** DAW + hosting platform
- **Output:** final episode audio
- Framed as archival material

---

## 4. WORKFLOW (STEP-BY-STEP)

### STEP 1 — Generate Session Script (TEXT ONLY)

Claude is instructed to:
- produce a single-session script
- follow the SYD protocol
- output only SYD's lines, clearly separated
- no stage directions
- no filler

**Deliverable:**
```
SYD_01: This session is being recorded...
SYD_02: This review was initiated...
SYD_Q1: How do you classify your work?
...
SYD_END: This concludes the current review.
```

### STEP 2 — Prepare TTS Inputs

For each SYD line:
- copy text exactly
- paste into TTS engine
- generate audio
- export as WAV or MP3

**File naming convention (mandatory):**
```
SYD_01_OPEN.wav
SYD_02_FLAG.wav
SYD_Q01_CLASS.wav
SYD_INT_PAUSE.wav
SYD_END.wav
```

This enables automation later.

### STEP 3 — Record Human Responses

**Recording rules:**
- SYD audio plays first
- human waits briefly
- human responds naturally
- silence is preserved
- no script reading

This is done sequentially, not live.

**Track layout:**
- Track 1: SYD audio clips
- Track 2: Human responses
- Track 3: Room tone (optional)

### STEP 4 — Assemble Conversation

**Assembly rules:**
- no crossfades between speakers
- minimal silence trimming
- no background music under dialogue
- interruptions are abrupt

The conversation must feel procedural, not edited.

### STEP 5 — Light Audio Processing

**SYD voice:**
- slightly compressed
- slightly band-limited
- dry (no room reverb)

**Human voice:**
- fuller
- more room presence

This establishes hierarchy.

### STEP 6 — Export & Publish

**Export:**
- WAV (archive)
- MP3 (distribution)

**Framing:**
- no intro music
- no greeting
- no call to action

Description written outside the audio, in memo style.

---

## 5. SYD CHARACTER CONSTRAINTS (FOR CLAUDE)

Claude must enforce these rules when writing SYD:

### SYD may:
- ask procedural questions
- interrupt once per session
- mark inconsistencies
- reframe answers clinically

### SYD may NOT:
- greet
- thank
- reassure
- speculate emotionally
- explain the institution
- acknowledge an audience

### Language constraints:
- short sentences
- administrative vocabulary
- neutral tone
- no metaphors

---

## 6. EPISODE TEMPLATE (STANDARD)

Each episode must include:
1. Recording declaration
2. Review purpose
3. Subject identification
4. 8–10 structured questions
5. One interruption
6. Abrupt closure

No deviations unless explicitly designed.

---

## 7. TECH STACK (MINIMUM VIABLE)

| Component | Tool |
|-----------|------|
| Script | Claude / ChatGPT (text) |
| TTS | ElevenLabs or equivalent |
| Recording | Audacity / Reaper |
| Hosting | Spotify / RSS / YouTube (static image) |

No live voice mode. No streaming.

---

## 8. SUCCESS CRITERIA

The episode is successful if:
- listeners understand what is happening within 90 seconds
- SYD never sounds conversational
- silence feels intentional
- no one asks "is this live AI?"

**Failure modes:**
- AI greeting behaviour
- over-explaining context
- theatrical delivery
- conversational filler

---

## 9. FUTURE EXTENSIONS (OPTIONAL)

- push-to-talk semi-live sessions
- state-driven SYD logic
- automated TTS batching
- secondary subsystems (e.g. SYD annotations)

These are not required for v1.

---

## 10. ONE-LINE SUMMARY (FOR CLAUDE)

> "This project produces procedural, archival-style audio interviews in which a constrained AI character (SYD) conducts structured reviews using scripted text rendered via TTS, with human-controlled timing and assembly to create a convincing conversational illusion."
