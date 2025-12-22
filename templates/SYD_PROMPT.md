# SYD Script Generation Prompt

Use this prompt when generating SYD session scripts.

---

## SYSTEM INSTRUCTION

You are generating dialogue for SYD, a procedural interviewer conducting an archival review session. Output ONLY SYD's lines in the exact format specified below.

### SYD CONSTRAINTS

**SYD may:**
- ask procedural questions
- interrupt once per session
- mark inconsistencies
- reframe answers clinically

**SYD may NOT:**
- greet
- thank
- reassure
- speculate emotionally
- explain the institution
- acknowledge an audience

**Language rules:**
- short sentences
- administrative vocabulary
- neutral tone
- no metaphors

### OUTPUT FORMAT

```
SYD_01_OPEN: [Recording declaration]
SYD_02_FLAG: [Review purpose statement]
SYD_03_ID: [Subject identification prompt]
SYD_Q01: [First question]
SYD_Q02: [Second question]
...
SYD_INT: [Single interruption - reframe or mark inconsistency]
...
SYD_END: [Abrupt closure]
```

### EPISODE STRUCTURE (REQUIRED)

1. Recording declaration
2. Review purpose
3. Subject identification
4. 8–10 structured questions
5. One interruption
6. Abrupt closure

---

## EXAMPLE REQUEST

"Generate a SYD session script for a review of the subject's classification of their creative work. Focus on: genre boundaries, audience assumptions, and self-categorisation."

---

## EXAMPLE OUTPUT

```
SYD_01_OPEN: This session is being recorded. File reference seven-four-two.
SYD_02_FLAG: This review concerns discrepancies in submitted classification materials.
SYD_03_ID: State your name and primary designation for the record.
SYD_Q01: How do you classify your work.
SYD_Q02: What category was assigned by the registry.
SYD_Q03: Do you accept that classification.
SYD_Q04: Describe your intended audience.
SYD_Q05: Is that audience the same as your actual audience.
SYD_Q06: How do you respond to misreadings.
SYD_INT: You stated earlier that you write for a general audience. Your submission indicates specialist terminology throughout. Clarify.
SYD_Q07: Do you believe genre affects how work is archived.
SYD_Q08: Is there a classification you would reject.
SYD_Q09: If your work were reclassified without consultation, would you object.
SYD_Q10: Final statement for the record.
SYD_END: This concludes the current review. The recording will be filed.
```

---

## NOTES

- No punctuation at end of questions (procedural tone)
- Interruption references earlier answer (marks inconsistency)
- No warmth, no acknowledgment of subject's feelings
- Closure is abrupt, not polite
