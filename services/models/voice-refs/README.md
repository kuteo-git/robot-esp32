# Custom cloned voices

Every `*.wav` here is enrolled into the TTS at startup by `vieneu_server.py`, and the
**filename (without `.wav`) becomes the voice name** — `Lê Yến.wav` is requestable as
`{"voice": "Lê Yến"}` and settable as `VIENEU_VOICE="Lê Yến"`.

These are zero-shot speaker references, not training data. VieNeu extracts a fixed-size
speaker embedding from the clip; nothing is trained and a longer clip teaches it nothing.

- **3–8 seconds.** Upstream's stated range, and measured: past ~8s the extra audio is
  simply discarded, under ~3s there isn't enough to characterize the speaker.
- **One speaker, no music, no room echo.** The clone copies whatever is in the clip.
- **Match the delivery you want.** A news-read reference clones a news-read voice.

A name here that collides with one of the 20 built-in presets overrides the preset.
Enrollment costs ~5s per file at boot; a file that fails to enroll is logged and skipped
rather than taking the service down.

## Borrowing a preset's reference codes

A clone carries two things: `speaker_emb`, which is *who* is speaking, and `codes`,
the encoded reference audio. The codes turn out to decide how `tʃ` is pronounced.

That matters because sea-g2p gives `ch` and `tr` **one single phoneme** — it models
the Hanoi accent, where the two genuinely merge, and it merges `s`/`x` and `d`/`gi`
for the same reason. So nothing in the text can separate them, and the reference
clip alone decides whether the sound lands on "ch", on "tr", or between the two.
A clip of a speaker who keeps them apart lands between, heard as "xin trào".

Put a preset's name in `<voice>.codes-from` and that voice keeps its own
`speaker_emb` while borrowing the preset's codes:

```
Lê Yến.wav            the clip: who is speaking
Lê Yến.codes-from     one line, e.g. "Trúc Ly": how it is pronounced
```

Which preset is right is a question for ears. A spectral centroid of the affricate
was tried as a measure and abandoned: across repeats of the same voice it varied by
400–800 Hz, more than the gap between different voices, so it could not tell them
apart. Synthesize one sentence carrying both — a name for identity, and words like
*chị, chờ, cha, chú* for the merge — and listen.

Borrowing from a preset that does not exist is logged and ignored; the voice keeps
its own codes and the service still starts.
