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
