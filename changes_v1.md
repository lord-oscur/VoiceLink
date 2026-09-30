# VoiceLink: changes v1

New requirements after the v1.0 MVP (commit `cfc23a4`), the decisions taken
with the user, and the order of implementation (easiest first). Status is
updated as each item lands.

## Requirements

| # | Requirement (as requested) | Status |
|---|---|---|
| 1 | Add a favicon.ico. | done |
| 2 | 100 % offline after the first install. | pending |
| 3 | Show the model's capabilities (thinking, tool use, vision…) as icons next to the loaded model name. | done |
| 4 | Audio/voice settings in a hover panel opened from a voice icon, in a small toolbar at the top right (more icons later). | done |
| 5 | MVP focuses on English and Spanish; keep the other voices (a multi-language selector comes later). | done |
| 6 | Voices grouped by language, then by gender; male/female filter buttons with icons outside the dropdown; sorted by approximate sounding age. | done |
| 7 | Input and output device selectors in the voice controls, limited to the main physical devices (no virtual devices). | done |
| 8 | Saved settings in the browser, export/import, and saved setting profiles. | done |
| 9 | Pitch, rate, timbre and volume controls applied after TTS generation. | done |
| 10 | Voice-age layer: Child 2–5, Preteen 8–12, Teen 13–19, Adult 20–50, Mature 50–70, Senior 70+ (base voices sound 20–30). | done |
| 11 | Emotion layer on top (order: normal audio → age → emotion). | done |
| 12 | Chat history on the left: stored conversations. | pending |
| 13 | Model settings on the right: system prompt, context, etc., even for a model loaded through the API. | pending |
| 14 | Attach files to the chat according to the model's capabilities (images, audio, video, documents). | pending |
| 15 | Spell check and grammar check in the text box, with coloured underlines, replacement suggestions, and a redundancy checker. | pending |
| 16 | STT noise filter (PC fans, ambient sound). | pending |
| 17 | STT input filter so a video playing on the same PC does not trigger the STT. | pending |

## Decisions (answered by the user)

* **Checker engine (15):** use offline EN/ES dictionaries and built-in rules
  (repeated words, pleonasms such as "free gift" or "subir arriba", a/an,
  capitalization), plus an optional **Check with AI** pass that asks the
  LM Studio model for grammar fixes.
* **Playback filter (17):** use both methods.
  * The server records what the PC itself plays (WASAPI loopback) and
    ignores microphone speech that matches it.
  * An optional **Only my voice** mode compares each utterance with an
    enrolled voice sample.
* **Chat history (12):** stored on disk in `data/conversations/` (one JSON
  file per conversation), not in the browser.
* **Model panel (13):** per-request settings, plus reloading the model with a
  new context length or flash attention, plus a model switcher that lists
  the downloaded models with their capability icons (LM Studio REST
  `/api/v1/models`, `/load`, `/unload`).

## Implementation order (easy → hard)

1. Favicon (1).
2. Offline operation (2): everything is served locally, the Hugging Face hub
   is forced offline at runtime, and all extra models and dictionaries are
   downloaded by `setup.bat`. A final check runs with the network blocked.
3. Capability icons (3), from `/api/v1/models`: vision, tool use, reasoning
   (with the loaded context length).
4. Top-right toolbar and voice hover panel (4).
5. English/Spanish first in the voice list, other languages grouped below (5).
6. Voices by language → gender, with filter buttons and age order (6). The
   order uses each voice's measured median pitch (lower pitch sounds older
   within the same gender) plus known character voices (the "santa"
   voices).
7. Microphone and speaker selectors (7): `enumerateDevices` with a filter for
   virtual drivers, `AudioContext.setSinkId` for output, and the `deviceId`
   constraint for input.
8. Settings store, profiles, export/import (8): one versioned JSON object in
   `localStorage`, with named profiles and file export/import.
9. Voice effects (9): speed through Kokoro's native `speed`; pitch, timbre
   (formant shift) and pitch range through a PSOLA/vocoder stage on the
   server; volume as gain.
10. Age presets (10): offsets on pitch, formants, rate and pitch range, plus
    breathiness and tremor for older voices.
11. Emotion presets (11), applied after age: pitch level and range, rate,
    volume, tremolo and breathiness.
12. Chat history (12): REST endpoints for list, load, rename and delete;
    automatic titles; the session history syncs to disk.
13. Model panel (13).
14. Attachments (14).
    * Images go to vision models as `image_url`.
    * Documents (PDF, DOCX, text and code) are converted to text.
    * Audio is transcribed with Parakeet.
    * Video: frames go to vision models and the audio track is
      transcribed.
    * Anything the model cannot use is refused with a clear message.
15. Checker (15): an overlay editor with coloured underlines (spelling, grammar,
    redundancy/style) and a suggestion popup.
16. Noise filter (16): stationary noise suppression (a spectral Wiener filter
    with a noise estimate learned while nobody speaks) on the 16 kHz
    microphone stream before VAD and STT, with Off / Light / Strong.
17. Playback filter (17): loopback reference capture, with a correlation and
    energy gate against the microphone, and speaker verification with a
    small ONNX speaker-embedding model and a 10-second enrollment.
