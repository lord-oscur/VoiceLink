# VoiceLink: development notes

This log records how the system was designed, what was measured, which
options were rejected and why. All numbers were measured on the target
machine unless stated otherwise.

## 1. Brief and constraints

* Full-duplex voice chat (STT → LLM → TTS) with the model **already loaded
  in LM Studio**, over its API at `http://127.0.0.1:1234`. The user must be
  able to talk over the assistant and interrupt it.
* Hardware limit: 64 GB RAM, an 8 GB VRAM GPU.
* Everything lives in this folder: a private Python venv, models downloaded
  from the internet, `.bat` launchers, documentation.
* Must be simpler, lighter and faster than an earlier prototype.

## 2. The machine

| Part | Detail | Consequence |
|---|---|---|
| CPU | Intel i7-6700K, 4 cores / 8 threads, AVX2 | All speech models must fit in about 4 + 4 threads without starving LM Studio. |
| GPU | GTX 1070 Ti 8 GB, Pascal (sm_61); it also drives the display | Owned by LM Studio. It idles in the P5 power state (about 721 MHz) and ramps slowly for short bursts. Recent CUDA / cuDNN releases no longer support it. |
| RAM | 64 GB | Not a constraint. VoiceLink uses about 1.1 GB. |
| OS | Windows 10, Python 3.12 | venv built with `py -3.12`. |
| LLM | `qwen3.5-2b` in LM Studio (llama.cpp CUDA), about 13–15 tok/s | A thinking model: see §3.1. |

## 3. Component choices

### 3.1 LLM access: turning thinking off

Qwen 3.5 thinks by default. `tools/lmstudio_probe.py` compares the time to
the first *spoken* token:

| Request | First content token | Total |
|---|---|---|
| plain request | **never** (711 characters of reasoning, cut by `max_tokens`) | 13.4 s |
| `reasoning_effort: "none"` | **0.31 s** | 4.0 s |
| prefilled assistant message with an empty `<think></think>` block | 0.32 s | 2.2 s |

`llm.py` sends `reasoning_effort: "none"`. If the server rejects it (HTTP
400) or the model still streams reasoning first, the client switches to the
prefill and retries transparently. Any `<think>` text that still arrives is
filtered out (`ThinkFilter`). The model is auto-detected from
`/api/v0/models` (the loaded one), with `/v1/models` as a fallback.

### 3.2 Speech-to-text: Parakeet TDT 0.6B v3 on the CPU

A Whisper model on the GPU was tried first and dropped. Whisper always
encodes a 30-second window, so a 2-second utterance costs as much as a long
one. On this card, running in a low-clock state and shared with LM Studio,
that was far too slow for a turn-taking loop.

NVIDIA Parakeet TDT 0.6B v3 (25 European languages, punctuation included),
INT8 ONNX via `onnx-asr`, CPU, 4 threads (`tools/bench_parakeet.py`):

| Audio | Time | Result |
|---|---|---|
| English, 2 s | **216 ms** | `Hey, what time is it right now?` |
| English, 7 s | 574 ms | exact, with punctuation |
| Spanish, 4 s | 325 ms | `Hola, me puedes recomendar una película para ver esta noche.` |

It is fast enough to transcribe the whole utterance again for every check,
so no streaming ASR is needed. It keeps the GPU free and has no
hallucination-on-silence problem. Its limitation is that it reports no
language id, so VoiceLink detects the language from the text (§5).

### 3.3 Text-to-speech: Kokoro-82M FP32 on the CPU

`tools/bench_tts.py`, `kokoro-onnx` 0.6.1, speed 1.05:

| Text | FP32, 6 threads | FP32, 4 threads | INT8, 4 threads |
|---|---|---|---|
| "Sure!" (0.7 s of audio) | 458 ms | 485–505 ms | 1670 ms |
| short sentence (1.6 s) | 822 ms | 815–845 ms | 2868 ms |
| long sentence (5.2 s) | 2307 ms (RTF 0.44) | 2318–2445 ms (RTF 0.44–0.47) | 7628 ms (RTF 1.45) |
| Spanish sentence (3.1 s) | 1361 ms | 1383–1398 ms | 4392 ms |

* **INT8 is 3.3× slower** on this CPU (slower than real time) and was
  removed.
* **The GPU is no faster** for an 82M-parameter model once the card is in
  its low-clock state and shared with the LLM. It also required pinning old
  CUDA libraries (§4). It remains an option (`tts.device = "cuda"`).
* **4 threads vs 6** costs about 30 ms on the first chunk but avoids
  oversubscribing the CPU when STT runs at the same time (§3.4 and §6).
  4 threads is the default.

Kokoro's voice ids encode their language in the first letter (`a`/`b`
English, `e` Spanish, `f` French, `h` Hindi, `i` Italian, `j` Japanese,
`p` Portuguese, `z` Chinese). The espeak phonemizer language is taken from
the voice, so the text and the voice always match.

### 3.4 Does CPU speech slow the LLM down?

`tools/bench_contention.py` measures LM Studio while Kokoro synthesizes
continuously:

| Kokoro threads | LLM first token | LLM speed |
|---|---|---|
| (TTS idle) | 477 ms | 15.5 tok/s |
| 8 | 1972 ms | 5.9 tok/s |
| 6 | 243 ms | 12.8 tok/s |
| 4 | 170 ms | 13.9 tok/s |

Using all 8 threads starves llama.cpp's CPU-side thread and makes the LLM
much slower. With 4 threads TTS still runs at about 2× real time and the
LLM barely notices.

### 3.5 Voice activity detection: Silero VAD (ONNX)

Silero v5 runs in streaming mode (512-sample frames at 16 kHz plus
64 samples of context and the recurrent state), on one thread with one
session per client, in about 0.2 ms per 32 ms frame.

### 3.6 Transport and client

* **Browser client** (Chrome recommended). `getUserMedia` with
  `echoCancellation`, `noiseSuppression` and `autoGainControl`. Chrome's
  echo canceller also covers audio played through WebAudio from the same
  page, which removes most of the assistant's voice from the microphone.
* An **AudioWorklet** decimates the mic to 16 kHz Int16 frames. Playback
  schedules 24 kHz segments back to back on one `AudioContext` clock, so
  there are no gaps and the exact position is known when interrupting.
* **aiohttp** serves the page and one WebSocket per client. Binary frames
  carry audio and JSON frames carry control messages. There is no build
  step and no external JavaScript.

## 4. ONNX Runtime and CUDA on a Pascal card

| Tried | Result |
|---|---|
| `onnxruntime-gpu` 1.30 | Needs CUDA 13 (`cublas64_13.dll`). CUDA 13 dropped Pascal. |
| `onnxruntime-gpu` 1.26 + cuDNN 9.26 | `CUDNN_STATUS_EXECUTION_FAILED_CUDART`: new cuDNN has no sm_61 kernels. |
| `onnxruntime-gpu` 1.26 + `nvidia-cudnn-cu12==9.10.2.21` | Works. Kokoro also needs `cudnn_conv_algo_search = HEURISTIC` (the exhaustive search is slow). |

That combination is pinned in `requirements-gpu.txt` and installed only by
`setup.bat gpu`. The default install is the CPU `onnxruntime` wheel, which
is smaller and has no DLL problems.

## 5. The duplex engine (`voicelink/session.py`)

### 5.1 Turn taking with speculation

```
speech ... 250 ms silence ──► speculative turn: STT → LLM → chunker → TTS
                                (output held in the turn's outbox)
           800 ms silence ──► commit: user bubble + held audio released
   user resumes before 800 ms ─► cancel the speculative turn, keep listening
```

The 550 ms between speculation and commit absorbs STT (about 300 ms) and
most of the LLM's first-token time. As a result the reply usually starts
0.9–1.3 s after the commit point, about 1.7–2.1 s after the user stops
talking.

### 5.2 Getting the first words out early

The time to the first audio is roughly LLM first token + the time to
generate the first chunk + TTS on the first chunk. `SentenceChunker` is
therefore progressive:

| Chunk | Cut at | Hard limit |
|---|---|---|
| 1st | first clause of ≥ 12 characters, or after 600 ms the words received so far (≥ 5 words, never ending on "the", "de", "and"…) | 70 characters |
| 2nd | clause of ≥ 12 characters | 80 |
| 3rd | clause of ≥ 30 characters | 140 |
| rest | full sentences | 220 |

Abbreviations ("Dr.", "e.g.", single initials) are not treated as sentence
ends, and CJK punctuation is supported. The first version cut after a fixed
number of characters and produced splits like "…do they / have?" that
sounded broken. After each chunk, 0.22 s, 0.08 s or 0.03 s of silence is
added depending on its final punctuation, so the rhythm stays natural.

### 5.3 Barge-in

1. **Duck**: speech-like frames reduce the playback volume to 22 %
   immediately. This is audible feedback at once, and less echo reaches the
   microphone while the decision is made.
2. **Confirm**: 280 ms of speech above 0.7 while the assistant is talking.
   With headphones the assistant stops right there: about **0.30 s**.
3. **Stop**: the server sends `stop`. The browser fades out in about 40 ms and
   drops queued segments, and the server cancels the LLM stream (closing the
   HTTP connection makes LM Studio stop generating) and the TTS.
4. **History**: from the playback schedule the server estimates which words
   were heard. It stores only those words plus "—", and the system prompt
   explains that an em dash means "cut off". The browser strikes through
   the unheard text.

### 5.4 Echo check (speakers, no headphones)

Chrome's echo canceller removes most of the echo, but not all of it: loud
speakers, a laptop chassis, or Edge/Firefox let some through. When the
assistant's voice leaks into the microphone, the VAD alone cannot tell it
from the user. So when headphones are off, the candidate speech is
transcribed and compared with the text of the segments playing at that
moment.

The first version:

* rejected backchannels (*yeah, ok, mhm, sí, vale…*);
* accepted stop words (*stop, wait, espera, basta…*) unless the assistant
  itself was saying them;
* otherwise required new words;
* retried with more audio up to 1.5 s, then gave up and applied a 0.8 s
  cooldown.

That version never interrupted itself, but a real barge-in over the echo
took **2.1–2.3 s**. The debug log showed why: continuous echo keeps the VAD
in "speech" for many seconds. The user's "Wait, stop." fell inside a window
that had already been judged as echo, and it was discarded together with
the echo audio.

The current version:

* **Slides the window.** Once the candidate reaches 1.5 s, only the last
  1.0 s is kept and checking continues (every 250 ms, then every 500 ms
  during long echo to leave CPU for the TTS). No user audio is thrown away.
* **Treats garbled echo as echo.** A transcript word that resembles one of
  the assistant's words (`difflib` ratio ≥ 0.6) is not new. Examples seen
  in the logs: "for its" → *Fruit.*, "hot" → *Hora.*
* **Needs a stronger signal.** A stop word interrupts on its own. Otherwise
  it takes at least two new words, making up at least half of the
  transcript. In a window that starts mid-word, the first word is ignored.

Result over repeated runs: **no self-interruptions**, and real barge-in over
raw echo in **1.0–1.5 s**.

### 5.5 Languages

The system prompt asks the model to reply in the user's language, but a 2B
model often drifts (an English question was once answered in Spanish). So:

* A stop-word language detector (`textproc._scores`) classifies the user's
  text among en/es/fr/it/pt. If the result is confident, "(Reply in X.)" is
  appended to that message *in the request only*, not in the history.
* The TTS language follows the **reply** text, so a reply in another
  language is still pronounced correctly with a matching voice.
* `stt.languages` limits the candidates (default English + Spanish).

### 5.6 Text cleanup for speech

Markdown emphasis, headings, bullets, links, URLs, code blocks and emojis
are removed. Line breaks become commas. Only letters, digits, combining
marks (needed for Devanagari vowel signs) and speakable punctuation reach
the phonemizer, and full-width CJK punctuation is mapped to ASCII pauses.

## 6. CPU budget during a turn

| Engine | Threads | When |
|---|---|---|
| Silero VAD | 1 | always (about 0.2 ms per frame) |
| Parakeet STT | 4 | end of utterance; live transcript once a second; echo checks |
| Kokoro TTS | 4 | while a reply is being synthesized |
| LM Studio | its own (model on the GPU) | while generating |

Each engine has its own single-worker executor, so jobs never pile up on
one engine. Live partial transcripts are skipped while another STT job
(the final transcript or an echo check) is pending, so they never delay
them.

With 6 TTS threads, continuous echo checks pushed Kokoro's real-time factor
from 0.5 to 0.8–0.98, close to audio underruns. With 4 + 4 threads and the
500 ms sliding step, the end-to-end tests are fast and stable (§7).

## 7. Testing

`test.bat` starts a server on port 8799 (so a browser tab left open on
8765 cannot join and pollute the test) and runs `tools/selftest.py`. The
test synthesizes the "user" with Kokoro voices and streams it to the
WebSocket in real time, paced by the wall clock (Windows `asyncio.sleep`
granularity made naive pacing drift). Optionally it feeds the assistant's
audio back into the "microphone" at 35 % gain, with no echo cancellation,
to simulate the worst case of loud speakers.

| # | Scenario | Checks |
|---|---|---|
| 1 | English question | transcript, voice-to-voice latency, metrics |
| 2 | Spanish question | Spanish voice chosen automatically |
| 3 | Barge-in with headphones during a long story | time to `stop`, the new turn is transcribed |
| 4 | 3 s of silence | no turn is created |
| 5 | Speakers, raw echo | echo alone never interrupts; a real barge-in over the echo does |

Final run (all passed):

| Metric | Result |
|---|---|
| English voice-to-voice | 1.72 s (STT 404 ms, LLM first token 311 ms, TTS first chunk 556 ms) |
| Spanish voice-to-voice | 1.65 s, reply voice `ef_dora` |
| Barge-in (headphones) | 0.32 s |
| Silence | no turn |
| Echo alone | no self-interruption |
| Barge-in over echo | 0.97–1.02 s |

Across runs: voice-to-voice 1.7–2.4 s, headphones barge-in 0.30–0.34 s.

A browser smoke test (Chromium) loaded the page without errors: the LM
Studio status appeared, all 54 voices were listed, and a typed message got
a spoken reply with metrics shown.

Other issues found and fixed along the way:

* A browser tab left open joined the test server and streamed a video's
  audio into the conversation. The test now uses its own port
  (`VOICELINK_PORT`).
* "Port already in use" after an unclean exit. The port is now checked
  before the models load, with a clear message, and a second browser tab is
  no longer opened in that case.
* The HTTP 400 path of the thinking-off fallback did not retry. Fixed.
* Stale cached `app.js` after an update. Static files are now served with
  `Cache-Control: no-cache`.
* **No sound in Chrome on the user's PC.** The page was fine: the context
  was running, all segments were scheduled, and the analyser showed signal.
  `tools/audio_sessions.ps1` reads the Windows audio sessions (Core Audio
  `IAudioSessionManager2`, the same data as the Volume Mixer). It showed
  Chrome's audio-service session muted on the default output (FxSound
  Speakers). A web page cannot see or change that state. After unmuting
  (`soundcheck.bat`), a real Chrome instance driven over DevTools, with the
  microphone and echo cancellation active, measured a peak of 0.64 on
  Chrome's session and 0.97 on the physical output behind FxSound. The page
  now also resumes a suspended `AudioContext` before scheduling audio.

## 8. Known limitations and next steps

* **Background audio from other apps** (videos, TV) is not removed by the
  browser's echo canceller and counts as user speech. Headphones or muting
  the mic is the fix today. A speaker-verification step could solve it
  later.
* The echo check is a text-level heuristic. A server-side reference echo
  canceller (the exact TTS signal is known) or a double-talk detector based
  on the echo-path gain would make barge-in on speakers as fast as with
  headphones.
* Parakeet covers European languages only; Japanese, Chinese and Hindi
  voices work for typed chat.
* All browser tabs share one STT and one TTS engine. That is fine for one
  user; several simultaneous conversations would queue.
* The VAD uses fixed silence timeouts. A semantic end-of-turn model
  (whether the sentence is finished) could shorten the 800 ms wait for
  complete sentences and lengthen it for "umm…".

## 9. Timeline

1. Probed LM Studio: model detection, streaming, thinking behaviour (§3.1).
2. Built the pipeline with Whisper and measured it. The GPU in its
   low-clock state and the 30 s window made STT too slow, so it was
   replaced by Parakeet on the CPU (§3.2).
3. Benchmarked Kokoro CPU/GPU/INT8 and the CPU contention with LM Studio
   (§3.3, §3.4). Chose the CPU FP32 model and pinned the Pascal-compatible
   GPU stack as an option (§4).
4. Wrote the duplex engine: speculative turns, streaming chunker, barge-in
   with ducking, heard-text history (§5).
5. Built the end-to-end self-test with synthetic speech and an echo
   loopback, and iterated on the chunker, language handling and echo check
   until all scenarios passed (§7).
6. Tuned threads (6 → 4 for TTS), isolated the test port, fixed text
   cleanup for non-Latin scripts and static caching, and wrote this
   documentation.
