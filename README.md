# VoiceLink

Real-time, full-duplex voice conversation with whatever model is loaded in
**LM Studio**. You talk, it answers out loud, and you can interrupt it at
any moment just by speaking, like on a phone call.

```
 Browser (Chrome)                      VoiceLink server (Python, localhost)              LM Studio
 ┌──────────────────┐  16 kHz PCM   ┌────────────────────────────────────────┐   SSE   ┌──────────┐
 │ mic + echo cancel├──────────────►│ Silero VAD ─► Parakeet STT ─► LLM ─────┼────────►│ your     │
 │ gapless player   │◄──────────────┤   ▲ barge-in      sentence chunker ◄───┼─────────┤ model    │
 │ chat UI          │  24 kHz PCM   │   └── duplex turn engine ◄── Kokoro TTS│         └──────────┘
 └──────────────────┘   + JSON      └────────────────────────────────────────┘
```

Measured on the target machine (i7-6700K, GTX 1070 Ti 8 GB, 64 GB RAM,
LM Studio running `qwen3.5-2b` on the GPU):

| What | Time |
|---|---|
| End of your sentence to first sound of the reply | **1.7 – 2.1 s** |
| You start talking to the assistant going quiet (headphones) | **0.30 s** |
| Same, over the assistant's own voice with no echo cancellation | **1.0 – 1.5 s** |
| Assistant interrupting itself because it hears its own voice | never (in tests) |

The speech engines run on the **CPU** and use about 1.1 GB of RAM, so all
8 GB of VRAM stay free for the LLM.

---

## Quick start

1. **Install Python 3.12** (3.10–3.13 work) from
   [python.org](https://www.python.org/downloads/) if you don't have it.
2. Double-click **`setup.bat`**. It creates a private `.venv`, installs the
   packages and downloads the models (about 1 GB, first time only). Nothing
   is installed system-wide.
3. In **LM Studio**, load a chat model and start the server (Developer tab,
   `http://127.0.0.1:1234`, the default).
4. Double-click **`run.bat`**. The page opens in your browser
   (<http://127.0.0.1:8765>). Click the orb, allow the microphone, and
   talk.

Use **Google Chrome**. Its echo cancellation also removes the assistant's
voice coming out of your speakers. With **headphones**, tick the
*Headphones* box: interruptions become instant because no echo check is
needed.

| File | Purpose |
|---|---|
| `setup.bat` | First-time install; run it again any time to repair or update. `setup.bat gpu` / `setup.bat cpu` switch the ONNX Runtime build (see *GPU option*). |
| `run.bat` | Starts the server and opens the browser. Ctrl+C or closing the window stops it. |
| `soundcheck.bat` | Fixes "no sound": unmutes Chrome/Edge in the Windows Volume Mixer and shows which app plays on which device (see *Troubleshooting*). |
| `test.bat` | Automatic end-to-end test (needs LM Studio running). It synthesizes speech, "talks" to the server and checks latency, barge-in and echo handling. It uses port 8799, so an open browser tab doesn't interfere. `test.bat --only=3,5` runs selected scenarios. |

### Using it

* **Talk**: the orb pulses green while you speak. About 0.8 s after you
  stop, the reply starts (the answer is already being prepared after
  0.25 s of silence, see below).
* **Languages**: English is always on. A second language (Spanish, French,
  Italian, Portuguese, Hindi, Japanese or Chinese) is chosen from this PC's
  display language or time zone. The voice list puts that language next to
  English.
* **Interrupt**: just speak. The reply drops in volume immediately and stops
  once your speech is confirmed. The part of the reply you did not hear is
  struck through in the chat, and the model is told where it was cut off.
* **Type** instead of talking in the box below the chat (also interrupts).
  Spelling, grammar and redundancy are underlined; click a word for
  suggestions. Optional **Check with AI** uses the loaded model.
* **Esc** stops the assistant, **M** mutes the mic. Conversations are stored
  on disk (left panel). Voice, devices and effects are saved in this browser
  (profiles, export/import in the person icon).
* The header pill shows the loaded model plus icons for thinking, tools and
  vision. Live search, weather and stocks are offered only when that tools
  icon is present. The right panel has the system prompt, context length and
  generation settings (including reload/switch in LM Studio). Each request
  includes the current date and time.
* The **voice icon** (top right) opens Voice (voices, pitch/age/emotion)
  and Devices (headphones, microphone, speaker, fan-noise filter, ignore-PC-playback,
  only-my-voice). The panel stays open until you use the orb or the text box.
* Paperclip attaches files the model can actually use (images if it has
  vision, documents as text, audio/video transcribed).

After `setup.bat`, speech (STT/TTS/VAD) and chat with LM Studio run
**offline**. Live tools (web search, weather, stocks) need the internet
and are offered only when the loaded model has tool use. `setup.bat` is
still the step that downloads models.

---

## How the full-duplex engine works

Everything happens in `voicelink/session.py`; the browser only captures,
plays and displays.

**Always listening.** The browser streams the microphone (with Chrome's
echo cancellation, noise suppression and auto-gain) as 16 kHz PCM over a
WebSocket, including while the assistant talks. Silero VAD scores every
32 ms frame.

**Speculative turns (low latency).** After `spec_silence_ms` (250 ms) of
silence a turn starts *speculatively*: speech recognition, the LLM request
and speech synthesis all begin, but the output is held back. If you are
still silent after `end_silence_ms` (800 ms) the turn is committed and the
already-prepared audio is released at once. If you start talking again
first, the speculative turn is cancelled and your utterance simply
continues. This hides almost all of the STT and LLM time inside the pause
you'd leave anyway.

**Streaming LLM to TTS.** Tokens stream from LM Studio into a sentence
chunker that sends a short first piece to TTS as early as possible: the
first clause of 12 or more characters, or after 600 ms whatever complete
words exist, never ending on a word like "the" or "de". Later pieces grow
to full sentences so the voice sounds natural. Kokoro synthesizes each
piece at about 2× real time while the previous one plays, and the browser
schedules the pieces back to back without gaps.

**Barge-in.** While the assistant is speaking:

1. Any speech detected by the VAD *ducks* the playback to 22 % volume
   immediately.
2. After `barge_in_ms` (280 ms) of confident speech the assistant is
   interrupted: the browser fades out and discards queued audio, and the
   LLM and TTS stop.
3. Without headphones, step 2 first runs an **echo check**. The candidate
   speech is transcribed and compared with the words the assistant is
   saying right now. Only real new speech interrupts: a stop word such as
   *stop, wait, espera, para…*, or at least two new words. Backchannels
   like *yeah, ok, mhm, sí, vale* never interrupt. Garbled echo is
   recognized because it resembles the assistant's own words. If the result
   is unsure the check repeats with more audio, sliding forward through
   long stretches of echo so words you say mid-sentence are not lost.
4. The reply stored in the conversation history is cut to what you actually
   heard (estimated from the playback schedule) and ends with "—". The
   system prompt tells the model what that means, so it can continue
   naturally ("…as I was saying").

**Thinking models.** Reasoning models (Qwen 3.x, etc.) would think for
seconds before saying anything. VoiceLink sends `reasoning_effort: "none"`;
if the model still starts thinking, it retries once with the documented
empty-think prefill and remembers which method works. `<think>` blocks are
also filtered out of the speech. Set `llm.disable_thinking` to `false` to
allow thinking.

**Mood tags.** When the model reads a story, it may wrap a character's line
in HTML-like tags (`<angry>Get out!</angry>`, also `sad`, `happy`, `excited`,
`calm`, `fearful`, `whisper`). The tag names are stripped from the chat and
are never spoken. Kokoro synthesizes the words as usual; VoiceLink then
applies that emotion filter to **that audio chunk only**, so the narrator
stays unchanged. A tag at the start of a span lasts until the matching
close (or the end of the reply). The effects panel still sets the default
voice; a tagged span overrides the emotion for those words. **Play all
emotions** (Voice panel) speaks one line per filter so you can hear them.
**Ask the model** sends a short scene that should use every mood tag once.

**Languages.** English is always first. A second language is taken from the
PC's OS locale, then its time zone, then the browser geolocation if that
permission was already granted (no prompt, no network). Parakeet recognizes
25 European languages automatically. The reply voice follows the language of
the reply text; Hindi, Japanese and Chinese voices can be chosen manually.
When your language is clear, "(Reply in Spanish.)" is appended to your
message, because small models often drift into the wrong language. The
browser sends `stt.languages` as `["en"]` or `["en", second]`.

---

## Configuration (`config.json`)

`config.json` is created with the defaults below. You only need to keep the
keys you change; missing keys fall back to the defaults. Restart
`run.bat` after editing.

### `server`
| Key | Default | Meaning |
|---|---|---|
| `host` | `127.0.0.1` | Interface to listen on. Use `0.0.0.0` for other devices, but browsers only allow the microphone on `localhost` or HTTPS. |
| `port` | `8765` | HTTP/WebSocket port. The `VOICELINK_PORT` environment variable overrides it. |
| `open_browser` | `true` | Open the page on start (Chrome is preferred if installed). |

### `llm`
| Key | Default | Meaning |
|---|---|---|
| `base_url` | `http://127.0.0.1:1234` | LM Studio server. |
| `model` | `auto` | `auto` uses the model currently loaded in LM Studio; or give a model id. |
| `api_key` | `lm-studio` | Sent as a Bearer token (LM Studio ignores it unless you enabled auth). |
| `temperature`, `top_p` | `0.7`, `0.9` | Sampling. |
| `max_tokens` | `512` | Upper bound for one reply. |
| `disable_thinking` | `true` | Turn reasoning off for fast replies (see above). |
| `history_messages` | `40` | How many past messages are sent with each request. |
| `auto_title` | `true` | Name new conversations with the LLM after the first reply. |
| `system_prompt` | voice-oriented prompt | Keep the parts about "no markdown" and the em dash. The current date/time is always appended at request time. |

### `mcp` (stateless tools)
| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Master switch. Off = no tools, no search instructions. |
| `servers` | `[{"name":"local","url":"local"}]` | `local` is VoiceLink's search / weather / stock tools. Extra entries must be Streamable HTTP: `{"name":"…","url":"http://host/mcp"}`. stdio, `command`, and `/sse` servers are rejected. |

Tools (`web_search`, `fetch_page`, `weather`, `stock_quote`) are sent to
LM Studio only when the loaded model reports tool use. Each request also
gets the current local date and time so the model knows how much time has
passed since it was trained. When tools are offered, the model is told
to search for anything it does not know — not only weather or stocks —
instead of saying it does not know or inventing an answer. Tool results
are never spoken; only the final reply is.

### `stt` (speech recognition, NVIDIA Parakeet TDT 0.6B v3)
| Key | Default | Meaning |
|---|---|---|
| `model`, `model_dir`, `quantization` | Parakeet, `int8` | Model id for `onnx-asr`, folder under `models/`, and its weights. |
| `device` | `cpu` | `cpu` or `cuda` (needs `setup.bat gpu`). |
| `cpu_threads` | `4` | ONNX Runtime threads. |
| `languages` | `["en"]` | English plus an optional second language. The browser fills the second from the PC locale / time zone (or geolocation if already allowed). |
| `live_transcript` | `true` | Show what you are saying while you speak. |
| `live_interval_ms` | `1000` | How often the live transcript updates. |
| `noise_filter` | `light` | `off`, `light` or `strong`. Spectral filter for fans and hum, before VAD/STT. |

### `tts` (speech synthesis, Kokoro-82M)
| Key | Default | Meaning |
|---|---|---|
| `device` | `cpu` | `cpu` or `cuda` (not faster on a GTX 10xx; see `docs/DEVELOPMENT.md`). |
| `model_file`, `voices_file` | `kokoro/kokoro-v1.0.onnx`, `kokoro/voices-v1.0.bin` | Under `models/`. |
| `cpu_threads` | `4` | Four threads each for TTS and STT on an 8-thread CPU. |
| `speed` | `1.05` | Default speaking rate (the page slider overrides it per browser). |
| `voices` | `af_heart`, `ef_dora`, … | Voice used for each language when *Voice* is "Auto". |
| `voices_male` | `am_michael`, `em_alex`, … | Auto voice when the gender filter is Male. |
| `first_chunk_min_chars` | `12` | Shortest clause that may be spoken as the first piece. |
| `first_chunk_wait_ms` | `600` | After this long without a clause break, speak the words received so far. |
| `chunk_max_chars` | `220` | Longest piece sent to TTS. |

### `vad` (turn taking and barge-in)
| Key | Default | Meaning |
|---|---|---|
| `threshold` / `neg_threshold` | `0.5` / `0.35` | Silero speech probability to enter / leave speech. |
| `min_speech_ms` | `160` | Speech needed to start a user utterance. |
| `spec_silence_ms` | `250` | Silence after which the answer is prepared speculatively. |
| `end_silence_ms` | `800` | Silence that ends your turn. Raise it (e.g. 1200) if you get cut off while thinking mid-sentence. |
| `preroll_ms` | `350` | Audio kept from before the speech was detected. |
| `max_utterance_s` | `30` | Force the end of a very long utterance. |
| `barge_in_ms` | `280` | Speech needed to interrupt the assistant. |
| `barge_in_threshold` | `0.7` | Stricter VAD threshold while the assistant is talking without headphones. |
| `echo_check` | `true` | Transcribe and compare before interrupting (speakers). Headphones mode always skips it. |
| `verify_max_ms` | `1500` | Longest audio window for one echo check; longer echo slides the window. |

---

## GPU option

The defaults run speech on the CPU on purpose. The GTX 1070 Ti stays in a
low-clock power state between short bursts, and small models are not faster
there. Keeping all 8 GB of VRAM for LM Studio matters more. To try CUDA
anyway:

```
setup.bat gpu
```

then set `"device": "cuda"` in `stt` and/or `tts`. It installs
`onnxruntime-gpu` 1.26 with cuDNN 9.10. These are the last builds that
support Pascal (GTX 10xx) cards; newer ones need CUDA 13 or dropped Pascal
kernels. `setup.bat cpu` switches back. If CUDA cannot start, VoiceLink
falls back to the CPU automatically.

---

## WebSocket protocol

`ws://127.0.0.1:8765/ws`. You can use it to build other clients (the
browser page is just one).

**Client → server**
* Binary: int16 little-endian mono PCM at **16 kHz**, any frame size.
* `{"type":"config","headphones":bool,"voice":"af_heart"|"","speed":1.1,"mic":bool}`: any subset.
* `{"type":"text","text":"..."}`: typed message (interrupts).
* `{"type":"interrupt"}`, `{"type":"reset"}`.
* `{"type":"playback_idle","turn":n}`: the client finished playing turn *n*.

**Server → client**
* Binary: `[uint32 turn][uint32 seg]` + int16 mono PCM at **24 kHz**, one
  per speech segment, preceded by
  `{"type":"seg","turn","seg","text","duration","lang"}`.
* `ready` (LLM status, voices), `state` (`listening`, `user_speaking`,
  `thinking`, `speaking`), `partial` (live transcript), `user` (final
  transcript of a committed turn), `ignored` (noise-only utterance),
  `assistant_delta` (reply text), `turn_done` (with timing metrics),
  `duck` / `unduck`, `stop` (discard audio of that turn now),
  `reset_done`, `error`,
  `tool` (`phase` `start`/`done`, `name`, `args` — never spoken).

`GET /api/status` returns the LM Studio status, voices, languages and
MCP tool list as JSON. `GET /api/mcp` is the same tool list with
descriptions. `POST /mcp` is the stateless MCP endpoint (JSON-RPC;
GET / SSE sessions are rejected).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Replies appear as text but there is **no sound** | Chrome is probably muted in the Windows **Volume Mixer** (right-click the speaker icon > *Open Volume mixer*), or Windows is sending sound to a device you are not listening to. Run **`soundcheck.bat`** and make VoiceLink speak within 20 s. It unmutes Chrome/Edge and shows each app's level per device. Then reload the page. |
| Header says **LM Studio offline** | Start the server in LM Studio (Developer tab) and load a model. VoiceLink reconnects by itself. |
| **"already running"** on start | Another VoiceLink is using port 8765: close its window, or change `server.port`. |
| The assistant stops for **background sounds** (TV, videos) | **Ignore speech playing on this PC** is on by default (voice panel → Devices). It listens to what the speakers play so a video does not start a turn. Untick it if you need the microphone to hear that. For people in the room or a TV, tick **Only my voice** and record 10 seconds. Headphones still help. |
| **PC fan / AC** keeps triggering listening | Voice panel → Devices → Background noise filter **Strong**. |
| It **interrupts itself** on speakers | Use Chrome (Edge and Firefox cancel less echo), lower the speaker volume, or use headphones. |
| **No microphone** prompt / "text-only mode" | Allow the mic for `127.0.0.1` in the browser's site settings, and open the page via `http://127.0.0.1:8765` (not a LAN IP). |
| **Cuts you off** when you pause | Raise `vad.end_silence_ms`. |
| Reply comes in the **wrong language** | Set `stt.languages` to the languages you speak; a bigger LLM helps too. |
| Slow first words | Check that the LM Studio model is fully on the GPU (GPU offload at max) and not a thinking model with thinking forced on. |

Logs appear in the `run.bat` window. For detailed timings per segment, run
`set VOICELINK_DEBUG=1` before `run.bat`.

---

## Project layout

```
setup.bat, run.bat, test.bat, soundcheck.bat   Windows entry points
config.json                    settings (see above)
voicelink/
  server.py     aiohttp web server, model loading, /ws endpoint
  session.py    duplex turn engine: VAD state machine, speculative turns,
                barge-in, echo check, LLM→TTS pipeline, history
  llm.py        LM Studio streaming client (thinking off, model auto-detect, tool loop)
  mcp.py        stateless MCP hub: local search/weather/stocks + remote HTTP
  stt.py        Parakeet TDT via onnx-asr
  tts.py        Kokoro via kokoro-onnx, voice/language mapping
  vad.py        Silero VAD (streaming ONNX)
  textproc.py   think-tag filter, speech text cleanup, chunker, language id
  gpu.py        CUDA/CPU execution-provider selection
  config.py     defaults and config loading
web/            browser client (no build step, no external scripts)
tools/          model download, end-to-end self-test, filter tests, benchmarks
models/         downloaded models (not part of the source)
data/           saved conversations
docs/DEVELOPMENT.md   design decisions, benchmarks and test results
changes_v1.md   v1 feature list and implementation notes
```

Models: [Silero VAD](https://github.com/snakers4/silero-vad) (MIT),
[NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
(CC-BY-4.0, ONNX export by
[istupakov](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx)),
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0, ONNX by
[kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)).
