"use strict";
// VoiceLink browser client: microphone capture, audio playback and UI.
// All speech processing happens on the Python server over one WebSocket.

const $ = (id) => document.getElementById(id);
const els = {
  orb: $("orb"), state: $("stateLabel"), metrics: $("metrics"), chat: $("chat"),
  llm: $("llmStatus"), mic: $("micBtn"), stop: $("stopBtn"), reset: $("resetBtn"),
  headphones: $("headphones"), voice: $("voice"), speed: $("speed"), speedOut: $("speedOut"),
  form: $("textForm"), input: $("textInput"),
};

const TTS_RATE = 24000;
const STATE_TEXT = {
  listening: "Listening…", user_speaking: "Hearing you…", thinking: "Thinking…",
  speaking: "Speaking — just talk to interrupt", muted: "Microphone muted",
};

let ws = null;
let started = false;
let micOn = true;
let state = "idle";
let capCtx = null, playCtx = null, outGain = null, outAnalyser = null;
let nextTime = 0;
const sources = new Set();          // {src, turn, seg, start, end}
const stopped = new Set();          // turn ids whose audio must be dropped
const segInfo = new Map();          // "turn:seg" -> {text, start, end}
const bubbles = new Map();          // turn id -> {el, raw}
let partialEl = null;
let micLevel = 0;
let reconnectDelay = 500;

const saved = JSON.parse(localStorage.getItem("voicelink") || "{}");
els.headphones.checked = !!saved.headphones;
els.speed.value = saved.speed || 1.05;
els.speedOut.textContent = (+els.speed.value).toFixed(2) + "×";

// ------------------------------------------------------------------ audio

function startPlayback() {
  playCtx = new AudioContext({ latencyHint: "interactive" });
  outGain = playCtx.createGain();
  outAnalyser = playCtx.createAnalyser();
  outAnalyser.fftSize = 512;
  outGain.connect(outAnalyser).connect(playCtx.destination);
}

async function startMic() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
  });

  try {
    capCtx = new AudioContext({ sampleRate: 16000, latencyHint: "interactive" });
    await capCtx.audioWorklet.addModule("/static/capture-worklet.js");
    capCtx.createMediaStreamSource(stream);
  } catch (e) {
    // Some browsers refuse a 16 kHz context for microphone input; resample in the worklet instead.
    if (capCtx) capCtx.close();
    capCtx = new AudioContext({ latencyHint: "interactive" });
    await capCtx.audioWorklet.addModule("/static/capture-worklet.js");
  }
  const src = capCtx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(capCtx, "capture", { numberOfOutputs: 1 });
  const sink = capCtx.createGain();
  sink.gain.value = 0;
  src.connect(node).connect(sink).connect(capCtx.destination);
  node.port.onmessage = (ev) => {
    micLevel = ev.data.level;
    if (micOn && ws && ws.readyState === WebSocket.OPEN) ws.send(ev.data.pcm);
  };
}

function playSegment(buf) {
  const view = new DataView(buf);
  const turn = view.getUint32(0, true);
  const seg = view.getUint32(4, true);
  if (stopped.has(turn)) return;
  if (playCtx.state !== "running") playCtx.resume();
  const pcm = new Int16Array(buf, 8);
  const f32 = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) f32[i] = pcm[i] / 32768;
  const ab = playCtx.createBuffer(1, f32.length, TTS_RATE);
  ab.copyToChannel(f32, 0);
  const src = playCtx.createBufferSource();
  src.buffer = ab;
  src.connect(outGain);
  const start = Math.max(playCtx.currentTime + 0.03, nextTime);
  src.start(start);
  nextTime = start + ab.duration;
  const item = { src, turn, seg, start, end: nextTime };
  sources.add(item);
  const info = segInfo.get(turn + ":" + seg);
  if (info) Object.assign(info, { start, end: nextTime });
  src.onended = () => {
    sources.delete(item);
    if (![...sources].some((s) => s.turn === turn)) send({ type: "playback_idle", turn });
  };
}

function stopPlayback(turn) {
  stopped.add(turn);
  const t = playCtx.currentTime;
  outGain.gain.cancelScheduledValues(t);
  outGain.gain.setTargetAtTime(0, t, 0.012);
  const victims = [...sources];
  sources.clear();
  setTimeout(() => {
    for (const s of victims) { try { s.src.stop(); } catch (_) {} }
    outGain.gain.cancelScheduledValues(playCtx.currentTime);
    outGain.gain.setValueAtTime(1, playCtx.currentTime);
  }, 60);
  nextTime = 0;
  markHeard(turn, t);
}

function duck(on) {
  if (!outGain) return;
  const t = playCtx.currentTime;
  outGain.gain.cancelScheduledValues(t);
  outGain.gain.setTargetAtTime(on ? 0.22 : 1, t, on ? 0.02 : 0.08);
}

// -------------------------------------------------------------- websocket

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

function sendConfig() {
  send({
    type: "config", headphones: els.headphones.checked, mic: micOn,
    voice: els.voice.value || null, speed: parseFloat(els.speed.value) || null,
  });
  localStorage.setItem("voicelink", JSON.stringify({
    headphones: els.headphones.checked, voice: els.voice.value, speed: els.speed.value,
  }));
}

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { reconnectDelay = 500; sendConfig(); };
  ws.onclose = () => {
    setState("idle", "Disconnected from server — retrying…");
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, 5000);
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data !== "string") { playSegment(ev.data); return; }
    handle(JSON.parse(ev.data));
  };
}

function handle(m) {
  switch (m.type) {
    case "ready":
      showLLM(m.llm);
      fillVoices(m.voices, m.voice_map, m.speed);
      break;
    case "state":
      setState(micOn || m.state !== "listening" ? m.state : "muted");
      if (m.state !== "user_speaking") clearPartial();
      break;
    case "partial":
      showPartial(m.text);
      break;
    case "user":
      clearPartial();
      addMsg("user", m.text);
      break;
    case "ignored":
      clearPartial();
      break;
    case "assistant_delta": {
      const b = bubble(m.turn);
      b.raw += m.text;
      b.el.textContent = b.raw.trimStart();
      scrollChat();
      break;
    }
    case "seg":
      segInfo.set(m.turn + ":" + m.seg, { text: m.text, start: 0, end: 0 });
      break;
    case "turn_done":
      showMetrics(m.metrics);
      break;
    case "duck": duck(true); break;
    case "unduck": duck(false); break;
    case "stop": stopPlayback(m.turn); break;
    case "reset_done":
      els.chat.innerHTML = "";
      bubbles.clear();
      addMsg("system", "New conversation");
      break;
    case "error":
      addMsg("error", m.message);
      break;
  }
}

// --------------------------------------------------------------------- UI

function setState(s, label) {
  state = s;
  els.orb.className = "orb " + s;
  els.state.textContent = label || STATE_TEXT[s] || "";
}

function addMsg(role, text) {
  const el = document.createElement("div");
  el.className = "msg " + role;
  el.textContent = text;
  els.chat.appendChild(el);
  scrollChat();
  return el;
}

function bubble(turn) {
  let b = bubbles.get(turn);
  if (!b) {
    b = { el: addMsg("assistant", ""), raw: "" };
    bubbles.set(turn, b);
  }
  return b;
}

function markHeard(turn, at) {
  const b = bubbles.get(turn);
  if (!b) return;
  let heardWords = 0;
  for (const [key, info] of segInfo) {
    if (!key.startsWith(turn + ":") || !info.end) continue;
    const n = info.text.split(/\s+/).length;
    if (at >= info.end) heardWords += n;
    else if (at > info.start) heardWords += Math.floor(n * (at - info.start) / (info.end - info.start));
  }
  const tokens = b.raw.trim().split(/(\s+)/);
  let words = 0, cut = tokens.length;
  for (let i = 0; i < tokens.length; i++) {
    if (!/^\s+$/.test(tokens[i]) && tokens[i]) {
      if (words === heardWords) { cut = i; break; }
      words++;
    }
  }
  b.el.textContent = "";
  b.el.append(tokens.slice(0, cut).join(""));
  if (cut < tokens.length) {
    const span = document.createElement("span");
    span.className = "unheard";
    span.textContent = tokens.slice(cut).join("");
    b.el.append(span);
  }
  b.el.classList.add("interrupted");
}

function showPartial(text) {
  if (!partialEl) partialEl = addMsg("user partial", "");
  partialEl.textContent = text;
  scrollChat();
}

function clearPartial() {
  if (partialEl) { partialEl.remove(); partialEl = null; }
}

function scrollChat() { els.chat.scrollTop = els.chat.scrollHeight; }

function showMetrics(m) {
  const parts = [];
  if (m.stt_ms != null) parts.push(`STT ${m.stt_ms} ms`);
  if (m.ttft_ms != null) parts.push(`LLM 1st token ${m.ttft_ms} ms`);
  if (m.tts_first_ms != null) parts.push(`TTS 1st chunk ${m.tts_first_ms} ms`);
  if (m.first_audio_ms != null) parts.push(`voice-to-voice ${(m.first_audio_ms / 1000).toFixed(2)} s`);
  els.metrics.textContent = parts.join("  ·  ");
}

function showLLM(llm) {
  els.llm.className = "pill " + (llm.ok ? "ok" : "bad");
  els.llm.textContent = llm.ok ? `LM Studio · ${llm.model}` : "LM Studio offline";
  els.llm.title = llm.ok ? llm.base_url : `${llm.base_url}: ${llm.error}`;
}

function fillVoices(voices, map, speed) {
  if (els.voice.options.length <= 1) {
    const names = { a: "US English", b: "UK English", e: "Spanish", f: "French", h: "Hindi",
                    i: "Italian", j: "Japanese", p: "Portuguese", z: "Chinese" };
    const groups = {};
    for (const v of voices) (groups[v[0]] = groups[v[0]] || []).push(v);
    for (const [k, list] of Object.entries(groups)) {
      const og = document.createElement("optgroup");
      og.label = names[k] || k;
      for (const v of list) og.append(new Option(v + (Object.values(map).includes(v) ? " ★" : ""), v));
      els.voice.append(og);
    }
    els.voice.value = saved.voice || "";
  }
  els.speed.value = saved.speed || speed;
  els.speedOut.textContent = (+els.speed.value).toFixed(2) + "×";
}

async function pollStatus() {
  try {
    const r = await fetch("/api/status");
    showLLM((await r.json()).llm);
  } catch (_) {}
  setTimeout(pollStatus, 10000);
}

function animate() {
  let s = 1;
  if (state === "user_speaking" || state === "listening") s = 1 + Math.min(micLevel * 6, 0.35);
  if (state === "speaking" && outAnalyser) {
    const d = new Uint8Array(outAnalyser.fftSize);
    outAnalyser.getByteTimeDomainData(d);
    let e = 0;
    for (const v of d) e += (v - 128) * (v - 128);
    s = 1 + Math.min(Math.sqrt(e / d.length) / 40, 0.35);
  }
  els.orb.style.setProperty("--s", s.toFixed(3));
  requestAnimationFrame(animate);
}

// --------------------------------------------------------------- controls

async function start() {
  if (started) return;
  started = true;
  setState("idle", "Starting microphone…");
  startPlayback();
  try {
    await startMic();
  } catch (e) {
    micOn = false;
    els.mic.className = "toggle off";
    els.mic.textContent = "No mic";
    addMsg("error", `Microphone unavailable (${e.message}). Text-only mode: type below.`);
  }
  connect();
  animate();
}

function toggleMic() {
  if (!capCtx) return;
  micOn = !micOn;
  els.mic.className = "toggle " + (micOn ? "on" : "off");
  els.mic.textContent = micOn ? "Mic on" : "Mic off";
  sendConfig();
  if (!micOn && state === "listening") setState("muted");
  if (micOn && state === "muted") setState("listening");
}

els.orb.onclick = () => (started ? toggleMic() : start());
els.mic.onclick = () => started && toggleMic();
els.stop.onclick = () => send({ type: "interrupt" });
els.reset.onclick = () => send({ type: "reset" });
els.headphones.onchange = sendConfig;
els.voice.onchange = sendConfig;
els.speed.oninput = () => { els.speedOut.textContent = (+els.speed.value).toFixed(2) + "×"; };
els.speed.onchange = sendConfig;
els.form.onsubmit = async (e) => {
  e.preventDefault();
  const text = els.input.value.trim();
  if (!text) return;
  if (!started) await start();
  const trySend = () => {
    if (ws && ws.readyState === WebSocket.OPEN) send({ type: "text", text });
    else setTimeout(trySend, 200);
  };
  trySend();
  els.input.value = "";
};
document.addEventListener("keydown", (e) => {
  if (e.target === els.input) return;
  if (e.key === "Escape") send({ type: "interrupt" });
  if (e.key === "m" || e.key === "M") started && toggleMic();
});

pollStatus();
