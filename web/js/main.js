// VoiceLink browser client: microphone capture, audio playback and UI.
// All speech processing happens on the Python server over one WebSocket.
import * as attach from "./attach.js";
import * as editor from "./editor.js";
import * as history from "./history.js";
import { hydrateIcons, icon } from "./icons.js";
import { initModelPanel, panelOpened, setMcp, setStatus } from "./modelpanel.js";
import { initProfiles } from "./profiles.js";
import { settings } from "./settings.js";
import { $, capsHTML, esc, toast } from "./ui.js";
import { languageList, resolveLanguages } from "./locale.js";
import { initVoicePanel, onFilterMessage, refreshDevices, setCatalog } from "./voicepanel.js";

hydrateIcons();

const els = {
  orb: $("orb"), state: $("stateLabel"), metrics: $("metrics"), chat: $("chat"),
  pill: $("modelPill"), mic: $("micBtn"), stop: $("stopBtn"), reset: $("newChatBtn"),
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
let capCtx = null, capNode = null, micStream = null, micSource = null;
let playCtx = null, outGain = null, outAnalyser = null;
let nextTime = 0;
let lastLang = "en";
const sources = new Set();          // {src, turn, seg, start, end}
const stopped = new Set();          // turn ids whose audio must be dropped
const segInfo = new Map();          // "turn:seg" -> {text, start, end}
const bubbles = new Map();          // turn id -> {el, raw}
let partialEl = null;
let micLevel = 0;
let reconnectDelay = 500;

// ------------------------------------------------------------------ audio

function ensurePlayback() {
  if (playCtx) return;
  playCtx = new AudioContext({ latencyHint: "interactive" });
  outGain = playCtx.createGain();
  outAnalyser = playCtx.createAnalyser();
  outAnalyser.fftSize = 512;
  outGain.connect(outAnalyser).connect(playCtx.destination);
  applySpeaker();
}

async function applySpeaker() {
  if (!playCtx || !("setSinkId" in playCtx)) return;
  const id = settings.get("devices.speaker") || "";
  try {
    await playCtx.setSinkId(id);
  } catch (e) {
    toast(`Speaker not available (${e.name}); using the system default`, "error");
    await playCtx.setSinkId("").catch(() => {});
  }
}

async function openMicStream() {
  const audio = { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 };
  const id = settings.get("devices.mic");
  if (id) {
    try {
      return await navigator.mediaDevices.getUserMedia({ audio: { ...audio, deviceId: { exact: id } } });
    } catch (e) {
      toast("The selected microphone is not available; using the system default", "error");
    }
  }
  return navigator.mediaDevices.getUserMedia({ audio });
}

async function startMic() {
  const stream = await openMicStream();
  try {
    capCtx = new AudioContext({ sampleRate: 16000, latencyHint: "interactive" });
    await capCtx.audioWorklet.addModule("/static/capture-worklet.js");
    capCtx.createMediaStreamSource(stream).disconnect();
  } catch (e) {
    // Some browsers refuse a 16 kHz context for microphone input; resample in the worklet instead.
    if (capCtx) capCtx.close();
    capCtx = new AudioContext({ latencyHint: "interactive" });
    await capCtx.audioWorklet.addModule("/static/capture-worklet.js");
  }
  capNode = new AudioWorkletNode(capCtx, "capture", { numberOfOutputs: 1 });
  const sink = capCtx.createGain();
  sink.gain.value = 0;
  capNode.connect(sink).connect(capCtx.destination);
  capNode.port.onmessage = (ev) => {
    micLevel = ev.data.level;
    if (micOn && ws && ws.readyState === WebSocket.OPEN) ws.send(ev.data.pcm);
  };
  attachStream(stream);
  refreshDevices();   // labels are only visible once the microphone is allowed
}

function attachStream(stream) {
  if (micSource) micSource.disconnect();
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  micStream = stream;
  micSource = capCtx.createMediaStreamSource(stream);
  micSource.connect(capNode);
  const track = stream.getAudioTracks()[0];
  if (track) track.onended = () => toast("The microphone was disconnected", "error");
}

async function switchMic() {
  if (!capCtx) return;
  try {
    attachStream(await openMicStream());
  } catch (e) {
    toast("Could not open the microphone: " + e.message, "error");
  }
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

let previewSrc = null;
async function playPreview(wav) {
  ensurePlayback();
  if (playCtx.state !== "running") await playCtx.resume();
  const ab = await playCtx.decodeAudioData(wav.slice(0));
  if (previewSrc) try { previewSrc.stop(); } catch (_) {}
  return new Promise((resolve) => {
    previewSrc = playCtx.createBufferSource();
    previewSrc.buffer = ab;
    previewSrc.connect(outGain);
    previewSrc.onended = resolve;
    previewSrc.start();
  });
}

function stopPreview() {
  if (previewSrc) try { previewSrc.stop(); } catch (_) {}
  previewSrc = null;
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
  const gender = settings.get("gender");
  const m = settings.get("model");
  const inp = settings.get("input");
  send({
    type: "config", headphones: !!settings.get("headphones"), mic: micOn,
    noise: inp.noise, playback_gate: !!inp.playbackGate && !settings.get("headphones"), loopback: inp.loopback || "",
    only_me: !!inp.onlyMe, only_me_strictness: inp.onlyMeStrict, voiceprint: inp.voiceprint || null,
    voice: settings.get("voice") || null, gender: gender === "all" ? null : gender,
    speed: parseFloat(settings.get("speed")) || null, fx: settings.get("fx"),
    system_prompt: m.systemPrompt || null,
    gen: { temperature: m.temperature, top_p: m.top_p, top_k: m.top_k, repeat_penalty: m.repeat_penalty,
           max_tokens: m.max_tokens, history: m.history, thinking: !!m.thinking },
    languages: languageList(settings),
    output_lang: settings.get("languages.output") || "voice",
    incognito: !!$("incognito")?.checked,
  });
}

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { reconnectDelay = 500; sendConfig(); attach.resend(); };
  ws.onclose = () => {
    if (started) setState("idle", "Disconnected from server — retrying…");
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, 5000);
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data !== "string") { playSegment(ev.data); return; }
    handle(JSON.parse(ev.data));
  };
}

function handle(m) {
  if (onFilterMessage(m)) return;
  switch (m.type) {
    case "ready":
      showLLM(m.llm, m.mcp);
      break;
    case "state":
      if (!started) break;
      setState(micOn || m.state !== "listening" ? m.state : "muted");
      if (m.state !== "user_speaking") clearPartial();
      break;
    case "partial":
      showPartial(m.text);
      break;
    case "user":
      clearPartial();
      addMsg("user", m.text, m.attachments);
      if (m.attachments && m.attachments.length) attach.consumed(m.attachments.map((a) => a.id));
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
      if (m.lang) lastLang = m.lang;
      break;
    case "turn_done":
      showMetrics(m.metrics);
      break;
    case "duck": duck(true); break;
    case "unduck": duck(false); break;
    case "stop": stopPlayback(m.turn); break;
    case "reset_done":
      clearChat();
      addMsg("system", m.incognito ? "New incognito conversation (not stored)" : "New conversation");
      history.setActive(null);
      break;
    case "conversation":
      clearChat();
      for (const msg of m.messages) {
        const el = addMsg(msg.role === "user" ? "user" : "assistant", msg.content, msg.attachments);
        if (msg.role === "assistant" && /\s—$/.test(msg.content)) el.classList.add("interrupted");
      }
      addMsg("system", `Continuing “${m.title}”`);
      history.setActive(m.id, { labels: m.labels, project_id: m.project_id });
      break;
    case "conv_saved":
      history.saved(m.id, m.title, { labels: m.labels, project_id: m.project_id });
      break;
    case "tool": {
      if (m.phase !== "start") break;
      const label = { web_search: "Searching the web", weather: "Checking the weather",
        stock_quote: "Looking up the market", fetch_page: "Reading a page" }[m.name] || m.name;
      const q = m.args && (m.args.query || m.args.place || m.args.symbol || m.args.url);
      addMsg("system", q ? `${label} · ${q}` : label);
      break;
    }
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

const KIND_ICON = { image: "image", document: "file", audio: "music", video: "film" };

function addMsg(role, text, attachments) {
  const el = document.createElement("div");
  el.className = "msg " + role;
  el.textContent = text;
  if (attachments && attachments.length) {
    const row = document.createElement("div");
    row.className = "msg-atts";
    row.innerHTML = attachments.map((a) => a.data
      ? `<img src="${a.data}" alt="${esc(a.name)}" title="${esc(a.name)}">`
      : `<span class="msg-att" title="${esc(a.name)}">${icon(KIND_ICON[a.kind] || "file")}${esc(a.name)}</span>`).join("");
    el.prepend(row);
  }
  els.chat.appendChild(el);
  scrollChat();
  return el;
}

function clearChat() {
  els.chat.innerHTML = "";
  bubbles.clear();
  partialEl = null;
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

function showLLM(llm, mcp) {
  els.pill.className = "model-pill " + (llm.ok ? "ok" : "bad");
  const name = llm.ok ? (llm.info && llm.info.name) || llm.model : "LM Studio offline";
  let title = llm.ok ? `${llm.model} @ ${llm.base_url}` : `${llm.base_url}: ${llm.error}`;
  if (llm.ok && llm.info) {
    const i = llm.info;
    title += [i.params && `\n${i.params} parameters`, i.quant && `${i.quant}`].filter(Boolean).join(" · ");
  }
  if (mcp && mcp.enabled && mcp.tools && mcp.tools.length) {
    title += mcp.offered
      ? `\nLive tools: ${mcp.tools.join(", ")}`
      : "\nLive tools off (this model has no tool use)";
  }
  els.pill.innerHTML = `<span class="dot"></span><span class="name">${esc(name)}</span>` +
    `<span class="caps">${llm.ok ? capsHTML(llm.info) : ""}</span>`;
  els.pill.title = title + "\nClick for model settings";
  setStatus(llm);
  if (mcp) setMcp(mcp);
  attach.setCaps(llm.ok ? llm.info : null);
}

async function pollStatus() {
  try {
    const r = await fetch("/api/status");
    const data = await r.json();
    showLLM(data.llm, data.mcp);
  } catch (_) {}
  setTimeout(pollStatus, 10000);
}

async function loadVoices() {
  try {
    setCatalog(await (await fetch("/api/voices")).json());
  } catch (_) {
    setTimeout(loadVoices, 3000);
  }
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
  ensurePlayback();
  try {
    await startMic();
  } catch (e) {
    micOn = false;
    renderMic("none");
    addMsg("error", `Microphone unavailable (${e.message}). Text-only mode: type below.`);
  }
  if (ws && ws.readyState === WebSocket.OPEN) {
    sendConfig();
    setState(micOn ? "listening" : "muted");
  }
  ensureSocket();
  animate();
}

function renderMic(mode) {
  els.mic.className = "round-btn " + (mode === "on" ? "on" : "off");
  els.mic.innerHTML = icon(mode === "on" ? "mic" : "micOff");
  els.mic.title = mode === "none" ? "No microphone" : mode === "on" ? "Mute microphone (M)" : "Unmute microphone (M)";
}

function toggleMic() {
  if (!capCtx) return;
  micOn = !micOn;
  renderMic(micOn ? "on" : "off");
  sendConfig();
  if (!micOn && state === "listening") setState("muted");
  if (micOn && state === "muted") setState("listening");
}

function sendText() {
  const text = els.input.value.trim();
  if (attach.isBusy()) { toast("Wait until the attachments are ready", "error"); return; }
  if (!text && !attach.hasReady()) return;
  const go = async () => {
    if (!started) await start();
    const trySend = () => {
      if (ws && ws.readyState === WebSocket.OPEN) send({ type: "text", text });
      else setTimeout(trySend, 200);
    };
    trySend();
  };
  go();
  els.input.value = "";
  editor.reset();
  autoGrow();
}

function autoGrow() {
  els.input.style.height = "auto";
  const border = els.input.offsetHeight - els.input.clientHeight;
  els.input.style.height = Math.min(els.input.scrollHeight + border, 180) + "px";
}

els.orb.onclick = () => (started ? toggleMic() : start());
els.mic.onclick = () => (started ? toggleMic() : start());
els.stop.onclick = () => send({ type: "interrupt" });
els.reset.onclick = newChat;
els.form.onsubmit = (e) => { e.preventDefault(); sendText(); };
els.input.addEventListener("input", autoGrow);
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); sendText(); }
});
document.addEventListener("keydown", (e) => {
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) {
    if (e.key === "Escape") t.blur();
    return;
  }
  if (e.key === "Escape") { send({ type: "interrupt" }); document.activeElement?.blur(); }
  if (e.key === "m" || e.key === "M") started && toggleMic();
});

// Opening a conversation or starting a new one works before the voice session
// is started too: the WebSocket connects first, without the microphone.
function ensureSocket() {
  if (!ws) connect();
}

function whenOpen(fn) {
  ensureSocket();
  if (ws.readyState === WebSocket.OPEN) fn();
  else setTimeout(() => whenOpen(fn), 150);
}

function newChat() {
  whenOpen(() => send({ type: "reset", incognito: !!$("incognito").checked }));
}
function openConversation(id) { whenOpen(() => send({ type: "open", id })); }

function applyPanels() {
  const left = !!settings.ui("left"), right = !!settings.ui("right");
  const pane = settings.ui("rightPane") || "model";
  $("leftPanel").hidden = !left;
  $("rightPanel").hidden = !right;
  $("toggleLeft").classList.toggle("active", left);
  $("toggleRight").classList.toggle("active", right && pane === "model");
  $("voiceBtn").classList.toggle("active", right && pane === "voice");
  $("devicesBtn").classList.toggle("active", right && pane === "io");
  $("pane-model").hidden = pane !== "model";
  $("pane-voice").hidden = pane !== "voice";
  $("pane-io").hidden = pane !== "io";
  $("rightTitleText").textContent = { model: "Model", voice: "Voice", io: "Devices" }[pane];
  $("rightTitleIcon").innerHTML = icon({ model: "chip", voice: "voice", io: "speaker" }[pane]);
  $("modelRefresh").hidden = pane !== "model";
  $("loadFoot").hidden = pane !== "model";
  if (right) {
    panelOpened();
    if (pane === "io") refreshDevices();
  }
}

function togglePanel(side) {
  settings.setUI(side, !settings.ui(side));
  applyPanels();
}

function openRight(pane) {
  if (settings.ui("right") && settings.ui("rightPane") === pane) {
    settings.setUI("right", false);
  } else {
    settings.setUI("rightPane", pane);
    settings.setUI("right", true);
  }
  applyPanels();
}

$("toggleLeft").onclick = () => togglePanel("left");
$("toggleRight").onclick = () => openRight("model");
$("voiceBtn").onclick = () => openRight("voice");
$("devicesBtn").onclick = () => openRight("io");
els.pill.onclick = () => openRight("model");

settings.onChange((paths) => {
  const all = paths.includes("*");
  if (all || paths.some((p) => ["voice", "gender", "speed", "headphones"].includes(p) ||
                               p.startsWith("fx") || p.startsWith("model") || p.startsWith("input") ||
                               p.startsWith("languages"))) sendConfig();
  if (all || paths.includes("devices.mic")) switchMic();
  if (all || paths.includes("devices.speaker")) applySpeaker();
});

initVoicePanel({
  playPreview, stopPreview, lang: () => lastLang, send: (obj) => whenOpen(() => send(obj)),
  sendUser: (text) => whenOpen(() => send({ type: "text", text })),
  ensureStarted: async () => {
    await start();
    if (!capCtx) { toast("The microphone is not available", "error"); return false; }
    if (!micOn) toggleMic();
    return true;
  },
});
initProfiles();
history.initHistory({
  newChat, open: openConversation,
  incognito: (on) => whenOpen(() => send({ type: "config", incognito: !!on })),
  meta: (m) => whenOpen(() => send({ type: "conv_meta", labels: m.labels, project_id: m.project_id })),
});
editor.initEditor();
attach.initAttach({ send: (obj) => whenOpen(() => send(obj)) });
initModelPanel({ statusChanged: (llm) => { if (llm) showLLM(llm); else pollStatus(); } });
applyPanels();
resolveLanguages(settings).then(() => sendConfig());
ensureSocket();
renderMic("on");
refreshDevices();
loadVoices();
pollStatus();
