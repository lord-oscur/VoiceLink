// File attachments for the next message: images (vision models), documents,
// audio (transcribed) and video (audio transcript + frames for vision models).
// Media is decoded in the browser, so no extra tools are needed on the server.
import { icon } from "./icons.js";
import { $, esc, fmtBytes, toast } from "./ui.js";

const DOC_EXT = new Set(("pdf docx txt md markdown csv tsv json jsonl xml html htm log ini toml yaml yml py js mjs ts tsx " +
  "jsx java c h cpp hpp cs go rs rb php sh bat ps1 sql css scss lua kt swift r m tex srt vtt rtf cfg conf env").split(" "));
const AUDIO_EXT = new Set("mp3 wav ogg oga opus m4a aac flac weba wma".split(" "));
const VIDEO_EXT = new Set("mp4 webm mov m4v mkv ogv avi".split(" "));
const IMAGE_EXT = new Set("png jpg jpeg webp gif bmp avif".split(" "));
const KIND_ICON = { image: "image", document: "file", audio: "music", video: "film" };
const MAX_MB = { image: 40, document: 60, audio: 400, video: 400 };
const IMAGE_MAX_SIDE = 1280;
const FRAME_MAX_SIDE = 768;

let items = [];          // {id, kind, name, size, status: busy|ready|error, note, record}
let vision = false;
let modelName = "";
let hooks = {};

export function initAttach(h) {
  hooks = h;
  $("attachBtn").onclick = () => $("attachFile").click();
  $("attachFile").onchange = () => { addFiles([...$("attachFile").files]); $("attachFile").value = ""; };
  $("attachTray").onclick = (e) => {
    const b = e.target.closest("[data-remove]");
    if (b) remove(b.dataset.remove);
  };
  const zone = document.querySelector(".center");
  zone.addEventListener("dragover", (e) => {
    if ([...e.dataTransfer.types].includes("Files")) { e.preventDefault(); zone.classList.add("drop"); }
  });
  zone.addEventListener("dragleave", (e) => { if (!zone.contains(e.relatedTarget)) zone.classList.remove("drop"); });
  zone.addEventListener("drop", (e) => {
    zone.classList.remove("drop");
    if (!e.dataTransfer.files.length) return;
    e.preventDefault();
    addFiles([...e.dataTransfer.files]);
  });
  $("textInput").addEventListener("paste", (e) => {
    const files = [...(e.clipboardData?.files || [])];
    if (files.length) { e.preventDefault(); addFiles(files); }
  });
  setCaps(null);
}

export function setCaps(info) {
  vision = !!(info && info.caps && info.caps.vision);
  modelName = info ? info.name : "";
  const kinds = ["documents (PDF, Word, text, code)", "audio (transcribed)", "video (transcribed" + (vision ? " + frames)" : ")")];
  if (vision) kinds.unshift("images");
  $("attachBtn").title = `Attach files: ${kinds.join(", ")}. You can also drop or paste them.` +
    (vision ? "" : "\nImages need a vision model.");
  const accept = [...DOC_EXT, ...AUDIO_EXT, ...VIDEO_EXT].map((e) => "." + e);
  if (vision) accept.push("image/*");
  $("attachFile").accept = accept.join(",");
}

export const hasReady = () => items.some((i) => i.status === "ready");
export const isBusy = () => items.some((i) => i.status === "busy");

// Called when the server used attachments in a user turn.
export function consumed(ids) {
  items = items.filter((i) => !ids.includes(i.id));
  render();
}

// After a reconnect the server session is new: send the ready attachments again.
export function resend() {
  for (const i of items) if (i.status === "ready") hooks.send({ type: "attach_add", attachment: i.record });
}

function kindOf(file) {
  const ext = (file.name.split(".").pop() || "").toLowerCase();
  if (file.type.startsWith("image/") || IMAGE_EXT.has(ext)) return "image";
  if (file.type.startsWith("audio/") || AUDIO_EXT.has(ext)) return "audio";
  if (file.type.startsWith("video/") || VIDEO_EXT.has(ext)) return "video";
  if (DOC_EXT.has(ext) || file.type.startsWith("text/")) return "document";
  return null;
}

function addFiles(files) {
  for (const file of files) {
    const kind = kindOf(file);
    if (!kind) { toast(`${file.name}: this file type is not supported`, "error"); continue; }
    if (kind === "image" && !vision) {
      toast(`${file.name}: ${modelName || "the loaded model"} cannot see images. Load a vision model in the Model panel.`, "error", 6000);
      continue;
    }
    if (file.size > MAX_MB[kind] * 1024 * 1024) {
      toast(`${file.name} is too large (${fmtBytes(file.size)}, limit ${MAX_MB[kind]} MB)`, "error");
      continue;
    }
    const item = { id: Math.random().toString(36).slice(2, 10), kind, name: file.name, size: file.size, status: "busy",
                   note: kind === "audio" || kind === "video" ? "transcribing…" : "reading…" };
    items.push(item);
    render();
    process(item, file).then((record) => {
      if (!items.includes(item)) return;            // removed meanwhile
      item.record = { id: item.id, kind, name: file.name, ...record };
      item.status = "ready";
      item.note = describe(item.record);
      hooks.send({ type: "attach_add", attachment: item.record });
      render();
    }).catch((e) => {
      item.status = "error";
      item.note = e.message || String(e);
      render();
      toast(`${file.name}: ${item.note}`, "error", 6000);
    });
  }
}

function describe(r) {
  if (r.kind === "document") return `${(r.text || "").length.toLocaleString()} chars${r.truncated ? " (truncated)" : ""}`;
  if (r.kind === "audio") return `${fmtTime(r.duration)} · ${r.text ? r.text.split(/\s+/).length + " words" : "no speech"}`;
  if (r.kind === "video") return `${fmtTime(r.duration)} · ${r.frames ? r.frames.length + " frames" : "no frames"} · ` +
    `${r.text ? r.text.split(/\s+/).length + " words" : "no speech"}`;
  return "";
}

const fmtTime = (s) => `${Math.floor((s || 0) / 60)}:${String(Math.round((s || 0) % 60)).padStart(2, "0")}`;

function remove(id) {
  const item = items.find((i) => i.id === id);
  items = items.filter((i) => i.id !== id);
  if (item && item.status === "ready") hooks.send({ type: "attach_remove", id });
  render();
}

function render() {
  const tray = $("attachTray");
  tray.hidden = !items.length;
  tray.innerHTML = items.map((i) => `
    <div class="att ${i.status}" title="${esc(i.name)} · ${esc(fmtBytes(i.size))}">
      ${i.kind === "image" && i.record ? `<img src="${i.record.data}" alt="">` : icon(KIND_ICON[i.kind])}
      <span class="att-text"><span class="att-name">${esc(i.name)}</span><span class="att-note">${esc(i.note || "")}</span></span>
      <button type="button" class="icon-btn small" data-remove="${i.id}" title="Remove">${icon("x")}</button>
    </div>`).join("");
}

// ------------------------------------------------------------ processing

async function process(item, file) {
  if (item.kind === "image") return { data: await imageDataURL(file, IMAGE_MAX_SIDE) };
  if (item.kind === "document") {
    const r = await fetch(`/api/extract?name=${encodeURIComponent(file.name)}`, { method: "POST", body: file });
    const d = await r.json();
    if (!d.ok) throw new Error(d.error);
    if (!d.text.trim()) throw new Error("the document is empty");
    return { text: d.text, truncated: d.truncated };
  }
  if (item.kind === "audio") {
    const { text, duration } = await transcribe(file);
    return { text, duration };
  }
  // video: transcript of the audio track + frames for vision models
  let text = "", duration = 0, frames = null;
  try {
    ({ text, duration } = await transcribe(file));
  } catch (e) {
    item.note = "no audio track";
  }
  if (vision) {
    item.note = "extracting frames…";
    render();
    const v = await videoFrames(file);
    frames = v.frames;
    duration = duration || v.duration;
  }
  if (!text && !frames) throw new Error("no audio track, and the model cannot see images");
  return { text, duration, frames };
}

function imageDataURL(file, maxSide) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, maxSide / Math.max(img.naturalWidth, img.naturalHeight));
      const c = document.createElement("canvas");
      c.width = Math.max(1, Math.round(img.naturalWidth * scale));
      c.height = Math.max(1, Math.round(img.naturalHeight * scale));
      const g = c.getContext("2d");
      g.fillStyle = "#fff";
      g.fillRect(0, 0, c.width, c.height);
      g.drawImage(img, 0, 0, c.width, c.height);
      URL.revokeObjectURL(url);
      resolve(c.toDataURL("image/jpeg", 0.88));
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("the image could not be decoded")); };
    img.src = url;
  });
}

async function transcribe(file) {
  const buf = await file.arrayBuffer();
  let audio;
  try {
    // An OfflineAudioContext decodes straight to its own rate: 16 kHz for the STT.
    audio = await new OfflineAudioContext(1, 1, 16000).decodeAudioData(buf);
  } catch (e) {
    throw new Error("the browser cannot decode this audio format");
  }
  const n = audio.length;
  const mono = new Float32Array(n);
  for (let c = 0; c < audio.numberOfChannels; c++) {
    const ch = audio.getChannelData(c);
    for (let i = 0; i < n; i++) mono[i] += ch[i] / audio.numberOfChannels;
  }
  const pcm = new Int16Array(n);
  for (let i = 0; i < n; i++) pcm[i] = Math.max(-32768, Math.min(32767, Math.round(mono[i] * 32767)));
  const r = await fetch("/api/transcribe", { method: "POST", body: pcm.buffer });
  const d = await r.json();
  if (!d.ok) throw new Error(d.error);
  return { text: d.text, duration: d.duration };
}

function once(el, ev) {
  return new Promise((resolve, reject) => {
    const ok = () => { cleanup(); resolve(); };
    const bad = () => { cleanup(); reject(new Error("the browser cannot play this video format")); };
    const cleanup = () => { el.removeEventListener(ev, ok); el.removeEventListener("error", bad); };
    el.addEventListener(ev, ok);
    el.addEventListener("error", bad);
  });
}

async function videoFrames(file) {
  const url = URL.createObjectURL(file);
  const v = document.createElement("video");
  v.muted = true;
  v.preload = "auto";
  v.src = url;
  try {
    await once(v, "loadedmetadata");
    const duration = v.duration && isFinite(v.duration) ? v.duration : 0;
    const count = Math.max(1, Math.min(8, Math.ceil(duration / 8)));
    const scale = Math.min(1, FRAME_MAX_SIDE / Math.max(v.videoWidth, v.videoHeight));
    const c = document.createElement("canvas");
    c.width = Math.round(v.videoWidth * scale);
    c.height = Math.round(v.videoHeight * scale);
    const g = c.getContext("2d");
    const frames = [];
    for (let i = 0; i < count; i++) {
      v.currentTime = duration * (i + 0.5) / count;
      await once(v, "seeked");
      g.drawImage(v, 0, 0, c.width, c.height);
      frames.push(c.toDataURL("image/jpeg", 0.8));
    }
    return { frames, duration };
  } finally {
    URL.revokeObjectURL(url);
  }
}
