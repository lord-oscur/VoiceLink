// Voice and Devices pages in the right panel: voice, effects, mic/speaker filters.
import { AGE_PRESETS, EMO_PRESETS, mergePreset, presetRows, readKey, writeKey } from "./fxpresets.js";
import { languageHint, NAMES } from "./locale.js";
import { canPickSpeaker, fillSelect, listDevices } from "./devices.js";
import { settings } from "./settings.js";
import { $, toast } from "./ui.js";

const GROUPS = [
  ["a", "English (US)"], ["b", "English (UK)"], ["e", "Spanish"],
  ["f", "French"], ["i", "Italian"], ["p", "Portuguese (BR)"], ["h", "Hindi"], ["j", "Japanese"], ["z", "Chinese"],
];
const GROUP_LANG = { a: "en", b: "en", e: "es", f: "fr", i: "it", p: "pt", h: "hi", j: "ja", z: "zh" };
const GENDERS = [["female", "Female"], ["male", "Male"]];

function mainGroups() {
  const second = settings.get("languages.second");
  const set = new Set(["a", "b"]);
  for (const [g, lang] of Object.entries(GROUP_LANG)) {
    if (lang === second) set.add(g);
  }
  return set;
}

// [settings path, label, min, max, step, default, format, left hint, right hint]
const FX_SLIDERS = [
  ["speed", "Rate", 0.6, 1.6, 0.05, 1.05, (v) => (+v).toFixed(2) + "×", "slower", "faster"],
  ["fx.pitch", "Pitch", -12, 12, 0.5, 0, (v) => (v > 0 ? "+" : "") + (+v) + " st", "lower", "higher"],
  ["fx.timbre", "Timbre", -100, 100, 5, 0, (v) => (v > 0 ? "+" : "") + v, "deeper, larger", "brighter, smaller"],
  ["fx.range", "Pitch range", 0, 200, 5, 100, (v) => v + " %", "monotone", "expressive"],
  ["fx.volume", "Volume", -18, 12, 1, 0, (v) => (v > 0 ? "+" : "") + v + " dB", "quieter", "louder"],
];
const EMO_SLIDERS = [
  ["fx.intensity", "Intensity", 0, 100, 5, 60, (v) => v + " %", "subtle", "strong"],
];
const ONLY_ME_SLIDER = [
  ["input.onlyMeStrict", "Strictness", 0, 100, 5, 50, (v) => v + " %", "accept more", "reject more"],
];
const ENROLL_TEXT = {
  en: "I am recording my voice so the assistant can recognise me. When I speak, it should answer; " +
    "when a video or another person talks, it should stay quiet. Today is a good day to try something new, " +
    "and tomorrow I will tell you how it went. One, two, three, four, five.",
  es: "Estoy grabando mi voz para que el asistente me reconozca. Cuando yo hable, debe responder; " +
    "cuando hable un vídeo u otra persona, debe quedarse en silencio. Hoy es un buen día para probar algo nuevo, " +
    "y mañana te contaré qué tal fue. Uno, dos, tres, cuatro, cinco.",
};
const AGES = [
  ["child", "Child", "2–5"], ["preteen", "Preteen", "8–12"], ["teen", "Teen", "13–19"],
  ["adult", "Adult", "20–50 · original"], ["mature", "Mature", "50–70"], ["senior", "Senior", "70+"],
];
const EMOTIONS = [
  ["none", "Normal"], ["happy", "Happy"], ["excited", "Excited"], ["calm", "Calm"],
  ["sad", "Sad"], ["angry", "Angry"], ["fearful", "Fearful"], ["whisper", "Whisper"],
];
const EMO_TOUR = [
  ["none", "Normal", "This is my normal speaking voice, with no extra feeling."],
  ["happy", "Happy", "I just heard the good news, and I cannot stop smiling."],
  ["excited", "Excited", "We did it! I can hardly believe this is really happening!"],
  ["calm", "Calm", "Everything is all right. There is no need to hurry."],
  ["sad", "Sad", "I did not think it would end this way. I am sorry."],
  ["angry", "Angry", "That is enough! I will not be spoken to like that!"],
  ["fearful", "Fearful", "Something is wrong. I think we should get out of here."],
  ["whisper", "Whisper", "Come closer. I can only say this very quietly."],
];
const EMO_ASK = "Read this short scene aloud. Wrap every line of dialogue in a mood tag, "
  + "and use each of these tags exactly once: <happy> <excited> <calm> <sad> <angry> <fearful> <whisper>. "
  + "Do not tag the narrator. Close every tag.\n\n"
  + "The friends were in the kitchen.\n"
  + "Maya laughed as she opened the letter.\n"
  + "Leo could not stay in his chair.\n"
  + "Jun poured tea and spoke slowly.\n"
  + "Rita sat down and looked at the floor.\n"
  + "Omar slammed his cup on the table.\n"
  + "Nia glanced at the dark window.\n"
  + "Sam leaned in so the others would not hear.";

let catalog = [];
let hooks = {};

export function initVoicePanel(h) {
  hooks = h;

  for (const b of document.querySelectorAll("#genderSeg button")) {
    b.onclick = () => settings.set("gender", b.dataset.g);
  }
  $("voice").onchange = () => {
    settings.set("voice", $("voice").value);
    previewPick($("previewBtn"));
  };
  $("outLang").onchange = () => settings.set("languages.output", $("outLang").value);
  $("headphones").onchange = () => settings.set("headphones", $("headphones").checked);
  $("previewBtn").onclick = () => preview($("previewBtn"));
  $("emoTour").onclick = () => emotionTour();
  $("emoAsk").onclick = () => askEveryMood();
  $("fxReset").onclick = () => {
    lastAge = lastEmo = null;
    settings.set("fx", { pitch: 0, timbre: 0, range: 100, volume: 0, age: "adult", emotion: "none", intensity: 60,
                         ageCustom: {}, emoCustom: {} });
    settings.set("speed", 1.05);
  };
  buildSliders($("fxSliders"), FX_SLIDERS);
  buildSliders($("emoSliders"), EMO_SLIDERS);
  buildChips($("ageChips"), AGES, "fx.age");
  buildChips($("emoChips"), EMOTIONS, "fx.emotion");

  $("micSel").onchange = () => settings.set("devices.mic", $("micSel").value);
  $("spkSel").onchange = () => settings.set("devices.speaker", $("spkSel").value);
  $("showVirtual").onchange = () => settings.set("devices.showVirtual", $("showVirtual").checked);
  if (!canPickSpeaker) {
    $("spkSel").disabled = true;
    $("spkSel").title = "This browser cannot choose the output device (use Chrome or Edge).";
  }
  navigator.mediaDevices?.addEventListener?.("devicechange", refreshDevices);
  initFilters();

  settings.onChange((paths) => {
    const all = paths.includes("*");
    if (all || paths.some((p) => ["voice", "gender", "headphones"].includes(p) || p.startsWith("languages"))) renderVoice();
    if (all || paths.some((p) => p === "speed" || p.startsWith("fx"))) renderFx();
    if (all || paths.some((p) => p.startsWith("devices"))) refreshDevices();
    if (all || paths.some((p) => p.startsWith("input") || p === "headphones")) renderFilters();
  });
  renderVoice();
  renderFx();
  renderFilters();
}

// ---------------------------------------------------------- microphone filters

let enrolling = false;
let lastMatch = null;
let loopDevices = null;

function initFilters() {
  for (const b of document.querySelectorAll("#noiseSeg button")) b.onclick = () => settings.set("input.noise", b.dataset.v);
  $("pbGate").onchange = () => settings.set("input.playbackGate", $("pbGate").checked);
  $("loopSel").onchange = () => settings.set("input.loopback", $("loopSel").value);
  $("onlyMe").onchange = () => {
    settings.set("input.onlyMe", $("onlyMe").checked);
    if ($("onlyMe").checked && !settings.get("input.voiceprint")) toast("Record your voice first (10 seconds of reading)");
  };
  buildSliders($("onlyMeSlider"), ONLY_ME_SLIDER);
  $("enrollBtn").onclick = startEnroll;
  $("enrollCancel").onclick = () => { hooks.send({ type: "enroll_cancel" }); endEnroll(); };
}

async function loadLoopDevices() {
  try {
    const d = await (await fetch("/api/loopback")).json();
    loopDevices = d;
  } catch (_) { loopDevices = { devices: [] }; }
  renderFilters();
}

function renderFilters() {
  const noise = settings.get("input.noise") || "light";
  for (const b of document.querySelectorAll("#noiseSeg button")) b.classList.toggle("active", b.dataset.v === noise);

  const gate = !!settings.get("input.playbackGate");
  $("pbGate").checked = gate;
  $("loopRow").classList.toggle("disabled", !gate);
  if (loopDevices === null) { loopDevices = { devices: [] }; loadLoopDevices(); }
  const sel = $("loopSel");
  const want = settings.get("input.loopback") || "";
  const def = loopDevices.devices.find((d) => d.default);
  sel.innerHTML = "";
  sel.append(new Option(`System default${def ? ` (${def.name})` : ""}`, ""));
  for (const d of loopDevices.devices) sel.append(new Option(d.name, d.name));
  if (want && !loopDevices.devices.some((d) => d.name === want)) sel.append(new Option(`${want} (not found)`, want));
  sel.value = want;
  $("pbHint").textContent = loopDevices.error ? `Cannot listen to the PC output: ${loopDevices.error}` :
    settings.get("headphones") ? "With headphones the microphone cannot hear the PC, so this filter is not needed." :
      "On by default. Compares what the microphone hears with what the PC is playing, so a video does not start a turn. " +
      "Choose the device your videos play on.";

  const only = !!settings.get("input.onlyMe");
  const vp = settings.get("input.voiceprint");
  $("onlyMe").checked = only;
  $("onlyMeBox").classList.toggle("disabled", !only && !enrolling);
  $("enrollBtn").querySelector("span").textContent = vp ? "Record my voice again" : "Record my voice (10 s)";
  $("enrollBtn").hidden = enrolling;
  const slider = $("onlyMeSlider").querySelector("input");
  slider.value = settings.get("input.onlyMeStrict") ?? 50;
  slider.parentElement.querySelector("output").textContent = slider._fmt(slider.value);
  let hint = loopDevices.speaker_model === false ? "The voice model is missing: run setup.bat. " : "";
  hint += vp ? `Voice recorded ${settings.get("input.voiceprintDate") || ""}. Other voices (videos, TV, people nearby, ` +
    "and the assistant's own voice) are ignored." : "No voice recorded yet.";
  if (lastMatch) {
    hint += ` Last match: <b class="${lastMatch.accepted ? "match-ok" : "match-no"}">${Math.round(lastMatch.score * 100)} %` +
      ` ${lastMatch.accepted ? "accepted" : "ignored"}</b> (needs ${Math.round(lastMatch.threshold * 100)} %).`;
  }
  $("onlyMeHint").innerHTML = hint;
}

async function startEnroll() {
  if (hooks.ensureStarted && !(await hooks.ensureStarted())) return;
  enrolling = true;
  const lang = (hooks.lang && hooks.lang()) === "es" ? "es" : "en";
  $("enrollText").textContent = ENROLL_TEXT[lang];
  $("enrollBar").style.width = "0%";
  $("enrollRun").hidden = false;
  renderFilters();
  hooks.send({ type: "enroll_start" });
}

function endEnroll() {
  enrolling = false;
  $("enrollRun").hidden = true;
  renderFilters();
}

// Server messages for the filters; returns true when handled.
export function onFilterMessage(msg) {
  switch (msg.type) {
    case "enroll_progress":
      $("enrollBar").style.width = msg.pct + "%";
      return true;
    case "enrolled":
      settings.set("input.voiceprint", msg.voiceprint);
      settings.set("input.voiceprintDate", new Date().toLocaleDateString());
      if (!settings.get("input.onlyMe")) settings.set("input.onlyMe", true);
      endEnroll();
      toast("Voice recorded - only your voice will be answered now");
      return true;
    case "enroll_error":
      endEnroll();
      toast("Voice recording failed: " + msg.message, "error");
      return true;
    case "speaker":
      lastMatch = msg;
      renderFilters();
      return true;
  }
  return false;
}

function buildSliders(box, defs) {
  for (const [path, label, min, max, step, def, fmt, lo, hi] of defs) {
    const wrap = document.createElement("label");
    wrap.className = "slider";
    wrap.innerHTML = `<span class="label">${label}<output></output></span>` +
      `<input type="range" min="${min}" max="${max}" step="${step}">` +
      `<span class="ends"><span>${lo}</span><span>${hi}</span></span>`;
    const input = wrap.querySelector("input");
    const out = wrap.querySelector("output");
    input.dataset.path = path;
    input._fmt = fmt;
    input.oninput = () => { out.textContent = fmt(input.value); };
    input.onchange = () => settings.set(path, parseFloat(input.value));
    input.ondblclick = () => settings.set(path, def);
    box.append(wrap);
  }
}

function buildChips(box, defs, path) {
  for (const [key, label, sub] of defs) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.dataset.key = key;
    b.innerHTML = sub ? `${label}<small>${sub}</small>` : label;
    b.onclick = () => {
      settings.set(path, key);
      previewPick($("previewBtn"));
    };
    box.append(b);
  }
  box.dataset.path = path;
}

function renderFx() {
  for (const input of document.querySelectorAll("#pane-voice .slider input:not([data-preset]), #pane-io .slider input:not([data-preset])")) {
    input.value = settings.get(input.dataset.path);
    input.parentElement.querySelector("output").textContent = input._fmt(input.value);
  }
  for (const box of [$("ageChips"), $("emoChips")]) {
    const value = settings.get(box.dataset.path);
    for (const b of box.children) {
      b.classList.toggle("active", b.dataset.key === value);
      if (box.id === "emoChips") b.classList.toggle("playing", tourMood === b.dataset.key);
    }
  }
  const emo = settings.get("fx.emotion");
  $("emoSliders").classList.toggle("disabled", !emo || emo === "none");
  renderPresetEditors();
}

let lastAge = null;
let lastEmo = null;

function customBag(kind) {
  return settings.get(kind === "age" ? "fx.ageCustom" : "fx.emoCustom") || {};
}

function livePreset(kind, name) {
  const base = kind === "age" ? AGE_PRESETS[name] : EMO_PRESETS[name];
  return mergePreset(base, customBag(kind)[name]);
}

function renderPresetEditors() {
  const age = settings.get("fx.age") || "adult";
  const emo = settings.get("fx.emotion") || "none";
  fillPresetBox($("ageDetail"), "age", age, age !== lastAge);
  fillPresetBox($("emoDetail"), "emo", emo, emo !== lastEmo);
  lastAge = age;
  lastEmo = emo;
}

function fillPresetBox(box, kind, name, rebuild) {
  const rows = presetRows(livePreset(kind, name));
  const labels = { age: "Age", emo: "Emotion" };
  const empty = { adult: "Adult uses the original voice — no extra levels.",
                  none: "No emotion overlay." };
  if (!rows.length) {
    box.hidden = !empty[name];
    box.innerHTML = empty[name] ? `<p class="hint">${empty[name]}</p>` : "";
    return;
  }
  box.hidden = false;
  if (rebuild || !box.querySelector("[data-preset]")) {
    box.innerHTML = `<div class="preset-head"><span>${labels[kind]} levels · <strong>${name}</strong></span>` +
      `<button type="button" class="link-btn" data-reset hidden>Reset</button></div>` +
      `<div class="sliders"></div>`;
    const grid = box.querySelector(".sliders");
    const reset = box.querySelector("[data-reset]");
    reset.onclick = () => {
      const bag = { ...customBag(kind) };
      delete bag[name];
      lastAge = lastEmo = null;
      settings.set(kind === "age" ? "fx.ageCustom" : "fx.emoCustom", bag);
      previewPick($("previewBtn"));
    };
    for (const row of rows) {
      const wrap = document.createElement("label");
      wrap.className = "slider";
      wrap.innerHTML = `<span class="label">${row.label}<output></output></span>` +
        `<input type="range" min="${row.min}" max="${row.max}" step="${row.step}" data-preset="${kind}">`;
      const input = wrap.querySelector("input");
      const out = wrap.querySelector("output");
      input._fmt = row.fmt;
      input._key = row.key;
      input.value = row.value;
      out.textContent = row.fmt(row.value);
      input.oninput = () => { out.textContent = row.fmt(input.value); };
      input.onchange = () => {
        const next = writeKey(livePreset(kind, name), input._key, parseFloat(input.value));
        const bag = { ...customBag(kind), [name]: next };
        settings.set(kind === "age" ? "fx.ageCustom" : "fx.emoCustom", bag);
        previewPick($("previewBtn"));
      };
      grid.append(wrap);
    }
    reset.hidden = !customBag(kind)[name];
    return;
  }
  const reset = box.querySelector("[data-reset]");
  if (reset) reset.hidden = !customBag(kind)[name];
  for (const input of box.querySelectorAll("input[data-preset]")) {
    const v = readKey(livePreset(kind, name), input._key);
    if (v == null) continue;
    input.value = v;
    input.parentElement.querySelector("output").textContent = input._fmt(v);
  }
}

export function setCatalog(data) {
  catalog = data.voices || [];
  renderVoice();
}

function renderVoice() {
  const gender = settings.get("gender");
  const main = mainGroups();
  const langEl = $("langHint");
  if (langEl) langEl.textContent = languageHint(settings);
  for (const b of document.querySelectorAll("#genderSeg button")) b.classList.toggle("active", b.dataset.g === gender);
  const sel = $("voice");
  const current = settings.get("voice");
  sel.innerHTML = "";
  sel.append(new Option(gender === "male" ? "Auto: male voice per language" :
                        gender === "female" ? "Auto: female voice per language" : "Auto: best voice per language", ""));
  let otherShown = false;
  for (const [g, gname] of GROUPS) {
    for (const [gen, genName] of GENDERS) {
      if (gender !== "all" && gender !== gen) continue;
      const list = catalog.filter((v) => v.group === g && v.gender === gen)
        .sort((x, y) => (x.age ?? 99) - (y.age ?? 99) || x.name.localeCompare(y.name));
      if (!list.length) continue;
      if (!main.has(g) && !otherShown) {
        otherShown = true;
        const sep = document.createElement("optgroup");
        sep.label = "──────── Other languages ────────";
        sel.append(sep);
      }
      const og = document.createElement("optgroup");
      og.label = `${gname} · ${genName}`;
      for (const v of list) {
        const opt = new Option(`${v.name}${v.age ? "   ~" + v.age : ""}${v.default ? "   ★" : ""}`, v.id);
        opt.title = `${v.id}${v.f0 ? ` · median pitch ${Math.round(v.f0)} Hz` : ""}${v.age ? ` · sounds about ${v.age}` : ""}`;
        og.append(opt);
      }
      sel.append(og);
    }
  }
  // Keep an explicitly chosen voice even if the filter hides its gender.
  if (current && !sel.querySelector(`option[value="${CSS.escape(current)}"]`)) {
    const v = catalog.find((x) => x.id === current);
    if (v) sel.insertBefore(new Option(`${v.name} (current)`, v.id), sel.options[1] || null);
  }
  sel.value = current || "";
  $("headphones").checked = !!settings.get("headphones");
  fillOutLang();
}

function fillOutLang() {
  const sel = $("outLang");
  if (!sel) return;
  const keep = settings.get("languages.output") || "voice";
  const voice = settings.get("voice") || "";
  const voiceLang = voice ? (GROUP_LANG[voice[0]] || "en") : "";
  const voiceName = voiceLang && NAMES[voiceLang] ? NAMES[voiceLang] : "";
  sel.innerHTML = "";
  sel.append(new Option(
    voice && voiceName ? `Stay in the voice's language (${voiceName})` : "Stay in the voice's language",
    "voice"));
  sel.append(new Option("Follow what I speak", "auto"));
  for (const [code, name] of Object.entries(NAMES)) sel.append(new Option(name, code));
  sel.value = keep;
}

let tourToken = 0;
let tourMood = null;
let touring = false;

function setTourLabel(running) {
  const btn = $("emoTour");
  const label = btn.querySelector("span");
  if (label) label.textContent = running ? "Stop tour" : "Play all emotions";
}

function stopTour() {
  tourToken += 1;
  touring = false;
  tourMood = null;
  setTourLabel(false);
  $("emoTourHint").textContent = "Hear every filter in order, using this voice, age and intensity. Ask the model sends a scene that should use each mood tag once.";
  if (hooks.stopPreview) hooks.stopPreview();
  renderFx();
}

async function emotionTour() {
  if (touring) { stopTour(); return; }
  const my = ++tourToken;
  touring = true;
  setTourLabel(true);
  const voice = settings.get("voice");
  const v = catalog.find((x) => x.id === voice);
  const gender = settings.get("gender");
  const baseFx = { ...settings.get("fx") };
  try {
    for (const [key, label, line] of EMO_TOUR) {
      if (my !== tourToken) return;
      tourMood = key;
      $("emoTourHint").textContent = `Playing ${label}…`;
      renderFx();
      const r = await fetch("/api/preview", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          voice: voice || null, lang: v ? v.lang : (hooks.lang ? hooks.lang() : "en"),
          speed: settings.get("speed"), gender: gender === "all" ? null : gender,
          fx: { ...baseFx, emotion: key },
          text: `${label}. ${line}`,
        }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      if (my !== tourToken) return;
      await hooks.playPreview(await r.arrayBuffer());
    }
    if (my === tourToken) $("emoTourHint").textContent = "Tour finished. Click a chip to keep one emotion, or Reset.";
  } catch (e) {
    if (my === tourToken) toast("Emotion tour failed: " + e.message, "error");
  } finally {
    if (my === tourToken) {
      touring = false;
      tourMood = null;
      setTourLabel(false);
      renderFx();
    }
  }
}

function askEveryMood() {
  if (!hooks.sendUser) return;
  hooks.sendUser(EMO_ASK);
  toast("Sent a scene that should use every mood tag");
}

let previewToken = 0;

function previewPick(btn) {
  if (touring) return;
  preview(btn);
}

async function preview(btn) {
  const my = ++previewToken;
  if (hooks.stopPreview) hooks.stopPreview();
  if (btn) btn.classList.add("busy");
  try {
    const voice = settings.get("voice");
    const v = catalog.find((x) => x.id === voice);
    const gender = settings.get("gender");
    const r = await fetch("/api/preview", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        voice: voice || null, lang: v ? v.lang : (hooks.lang ? hooks.lang() : "en"),
        speed: settings.get("speed"), gender: gender === "all" ? null : gender, fx: settings.get("fx"),
      }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    if (my !== previewToken) return;
    await hooks.playPreview(await r.arrayBuffer());
  } catch (e) {
    if (my === previewToken) toast("Preview failed: " + e.message, "error");
  } finally {
    if (my === previewToken && btn) btn.classList.remove("busy");
  }
}

let refreshing = null;
export async function refreshDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  if (refreshing) return refreshing;
  refreshing = (async () => {
    const want = { mic: settings.get("devices.mic"), speaker: settings.get("devices.speaker") };
    const lists = await listDevices(settings.get("devices.showVirtual"), want);
    fillSelect($("micSel"), lists.mic, want.mic);
    fillSelect($("spkSel"), lists.speaker, want.speaker);
    $("showVirtual").checked = !!settings.get("devices.showVirtual");
    $("devHint").textContent = lists.labelled ? "" :
      "Device names appear after the microphone is allowed (click the orb).";
  })().finally(() => { refreshing = null; });
  return refreshing;
}
