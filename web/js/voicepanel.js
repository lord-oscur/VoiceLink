// Voice & audio hover panel: voice choice, speed, devices (and effects).
import { canPickSpeaker, fillSelect, listDevices } from "./devices.js";
import { settings } from "./settings.js";
import { $, toast } from "./ui.js";

const GROUPS = [
  ["a", "English (US)"], ["b", "English (UK)"], ["e", "Spanish"],
  ["f", "French"], ["i", "Italian"], ["p", "Portuguese (BR)"], ["h", "Hindi"], ["j", "Japanese"], ["z", "Chinese"],
];
const MAIN = new Set(["a", "b", "e"]);   // English and Spanish first
const GENDERS = [["female", "Female"], ["male", "Male"]];

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
const AGES = [
  ["child", "Child", "2–5"], ["preteen", "Preteen", "8–12"], ["teen", "Teen", "13–19"],
  ["adult", "Adult", "20–50 · original"], ["mature", "Mature", "50–70"], ["senior", "Senior", "70+"],
];
const EMOTIONS = [
  ["none", "Normal"], ["happy", "Happy"], ["excited", "Excited"], ["calm", "Calm"],
  ["sad", "Sad"], ["angry", "Angry"], ["fearful", "Fearful"], ["whisper", "Whisper"],
];

let catalog = [];
let hooks = {};

export function initVoicePanel(h) {
  hooks = h;
  // Tabs
  const tabs = document.querySelectorAll("#voicePanel .tabs button");
  const showTab = (name) => {
    for (const b of tabs) b.classList.toggle("active", b.dataset.tab === name);
    for (const p of document.querySelectorAll("#voicePanel [data-pane]")) p.hidden = p.dataset.pane !== name;
    settings.setUI("voiceTab", name);
    if (name === "io") refreshDevices();
  };
  for (const b of tabs) b.onclick = () => showTab(b.dataset.tab);
  showTab(settings.ui("voiceTab") || "voice");

  for (const b of document.querySelectorAll("#genderSeg button")) {
    b.onclick = () => settings.set("gender", b.dataset.g);
  }
  $("voice").onchange = () => settings.set("voice", $("voice").value);
  $("headphones").onchange = () => settings.set("headphones", $("headphones").checked);
  $("previewBtn").onclick = () => preview($("previewBtn"));
  $("fxPreview").onclick = () => preview($("fxPreview"));
  $("fxReset").onclick = () => {
    settings.set("fx", { pitch: 0, timbre: 0, range: 100, volume: 0, age: "adult", emotion: "none", intensity: 60 });
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

  settings.onChange((paths) => {
    const all = paths.includes("*");
    if (all || paths.some((p) => ["voice", "gender", "headphones"].includes(p))) renderVoice();
    if (all || paths.some((p) => p === "speed" || p.startsWith("fx"))) renderFx();
    if (all || paths.some((p) => p.startsWith("devices"))) refreshDevices();
  });
  renderVoice();
  renderFx();
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
    b.onclick = () => settings.set(path, key);
    box.append(b);
  }
  box.dataset.path = path;
}

function renderFx() {
  for (const input of document.querySelectorAll("#voicePanel .slider input")) {
    input.value = settings.get(input.dataset.path);
    input.parentElement.querySelector("output").textContent = input._fmt(input.value);
  }
  for (const box of [$("ageChips"), $("emoChips")]) {
    const value = settings.get(box.dataset.path);
    for (const b of box.children) b.classList.toggle("active", b.dataset.key === value);
  }
  const emo = settings.get("fx.emotion");
  $("emoSliders").classList.toggle("disabled", !emo || emo === "none");
}

export function setCatalog(data) {
  catalog = data.voices || [];
  renderVoice();
}

function renderVoice() {
  const gender = settings.get("gender");
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
      if (!MAIN.has(g) && !otherShown) {
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
}

async function preview(btn) {
  if (btn.classList.contains("busy")) return;
  btn.classList.add("busy");
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
    await hooks.playPreview(await r.arrayBuffer());
  } catch (e) {
    toast("Preview failed: " + e.message, "error");
  } finally {
    btn.classList.remove("busy");
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
