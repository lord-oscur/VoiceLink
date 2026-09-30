// Settings store: named profiles in localStorage, with export/import to JSON files.
//
// localStorage["voicelink.settings"] = {
//   version, active: "<profile name>", profiles: { name: {...settings} }, ui: {...}
// }
// Settings are deep-merged with DEFAULTS on read, so new keys appear automatically.

const KEY = "voicelink.settings";
const VERSION = 3;

export const DEFAULTS = {
  voice: "",                 // "" = automatic voice for the reply's language
  gender: "all",             // voice list filter: all | female | male
  speed: 1.05,
  headphones: false,
  devices: { mic: "", speaker: "", showVirtual: false },
  fx: { pitch: 0, timbre: 0, range: 100, volume: 0, age: "adult", emotion: "none", intensity: 60 },
  input: {
    noise: "light", playbackGate: true, loopback: "",
    onlyMe: false, onlyMeStrict: 50, voiceprint: null, voiceprintDate: "",
  },
  model: {
    systemPrompt: "", temperature: null, top_p: null, top_k: null, repeat_penalty: null,
    max_tokens: null, thinking: false, history: null,
  },
  editor: { check: true, lang: "auto" },
  languages: { second: "", source: "", manual: false },
};

const UI_DEFAULTS = { left: true, right: false, voiceTab: "voice" };

const isObj = (v) => v && typeof v === "object" && !Array.isArray(v);
const clone = (v) => JSON.parse(JSON.stringify(v));

function merge(base, over) {
  const out = clone(base);
  if (!isObj(over)) return out;
  for (const [k, v] of Object.entries(over)) {
    out[k] = isObj(v) && isObj(out[k]) ? merge(out[k], v) : v;
  }
  return out;
}

function getPath(obj, path) {
  return path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj);
}

function setPath(obj, path, value) {
  const keys = path.split(".");
  let o = obj;
  for (const k of keys.slice(0, -1)) o = isObj(o[k]) ? o[k] : (o[k] = {});
  o[keys[keys.length - 1]] = value;
}

function load() {
  let data = null;
  try { data = JSON.parse(localStorage.getItem(KEY)); } catch (_) {}
  if (!isObj(data) || !isObj(data.profiles)) {
    // v1 stored {headphones, voice, speed} under "voicelink".
    let old = {};
    try { old = JSON.parse(localStorage.getItem("voicelink")) || {}; } catch (_) {}
    const first = {};
    if (old.voice) first.voice = old.voice;
    if (old.speed) first.speed = parseFloat(old.speed);
    if (old.headphones) first.headphones = true;
    data = { version: VERSION, active: "Default", profiles: { Default: first }, ui: {} };
  }
  if (!data.profiles[data.active]) data.active = Object.keys(data.profiles)[0] || "Default";
  if (!data.profiles[data.active]) data.profiles[data.active] = {};
  data.ui = merge(UI_DEFAULTS, data.ui);
  if ((data.version || 0) < 3) {
    for (const p of Object.values(data.profiles)) {
      if (isObj(p) && isObj(p.input)) p.input.playbackGate = true;
    }
    data.version = VERSION;
  }
  return data;
}

class Store {
  constructor() {
    this.data = load();
    this.current = merge(DEFAULTS, this.data.profiles[this.data.active]);
    this.listeners = new Set();
    this.save();
  }

  get(path) { return getPath(this.current, path); }
  ui(key) { return this.data.ui[key]; }
  setUI(key, value) { this.data.ui[key] = value; this.save(); }

  set(path, value) {
    if (JSON.stringify(this.get(path)) === JSON.stringify(value)) return;
    setPath(this.current, path, value);
    this.data.profiles[this.data.active] = clone(this.current);
    this.save();
    this.emit([path]);
  }

  onChange(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  emit(paths) { for (const fn of this.listeners) fn(paths, this); }
  save() { localStorage.setItem(KEY, JSON.stringify(this.data)); }

  // ------------------------------------------------------------ profiles

  get active() { return this.data.active; }
  get profileNames() { return Object.keys(this.data.profiles); }

  switchTo(name) {
    if (!this.data.profiles[name]) return;
    this.data.active = name;
    this.current = merge(DEFAULTS, this.data.profiles[name]);
    this.save();
    this.emit(["*"]);
  }

  saveAs(name) {
    name = (name || "").trim();
    if (!name) return false;
    this.data.profiles[name] = clone(this.current);
    this.switchTo(name);
    return true;
  }

  rename(name) {
    name = (name || "").trim();
    if (!name || this.data.profiles[name]) return false;
    const profiles = {};
    for (const [k, v] of Object.entries(this.data.profiles)) profiles[k === this.active ? name : k] = v;
    this.data.profiles = profiles;
    this.data.active = name;
    this.save();
    this.emit(["profile"]);
    return true;
  }

  remove() {
    if (this.profileNames.length <= 1) return false;
    delete this.data.profiles[this.active];
    this.switchTo(this.profileNames[0]);
    return true;
  }

  resetProfile() {
    this.current = clone(DEFAULTS);
    this.data.profiles[this.active] = clone(this.current);
    this.save();
    this.emit(["*"]);
  }

  // ------------------------------------------------------ export / import

  exportJSON() {
    return JSON.stringify({
      app: "VoiceLink", version: VERSION, exported: new Date().toISOString(),
      active: this.active, profiles: this.data.profiles,
    }, null, 2);
  }

  // Adds (or replaces) the file's profiles. Returns the names imported.
  importJSON(text) {
    const data = JSON.parse(text);
    let profiles = data.profiles;
    if (!isObj(profiles)) {
      if (!isObj(data) || !Object.keys(data).some((k) => k in DEFAULTS)) {
        throw new Error("not a VoiceLink settings file");
      }
      profiles = { Imported: data };   // a bare settings object
    }
    const names = [];
    for (const [name, p] of Object.entries(profiles)) {
      if (!isObj(p)) continue;
      this.data.profiles[name] = merge(DEFAULTS, p);
      names.push(name);
    }
    if (!names.length) throw new Error("the file contains no profiles");
    this.switchTo(profiles[data.active] ? data.active : names[0]);
    return names;
  }
}

export const settings = new Store();
