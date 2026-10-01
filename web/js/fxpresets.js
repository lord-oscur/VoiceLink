// Built-in age / emotion levels (keep in sync with voicelink/voicefx.py).
// The panel shows these when you pick a chip; edits are stored in fx.ageCustom / fx.emoCustom.

export const AGE_PRESETS = {
  none: {},
  child:   { target_f0: [300, 285], formant: 1.22, range: 1.30, rate: 0.92, breath: 0.02 },
  preteen: { target_f0: [250, 230], formant: 1.14, range: 1.20, rate: 0.97 },
  teen:    { target_f0: [215, 128], formant: 1.05, range: 1.10, rate: 1.03 },
  adult:   {},
  mature:  { shift: [-1.5, -0.5], formant: 0.98, range: 0.90, rate: 0.95, breath: 0.06,
             tremor: [0.10, 5.5], tremolo: 0.03 },
  senior:  { shift: [-2.5, 0.8], formant: 0.95, range: 0.80, rate: 0.86, breath: 0.16,
             tremor: [0.35, 5.5], tremolo: 0.08 },
};

export const EMO_PRESETS = {
  none:    {},
  happy:   { shift: 2.0, range: 1.35, rate: 1.06, volume: 1.5, formant: 1.03 },
  excited: { shift: 3.0, range: 1.60, rate: 1.12, volume: 3.0, formant: 1.04 },
  calm:    { shift: -1.0, range: 0.75, rate: 0.92, volume: -1.5, breath: 0.03 },
  sad:     { shift: -2.0, range: 0.60, rate: 0.86, volume: -3.0, formant: 0.98, breath: 0.07 },
  angry:   { shift: 1.0, range: 1.30, rate: 1.08, volume: 4.0, formant: 1.02, drive: 0.6 },
  fearful: { shift: 2.5, range: 1.20, rate: 1.10, breath: 0.05, tremor: [0.5, 7.0], tremolo: 0.06 },
  whisper: { volume: -6.0, rate: 0.70 },
};

const F = {
  rate:    { label: "Rate",        min: 0.55, max: 1.45, step: 0.01, fmt: (v) => (+v).toFixed(2) + "×" },
  formant: { label: "Timbre",      min: 0.85, max: 1.35, step: 0.01, fmt: (v) => (+v).toFixed(2) },
  range:   { label: "Pitch range", min: 0.50, max: 1.80, step: 0.05, fmt: (v) => Math.round(v * 100) + " %" },
  volume:  { label: "Volume",      min: -12,  max: 8,    step: 0.5,  fmt: (v) => (v > 0 ? "+" : "") + (+v) + " dB" },
  breath:  { label: "Breath",      min: 0,    max: 0.40, step: 0.01, fmt: (v) => Math.round(v * 100) + " %" },
  tremolo: { label: "Tremolo",     min: 0,    max: 0.20, step: 0.01, fmt: (v) => Math.round(v * 100) + " %" },
  drive:   { label: "Drive",       min: 0,    max: 1,    step: 0.05, fmt: (v) => (+v).toFixed(2) },
  shift:   { label: "Pitch",       min: -6,   max: 6,    step: 0.1,  fmt: (v) => (v > 0 ? "+" : "") + (+v).toFixed(1) + " st" },
};

export function presetRows(preset) {
  const rows = [];
  if (!preset) return rows;
  const add = (key, label, value, spec) => {
    rows.push({ key, label, value, min: spec.min, max: spec.max, step: spec.step, fmt: spec.fmt });
  };
  for (const [k, v] of Object.entries(preset)) {
    if (k === "target_f0" && Array.isArray(v)) {
      add("target_f0.0", "Pitch ♀", v[0], { min: 160, max: 360, step: 5, fmt: (x) => Math.round(x) + " Hz" });
      add("target_f0.1", "Pitch ♂", v[1], { min: 100, max: 340, step: 5, fmt: (x) => Math.round(x) + " Hz" });
    } else if (k === "shift" && Array.isArray(v)) {
      add("shift.0", "Pitch ♀", v[0], F.shift);
      add("shift.1", "Pitch ♂", v[1], F.shift);
    } else if (k === "tremor" && Array.isArray(v)) {
      add("tremor.0", "Tremor", v[0], { min: 0, max: 0.6, step: 0.01, fmt: (x) => (+x).toFixed(2) });
      add("tremor.1", "Tremor rate", v[1], { min: 3, max: 9, step: 0.1, fmt: (x) => (+x).toFixed(1) + " Hz" });
    } else if (F[k] && typeof v === "number") {
      add(k, F[k].label, v, F[k]);
    }
  }
  return rows;
}

export function readKey(obj, key) {
  if (key.includes(".")) {
    const [a, b] = key.split(".");
    const pair = obj[a];
    return Array.isArray(pair) ? pair[+b] : undefined;
  }
  return obj[key];
}

export function writeKey(obj, key, value) {
  const out = { ...obj };
  if (key.includes(".")) {
    const [a, b] = key.split(".");
    const pair = Array.isArray(out[a]) ? [...out[a]] : [0, 0];
    pair[+b] = value;
    out[a] = pair;
  } else {
    out[key] = value;
  }
  return out;
}

export function mergePreset(base, over) {
  const out = JSON.parse(JSON.stringify(base || {}));
  if (!over || typeof over !== "object") return out;
  for (const [k, v] of Object.entries(over)) {
    if (Array.isArray(v) && Array.isArray(out[k])) out[k] = v.slice(0, out[k].length);
    else if (typeof v === "number") out[k] = v;
  }
  return out;
}
