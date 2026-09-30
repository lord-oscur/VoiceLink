// Primary language is always English. A second language is taken from the
// OS locale, then the timezone (country hint), then the browser geolocation
// if permission was already granted. Nothing is fetched from the internet.

export const NAMES = {
  en: "English", es: "Spanish", fr: "French", it: "Italian", pt: "Portuguese",
  hi: "Hindi", ja: "Japanese", zh: "Chinese",
};

const TAG = [
  [/^es\b/, "es"], [/^pt\b/, "pt"], [/^fr\b/, "fr"], [/^it\b/, "it"],
  [/^hi\b/, "hi"], [/^ja\b/, "ja"], [/^zh\b/, "zh"], [/^ca\b/, "es"],
  [/^gl\b/, "es"], [/^eu\b/, "es"], [/^gn\b/, "es"],
];

// IANA zones that are a reliable country/language hint when the OS is English.
const TZ = {
  "America/Mexico_City": "es", "America/Merida": "es", "America/Monterrey": "es",
  "America/Tijuana": "es", "America/Chihuahua": "es", "America/Hermosillo": "es",
  "America/Mazatlan": "es", "America/Bahia_Banderas": "es", "America/Cancun": "es",
  "America/Bogota": "es", "America/Lima": "es", "America/Guayaquil": "es",
  "America/Caracas": "es", "America/La_Paz": "es", "America/Asuncion": "es",
  "America/Montevideo": "es", "America/Argentina/Buenos_Aires": "es",
  "America/Argentina/Cordoba": "es", "America/Santiago": "es",
  "America/Santo_Domingo": "es", "America/Havana": "es", "America/Puerto_Rico": "es",
  "America/Panama": "es", "America/Costa_Rica": "es", "America/Guatemala": "es",
  "America/El_Salvador": "es", "America/Tegucigalpa": "es", "America/Managua": "es",
  "Europe/Madrid": "es", "Atlantic/Canary": "es", "Africa/Ceuta": "es",
  "America/Sao_Paulo": "pt", "America/Fortaleza": "pt", "America/Recife": "pt",
  "America/Bahia": "pt", "America/Manaus": "pt", "America/Belem": "pt",
  "America/Maceio": "pt", "America/Noronha": "pt", "Europe/Lisbon": "pt",
  "Atlantic/Azores": "pt", "Atlantic/Madeira": "pt",
  "Europe/Paris": "fr", "Europe/Brussels": "fr", "Europe/Luxembourg": "fr",
  "Europe/Monaco": "fr", "Africa/Casablanca": "fr", "Indian/Antananarivo": "fr",
  "America/Martinique": "fr", "America/Guadeloupe": "fr",
  "Europe/Rome": "it", "Europe/Vatican": "it", "Europe/San_Marino": "it",
  "Asia/Tokyo": "ja", "Asia/Shanghai": "zh", "Asia/Hong_Kong": "zh",
  "Asia/Taipei": "zh", "Asia/Macau": "zh", "Asia/Singapore": "zh",
  "Asia/Kolkata": "hi", "Asia/Calcutta": "hi",
};

// Rough bounding boxes [south, west, north, east] for when geolocation is already allowed.
const BOXES = [
  ["es", [13.5, -118.5, 32.8, -86.5]],   // Mexico
  ["es", [35.1, -9.4, 43.9, 4.4]],         // Spain
  ["es", [-56.0, -75.0, 13.0, -34.0]],     // Spanish South / Central America (minus Brazil)
  ["pt", [-34.0, -74.1, 5.3, -34.7]],      // Brazil
  ["pt", [36.9, -9.6, 42.2, -6.1]],        // Portugal
  ["fr", [42.3, -5.2, 51.2, 8.3]],         // France
  ["it", [36.6, 6.6, 47.1, 18.6]],         // Italy
  ["ja", [24.2, 122.9, 45.6, 146.0]],      // Japan
  ["zh", [18.1, 73.5, 53.6, 135.1]],       // China
  ["hi", [6.7, 68.1, 35.7, 97.4]],         // India
];

function fromTags(tags) {
  for (const raw of tags) {
    if (!raw) continue;
    const t = String(raw).toLowerCase().replace("_", "-");
    if (t.startsWith("en")) continue;
    for (const [re, code] of TAG) {
      if (re.test(t)) return { second: code, source: "locale" };
    }
  }
  return null;
}

function fromTimezone() {
  let tz = "";
  try { tz = Intl.DateTimeFormat().resolvedOptions().timeZone || ""; } catch (_) {}
  if (TZ[tz]) return { second: TZ[tz], source: "timezone" };
  if (tz.startsWith("America/Argentina/")) return { second: "es", source: "timezone" };
  return null;
}

function fromCoords(lat, lon) {
  for (const [lang, [s, w, n, e]] of BOXES) {
    if (lat >= s && lat <= n && lon >= w && lon <= e) {
      if (lang === "pt" && lat >= 13.5 && lon <= -86.5) continue; // Mexico is not Brazil
      if (lang === "es" && lat <= 5.3 && lon >= -74.1 && lon <= -34.7 && lat >= -34) {
        // Brazil overlaps the loose South America box; skip ES there.
        continue;
      }
      return { second: lang, source: "geolocation" };
    }
  }
  return null;
}

function localeGuess() {
  const tags = [...(navigator.languages || []), navigator.language, navigator.userLanguage];
  return fromTags(tags) || fromTimezone();
}

function geoGuess() {
  if (!navigator.geolocation || !navigator.permissions) return Promise.resolve(null);
  return navigator.permissions.query({ name: "geolocation" }).then((p) => {
    if (p.state !== "granted") return null;
    return new Promise((resolve) => {
      navigator.geolocation.getCurrentPosition(
        (pos) => resolve(fromCoords(pos.coords.latitude, pos.coords.longitude)),
        () => resolve(null),
        { maximumAge: 86400000, timeout: 4000, enableHighAccuracy: false },
      );
    });
  }).catch(() => null);
}

/** ["en"] or ["en", second]. Always English first. */
export function languageList(settings) {
  const second = settings.get("languages.second");
  return second && second !== "en" && NAMES[second] ? ["en", second] : ["en"];
}

export function languageHint(settings) {
  const second = settings.get("languages.second");
  const src = settings.get("languages.source");
  if (!second || !NAMES[second]) return "Languages: English (default).";
  const how = src === "locale" ? "from this PC's language" :
    src === "timezone" ? "from this PC's time zone" :
      src === "geolocation" ? "from this PC's location" : "";
  return `Languages: English + ${NAMES[second]}${how ? " (" + how + ")" : ""}.`;
}

/** Detects and stores the second language. Resolves when the list is ready. */
export async function resolveLanguages(settings) {
  if (settings.get("languages.manual")) return languageList(settings);
  const local = localeGuess();
  if (local) {
    settings.set("languages.second", local.second);
    settings.set("languages.source", local.source);
    return languageList(settings);
  }
  const geo = await geoGuess();
  if (geo) {
    settings.set("languages.second", geo.second);
    settings.set("languages.source", geo.source);
  } else {
    settings.set("languages.second", "");
    settings.set("languages.source", "");
  }
  return languageList(settings);
}
