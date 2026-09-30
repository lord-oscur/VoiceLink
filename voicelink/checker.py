"""Offline spelling, grammar and redundancy checker for English and Spanish.

* Spelling: pyspellchecker frequency dictionaries (bundled with the package),
  topped up with models/dict/es_50k.txt for Spanish, plus an accent-folding
  index so "cancion" suggests "canción" instantly.
* Grammar: a small set of high-confidence rules (repeated words, a/an,
  capitals, spacing, common confusions such as "could of" or "a sido").
* Redundancy: pleonasms ("free gift", "subir arriba") and the same long word
  repeated close together.
Issue offsets are UTF-16 code units so the browser can use them directly.
"""
import functools
import os
import re
import threading
import unicodedata

from .config import MODELS_DIR

WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
SKIP_TOKEN = re.compile(r"(https?://|www\.)\S+|\S+@\S+\.\S+|[#@]\w+|`[^`]*`", re.UNICODE)

_lock = threading.Lock()
_spell = {}
_folded = {}


def _fold(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _checker(lang):
    with _lock:
        if lang not in _spell:
            from spellchecker import SpellChecker

            sp = SpellChecker(language=lang, distance=1)
            _add_frequency_list(sp, lang)
            _spell[lang] = sp
            index = {}
            for w in sp.word_frequency.words():
                f = _fold(w)
                if f != w:
                    index.setdefault(f, []).append(w)
            _folded[lang] = index
        return _spell[lang]


def warm():
    """Loads both dictionaries (about 1 s) so the first check is instant."""
    for lang in ("en", "es"):
        _checker(lang)


def _add_frequency_list(sp, lang):
    """Merges models/dict/<lang>_50k.txt ("word count" lines) into the dictionary.
    Subtitles often drop accents, so an unaccented form is skipped when an
    accented spelling of it is more frequent ("cancion" vs "canción")."""
    path = os.path.join(MODELS_DIR, "dict", f"{lang}_50k.txt")
    if not os.path.exists(path):
        return
    counts = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit() and WORD.fullmatch(parts[0]):
                counts[parts[0].lower()] = int(parts[1])
    best = {}
    for w, c in counts.items():
        f = _fold(w)
        if f != w and c > best.get(f, 0):
            best[f] = c
    scale = (sp.word_frequency.total_words or 1) / max(1, sum(counts.values()))  # comparable frequencies
    add = {}
    for w, c in counts.items():
        if w in sp or c < 30 or (len(w) == 1 and w not in "aeouy"):
            continue
        if _fold(w) == w and best.get(w, 0) > c:
            continue
        add[w] = max(1, int(c * scale))
    sp.word_frequency.load_json(add)


@functools.lru_cache(maxsize=4096)
def _suggest(lang, word):
    sp = _checker(lang)
    out = list(_folded[lang].get(_fold(word), []))
    with _lock:
        cands = set(sp.candidates(word) or ())
        if not cands - {word} and not out and len(word) <= 12:
            sp.distance = 2
            try:
                cands = set(sp.candidates(word) or ())
            finally:
                sp.distance = 1
    cands.discard(word)
    out += sorted(cands, key=lambda w: -sp.word_usage_frequency(w))
    seen, uniq = set(), []
    for w in out:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return tuple(uniq[:5])


def _match_case(src, dst):
    if src.isupper() and len(src) > 1:
        return dst.upper()
    if src[:1].isupper():
        return dst[:1].upper() + dst[1:]
    return dst


# ------------------------------------------------------------------ rules

AN_EXCEPT = {"hour", "hours", "honest", "honestly", "honor", "honour", "honorable", "heir", "herb"}
A_EXCEPT = ("uni", "use", "usu", "uti", "eu", "one", "once", "ubiq", "ure", "uro")

# (pattern, replacement, message) - replacement may use \1 etc.
RULES = {
    "en": [
        (r"\b(could|should|would|must|might) of\b", r"\1 have", "Use “have” after modal verbs"),
        (r"\balot\b", "a lot", "“a lot” is two words"),
        (r"\birregardless\b", "regardless", "Use “regardless”"),
        (r"\b(more|less|better|worse|rather|other|greater|higher|lower|bigger|smaller) then\b", r"\1 than",
         "Use “than” for comparisons"),
        (r"\byour welcome\b", "you're welcome", "Did you mean “you're welcome”?"),
        (r"\btheir (is|are|was|were)\b", r"there \1", "Did you mean “there”?"),
        (r"\bits (a|an|the|been|not|going|very|so|too)\b", r"it's \1", "Did you mean “it's” (it is)?"),
        (r"\bme and (him|her|them|you)\b(?= (?:are|were|went|have|will)\b)", r"\1 and I", "Use “… and I” as a subject"),
        (r"\bI is\b", "I am", "Use “I am”"),
        (r"\b(he|she|it) don't\b", r"\1 doesn't", "Use “doesn't” with he/she/it"),
        (r"\b(have|has|had) went\b", r"\1 gone", "Use the participle “gone”"),
        (r"\bloose (weight|the game|my|your|his|her|our|their)\b", r"lose \1", "Did you mean “lose”?"),
    ],
    "es": [
        (r"\ba (sido|estado|habido|hecho|tenido|dicho|visto|ido|venido|podido|querido)\b", r"ha \1",
         "Con el participio se usa el verbo haber: “ha”"),
        (r"\ba echo\b", "ha hecho", "Se escribe “ha hecho”"),
        (r"\bhaber si\b", "a ver si", "Se escribe “a ver si”"),
        (r"\bhaber que\b(?= (?:pasa|dice|hace|opina))", "a ver qué", "Se escribe “a ver qué”"),
        (r"\bosea\b", "o sea", "Se escribe separado: “o sea”"),
        (r"\baveces\b", "a veces", "Se escribe separado: “a veces”"),
        (r"\benserio\b", "en serio", "Se escribe separado: “en serio”"),
        (r"\bporfavor\b", "por favor", "Se escribe separado: “por favor”"),
        (r"\bapesar\b", "a pesar", "Se escribe separado: “a pesar”"),
        (r"\bdesque\b", "desde que", "Se escribe “desde que”"),
        (r"\bhechar\b", "echar", "“echar” va sin h"),
        (r"\bhalla (que|sido)\b", r"haya \1", "Del verbo haber: “haya”"),
        (r"\b(dijiste|hiciste|tuviste|estuviste|pudiste|quisiste|viniste|fuiste|diste|comiste|bebiste|trajiste|"
         r"supiste|pusiste|hablaste|llegaste|pensaste|miraste|jugaste|dejaste|compraste|entraste|llamaste)s\b",
         r"\1", "El pretérito de “tú” no lleva -s final"),
        (r"\bpienso de que\b", "pienso que", "Dequeísmo: “pienso que”"),
        (r"\bcreo de que\b", "creo que", "Dequeísmo: “creo que”"),
        (r"\bme dijo de que\b", "me dijo que", "Dequeísmo: “me dijo que”"),
    ],
}

PLEONASMS = {
    "en": {
        "free gift": "gift", "end result": "result", "past history": "history", "future plans": "plans",
        "added bonus": "bonus", "close proximity": "proximity", "unexpected surprise": "surprise",
        "each and every": "each", "revert back": "revert", "return back": "return", "repeat again": "repeat",
        "advance planning": "planning", "basic fundamentals": "fundamentals", "completely finished": "finished",
        "absolutely essential": "essential", "final outcome": "outcome", "true fact": "fact",
        "new innovation": "innovation", "join together": "join", "combine together": "combine",
        "merge together": "merge", "collaborate together": "collaborate", "atm machine": "ATM",
        "pin number": "PIN", "at this point in time": "now", "due to the fact that": "because",
        "the reason why is because": "the reason is that", "very unique": "unique", "12 noon": "noon",
        "sum total": "total", "exact same": "same", "still remains": "remains", "brief summary": "summary",
        "first began": "began", "plan ahead": "plan", "postpone until later": "postpone",
        "empty space": "space", "kneel down": "kneel", "may possibly": "may", "might possibly": "might",
        "circle around": "circle", "mutual cooperation": "cooperation", "over exaggerate": "exaggerate",
        "armed gunman": "gunman", "safe haven": "haven", "consensus of opinion": "consensus",
        "in order to": "to", "whether or not": "whether",
    },
    "es": {
        "subir arriba": "subir", "subir para arriba": "subir", "bajar abajo": "bajar", "bajar para abajo": "bajar",
        "salir afuera": "salir", "salir fuera": "salir", "entrar adentro": "entrar", "entrar dentro": "entrar",
        "volver a repetir": "repetir", "lapso de tiempo": "lapso", "prever de antemano": "prever",
        "persona humana": "persona", "mas sin embargo": "sin embargo", "erario público": "erario",
        "nexo de unión": "nexo", "base fundamental": "base", "falso pretexto": "pretexto", "regalo gratis": "regalo",
        "a la misma vez": "a la vez", "hemorragia de sangre": "hemorragia", "hecho real": "hecho",
        "accidente fortuito": "accidente", "resultado final": "resultado", "planes de futuro": "planes",
        "sorpresa inesperada": "sorpresa", "colaborar juntos": "colaborar", "cooperar juntos": "cooperar",
        "adelantar hacia delante": "adelantar", "retroceder hacia atrás": "retroceder", "vuelvo a reiterar": "reitero",
        "reiterar de nuevo": "reiterar", "muy óptimo": "óptimo", "muy excelente": "excelente", "más mejor": "mejor",
        "más peor": "peor", "divisas extranjeras": "divisas", "totalmente gratis": "gratis",
        "completamente lleno": "lleno", "cita previa": None, "insistir de nuevo": "insistir", "mendrugo de pan": "mendrugo",
    },
}
_HACE_ATRAS = re.compile(r"\bhace ([\w\s]{1,20}?) atrás\b", re.IGNORECASE)

STOP = {
    "en": set("about after again also always another around because before being between could every first "
              "going great other people really should since still their there these thing things think those "
              "through under until where which while would years your yours".split()),
    "es": set("porque cuando donde desde entre hasta mientras nuestro nuestra otros otras pero sobre también "
              "tiene tienen todos todas puede pueden siempre alguien algunos algunas mucho muchos muchas cosas "
              "después antes estos estas estaba estaban hacer quiero".split()),
}


def detect(text, hint="auto"):
    if hint in ("en", "es"):
        return hint
    words = [w.lower() for w in WORD.findall(text)][:200]
    if not words:
        return "en"
    if re.search(r"[ñ¿¡áéíóú]", text.lower()):
        return "es"
    scores = {lang: len(_checker(lang).known(words)) for lang in ("en", "es")}
    return "es" if scores["es"] > scores["en"] else "en"


def _u16(text):
    """Maps Python string offsets to UTF-16 offsets (only differs with emoji etc.)."""
    if all(ord(c) < 0x10000 for c in text):
        return lambda i: i
    table, n = [], 0
    for c in text:
        table.append(n)
        n += 2 if ord(c) >= 0x10000 else 1
    table.append(n)
    return lambda i: table[i]


def check(text, lang="auto", ignore=()):
    lang = detect(text, lang)
    sp = _checker(lang)
    ignore = {w.lower() for w in ignore}
    issues = []
    skip = [(m.start(), m.end()) for m in SKIP_TOKEN.finditer(text)]

    def skipped(a, b):
        return any(s <= a and b <= e for s, e in skip)

    def add(kind, a, b, message, suggestions=()):
        if skipped(a, b) or any(i["_a"] < b and a < i["_b"] for i in issues):
            return
        issues.append({"_a": a, "_b": b, "type": kind, "message": message, "suggestions": list(suggestions)})

    # --- redundancy (pleonasms) first: they span several words
    low = text.lower()
    for phrase, better in PLEONASMS[lang].items():
        if better is None:
            continue
        for m in re.finditer(r"\b" + re.escape(phrase) + r"\b", low):
            add("style", m.start(), m.end(), f"Redundant: “{text[m.start():m.end()]}” can be just “{better}”"
                if lang == "en" else f"Redundancia: “{text[m.start():m.end()]}” puede ser solo “{better}”",
                [_match_case(text[m.start():m.end()], better)])
    if lang == "es":
        for m in _HACE_ATRAS.finditer(text):
            add("style", m.start(), m.end(), "Redundancia: “hace … atrás”; basta con “hace …”",
                [f"hace {m.group(1)}"])

    # --- grammar rules
    for pattern, repl, message in RULES[lang]:
        if repl is None:
            continue
        for m in re.finditer(pattern, text, re.IGNORECASE):
            add("grammar", m.start(), m.end(), message, [_match_case(m.group(0), m.expand(repl))])

    words = list(WORD.finditer(text))
    for i, m in enumerate(words):
        w = m.group(0)
        prev = words[i - 1] if i else None
        # repeated word: "the the"
        if prev and w.lower() == prev.group(0).lower() and text[prev.end():m.start()].strip() == "" \
                and w.lower() not in ("that", "had", "is", "no", "bye", "très", "muy", "que", "ja", "ha"):
            add("grammar", prev.end(), m.end(), "Repeated word" if lang == "en" else "Palabra repetida", [""])
            continue
        if lang == "en" and prev and prev.group(0).lower() in ("a", "an") and text[prev.end():m.start()] == " ":
            lw = w.lower()
            vowel = lw[0] in "aeiou" and not lw.startswith(A_EXCEPT)
            vowel = vowel or lw in AN_EXCEPT or (len(w) > 1 and w.isupper() and w[0] in "AEFHILMNORSX")
            want = "an" if vowel else "a"
            if prev.group(0).lower() != want:
                add("grammar", prev.start(), prev.end(), f"Use “{want}” before “{w}”", [_match_case(prev.group(0), want)])
        if lang == "en" and w == "i":
            add("grammar", m.start(), m.end(), "Capitalize “I”", ["I"])

    # capital letter at the start of each sentence (only if the text has several words)
    if len(words) > 3:
        for m in re.finditer(r"(?:^|[.!?]\s+|\n\s*)([¿¡\"“]?)([a-zà-ÿ])", text):
            a = m.start(2)
            nxt = WORD.match(text, a)
            if nxt and not skipped(a, nxt.end()) and len(nxt.group(0)) > 1:
                add("grammar", a, nxt.end(), "Start the sentence with a capital letter" if lang == "en"
                    else "La oración empieza con mayúscula", [nxt.group(0)[0].upper() + nxt.group(0)[1:]])

    if lang == "es":
        for m in re.finditer(r"(^|[.!?]\s+|\n)([^¿¡.!?\n]{3,}?)([?!])", text):
            open_mark = "¿" if m.group(3) == "?" else "¡"
            sentence = m.group(2)
            if open_mark not in sentence:
                a = m.start(2)
                first = WORD.match(text, a) or re.compile(r"\S+").match(text, a)
                if first:
                    add("grammar", first.start(), first.end(), f"Falta el signo de apertura “{open_mark}”",
                        [open_mark + first.group(0)])

    # spacing
    for m in re.finditer(r"(?<=\S)  +(?=\S)", text):
        add("grammar", m.start(), m.end(), "Double space" if lang == "en" else "Espacio doble", [" "])
    for m in re.finditer(r"(?<=\w) +([,.;:!?])(?!\S*\.\w)", text):
        add("grammar", m.start(), m.end(), "No space before punctuation" if lang == "en"
            else "Sin espacio antes del signo", [m.group(1)])

    # --- spelling
    for m in words:
        w = m.group(0)
        lw = w.lower().replace("’", "'")
        if len(w) < 2 or lw in ignore or skipped(m.start(), m.end()):
            continue
        if w.isupper() or (w[0].isupper() and m.start() > 0 and not re.search(r"(^|[.!?¿¡]\s*)$", text[:m.start()])):
            continue  # acronyms and names in mid-sentence
        if sp.known([lw]) or ("'" in lw and all(sp.known([p]) or len(p) <= 2 for p in lw.split("'"))):
            continue
        sug = [_match_case(w, s) for s in _suggest(lang, lw)]
        add("spelling", m.start(), m.end(), "Possible spelling mistake" if lang == "en" else "Posible falta de ortografía",
            sug)

    # --- nearby repetition of long words (style)
    seen = {}
    for idx, m in enumerate(words):
        lw = m.group(0).lower()
        if len(lw) < 6 or lw in STOP[lang]:
            continue
        if lw in seen and idx - seen[lw] <= 12:
            add("style", m.start(), m.end(), f"“{m.group(0)}” is repeated close together" if lang == "en"
                else f"“{m.group(0)}” se repite muy cerca", [])
        seen[lw] = idx

    conv = _u16(text)
    out = []
    for i in sorted(issues, key=lambda i: i["_a"]):
        a, b = i.pop("_a"), i.pop("_b")
        i.update(start=conv(a), end=conv(b), text=text[a:b])
        out.append(i)
    return {"lang": lang, "issues": out}


def ai_issues(original, corrected, lang):
    """Word-level differences between the text and the model's corrected version."""
    from difflib import SequenceMatcher

    tok = re.compile(r"\s+|[^\s]+")
    a = [m for m in tok.finditer(original)]
    b = [m.group(0) for m in tok.finditer(corrected)]
    sm = SequenceMatcher(None, [m.group(0) for m in a], b, autojunk=False)
    conv = _u16(original)
    issues = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        new = "".join(b[j1:j2])
        if i1 == i2:  # insertion: attach to the previous token
            if i1 == 0:
                continue
            i1 -= 1
            new = a[i1].group(0) + new
        start, end = a[i1].start(), a[i2 - 1].end()
        old = original[start:end]
        if old.strip() == new.strip():
            continue
        issues.append({"type": "ai", "start": conv(start), "end": conv(end), "text": old,
                       "message": "Suggested by the AI" if lang == "en" else "Sugerencia de la IA",
                       "suggestions": [new]})
    return issues
