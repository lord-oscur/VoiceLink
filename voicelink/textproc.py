"""Text utilities: think-tag filtering, speech cleanup, streaming sentence
chunking and a tiny stop-word language detector."""
import re
import unicodedata

# ---------------------------------------------------------------- think tags


class ThinkFilter:
    """Removes <think>...</think> blocks from a token stream (for models that
    put their reasoning inside the content field)."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.buf = ""
        self.inside = False

    def feed(self, text):
        self.buf += text
        out = []
        while self.buf:
            tag = self.CLOSE if self.inside else self.OPEN
            i = self.buf.find(tag)
            if i >= 0:
                if not self.inside:
                    out.append(self.buf[:i])
                self.buf = self.buf[i + len(tag):]
                self.inside = not self.inside
                continue
            keep = 0
            for n in range(min(len(tag) - 1, len(self.buf)), 0, -1):
                if tag.startswith(self.buf[-n:]):
                    keep = n
                    break
            if not self.inside:
                out.append(self.buf[:len(self.buf) - keep])
            self.buf = self.buf[len(self.buf) - keep:]
            break
        return "".join(out)

    def flush(self):
        rest, self.buf = ("" if self.inside else self.buf), ""
        return rest


# ------------------------------------------------------------ speech cleanup

_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL = re.compile(r"https?://\S+|www\.\S+")
_CODE_FENCE = re.compile(r"```.*?(```|$)", re.S)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.M)
_HEADING = re.compile(r"^\s*#{1,6}\s*", re.M)
_EMPH = re.compile(r"(\*\*|__|\*|~~)")
_SPACES = re.compile(r"[ \t\r\f\v]+")


_SPEAKABLE_PUNCT = set(".,!?;:'\"-()¿¡…—–«»“”‘’%$€£+=&@°/\n")
_FULLWIDTH_PUNCT = str.maketrans({"。": ". ", "，": ", ", "、": ", ", "？": "? ", "！": "! ",
                                  "：": ": ", "；": "; ", "（": " (", "）": ") "})


def _is_speakable(ch):
    # Marks (M*) are the vowel signs of Devanagari and other scripts;
    # variation selectors only style emoji.
    cat = unicodedata.category(ch)
    return (cat.startswith(("L", "N", "Z")) or (cat.startswith("M") and ch not in "\ufe0e\ufe0f")
            or ch in _SPEAKABLE_PUNCT)


def clean_for_speech(text):
    text = _CODE_FENCE.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _URL.sub(" ", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _HEADING.sub("", text)
    text = _BULLET.sub("", text)
    text = _EMPH.sub("", text)
    text = text.translate(_FULLWIDTH_PUNCT)
    text = "".join(ch if _is_speakable(ch) else " " for ch in text)
    lines = [_SPACES.sub(" ", ln).strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    for i in range(len(lines) - 1):
        if lines[i][-1] not in ".,!?;:…":
            lines[i] += ","
    return " ".join(lines)


def has_words(text):
    return any(ch.isalnum() for ch in text)


# ---------------------------------------------------------- sentence chunker

_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "sra", "srta", "st", "vs", "etc", "e.g", "i.e", "approx",
    "no", "vol", "fig", "jr", "inc", "ltd", "ud", "uds", "av", "p.ej",
}
_DANGLING = set(
    "the a an to of and or but in on at for with from by as my your his her its our their is are was were be "
    "that this these those very so than if do does did how what why when where who which will would can could "
    "have has had not "
    "como qué cómo cuándo dónde quién cuál no muy más "
    "el la los las un una unos unas de del y o u que en con por para a al mi tu su sus se lo le es muy "
    "le la les des et ou à au du en un une mon ton son ce cette il lo gli di da che per con é"
    .split())
_SENT_END = re.compile(r"[.!?…]+[\"')\]»”]*(?=\s)|[。！？]+[」』）]*|\n+")
_CLAUSE = re.compile(r"[,;:—–](?=\s)|[，、；：]")


class SentenceChunker:
    """Turns a token stream into speakable chunks.

    Chunks grow progressively: the first is cut at the first clause boundary
    (or after a few words) so audio starts quickly, the second is still short,
    and later ones are whole sentences. Each chunk then finishes synthesizing
    while the previous one is playing, so there are no gaps."""

    def __init__(self, first_min_chars=12, max_chars=220, first_wait_s=0.6):
        self.first_min = first_min_chars
        self.max_chars = max_chars
        self.first_wait = first_wait_s
        self.buf = ""
        self.emitted = 0
        self.t_first = None

    def _word_cut(self, limit):
        """Last word boundary before `limit` that does not leave a dangling
        article/preposition at the end of the chunk."""
        pos = self.buf.rfind(" ", 0, limit)
        fallback = pos
        while pos > 0:
            prev = self.buf[:pos].rstrip().rsplit(" ", 1)[-1].lower()
            if prev not in _DANGLING:
                return pos
            pos = self.buf.rfind(" ", 0, pos)
        return fallback

    def _limits(self):
        """(minimum length for a clause cut, length that forces a cut)."""
        if self.emitted == 0:
            return self.first_min, 70
        if self.emitted == 1:
            return self.first_min, 80
        if self.emitted == 2:
            return self.first_min * 2.5, 140
        return None, self.max_chars

    def _sentence_cut(self, start=0):
        for m in _SENT_END.finditer(self.buf, start):
            end = m.end()
            if self.buf[m.start()] == ".":
                word = re.search(r"(\S+)\.$", self.buf[:m.start() + 1])
                if word:
                    w = word.group(1).lower().strip("(\"'¿¡")
                    if w in _ABBREV or (len(w) == 1 and w.isalpha()):
                        continue
            return end
        return -1

    def feed(self, text, now=None):
        """`now` (seconds, any clock) enables the early first-chunk cut."""
        self.buf += text
        if self.t_first is None and now is not None and self.buf.strip():
            self.t_first = now
        out = []
        while True:
            clause_min, hard = self._limits()
            cut = self._sentence_cut()
            if cut < 0 and clause_min is not None:
                clause = [m.end() for m in _CLAUSE.finditer(self.buf) if m.end() >= clause_min]
                if clause:
                    cut = clause[0]
            if (cut < 0 and self.emitted == 0 and now is not None and self.t_first is not None
                    and now - self.t_first >= self.first_wait and len(self.buf.split()) >= 5):
                complete = len(self.buf) - 1 if self.buf.endswith(" ") else self.buf.rfind(" ")
                cut = self._word_cut(complete + 1)
            if cut < 0 and len(self.buf) > hard:
                clause = [m.end() for m in _CLAUSE.finditer(self.buf, 0, hard)]
                space = self._word_cut(hard)
                cut = clause[-1] if clause else (space if space > 0 else hard)
            if cut < 0:
                break
            piece, self.buf = self.buf[:cut].strip(), self.buf[cut:]
            if piece:
                out.append(piece)
                self.emitted += 1
        return out

    def flush(self):
        piece, self.buf = self.buf.strip(), ""
        if piece:
            self.emitted += 1
            return [piece]
        return []


# -------------------------------------------------------- language detection

_STOPWORDS = {
    "en": "the and is are you i to of a in that it for on with this what do not be have was my your "
          "can me so we they he she how there here just like but about yes no please thanks hello",
    "es": "el la los las de que y en un una es por para con no se lo le me mi tu te qué como pero "
          "más muy sí hola gracias está estoy eres soy del al yo tú usted cómo cuál dónde también",
    "fr": "le la les de des et est un une que qui pas je tu vous nous il elle ce pour avec dans sur "
          "oui non merci bonjour mais très c'est suis êtes",
    "it": "il lo la gli le di che e è un una per non con sono sei io tu mi ti ciao grazie ma molto "
          "anche come questo della",
    "pt": "o a os as de que e é um uma não com para eu você por mas muito sim olá obrigado está "
          "estou isso também como",
}
_STOP = {k: set(v.split()) for k, v in _STOPWORDS.items()}
_WORD = re.compile(r"[a-záéíóúüñàèìòùâêîôûçãõœ']+", re.I)


LANGUAGE_NAMES = {"en": "English", "es": "Spanish", "fr": "French", "it": "Italian", "pt": "Portuguese"}


def _scores(text, allowed):
    words = _WORD.findall(text.lower())
    scores = {}
    for lang, stop in _STOP.items():
        if allowed and lang not in allowed:
            continue
        scores[lang] = sum(1 for w in words if w in stop)
    if "es" in scores:
        scores["es"] += 2 * (text.count("¿") + text.count("¡") + text.lower().count("ñ"))
    if "pt" in scores:
        scores["pt"] += 2 * sum(text.lower().count(c) for c in "ãõ")
    if "fr" in scores:
        scores["fr"] += 2 * sum(text.lower().count(c) for c in "èêç") + text.lower().count("œ")
    return scores


def confident_language(text, allowed=None):
    """The language of `text` if the evidence is clear, else None."""
    scores = _scores(text, allowed)
    if not scores:
        return None
    ranked = sorted(scores.values(), reverse=True)
    best = max(scores, key=scores.get)
    if ranked[0] >= 2 and (len(ranked) == 1 or ranked[0] >= 2 * ranked[1]):
        return best
    return None


def detect_language(text, allowed=None, default="en"):
    scores = _scores(text, allowed)
    if not scores:
        return default
    best = max(scores, key=scores.get)
    if scores[best] == 0 or list(scores.values()).count(scores[best]) > 1:
        return default if (not allowed or default in allowed) else best
    return best


def normalized_words(text):
    return _WORD.findall(text.lower())
