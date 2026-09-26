"""Rule-based transliteration of Indic scripts to Latin, and a cross-script phonetic key.

`romanize` handles all nine Brahmic scripts in U+0900–U+0DFF (Devanagari, Bengali, Gurmukhi,
Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam). These scripts share one offset layout inside
their 128-code-point blocks, so a single table covers them all. A consonant carries an inherent
"a" unless a vowel sign or virama follows, and a word-final inherent "a" is dropped
("राम" -> "ram").

`phonetic` maps Latin or romanised text to a consonant skeleton. It removes vowels, h and y, and
merges letters that the scripts or common romanisations confuse (voiced/unvoiced, aspirated,
v/w/b/bh, m/n, s/sh/z). Spelling variants and cross-script forms of one word then share a key:
  "Maharashtra" == "महाराष्ट्र"   "Kancheepuram" == "Kanchipuram"   "private" == "ప్రైవేట్"
It is a similarity key, not an identity. Use it for blocking and fuzzy features, never as proof.
"""
import re
from functools import lru_cache

_CONS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng", 0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh",
    0x1E: "ny", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d",
    0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y",
    0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l",  # 0x34 = Tamil/Malayalam zha: 'l' keeps Tamil == Tamil
    0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x58: "q", 0x59: "kh", 0x5A: "gh", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
_INDEP = {0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu", 0x0B: "ri", 0x0C: "li",
          0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
          0x60: "ri", 0x61: "li", 0x72: "i", 0x73: "u"}
_SIGN = {0x3E: "aa", 0x3F: "i", 0x40: "ii", 0x41: "u", 0x42: "uu", 0x43: "ri", 0x44: "ri", 0x45: "e",
         0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x55: "", 0x56: "ai",
         0x57: "au", 0x62: "li", 0x63: "li"}
_NASAL = {0x01: "n", 0x02: "n", 0x03: "h", 0x70: "n"}          # candrabindu, anusvara, visarga, Gurmukhi tippi
_CHILLU = {0x7A: "n", 0x7B: "n", 0x7C: "r", 0x7D: "l", 0x7E: "l", 0x7F: "k"}  # Malayalam
_NUKTA = {"j": "z", "ph": "f", "k": "q", "d": "r", "dh": "rh", "g": "gh"}
_VIRAMA, _NUKTA_CP = 0x4D, 0x3C
_INDIC_RE = re.compile(r"[ऀ-෿]")


def has_indic(s: str) -> bool:
    return bool(_INDIC_RE.search(s))


@lru_cache(maxsize=500_000)
def romanize(text: str) -> str:
    """Indic -> simple ASCII romanisation. Non-Indic characters pass through unchanged."""
    if not _INDIC_RE.search(text):
        return text
    out = []
    pending = False      # last consonant still owns its inherent 'a'
    prev_virama = False  # previous character was a virama (we are inside a conjunct)
    last_rra = False     # the last consonant emitted was Malayalam റ (U+0D31)
    for ch in text:
        cp = ord(ch)
        is_virama = 0x0900 <= cp <= 0x0DFF and (cp & 0x7F) == _VIRAMA
        if 0x0900 <= cp <= 0x0DFF:
            off = cp & 0x7F
            block = cp & ~0x7F
            if off in _CONS:
                if pending:
                    out.append("a")
                cons = _CONS[off]
                if cp == 0x0D31 and prev_virama:
                    # Malayalam conjuncts: റ്റ is "tt", ന്റ is "nt"
                    if last_rra and out and out[-1] == "r":
                        out[-1] = "t"
                    cons = "t"
                out.append(cons); pending = True
                last_rra = cp == 0x0D31 and not prev_virama
            elif off in _SIGN:
                out.append(_SIGN[off]); pending = False
            elif off == _VIRAMA:
                pending = False
            elif off == _NUKTA_CP:
                if out and out[-1] in _NUKTA:
                    out[-1] = _NUKTA[out[-1]]
            elif off in _INDEP:
                if pending:
                    out.append("a")
                out.append(_INDEP[off]); pending = False
            elif off in _NASAL:
                if pending:
                    out.append("a")
                out.append(_NASAL[off]); pending = False
            elif block == 0x0D00 and off in _CHILLU:
                if pending:
                    out.append("a")
                out.append(_CHILLU[off]); pending = False
            elif block == 0x0980 and off == 0x4E:   # Bengali khanda ta
                if pending:
                    out.append("a")
                out.append("t"); pending = False
            elif 0x66 <= off <= 0x6F:
                if pending:
                    out.append("a")
                out.append(str(off - 0x66)); pending = False
            # everything else (avagraha, addak, length marks, om...) is dropped
        else:
            pending = False  # word-final schwa deletion
            if cp not in (0x200C, 0x200D):
                out.append(ch)
        prev_virama = is_virama
    return "".join(out)


# order matters: longer patterns first. "tio"/"sio" -> S covers -tion/-sion (Hindi writes शन).
_DIGRAPHS = [("chh", "C"), ("tio", "S"), ("sio", "S"), ("sh", "S"), ("th", "T"), ("dh", "T"), ("kh", "K"),
             ("gh", "K"), ("ph", "P"), ("bh", "P"), ("ck", "K"), ("zh", "L"), ("jh", "C"), ("rh", "R")]
_VOWELS = set("aeiouy")
_SINGLE = {"b": "P", "p": "P", "f": "P", "v": "P", "w": "P", "d": "T", "t": "T", "g": "K", "k": "K", "q": "K",
           "j": "C", "z": "S", "s": "S", "x": "KS", "l": "L", "r": "R", "m": "N", "n": "N"}


@lru_cache(maxsize=1_000_000)
def phonetic_token(tok: str) -> str:
    """Consonant skeleton of one lower-case token; digits are kept as-is."""
    t = romanize(tok).lower()
    out = []
    i = 0
    while i < len(t):
        for dg, sym in _DIGRAPHS:
            if t.startswith(dg, i):
                out.append(sym); i += len(dg); break
        else:
            c = t[i]
            if t.startswith("ch", i):
                # "ch" + consonant is a hard k (technology, school); otherwise the Indic "ch"
                nxt = t[i + 2:i + 3]
                hard = (nxt and nxt.isalpha() and nxt not in _VOWELS) or (i > 0 and t[i - 1] == "s" and nxt in ("o", "e", "a"))
                out.append("K" if hard else "C")
                i += 2
                continue
            if c == "c":
                out.append("S" if t[i + 1:i + 2] in ("e", "i", "y") else "K")
            elif c.isdigit():
                out.append(c)
            else:
                s = _SINGLE.get(c)
                if s:
                    out.append(s)
            i += 1
    key = "".join(out)
    return re.sub(r"([A-Z])\1+", r"\1", key)


def phonetic(text: str) -> str:
    """Space-joined phonetic_token of every alphanumeric token (Indic tokens are romanised first)."""
    toks = re.findall(r"[0-9a-zऀ-෿‌‍]+", text.lower())
    return " ".join(k for k in map(phonetic_token, toks) if k)
