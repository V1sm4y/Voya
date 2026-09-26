"""Business-name parsing: core name, legal form, alias.

The legal form is never thrown away. `parse_name` returns it next to the core name:
  core_name   basic()-normalised name without legal-form tokens, bracket tags, '&'/'and'/'the'.
              Indic script is kept as written. Use it for exact-match blocking keys.
  core_key    cross-script phonetic key of core_name (translit.phonetic), for fuzzy blocking and features.
  legal_form  canonical legal forms found, sorted and '+'-joined (e.g. 'ltd+pvt', 'llc', ''),
              including forms written in Indic scripts (लिमिटेड, ప్రైవేట్, प्रा. लि., एलएलपी …).
  alias       core_name of the part after f/k/a, formerly, t/a, d/b/a, a/k/a (S3 only), or None.

Blocking may ignore legal form. Features must compare forms with `legal_relation`, because the
US decoys differ from the real record only in LLC vs Corp and similar.
"""
import re
from functools import lru_cache
from typing import Optional

from .normalize import strip_accents
from .translit import has_indic, phonetic, phonetic_token, romanize

LEGAL_MAP = {
    "llc": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt",
    "llp": "llp", "lp": "lp", "pc": "pc", "pllc": "pllc", "plc": "plc",
    "sarl": "sarl", "eurl": "eurl", "sas": "sas", "sasu": "sasu", "sa": "sa", "sci": "sci", "snc": "snc",
}
_MULTI = [(("l", "l", "c"), "llc"), (("l", "l", "p"), "llp"), (("p", "l", "l", "c"), "pllc"),
          (("public", "limited"), "public+ltd"), (("public", "ltd"), "public+ltd")]
# Indic legal words, matched on the romanised token or on its phonetic skeleton
_INDIC_EXACT = {"praa": "pvt", "pra": "pvt", "li": "ltd", "elaelapii": "llp", "elelpii": "llp"}
_INDIC_PHON = {phonetic_token(w): f for w, f in
               [("private", "pvt"), ("limited", "ltd"), ("company", "co"), ("corporation", "corp"),
                ("elelpi", "llp"), ("publik", "public")]}
_ALIAS_RE = re.compile(r"\b(?:f/?k/?a|formerly(?: known as)?|t/a|d/?b/?a|a/?k/?a|trading as)\b\.?", re.I)
_BRACKET = re.compile(r"\[([^\]]*)\]")
_TOKEN = re.compile(r"[0-9a-zऀ-෿‌‍]+")
_STOP = {"and", "the"}


_DOMAIN = re.compile(r"\.(?:co\.in|com|net|org|biz|info|in|fr|co)\b")


def _tokens(s: str):
    s = _DOMAIN.sub(" ", strip_accents(s.lower()))          # "canaraagro.com" -> "canaraagro"
    s = s.replace("&", " ").replace(".", "").replace("'", "")
    return _TOKEN.findall(s)


def _split_legal(tokens):
    core, forms = [], set()
    i = 0
    while i < len(tokens):
        for seq, form in _MULTI:
            if tuple(tokens[i:i + len(seq)]) == seq:
                forms.update(form.split("+")); i += len(seq); break
        else:
            t = tokens[i]
            if t in LEGAL_MAP:
                forms.add(LEGAL_MAP[t])
            elif has_indic(t):
                r = romanize(t)
                f = _INDIC_EXACT.get(r) or _INDIC_PHON.get(phonetic_token(r))
                if f:
                    forms.add(f)
                else:
                    core.append(t)
            elif t not in _STOP:
                core.append(t)
            i += 1
    return core, forms


@lru_cache(maxsize=200_000)
def parse_name(name: str) -> dict:
    main, alias = name, None
    m = _ALIAS_RE.search(name)
    if m:
        main, alias = name[:m.start()], name[m.end():]
    # bracket tags ("[Service]", "[Inc]") are junk apart from any legal form inside them
    bracket_forms = set()
    for inner in _BRACKET.findall(main):
        bracket_forms |= _split_legal(_tokens(inner))[1]
    core, forms = _split_legal(_tokens(_BRACKET.sub(" ", main)))
    forms |= bracket_forms
    core_name = " ".join(core) or None
    alias_core = None
    if alias:
        a_core, a_forms = _split_legal(_tokens(_BRACKET.sub(" ", alias)))
        alias_core = " ".join(a_core) or None
    return {
        "core_name": core_name,
        "core_key": phonetic(core_name) if core_name else None,
        "legal_form": "+".join(sorted(forms)),
        "alias": alias_core,
    }


def legal_relation(a: str, b: str) -> str:
    """Compare two legal_form strings: 'missing' (either empty), 'same', 'subset', 'overlap', or 'conflict'.
    'pvt' implies 'ltd' (Indian S2/S3 often drop one half of 'Pvt Ltd')."""
    A, B = set(filter(None, a.split("+"))), set(filter(None, b.split("+")))
    if not A or not B:
        return "missing"
    for s in (A, B):
        if "pvt" in s:
            s.add("ltd")
    if A == B:
        return "same"
    if A <= B or B <= A:
        return "subset"
    return "overlap" if A & B else "conflict"
