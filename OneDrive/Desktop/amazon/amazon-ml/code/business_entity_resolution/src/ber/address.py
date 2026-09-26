"""Address parsing into one schema for every country, with India-specific structure.

parse_address(addr, country) returns a dict of str/None (multi-valued fields are '|'-joined):
  state        canonical state/region name ('maharashtra', 'texas', 'hauts de france'); French
               departments are mapped to their region
  city         canonical city (old/new names unified, ' city'/' suburban' suffixes dropped). This is
               the component next to the state, which in Indian S1 data is often the district.
  locality     locality/area phrases: '... nagar', '... colony', 'sector 19d', 'phase 5', 'block g',
               villages, and other plain components
  street       road/street/marg/lane/cross names with abbreviations expanded ('1st main road')
  building     building/complex names ('emjay towers', 'satra plaza')
  premise_no   canonical building/plot/house/flat/shop/door/survey numbers ('9062/17', 'b/53a', '1795')
  premise_typed  the same numbers with their type ('plot:35b', 'flat:104', 'house:1795')
  unit         US/FR unit, suite or apartment ids ('136')
  floor        floor ('g', '1', '2', 'b')
  landmark     text after near/nr/opp/opposite/behind/beside/next to ('state bank colony')
  admin        district/taluka/tehsil/mandal/post-office values
  careof       c/o, w/o, s/o text
  pin          6-digit PIN or 5-digit ZIP / code postal when present
  *_key        translit.phonetic() of locality/street/building (same '|' order). These keys are
               script- and spelling-insensitive, so "Kancheepuram" == "Kanchipuram" and a Devanagari
               locality meets its Latin spelling.

Components in an Indic script are romanised first (translit.romanize), and null tokens are dropped.
"""
import re
from functools import lru_cache
from typing import Optional

from .normalize import IN_STATES, US_STATES, basic, strip_accents
from .translit import has_indic, phonetic, romanize

# ---------------------------------------------------------------- geography
FR_DEPT_REGION = {
    **{d: "hauts de france" for d in ["nord", "pas de calais", "somme", "oise", "aisne"]},
    **{d: "nouvelle aquitaine" for d in ["gironde", "landes", "dordogne", "lot et garonne", "pyrenees atlantiques",
                                           "charente", "charente maritime", "vienne", "deux sevres", "haute vienne",
                                           "creuse", "correze"]},
    **{d: "pays de la loire" for d in ["loire atlantique", "maine et loire", "vendee", "sarthe", "mayenne"]},
}
FR_REGIONS = ["auvergne rhone alpes", "bourgogne franche comte", "bretagne", "centre val de loire", "corse",
              "grand est", "hauts de france", "ile de france", "normandie", "nouvelle aquitaine", "occitanie",
              "pays de la loire", "provence alpes cote d azur"]
IN_ALIASES = {"dilli": "delhi", "hariyana": "haryana", "asam": "assam", "odisa": "odisha", "orisa": "odisha",
              "keralam": "kerala", "paschim banga": "west bengal", "paschimbanga": "west bengal",
              "pashchim bangal": "west bengal", "orissa": "odisha", "uttaranchal": "uttarakhand",
              "pondicherry": "puducherry", "nct of delhi": "delhi", "chhatisgarh": "chhattisgarh",
              "tamilnadu": "tamil nadu", "andhra": "andhra pradesh"}
IN_CANON = {v: v for v in IN_STATES.values()}
IN_CANON.update({"orissa": "odisha", "odisha": "odisha"})


def loose(s: str) -> str:
    """basic() without spaces and with doubled vowels collapsed: 'bihaar' -> 'bihar', 'Tamil Nadu' -> 'tamilnadu'."""
    return re.sub(r"([aeiou])\1+", r"\1", basic(s)).replace(" ", "")


def _build_state_index():
    """country -> (code->state, {name variant -> state}). A component matches a state when its loose()
    form equals a variant, or (keys of 3+ consonants only) its phonetic key AND first letter match.
    Short keys ('PR' = Bihar, 'TL' = Delhi) are too collision-prone for phonetic-only matching."""
    def variants(pairs):
        exact, phon = {}, {}
        for name, canon in pairs:
            exact[loose(name)] = canon
            k = phonetic(name).replace(" ", "")
            if len(k) >= 3:
                phon[(k, loose(name)[0])] = canon
        return exact, phon
    idx = {}
    idx["US"] = (dict(US_STATES), *variants((v, v) for v in US_STATES.values()))
    ind_codes = {k: IN_CANON.get(v, v) for k, v in IN_STATES.items()}
    pairs = [(v, IN_CANON.get(v, v)) for v in IN_STATES.values()] + \
            [(a, IN_CANON.get(c, c)) for a, c in IN_ALIASES.items()]
    idx["India"] = (ind_codes, *variants(pairs))
    idx["France"] = ({}, *variants([(r, r) for r in FR_REGIONS] + list(FR_DEPT_REGION.items())))
    return idx


_STATE_IDX = _build_state_index()

CITY_ALIASES = {
    "bangalore": "bengaluru", "bangalore urban": "bengaluru", "bengaluru urban": "bengaluru",
    "bangalore north": "bengaluru", "bangalore south": "bengaluru", "bombay": "mumbai",
    "mumbai city": "mumbai", "mumbai suburban": "mumbai", "calcutta": "kolkata", "madras": "chennai",
    "gurgaon": "gurugram", "hissar": "hisar", "poona": "pune", "baroda": "vadodara", "mysore": "mysuru",
    "trivandrum": "thiruvananthapuram", "cochin": "kochi", "calicut": "kozhikode", "benaras": "varanasi",
    "banaras": "varanasi", "allahabad": "prayagraj", "darjiling": "darjeeling", "vizag": "visakhapatnam",
    "new york city": "new york", "nyc": "new york",
}
_CITY_SUFFIX = re.compile(r"\s+(city|suburban|urban|district|dist)$")


def canon_city(c: Optional[str], country: str = "") -> Optional[str]:
    """basic() + old/new name aliases; Indian ' city'/' suburban'/' urban' suffixes dropped (not 'Iowa City')."""
    if not c:
        return None
    c = basic(c)
    c = CITY_ALIASES.get(c, c)
    if country == "India":
        c = _CITY_SUFFIX.sub("", c)
        c = CITY_ALIASES.get(c, c)
    return c or None


@lru_cache(maxsize=300_000)
def detect_state(component: str, country: str) -> Optional[str]:
    """Canonical state if the whole component is a state/region (code, name, alias or Indic spelling)."""
    idx = _STATE_IDX.get(country)
    if idx is None:
        return None
    codes, exact, phon = idx
    c = component.strip()
    if re.fullmatch(r"[A-Za-z]{2}", c):
        return codes.get(c.upper()) if (c.isupper() or country != "US") else None
    if not c or re.search(r"\d", c):
        return None
    c = romanize(c)
    lc = loose(c)
    if lc in exact:
        return exact[lc]
    key = phonetic(basic(c)).replace(" ", "")
    return phon.get((key, lc[:1])) if len(key) >= 3 else None


# ---------------------------------------------------------------- component patterns
_NULLS = {"null", "<null>", "n/a", "na", "none", "nan", "-", "--"}
_CAREOF = re.compile(r"^\s*(c\s*/\s*[o0]|w\s*/\s*o|s\s*/\s*o|d\s*/\s*o|care of)\b[\s.:-]*", re.I)
_LANDMARK = re.compile(r"\b(near|nr|opp|opposite|behind|beside|besides|next to|adjacent to|adj|"
                       r"in front of|infront of|facing)\b\.?", re.I)
_PIN = re.compile(r"\b(\d{3}\s?\d{3})\b")
_ZIP = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
_FLOOR = re.compile(r"\b(?:(ground|grd|gr|g|first|second|third|fourth|fifth|top|upper|lower|basement|mezzanine|"
                    r"\d{1,2})\s*(?:st|nd|rd|th)?\s*(?:floor|flr|fl)\b|(?:floor|flr|fl)\.?\s*(?:no\.?)?\s*"
                    r"(ground|g|\d{1,2}))\b\.?", re.I)
_FLOOR_WORD = {"ground": "g", "grd": "g", "gr": "g", "first": "1", "second": "2", "third": "3", "fourth": "4",
               "fifth": "5", "basement": "b", "top": "top", "upper": "u", "lower": "l", "mezzanine": "m"}
_ID = r"(?:[a-z]{1,3}\s?[-/]?\s?)?\d[\w/-]*(?:\s?[a-z](?![a-z]))?"
_KW_ID = re.compile(r"(?:\b(?P<kw>[a-z]{1,10})\.?\s*(?:no|num|number)\b\.?|(?P<kw2>\b(?:plot|flat|shop|office|"
                    r"door|khasra|kh|gala|room|house|hn|unit|bldg|building|survey|sy|gat|site|stall|godown)\b)|"
                    r"(?P<kw3>\bno\b\.?|#))\s*[.:#-]*\s*(?P<id>" + _ID + r")", re.I)
_KW_TYPE = {"h": "house", "house": "house", "hn": "house", "hno": "house", "plot": "plot", "p": "plot",
            "flat": "flat", "f": "flat", "fl": "flat", "shop": "shop", "office": "office", "off": "office",
            "door": "door", "d": "door", "sy": "survey", "sr": "survey", "s": "survey", "survey": "survey",
            "kh": "khasra", "khasra": "khasra", "gala": "gala", "room": "room", "unit": "unit", "bldg": "bldg",
            "building": "bldg", "gat": "gat", "g": "gat", "site": "site", "stall": "shop", "godown": "godown"}
_LEAD_ID = re.compile(r"^\s*#?\s*(?:part\s+of\s+)?(?P<id>(?:[a-z]{1,4}\s?[-/]\s?){0,2}\d[\w/-]*(?:\s?[a-z](?![a-z]))?)(?!\s*(?:st|nd|rd|th)\b)"
                      r"(?=\s|$|,)", re.I)
_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7", "viii": "8", "ix": "9",
          "x": "10", "xi": "11", "xii": "12"}
_QUAL = re.compile(r"\b(sector|sec|phase|ph|block|blk|stage|stg|pocket|pkt|ward|zone|extn|extension)\s*"
                   r"[-.:#]?\s*(?:no\.?\s*)?(?P<v>[ivx]{1,4}|\d{1,3}[a-z]?|[a-z])\b(?:\s*-?\s*(?P<v2>[a-z])\b)?", re.I)
_QUAL_REV = re.compile(r"\b(?P<v>\d{1,3}|[a-z])\s*(?:st|nd|rd|th)?\s*(block|phase|stage|sector)\b", re.I)
_QUAL_CANON = {"sec": "sector", "ph": "phase", "blk": "block", "stg": "stage", "pkt": "pocket",
               "extn": "extension"}
_STREET_WORDS = {
    "India": "road|rd|street|st|marg|lane|ln|path|gali|galli|cross|main road|highway|hwy|bypass|salai|veedhi|veethi|rasta",
    "US": "road|rd|street|st|lane|ln|highway|hwy|avenue|ave|drive|dr|drve|boulevard|blvd|court|ct|place|pl|parkway|"
          "pkwy|circle|cir|way|trail|trl|terrace|ter|pike|square|sq|loop|run|pass|path|row|alley|expressway|expy|"
          "freeway|fwy|turnpike|tpke",
    "France": "rue|r|avenue|av|ave|boulevard|bd|bld|chemin|ch|route|rte|impasse|imp|allee|all|quai|cours|place|pl|"
              "square|sq|voie|passage|chaussee|sentier|faubourg|fbg|cite|residence|lotissement|lot",
}
_STREET_DEFAULT = "|".join(sorted(set("|".join(_STREET_WORDS.values()).split("|"))))
_STREET = {c: re.compile(r"\b(" + w + r")\b", re.I) for c, w in _STREET_WORDS.items()}


def _street_re(country):
    return _STREET.get(country) or _STREET.setdefault(country, re.compile(r"\b(" + _STREET_DEFAULT + r")\b", re.I))
_BUILDING = re.compile(r"\b(tower|towers|plaza|complex|apartment|apartments|apts|building|bldg|bhavan|bhawan|"
                       r"house|chambers|arcade|mall|residency|regency|heights|mansion|manor|villa|villas|chs|"
                       r"co-?op|hsg|niwas|nivas|sadan|kunj|mahal|palace|centre|center|hub|tech park|it park|"
                       r"business park|trade centre|trade center|estate|enclave apartments)\b", re.I)
_ADMIN = re.compile(r"^\s*(dist|distt|district|tal|taluka|taluk|tq|tehsil|teh|mandal|p\.?\s?o|post|ps|via)\b"
                    r"\.?\s*[-:.]?\s*(?P<v>.*)$", re.I)
_VILLAGE = re.compile(r"^\s*(vill|village|vil|vii|gram|grm|at)\b\.?\s*[-:.]?\s*(?P<v>.*)$", re.I)
_UNIT = re.compile(r"(?:\b(?:unit|apt|apartment|suite|ste|rm|room|spc|space)\b|#)\s*(?:unit|apt|apartment|suite|ste|"
                   r"#)?\s*#?\s*(?P<id>[a-z]?-?\d[\w-]*|[a-z])\b", re.I)
_POBOX = re.compile(r"\bp\.?\s?o\.?\s?box\s*(?P<id>\d+)", re.I)
_DIR_SUFFIX = re.compile(r"\((w|e|n|s|west|east|north|south)\)", re.I)
_DIRS = {"w": "west", "e": "east", "n": "north", "s": "south"}

# abbreviation expansion applied to street/locality/building text (after basic()), per country
_ABBR_COMMON = {"rd": "road", "st": "street", "ln": "lane", "hwy": "highway", "sq": "square", "apts": "apartments",
                "bldg": "building", "stn": "station"}
_ABBR = {
    "India": {**_ABBR_COMMON, "mg": "mahatma gandhi", "ngr": "nagar", "nagr": "nagar", "colany": "colony",
              "extn": "extension", "ext": "extension", "indl": "industrial", "inds": "industrial", "ind": "industrial",
              "mkt": "market", "rly": "railway", "govt": "government", "sec": "sector", "blk": "block",
              "apt": "apartments", "opp": "opposite"},
    "US": {**_ABBR_COMMON, "ave": "avenue", "av": "avenue", "dr": "drive", "drve": "drive", "blvd": "boulevard",
           "ct": "court", "pl": "place", "pkwy": "parkway", "cir": "circle", "trl": "trail", "ter": "terrace",
           "expy": "expressway", "fwy": "freeway", "tpke": "turnpike", "mt": "mount", "ft": "fort",
           "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
           "se": "southeast", "sw": "southwest"},
    "France": {"r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "bld": "boulevard", "ch": "chemin",
               "rte": "route", "imp": "impasse", "all": "allee", "pl": "place", "sq": "square", "fbg": "faubourg",
               "st": "saint", "ste": "sainte"},
}
_ORD = re.compile(r"\b(\d+)\s+(st|nd|rd|th)\b")


def canon_text(s: str, country: str = "") -> str:
    """basic() + ordinal glue ('1 st' -> '1st') + country abbreviation expansion + '(W)' -> 'west'.
    A leading US 'St' followed by more words is 'saint' (St Louis), elsewhere it is 'street'."""
    s = _DIR_SUFFIX.sub(lambda m: " " + _DIRS.get(m.group(1).lower(), m.group(1).lower()), s)
    s = basic(romanize(s))
    s = _ORD.sub(r"\1\2", s)
    toks = s.split()
    abbr = _ABBR.get(country, _ABBR_COMMON)
    return " ".join("saint" if (t == "st" and i == 0 and len(toks) > 1 and country == "US") else abbr.get(t, t)
                    for i, t in enumerate(toks))


def canon_id(raw: str) -> Optional[str]:
    """'B-53A' -> 'b/53a', '0203' -> '203', '6-3-1239/2' -> '6/3/1239/2', '504 / A' -> '504/a'."""
    s = re.sub(r"\s+", "", raw.lower())
    s = re.sub(r"[-.\\]+", "/", s).strip("/")
    s = re.sub(r"/+", "/", s)
    parts = [p.lstrip("0") or "0" if p.isdigit() else p for p in s.split("/")]
    s = "/".join(p for p in parts if p)
    return s or None


def _is_plain(c: str, country: str) -> bool:
    return bool(c) and not re.search(r"\d", c) and not _LANDMARK.search(c) and not _CAREOF.match(c) \
        and not _street_re(country).search(c) and not _ADMIN.match(c) and not _BUILDING.search(c)


# ---------------------------------------------------------------- main parser
_FIELDS = ["state", "city", "locality", "street", "building", "premise_no", "premise_typed", "unit", "floor",
           "landmark", "admin", "careof", "pin", "locality_key", "street_key", "building_key"]


def _components(addr: str):
    out = []
    for c in addr.split(","):
        c = c.strip()
        if not c or c.lower() in _NULLS:
            continue
        if has_indic(c):
            c = romanize(c)
        out.append(strip_accents(c))
    return out


def locate(comps, country):
    """(state, state_index, city) from components. City = neighbour of the state, else last plain comp."""
    st_idx, state = None, None
    # an explicit 2-letter code wins over a state *name*: "DC, WASHINGTON" and "NY, Alabama" name cities
    for codes_only in (True, False):
        for i in range(len(comps) - 1, -1, -1):
            if codes_only != bool(re.fullmatch(r"\s*[A-Za-z]{2}\s*", comps[i])):
                continue
            s = detect_state(comps[i], country)
            if s:
                st_idx, state = i, s
                break
        if state:
            break
    city = None
    if st_idx is not None:
        if st_idx > 0 and _is_plain(comps[st_idx - 1], country) and not detect_state(comps[st_idx - 1], country):
            city = comps[st_idx - 1]
        elif st_idx == 0 and len(comps) > 1 and _is_plain(comps[1], country):
            city = comps[1]
    if city is None:
        for i in range(len(comps) - 1, -1, -1):
            if i != st_idx and _is_plain(comps[i], country) and not detect_state(comps[i], country):
                city = comps[i]
                break
    return state, st_idx, canon_city(city, country)


def geo(addr: str, country: str):
    """Fast path used for DEV sampling: (state, city) only."""
    comps = _components(addr)
    state, _, city = locate(comps, country)
    return state, city


@lru_cache(maxsize=200_000)
def parse_address(addr: str, country: str) -> dict:
    out = {f: [] for f in _FIELDS}
    comps = _components(addr)
    state, st_idx, city = locate(comps, country)
    for i, c in enumerate(comps):
        if i == st_idx:
            continue
        if city and _is_plain(c, country) and canon_city(c, country) == city:
            continue  # every spelling of the city ("Mumbai", "Mumbai City") is the city, not a locality
        _parse_component(c, country, out)
    res = {k: ("|".join(dict.fromkeys(v)) if v else None) for k, v in out.items()}
    res["state"], res["city"] = state, city
    for f in ("locality", "street", "building"):
        vals = out[f]
        res[f + "_key"] = "|".join(dict.fromkeys(phonetic(v) for v in vals)) if vals else None
    return res


_LEFTOVER = {"fl", "floor", "flr", "unit", "apt", "suite", "ste", "no", "dist", "tal"}


def _add_id(out, kind, raw):
    cid = canon_id(raw)
    if cid:
        out["premise_no"].append(cid)
        out["premise_typed"].append(f"{kind}:{cid}")


def _parse_component(c: str, country: str, out: dict) -> None:
    m = _CAREOF.match(c)
    if m:
        v = canon_text(c[m.end():], country)
        if v:
            out["careof"].append(v)
        return
    m = _LANDMARK.search(c)
    if m:
        lm = canon_text(c[m.end():], country)
        if lm:
            out["landmark"].append(lm)
        c = c[:m.start()]
    m = _POBOX.search(c)
    if m:
        out["unit"].append("pobox " + m.group("id").lstrip("0"))
        c = c[:m.start()] + " " + c[m.end():]
    if country == "India":
        m = _PIN.search(c)
        if m and len(re.sub(r"\D", "", m.group(1))) == 6 and not re.search(r"[/-]" + re.escape(m.group(1)), c):
            out["pin"].append(m.group(1).replace(" ", ""))
            c = c[:m.start()] + " " + c[m.end():]
    for m in list(_FLOOR.finditer(c)):
        v = (m.group(1) or m.group(2) or "").lower()
        out["floor"].append(_FLOOR_WORD.get(v, v.lstrip("0") or "0"))
    c = _FLOOR.sub(" ", c)
    m = _ADMIN.match(c)
    if m:
        v = canon_text(m.group("v"), country)
        if v:
            out["admin"].append(v)
        return
    m = _VILLAGE.match(c)
    if m and m.group("v").strip():
        c = m.group("v")
    if country != "India":
        for m in list(_UNIT.finditer(c)):
            out["unit"].append(canon_id(m.group("id")) or m.group("id").lower())
        c = _UNIT.sub(" ", c)
    # sector / phase / block / ward qualifiers (before keyword ids, so 'Ward No 5' stays a locality)
    for m in list(_QUAL.finditer(c)):
        kind = _QUAL_CANON.get(m.group(1).lower(), m.group(1).lower())
        v = m.group("v").lower()
        v = _ROMAN.get(v, v) if kind in ("phase", "stage", "part") or len(v) > 1 else v
        out["locality"].append(f"{kind} {v}{m.group('v2').lower() if m.group('v2') else ''}")
    c = _QUAL.sub(" ", c)
    for m in list(_QUAL_REV.finditer(c)):
        out["locality"].append(f"{m.group(2).lower()} {m.group('v').lower()}")
    c = _QUAL_REV.sub(" ", c)
    # keyword ids: "Plot No 35B", "H.No 442", "Flat 104", "No. 78", "#246"
    for m in list(_KW_ID.finditer(c)):
        kw = (m.group("kw") or m.group("kw2") or "").lower()
        _add_id(out, _KW_TYPE.get(kw, "no"), m.group("id"))
    c = _KW_ID.sub(" ", c)
    # leading bare id: "9062/17 Rani Bagh Road", "B-84", "33/3115 C Paroppady", "1795 Westchester Drive"
    m = _LEAD_ID.match(c)
    lead_house = False
    if m and re.search(r"\d", m.group("id")) and not re.fullmatch(r"\d+(st|nd|rd|th)", m.group("id"), re.I):
        _add_id(out, "house" if country != "India" else "num", m.group("id"))
        c = c[m.end():]
        lead_house = country != "India"   # US/FR: "<number> <street>"
    text = canon_text(c, country)
    text = re.sub(r"^(no|number)\b\s*", "", text).strip()
    text = " ".join(t for t in text.split() if t not in _LEFTOVER)
    if not text or text.isdigit():
        if text.isdigit():
            _add_id(out, "num", text)
        return
    if lead_house:
        out["street"].append(text)
    elif _BUILDING.search(c):
        out["building"].append(text)
    elif _street_re(country).search(c) and not re.fullmatch(r"(.* )?(nagar|colony|layout|vihar|enclave|market|bazar|bazaar|"
                                                r"area|park|garden|gardens)", text):
        out["street"].append(text)
    else:
        out["locality"].append(text)
