"""String normalisation shared by blocking, features and baselines.

Name keys live in names.py (core_name + legal_form, never a key with the legal form silently dropped).

The geography lists are hand-written general knowledge (state and region names, not a lookup
service), so they are allowed under the no-external-data rule.
"""
import re
import unicodedata
from typing import Optional

US_STATES = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas", "CA": "california", "CO": "colorado",
    "CT": "connecticut", "DE": "delaware", "FL": "florida", "GA": "georgia", "HI": "hawaii", "ID": "idaho",
    "IL": "illinois", "IN": "indiana", "IA": "iowa", "KS": "kansas", "KY": "kentucky", "LA": "louisiana",
    "ME": "maine", "MD": "maryland", "MA": "massachusetts", "MI": "michigan", "MN": "minnesota",
    "MS": "mississippi", "MO": "missouri", "MT": "montana", "NE": "nebraska", "NV": "nevada",
    "NH": "new hampshire", "NJ": "new jersey", "NM": "new mexico", "NY": "new york", "NC": "north carolina",
    "ND": "north dakota", "OH": "ohio", "OK": "oklahoma", "OR": "oregon", "PA": "pennsylvania",
    "RI": "rhode island", "SC": "south carolina", "SD": "south dakota", "TN": "tennessee", "TX": "texas",
    "UT": "utah", "VT": "vermont", "VA": "virginia", "WA": "washington", "WV": "west virginia",
    "WI": "wisconsin", "WY": "wyoming", "DC": "district of columbia", "PR": "puerto rico",
}
IN_STATES = {
    "AP": "andhra pradesh", "AR": "arunachal pradesh", "AS": "assam", "BR": "bihar", "CG": "chhattisgarh",
    "CT": "chhattisgarh", "GA": "goa", "GJ": "gujarat", "HR": "haryana", "HP": "himachal pradesh",
    "JH": "jharkhand", "KA": "karnataka", "KL": "kerala", "MP": "madhya pradesh", "MH": "maharashtra",
    "MN": "manipur", "ML": "meghalaya", "MZ": "mizoram", "NL": "nagaland", "OD": "odisha", "OR": "orissa",
    "PB": "punjab", "RJ": "rajasthan", "SK": "sikkim", "TN": "tamil nadu", "TG": "telangana",
    "TS": "telangana", "TR": "tripura", "UP": "uttar pradesh", "UK": "uttarakhand", "UT": "uttarakhand",
    "WB": "west bengal", "AN": "andaman and nicobar islands", "CH": "chandigarh", "DL": "delhi",
    "JK": "jammu and kashmir", "LA": "ladakh", "LD": "lakshadweep", "PY": "puducherry",
    "DN": "dadra and nagar haveli and daman and diu",
}
IN_STATE_ALIASES = {"keralam", "uttaranchal", "pondicherry", "nct of delhi", "chhatisgarh"}
FR_REGIONS = {
    "auvergne-rhone-alpes", "bourgogne-franche-comte", "bretagne", "centre-val de loire", "corse",
    "grand est", "hauts-de-france", "ile-de-france", "normandie", "nouvelle-aquitaine", "occitanie",
    "pays de la loire", "provence-alpes-cote d'azur",
    # departments of the regions seen in test (DATA_NOTES §4.1)
    "nord", "pas-de-calais", "somme", "oise", "aisne", "gironde", "landes", "dordogne", "lot-et-garonne",
    "pyrenees-atlantiques", "charente", "charente-maritime", "vienne", "deux-sevres", "haute-vienne",
    "creuse", "correze", "loire-atlantique", "maine-et-loire", "vendee", "sarthe", "mayenne",
}
_STATE_WORDS = ({v for v in US_STATES.values()} | {v for v in IN_STATES.values()} | IN_STATE_ALIASES
                | FR_REGIONS)

_NON_ALNUM = re.compile(r"[^0-9a-zऀ-෿]+")
_DIGIT = re.compile(r"\d")
_UNITLIKE = re.compile(r"^(unit|apt|suite|ste|fl|floor|#|po box|p\.?o\.? box)\b", re.I)
_NULLS = {"null", "<null>", "n/a", "na", "none", "nan"}


def strip_accents(s: str) -> str:
    """Drop Latin diacritics only (U+0300–U+036F). Indic virama/nukta are also 'combining' and must survive."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not "̀" <= c <= "ͯ")


def basic(s: str) -> str:
    """lower + strip accents + punctuation to space + collapse spaces ('.' removed, so L.L.C. -> llc)."""
    s = strip_accents(s.lower()).replace(".", "").replace("'", "")
    return " ".join(_NON_ALNUM.sub(" ", s).split())



_STATE_KEYS = {basic(w) for w in _STATE_WORDS}  # basic() maps '-' to ' ', so both spellings collapse


def is_state(component: str) -> bool:
    c = component.strip()
    return bool(re.fullmatch(r"[A-Z]{2}", c)) or basic(c) in _STATE_KEYS


def address_components(addr: str) -> list:
    return [c.strip() for c in addr.split(",") if c.strip() and c.strip().lower() not in _NULLS]


def s1_city(addr: str) -> Optional[str]:
    """City of a (clean) S1 address: the last component that has no digits and is not a state
    or unit. Returns the basic()-normalised form, or None."""
    for c in reversed(address_components(addr)):
        if _DIGIT.search(c) or _UNITLIKE.match(c) or is_state(c):
            continue
        b = basic(c)
        if b:
            return b
    return None


def contains_phrase(haystack_basic: str, needle_basic: str) -> bool:
    """Word-boundary containment on basic()-normalised strings."""
    return bool(needle_basic) and f" {needle_basic} " in f" {haystack_basic} "
