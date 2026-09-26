import pytest

from ber.normalize import strip_accents
from ber.translit import phonetic, romanize

SAME = [  # (Indic or variant spelling, Latin) pairs that must share a phonetic key
    ("लिमिटेड", "limited"), ("ಲಿಮಿಟೆಡ್", "limited"), ("லிமிடெட்", "limited"), ("ലിമിറ്റഡ്", "limited"),
    ("ଲିମିଟେଡ୍", "limited"), ("प्राइवेट", "private"), ("ప్రైవేట్", "private"), ("প্রাইভেট", "private"),
    ("પ્રાઇવેટ", "private"), ("പ്രൈവറ്റ്", "private"),
    ("महाराष्ट्र", "maharashtra"), ("ಕರ್ನಾಟಕ", "karnataka"), ("தமிழ்நாடு", "tamil nadu"), ("ગુજરાત", "gujarat"),
    ("తెలంగాణ", "telangana"), ("राजस्थान", "rajasthan"), ("ఆంధ్రప్రదేశ్", "andhra pradesh"), ("ਪੰਜਾਬ", "punjab"),
    ("राम मार्केटिंग", "ram marketing"), ("मॉडर्न फाउंडेशन", "modern foundation"), ("इंटरनेशनल", "international"),
    ("ജൈൻ എന്റർപ്രൈസസ്", "jain enterprises"), ("लक्ष्मी", "laxmi"), ("श्री", "shri"),
    ("Kancheepuram", "Kanchipuram"), ("Haweli", "Haveli"), ("Kozhikode", "കോഴിക്കോട്"), ("Chennai", "சென்னை"),
]


@pytest.mark.parametrize("a,b", SAME)
def test_cross_script_keys_match(a, b):
    assert phonetic(a).replace(" ", "") == phonetic(b).replace(" ", "")


def test_distinct_words_differ():
    assert phonetic("mumbai") != phonetic("madurai")
    assert phonetic("private") != phonetic("public")


def test_romanize_schwa_and_passthrough():
    assert romanize("राम") == "raam"          # long aa kept, final inherent a dropped
    assert romanize("कमल") == "kamal"
    assert romanize("Plot 5, राम") == "Plot 5, raam"
    assert romanize("Main Road") == "Main Road"


def test_malayalam_rra_conjuncts():
    assert romanize("ലിമിറ്റഡ്") == "limittad"      # റ്റ -> tt
    assert "nt" in romanize("എന്റർപ്രൈസസ്")        # ന്റ -> nt


def test_strip_accents_keeps_indic_virama():
    assert strip_accents("प्राइवेट") == "प्राइवेट"
    assert strip_accents("Àmicale Çentre") == "Amicale Centre"
