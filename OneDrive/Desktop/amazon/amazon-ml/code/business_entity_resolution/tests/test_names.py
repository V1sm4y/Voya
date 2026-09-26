import pytest

from ber.names import legal_relation, parse_name


@pytest.mark.parametrize("name,core,form", [
    ("Crawley Allied Muniyield L.L.C.", "crawley allied muniyield", "llc"),
    ("Crawley Allied Muniyield Corp", "crawley allied muniyield", "corp"),
    ("Green & Auclair L.L.C.", "green auclair", "llc"),
    ("AURIFY CALAMOS (CORP.)", "aurify calamos", "corp"),
    ("3424 Mountain Vwnba Drive Management [Inc]", "3424 mountain vwnba drive management", "inc"),
    ("Wonderful Co-&", "wonderful", "co"),
    ("Infrastructure Rab Infolabs Public Limited", "infrastructure rab infolabs", "ltd+public"),
    ("Good Exports Private Limited", "good exports", "ltd+pvt"),
    ("Maison Adi SAS", "maison adi", "sas"),
    ("LLC Moncada Léarning Center", "moncada learning center", "llc"),
    ("Private Limited", None, "ltd+pvt"),
    ("HITECHENGINEERING.COM", "hitechengineering", ""),
    ("trustaguilar.com", "trustaguilar", ""),
    ("Jaipur Traders Pvt. Ltd.", "jaipur traders", "ltd+pvt"),
])
def test_core_and_legal_form(name, core, form):
    r = parse_name(name)
    assert r["core_name"] == core
    assert r["legal_form"] == form


def test_indic_legal_forms():
    assert parse_name("राम मार्केटिंग प्राइवेट लिमिटेड")["legal_form"] == "ltd+pvt"
    assert parse_name("राम मार्केटिंग प्रा. लि.")["legal_form"] == "ltd+pvt"
    assert parse_name("आदित्य प्रॉपर्टीज एलएलपी")["legal_form"] == "llp"
    assert parse_name("ജൈൻ എന്റർപ്രൈസസ് ലിമിറ്റഡ്")["legal_form"] == "ltd"
    r = parse_name("राम मार्केटिंग प्राइवेट लिमिटेड")
    assert r["core_key"] == parse_name("Ram Marketing Private Limited")["core_key"]


def test_alias_split():
    r = parse_name("Beloarcgild formerly MB Milk Private Limited")
    assert r["core_name"] == "beloarcgild" and r["alias"] == "mb milk"


@pytest.mark.parametrize("a,b,rel", [
    ("llc", "corp", "conflict"), ("llc", "llc", "same"), ("", "llc", "missing"),
    ("ltd+pvt", "ltd", "subset"), ("pvt", "ltd+pvt", "same"), ("ltd+pvt", "ltd+public", "overlap"),
    ("llc", "co+llc", "subset"),
])
def test_legal_relation(a, b, rel):
    assert legal_relation(a, b) == rel
