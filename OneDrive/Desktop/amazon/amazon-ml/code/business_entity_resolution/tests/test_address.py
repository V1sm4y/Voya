from ber.address import canon_id, detect_state, parse_address


def f(addr, country="India"):
    return parse_address(addr, country)


def test_india_premise_locality_street_admin():
    r = f("Bldg No.1, Fl No.5, F No.504/A, Lotus Sanskruti, Sr No.74/2, Kiwale Dehu Road, Tal Haweli, Dist, Pune, Maharashtra")
    assert r["state"] == "maharashtra" and r["city"] == "pune"
    assert r["premise_typed"] == "bldg:1|flat:504/a|survey:74/2"
    assert r["floor"] == "5" and r["admin"] == "haweli"
    assert r["street"] == "kiwale dehu road" and r["locality"] == "lotus sanskruti"


def test_india_sector_phase_block():
    r = f("No 10 Enkay Square, 448A, Udyog Vihar Phase V, Gurugram, Gurgaon, HR")
    assert r["state"] == "haryana" and r["city"] == "gurugram"
    assert "phase 5" in r["locality"].split("|") and "udyog vihar" in r["locality"].split("|")
    assert "gurgaon" not in (r["locality"] or "")          # alias of the city, not a locality
    r = f("Office No 154, Satra Plaza, Plot-20, Sector-19D, Palm Beach Road, Vashi, Navi Mumbai, Raigarh, Maharashtra")
    assert r["premise_typed"] == "office:154|plot:20" and r["building"] == "satra plaza"
    assert "sector 19d" in r["locality"].split("|")
    assert "block 1" in f("1St Block, Rajajinagar, Bengaluru, Karnataka")["locality"].split("|")


def test_india_landmark_careof_direction():
    r = f("Stores No 13, Flat 502, Vi Sunder Nagar S V Road Near Dalmiha College, Malad (W), Mumbai, Mumbai City, MH")
    assert r["landmark"] == "dalmiha college" and r["city"] == "mumbai"
    assert r["locality"] == "malad west" and "s v road" in r["street"]
    r = f("C/O- Yogika Gautam, Vii.- Barola, Dadri, Gautam Buddha Nagar, Uttar Pradesh")
    assert r["careof"] == "yogika gautam" and "barola" in r["locality"]


def test_india_indic_state_and_ids():
    assert f("DSS-68 SECTOR 15-A, HISAR, null, हरियाणा")["state"] == "haryana"
    assert f("2/3 SARAT BOSE ROAD, KOLKATA, পশ্চিমবঙ্গ")["state"] == "west bengal"
    assert f("കേരളം, Marikunnu, Kozhikode, 33/3115 C Paroppady")["state"] == "kerala"
    assert f("Plot No-#3 Kh-1130/14, Shankar Vihar-1 Lal Kuan, Ghaziabad, UP")["premise_typed"] == "plot:3|khasra:1130/14"
    assert f("Lucknow, Uttar Pradesh, L/D-6, Lucknow, Sector-1, Gomti Nagar Vistar")["premise_no"] == "l/d/6"


def test_same_locality_across_spellings_and_scripts():
    a = f("12, Kancheepuram Road, Chennai, Tamil Nadu")
    b = f("12, KANCHIPURAM RD, CHENNAI, தமிழ்நாடு")
    assert a["street_key"] == b["street_key"] and a["state"] == b["state"] == "tamil nadu"


def test_us_and_france():
    r = parse_address("7400 Gainey Club Drive, Unit UNIT 136, Scottsdale, AZ", "US")
    assert (r["state"], r["city"], r["street"], r["premise_no"], r["unit"]) == \
        ("arizona", "scottsdale", "gainey club drive", "7400", "136")
    r = parse_address("MINNEAPOLIS, 424 HUMBOLDT AVE, MN", "US")
    assert (r["city"], r["street"], r["premise_no"]) == ("minneapolis", "humboldt avenue", "424")
    assert parse_address("2500 Q ST, DC, WASHINGTON", "US")["state"] == "district of columbia"
    assert parse_address("IA, Iowa City, 1064 Newton Rd, Unit 11", "US")["city"] == "iowa city"
    r = parse_address("63 R. DE DIEPPE, LILLE, Hauts-de-France", "France")
    assert (r["state"], r["city"], r["street"]) == ("hauts de france", "lille", "rue de dieppe")
    assert parse_address("18 RUE JEN ZAY, Dunkerque, Nord", "France")["state"] == "hauts de france"


def test_short_state_keys_do_not_collide():
    assert detect_state("Bihar", "India") == "bihar"
    assert detect_state("Pur", "India") is None and detect_state("Bera", "India") is None


def test_canon_id():
    assert canon_id("B-53A") == "b/53a"
    assert canon_id("0203") == "203"
    assert canon_id("6-3-1239/2") == "6/3/1239/2"
