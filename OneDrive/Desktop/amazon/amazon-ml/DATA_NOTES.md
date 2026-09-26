# DATA_NOTES — Amazon ML Challenge 2026: Business Entity Resolution

EDA of `student_resource/` (2026-09-25). No model has been built yet. All numbers come from the full files unless marked *sample*.
Files were read with `pd.read_csv(sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)`. Every file parsed with rows = lines − 1, so there are no quoting problems.

## TL;DR: facts that should drive the design

1. **The mapping is a partition.** Every S2/S3 id appears in **at most one** S1 match set (7,638,365 links, 7,638,365 distinct ids). So a one-S2/S3-to-at-most-one-S1 assignment constraint is valid and is a strong precision lever.
2. **Many matches from the same source is the norm.** 76.8% of S1 entities match ≥2 records from S2 or from S3. The mean match-set size is 3.46 (3.67 for non-singletons), and the max is 11.
3. **Singletons are rare (5.58%)** in train, and the rate is the same for US and India. Predicting all-empty would score ≈0.056.
4. **About 26% of S2/S3 records match no S1** (1.34M in each source), and many of them look like **deliberately planted decoys**: the same name ± one word, a swapped legal form, or a house number shifted by 1–15 on the same street. In a hard-negative search, 1,386 of 1,585 hard negatives were unlinked records. Precision depends on catching these.
5. **The country label always agrees between S1 and its matches** (100.000% of links), so it can be used as a hard blocking key. **Test adds `France`** (15% of test S1), which never appears in train.
6. **Test has more S2/S3 records per S1 than train** (5.5–5.8 vs 4.67). The test will therefore have larger clusters and/or more decoys. Priors tuned on train (cluster size, thresholds) may shift.
7. **Name alone is insufficient.** 13–24% of India matches have their name in an Indic script (Devanagari, Bengali, Malayalam, Odia, …). About 5–8% of Latin-script matches have a completely different name (synthetic brand like "Fluxveo", initialism "DCH", domain "taolabs.com", handle "@SWEETHAIR"). These can only be matched through the address.
8. **S1 is clean and consistently formatted** (Title Case, full street words, US state codes, full Indian state names, no empties). S2 and S3 each have a distinct "house style" (below), so normalisation should be source-aware.

---

## 1. Files, schema, row counts, null rates

```
student_resource/
├── README.md                    # problem statement (authoritative spec)
├── Documentation_template.md    # methodology write-up template (goes in the final zip)
├── utils/validate_submission.py # stdlib-only format validator
└── dataset/
    ├── train/ train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
    └── test/  test_source1.tsv   test_source2.tsv   test_source3.tsv
```

**Source files** all have the schema `entity_id, business_name, business_address, country`, all strings. There is no source column: the source is the `S1-/S2-/S3-` id prefix, and 100% of ids carry the right prefix. Ids are unique within every file. The numeric part of an id is random and carries no information.
**Ground truth** has the schema `source1_entity_id, matched_entity_ids`, where the second column is a comma-separated list with no spaces, and an empty string means no match.

| File | Rows | Size | Country counts | Empty name | Empty address | Exact dup (name,addr,country) |
|---|---:|---:|---|---:|---:|---:|
| train_source1 | 2,206,821 | 210 MB | US 1,323,633 · India 883,188 | 0% | 0% | 0 |
| train_source2 | 5,034,616 | 489 MB | US 3,016,817 · India 2,017,799 | 0% | 3.36% (US 3.68, IN 2.87) | 25,873 |
| train_source3 | 5,285,603 | 504 MB | US 3,170,056 · India 2,115,547 | 0% | 3.33% (US 3.50, IN 3.07) | 18,860 |
| train_ground_truth | 2,206,821 | 127 MB | one row per train S1 (id set identical to train_source1) | – | – | – |
| test_source1 | 1,732,544 | 175 MB | India 809,986 · US 663,106 · **France 259,452** | 0% | 0% | 0 |
| test_source2 | 4,887,273 | 509 MB | India 2,312,565 · US 1,871,330 · France 703,378 | 0% | 2.65% | 22,641 |
| test_source3 | 5,082,316 | 506 MB | India 2,405,000 · US 1,945,701 · France 731,615 | 0% | 2.68% | 16,293 |

Notes on nulls:
- The `country` column is never empty. `business_name` is never empty, but 0.001% of test S2/S3 names are placeholder-like.
- **Hidden nulls inside addresses:** the tokens `null`, `NULL`, `<NULL>` and `N/A` appear as a comma component in about 3–4% of S2/S3 addresses (never in S1). Example: `34 PUTNAM ST, <NULL>, HARTFORD, CT`. Strip them before any similarity computation.
- **Missing components:** 33–35% of test France S2/S3 addresses have ≤2 comma parts (region/department dropped). This drops to about 3% for US and India. Missing house numbers, truncated numbers and missing cities are common in S2/S3 (see §4).
- String lengths: names are 24–26 characters on average (max about 123), and addresses are 46–57 characters (max about 269).
- **Train and test are the same distribution except for France.** Both the US and India rates in §4 match to within about 1 percentage point between train and test.

| Split | S2+S3 records per S1 (India / US / France) |
|---|---|
| train | 4.68 / 4.67 / – |
| test | 5.82 / 5.76 / 5.53 |

### 10 sample rows per source (random, seed 7)

#### train_source1.tsv

| entity_id | business_name | business_address | country |
|---|---|---|---|
| S1-143205451 | Seal & Sons Associates | Bldg No.1, Fl No.5, F No.504/A, Lotus Sanskruti, Sr No.74/2, Kiwale Dehu Road, Tal Haweli, Dist, Pune, Maharashtra | India |
| S1-697347685 | Jackson Total Starry Inc | 2430 23rd Street, Henrico County, VA | US |
| S1-832591575 | Good Exports Private Limited | Shop No. 1, Basement Floor, United Medicine Complex, Reti Street, Moradabad, Uttar Pradesh | India |
| S1-105982737 | Electricians Local 210 | 1275 5, Unit Unit 52, Elbridge, NY | US |
| S1-451092852 | Kackley Minerals | 805 Ford Street, Joiner, AR | US |
| S1-119103715 | Celebrity Management (India) Private Limited | 9062/17 Rani Bagh Road Pul Bangesh, Azad Market, New Delhi, Delhi | India |
| S1-675888289 | Al Apex Infrastructure Private Limited | No. 78, 3Rd Main Road, Ramanjaneya Layout Marathalli, Bengaluru, Bangalore, Karnataka | India |
| S1-685157977 | Hester Kim Optimal Atlanta, LLC | 7400 Gainey Club Drive, Unit UNIT 136, Scottsdale, AZ | US |
| S1-996022607 | C/B Pimco LP | 2583 Birch Court, Milton, WA | US |
| S1-162343011 | Toify LLC | 1004 Marlin Street, Village Of Holmen, WI | US |

#### train_source2.tsv

| entity_id | business_name | business_address | country |
|---|---|---|---|
| S2-87432955 | WEBSTER APEX OMEGA, CORPORATION | 65 PENNSYLVANIA AVENUE, LONG BEACH, NY | US |
| S2-84120274 | Inc Thomas and Dgvei | 5619 REGENCY PARK CT, SUITLAND, MD | US |
| S2-298933248 | Holdings Atlantic Project | 40 CLOVELLY WAY, ASHEVILLE, NC | US |
| S2-293805369 | The Ear Nose & Throat Choice Associates LLC | 280 2ND ST, LA GRANDE, OR | US |
| S2-281300882 | NAVARRO  INC DRUGS | 1931 HORSESHOE RD, WESTFIELD, NC | US |
| S2-395587088 | Onyxwex | 34 PUTNAM ST, <NULL>, HARTFORD, CT | US |
| S2-381586774 | Jaipur Traders Private Ltd | DSS-68 SECTOR 15-A, HISAR, null, हरियाणा | India |
| S2-357839016 | Urology Specialists Partners White of | 2334 OAKLAND AVENUE, WHITE, PA | US |
| S2-519640855 | Stenger Midwest Home Corp | 5521 KEYWAY BLVD, CHARLOTTE, NC | US |
| S2-650203340 | Houston Borough Veterans Charities Inc | 331-B GRANT ST, HOUSTON, PA | US |

#### train_source3.tsv

| entity_id | business_name | business_address | country |
|---|---|---|---|
| S3-320159684 | Premier Ventures General  Pvt Ltd | No 112 Springboard Business Hub Private Limited, B-wing 5 Th Floor, Ackruti Trade Centre, Midc, A, Ndheri (E), Mumbai, Mumbai City, MH | India |
| S3-534674466 | -- Historical Sfhicety III | 505-509 Brindle Rd, NULL, Ostrander, Ohio | US |
| S3-341747108 | Smt Banerjee Welfare Sóciety | No 05, Software Units Layout, Madhapur, Hyderabad, NULL, Andhra Pradesh | India |
| S3-786239957 | Chrysolite Management  Pvt | Plot No-#3 Kh-1130/14, Shankar Vihar-1 Lal Kuan, Ghaziabad, UP | India |
| S3-296482968 | Beloarcgild formerly MB Milk Private Limited | 163-A Rest Camp Colonycharbagh Lucknow, Uttar Pradesh, Lucknow, Uttar Pradesh | India |
| S3-696396762 | Dental Associates | 1278 1/2 State Highway 37, Minco, Oklahoma | US |
| S3-857726215 | Fayeyumaio F/K/A Seguin Preparatory Academy | 2017 Bailey Avenue, Richmond City, VA | US |
| S3-915849742 | Urgent Care Star Specialists Ltd | 70-16 60 Avenue, null, Maspeth, New York | US |
| S3-359508363 | Sky-Hospitality | Shop-204, WB, Darjiling, Darjeeling | India |
| S3-501574263 | Velvet Lozano, DO South LP | #1117 Harvest Mill Ct, Raleigh, North Carolina | US |

#### test_source1.tsv

| entity_id | business_name | business_address | country |
|---|---|---|---|
| S1-937916806 | Decision Construction Private Limited | No. 21/202K, 1St Floor, Emjay Towers Railway Station Road, Angamaly, Ernakulam, Kerala | India |
| S1-900566251 | Pediatric Care Associates Inc. | 40 Eagle Street, North Adams, MA | US |
| S1-790112362 | Prevention Compagnie (France) SARL | 15 Place Viarme, Nantes, Pays de la Loire | France |
| S1-67268817 | Maison Adi SAS | 3 AV des Amandines, Nantes, Pays de la Loire | France |
| S1-418813176 | Ets Bellegarde SAS | 3319 AV de la Liberation, Dunkerque, Hauts-de-France | France |
| S1-577542272 | Jai It Limited | Flat No.10 Archit Regency, Near Nava Ranchana School, Nashik, Maharashtra | India |
| S1-25127202 | American Eco LLC | 8373 De Ventura, Unit APT 226, Scottsdale, AZ | US |
| S1-613837207 | Swastik Hitech Impex | No 14 Sudha Avenue 5 Th Street Chitlapakkam, Kancheepuram, Chennai, Tamil Nadu | India |
| S1-993746850 | Mcmahon, Mcmillon and Miles LLC | Sophia, 2822 Sawyer Road, NC | US |
| S1-920162272 | Metropolitan Network | 12220 5th Street, Unit UNIT 156, Yucaipa, CA | US |

#### test_source2.tsv

| entity_id | business_name | business_address | country |
|---|---|---|---|
| S2-683567035 | प्राइम सॉल्यूशंस प्राइवेट लिमिटेड | H.NO 442 B-53A, NEW DELHI, EAST DELHI, Delhi | India |
| S2-766015953 | ওম ইনভেস্টমেন্ট এলএলপি | 2/3 SARAT BOSE ROAD, KOLKATA, West Bengal | India |
| S2-971443413 | Wonderful Co-& | C/O VEDHAVATHI, MINASANDRA, VILLAGE, LAKKURU HOBLI, MALUR, Karnataka | India |
| S2-287137930 | *** BALAJI  PVT. LTD. CENTER | #246 1-11-224/1/A, GURUMURTHY LANE BEGUMPET, HYDERABAD, Telangana | India |
| S2-586675782 | Jobina Biggerstaff Exploration  LLC | 2708 Rhodes Avenue, ROANOKE COUNTY, VA | US |
| S2-665442254 | Strickland, Jemima E., MD LLC | 17923 FOLLY POINT DR, CYPRESS, TX | US |
| S2-122344563 | Sri PRAGATIDIES.COM | HN #525 H.N. 144 NARGUDA GATE TIKAMGARH, TIKAMGARH, Madhya Pradesh | India |
| S2-138743191 | SAS Lecole Çentre | NO. 44 COUR2 ARISTIDE BRIAND, Bordeaux | France |
| S2-501943333 | Bison  PC | TN, GLENDALE RD, WHITE BLUFF | US |
| S2-454185147 | @jaincotrading | 0837 INDUSTRIAL AREA ALUDHIANA, LUDHIANA, Punjab | India |

#### test_source3.tsv

| entity_id | business_name | business_address | country |
|---|---|---|---|
| S3-717325163 | sarl reseau maternelle | No 3 Rue Schepers, Lille, Hauts-de-France | France |
| S3-546044758 | premier tech private limited | S-419 Shool Block Shakarpur, New Delhi, East Delhi, DL | India |
| S3-934095055 | [Service] Mukundapuram Company | Keralam, Thrissur, 159/B | India |
| S3-547069708 | Peggie Willoughby, DDS PC Center | 119-17 234 Street, Cambria Heights, New York | US |
| S3-852805165 | Southern Developers  Stores | No 13, Flat 502, Vi Sunder Nagar S V Road Near Dalmiha College, Malad (W), Mumbai, Mumbai City, MH | India |
| S3-997650136 | suresh lifesciences private limited | 6 Bentinck Street, Kolkata, Calcutta, WB | India |
| S3-799032119 | Orbivera t/a Mangal Foundation | Kudlu Gate, Bengaluru, Tower C - 406 Purva Westend, Karnataka, Bangalore, Bangalore South | India |
| S3-735373702 | Department of Plhns and Recreation | Indiana, 006596 Central Ave, Portage | US |
| S3-396725357 | Siis  Co | *(empty)* | US |
| S3-761360216 | Accu  Foundation Holdings | Rangareddy, Medchal Malkajgiri, Plot No 94., Shaili Gardens, Yapral, Andhra Pradesh | India |

---

## 2. Ground-truth structure (train)

- There are 2,206,821 rows, one per train S1. There are no duplicate rows, no whitespace in the lists, and no duplicate ids within a list. Every referenced id exists in train S2/S3. The lists are always ordered with S2 ids before S3 ids, which is irrelevant for scoring.
- **Singletons (empty list): 123,247 = 5.585%** (India 5.59%, US 5.58%).

**Match-set size distribution**

| size | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| % of S1 | 5.59 | 5.40 | 17.00 | 24.06 | 21.94 | 14.59 | 7.47 | 2.90 | 0.85 | 0.19 | 0.02 | 0.002 |

The mean size is 3.46 overall and 3.67 for non-singletons. The distribution is the same for US and India (within 0.1 percentage point for every size).

**S2 vs S3 split**
- Total links: 7,638,365. Of these, **S2 has 3,693,619 (48.4%)** and **S3 has 3,944,746 (51.6%)**.
- S1 entities with ≥1 S2 match: 86.96%. With ≥1 S3 match: 87.93%. With both: 80.48%. S2 only: 6.48%. S3 only: 7.45%.
- The number of S2 matches per S1 ranges over 0–5 ({0: 287,745, 1: 789,108, 2: 652,779, 3: 333,957, 4: 119,078, 5: 24,154}). The number of S3 matches ranges over 0–6 ({0: 266,276, 1: 716,417, 2: 668,375, 3: 372,443, 4: 145,116, 5: 35,378, 6: 2,816}). The two counts look roughly independent.

Joint distribution (% of all S1; rows = number of S2 matches, columns = number of S3 matches, clipped at 4):

| #S2 \ #S3 | 0 | 1 | 2 | 3 | 4+ |
|---|---|---|---|---|---|
| 0 | 5.58 | 2.74 | 2.58 | 1.43 | 0.70 |
| 1 | 2.66 | 12.22 | 11.38 | 6.36 | 3.13 |
| 2 | 2.20 | 10.11 | 9.43 | 5.26 | 2.58 |
| 3 | 1.13 | 5.18 | 4.79 | 2.69 | 1.33 |
| 4+ | 0.49 | 2.21 | 2.09 | 1.13 | 0.56 |

**Is the mapping one-to-one?**
- **S2/S3 to S1 is many-to-one, with no overlap:** no S2/S3 id appears in more than one S1's match set (maximum multiplicity 1). The S1 match sets are therefore disjoint clusters.
- **S1 to S2/S3 is one-to-many, including within the same source.** 51.2% of S1 entities have ≥2 S2 matches, 55.5% have ≥2 S3 matches, and **76.8% have ≥2 from at least one source**. Each source is therefore itself un-deduplicated. Two S2 matches of the same S1 are usually near-copies of each other (median name token-set ratio 97, median address ratio 95), but the 10th percentile of name similarity is only 22 (one copy transliterated or renamed).
- Coverage: 73.4% of train S2 records and 74.6% of train S3 records are linked to some S1. **1,340,997 S2 and 1,340,857 S3 records are linked to nothing** (distractors and decoys).
- The country label agrees between S1 and each matched record in 100.000% of links.

---

## 3. True matches vs hard non-matches

The hard non-matches were mined from 4,000 random train S1 records. A candidate had to meet all of these conditions:
- be in the same country;
- contain the S1 "city" (the second-to-last address component) in its address;
- share at least one distinctive name token (document frequency ≤ 3000, stop-words and legal words removed);
- have a normalised-name `token_set_ratio` ≥ 70;
- not be in the S1's true match set.

1,585 of the 4,000 S1 records (40%) have such a candidate, with a median name similarity of 92. **1,386 of the 1,585 hard negatives are unlinked records** and 199 belong to another S1.

### 30 random TRUE match pairs

| C | S1 name | S1 address | Src | Match name | Match address |
|---|---|---|---|---|---|
| US | Davis Society Care | 3424 Humboldt Avenue, Minneapolis, MN | S2 | Davis Society | MINNEAPOLIS, 424 HUMBOLDT AVE, MN |
| US | Peak Highland Fast Inc | 3029 Veterans Parkway, City Of Marshfield, WI | S3 | Peak Highland Fast Inc | 3029 Veterans Parkway, City Of Marshfield, Wisconsin |
| US | Coastal Maritime Inc | 805 Park Avenue, Unit A, Victoria, TX | S3 | Coastal Inc Services | Park Ave, Unit A, Vcitoria, Texas |
| India | Modern Foundation | Delhi, New Delhi, 5, Ground Floor, Park End Colony, Vikas Marg, Preet Vihar, New Delhi | S2 | मॉडर्न फाउंडेशन | HN 372 5, Delhi, GROUND FLOOR, PARK END COLONY, VIKAS MARG, PREET VIHAR, NEW DELHI |
| US | Internal Medicine Health LLC | 100 Sundown Drive, Unit #14, 06, IA | S2 | Internal Medicine Health | 100 SUNDOWN DR, GLIDDEN, IA |
| US | Family Keystone Center | 136 Stella Lane, Troutman, NC | S3 | Family Keystone | 36 Stella Ln, Troutman, North Carolina |
| India | Jain Enterprises Limited | 33/3115 C Paroppady, Marikunnu, Kozhikode, Kerala | S3 | ജൈൻ എന്റർപ്രൈസസ് ലിമിറ്റഡ് | കേരളം, Marikunnu, Kozhikode, 33/3115 C Paroppady |
| India | Vishvam Farming Limited | H. No. 867, Girwa, Amrit Nagar, Jharnon Ki Sarai, Girwa, Udaipur, Rajasthan, Udaipur | S3 | Vishvam Farming  Limited |  |
| US | Valley Holdings | 719 Mountain Road, Addison, VT | S3 | Vlley Holdings | 719 Mountain Rd, Addison, Vermont |
| US | Strategic Dynamic Graf Inc | 320 Village Green Drive, Nashville, TN | S3 | Strategic Dynamic Graf | Tennessee, 20 Village Green Dr, Nashville |
| India | Thomas Polytechnic | Lucknow, Uttar Pradesh, L/D-6, Lucknow, Sector-1, Gomti Nagar Vistar | S2 | THOMAS POLYTECHNIC COMPANY | L/D-6, SECTOR-1, GOMTI NAGAR VISTAR, LUCKNOW, उत्तर प्रदेश |
| US | Peace Commercial Inc | NY, Troy, 20 Hamilton Avenue | S2 | Peace Commercial | HAMILTON AVE, null, TROY, NY |
| US | Norah Pletcher Bain Care | AL, 658 Wasson Avenue, Prichard | S2 | Norah Pletcher Báin Care | WASSON AVE, PRICHARD, AL |
| US | Filler Interests Inc | 1201 Ponderosa Parkway, Unit Apt 140, Flagstaff, AZ | S3 | Filler Interests | # Apt 140, 120 Ponderosa Parkway, Arizona, Flagstaff |
| US | Bondi Evolution LLC | 514 Meigs Avenue, Grand River, OH | S3 | Bondi Evolution | Meigs Ave, Grand River, Ohio |
| India | Ciro Builders Private Limited | 01 Bhandarisome Mangichhi, Darbhanga, Bihar | S3 | Dovaflux | 01 Bhandarisome Mangichhi, Darbhanga, Bihar |
| US | Green & Auclair L.L.C. | 617 Amanda Drive, Weddington, NC | S3 | Green Áuclair L.L.C. Services | 617 Amanda Drive, Matthews, North Carolina |
| US | Culberson Fortunex Corporation | 31840 Dodge Park Boulevard, Gresham, OR | S3 | Culberson Fortunex | 31840 Dodge Park Boulevard, Gresham, OR |
| US | Vivanco Triton of Atlanta | 1043 Joseph E Boone Boulevard, Atlanta, GA | S2 | Vivanco Triton of Atlanta Corp | 1043 JOSEPH E BOONE BLVD, ATLANTA, GA |
| India | Rao Green Private Limited | Maharashtra, Plot No:85, Mumbai City, A-703, Suryakirti Towers, Tilak Nagar, Chembur, Mumbai | S3 | Rao Green Private | Plbt No:85, A-703, Suryakirti Towers, Tilak Nagar, Chembur, Mumbai, Mumbai City, महाराष्ट्र |
| US | 3424 Mountain View Drive Management Inc | 719 Lynn St, Unit 101, Louisville, KY | S2 | 3424 Mountain Vwnba Drive Management [Inc] | 719A LYNN ST, LOUISVILLE, KY |
| US | Aurify Calamos Corp | 1068 Westridge Drive, Waterville, OH | S2 | AURIFY CALAMOS (CORP.) | 1068 WESTRIDGE DRVE, WATERVILLE, OH |
| US | Star Maywood PC | 3200 Truxel Road, Unit 401, Sacramento, CA | S3 | Star Maywood Pc | California, Sacramento, # 401, 3200-3202 Truxel Rd |
| India | Laxmi Healthcare Private Limited | C/O- Yogika Gautam, W/O-, Ajendra Kr, Vii.- Barola, Dadri, Gautam Buddha Nagar, Uttar Pradesh | S2 | Laxmi Hea1thcare Private Limited Limited | C/O- YOGIKA GAUTAM, W/O-, AJENDRA KR, VII.- BAROLA, DADRI, Uttar Pradesh |
| India | Infra Arya Consultantsprivate Pvt Ltd | H No.162 Shiv Nagar, Azad Nagar, Hisar, Hissar, Haryana | S3 | Infra ARYA Consultantsprivate PVT LTD | H No.##162 Shiv Nagar, Hisar, Hissar, HR |
| US | 1194 Third Court Partners Inc | 13533 Dixie Highway, Fl FLOOR 1, Walton Ky, KY | S3 | 1194 Third Court Parns Inc Inc | 13533 Dixie Highway, Fl. Floor 1, Walton Ky, Kentucky |
| US | West Veterinary Clinic Inc | 715 Bristol Drive, Bismarck, ND | S2 | West Veterinary Clinic-Inc | 715 BRISTOL DR, BISMARCK, ND |
| US | Bushman Financial | 10615 Canoe Drive, TX, Irving | S3 | Bushman Financial | 10615 Canoe Drive, Coppell, Texas |
| India | Big Consultancy Limited | C/O Suresh Sharma Near Singh Building, Main Road, Rourkela, Sundargarh, Orissa | S3 | ବିଗ୍ କନସଲଟାନ୍ସି ଲିମିଟେଡ୍ | C/o Suresh Sharma Near Singh Building, Main Road, Rourkela, Sundargarh, OD |
| US | Valley Allied Xsolla P.C. | 203 Millbrook Road, Raleigh, NC | S2 | Valley Allied Xnsliola P.C. | 0203 MILLBROOK RD, RALEIGH, NC |

### 30 HARD NON-matches (same country, S1 city appears in candidate address, shared distinctive name token, token_set_ratio ≥ 70)

| C | S1 name | S1 address | Src | Non-match name | Non-match address | name sim | belongs to |
|---|---|---|---|---|---|---|---|
| India | NHI Farm Private Limited | P.No 18, A.K Enclave, Shanthi Vihar Green Park, Tirumalagiri, Hyderabad, Telangana | S2 | NHI FARM | FLAT NO. 104, KOCHAR APTS BEGUMPET MAIN ROAD, HYDERABAD, Telangana | 100 | other S1 |
| India | Cratus Services Corp | Office No 154, Satra Plaza, Plot-20, Sector-19D, Palm Beach Road, Vashi, Navi Mumbai, Raigarh, Maharashtra | S3 | Cratus Services Industries Corp | Office No 158, Satra Plaza, Plot-20, Sector-19d, Palm Beach Road, Vashi, Navi Mumbai, Raigarh, महाराष्ट्र | 100 | unlinked |
| India | Howrah Solutions Limited | Lake Land Country Club, Flat No. 10C Opp. Kona Expressway, Howrah, West Bengal | S3 | Howrah Solutions Limited | C/o, Howrah, Ravi Kumar, 4Th Floor Flat Room-1 52 Nanda Ghosh Road, WB | 100 | other S1 |
| India | Pune Bit Private Limited | Sno.14/1, Shivneri Colony, Pimple Gurav, Pune, Maharashtra | S3 | Pune Bit Industries Private Limited | Block B-84 Sno.14/22, Shivneri Colony, Pimple Gurav, Pune, MH | 100 | unlinked |
| India | Droid Advisory (India) Private Limited | Amar, 4Th Floor, 6-3-1239/2, Raj Bhavan Road, Somajiguda, Hyderabad, Telangana | S2 | DROID ADVISORY (INDIA) INDUSTRIES  PRIVATE LIMITED | NO 16 AMAR, 4TH FLOOR, 6-3-1239/2, RAJ BHAVAN ROAD, SOMAJIGUDA, HYDERABAD, Andhra Pradesh | 100 | unlinked |
| India | Three Infra (India) Care | 7Th Floor, Pushpak Building Nr Cama Hotel, Khanpur, Gujarat, Ahmedabad | S3 | Three | 55, Abad Nagar, Bhalej Road, Nroverbridge, Anand, Gujarat | 100 | other S1 |
| US | IN Bright Tankers LLC | 425 Annette Avenue, Vallejo, CA | S2 | IN Bright Tankers Downtown LLC | 430 ANNETTE AVE, VALLEJO, CA | 100 | unlinked |
| US | J D Russell Btc | 15297 Manchester Road, Ballwin, MO | S3 | LLC J D Russell Btc | 15298 Manchester Road, Ballwin, Missouri | 100 | unlinked |
| India | Ahmednagar Technologies | Ahmednagar, 16 Goverdhan Apptstation Rd Market Yard, Ahmednagar, Maharashtra | S2 | Ahmednagar Technologies | NO 6/8 , VEER SAVARKAR ROAD, VASANT TEKDI, OPP. STATE BANK COLONY, AHMEDNAGAR, Maharashtra | 100 | unlinked |
| India | Mayur Hospital | 320/6, Valmiki Marg, Raja Park, Jaipur, Rajasthan | S2 | Mayur Hospital Private Limited | 320/11, VALMIKI MARG, RAJA PARK, JAIPUR, Rajasthan | 100 | unlinked |
| US | Montance Chimera LLC | Unit UNIT 1029, 2040 Scottsdale Road, Scottsdale, AZ | S3 | Montanke Chimera Llc | 2061 Scottsdale Rd, Unit UNIT 1029, Scottsdale, Arizona | 95 | unlinked |
| India | Lucrative Infrastructures Private Limited | G No. 549, Kasarwadi Road Barshi, Dist. Solapur, Solapur, Maharashtra | S3 | XK Infrastructures Limited | Tal- Malshiras, MH, B. S. No. 0087/1A/12, Solapur | 94 | unlinked |
| India | CYY Palace Private Limited | Raj Classic, Bda No 7, Municipal No 113, 1St Main Road, 1St Block, Rajajinagar, Bangalore North, Bangalore, Karnataka | S3 | Palace Cross Roadbangalore Karnataka Training Private Limited | No 14, Palace Cross Roadbangalore Karnataka, Karnataka, KA | 92 | unlinked |
| India | Infrastructure Rab Infolabs Private Limited | C/O Shatrudhan Kuhswaha Sukhdewpur Kaithwliya Ghazipur, Ghazipur, Uttar Pradesh | S2 | Infrastructure Rab Infolabs Public Limited | NO 9 C/O SHATRUDHAN KUHSWAHA SUKHDEWPUR KAITHWLIYA GHAZIPUR, GHAZIPUR, Uttar Pradesh | 91 | unlinked |
| US | Crawley Allied Muniyield L.L.C. | 613 Scissortail Lane, Cushing, OK | S2 | Crawley Allied Muniyield Corp | 617 SCISSORTAIL LN, CUSHING, OK | 91 | unlinked |
| US | Clarendon Hills Family Total Guild, LLC | 40 Norfolk Avenue, Clarendon Hills, IL | S2 | CLARENDON HILLS FMILY TOTAL GUILD VALLEY, LLC | 43 NORFOLK AVE, CLARENDON HILLS, IL | 90 | unlinked |
| India | Accord Seva Private Limited | Part Of 520A, Gt Road Inds. Area Near Ganpati Dharma Kanta, Shahdara, Delhi, East Delhi, Delhi | S3 | Dr Accord & Brothers Private  Limited | A-1 Dsidc Opposite Plot No. 129, Patparganj, Industria, Delhi, East Delhi, DL | 90 | other S1 |
| US | Sicard, Stoltz and Wyatt Chase, Inc | 896 8th Street, Upland, CA | S2 | Soto, Stoltz and Wyatt Chase, | 996 EIGHTH ST, UPLAND, CA | 88 | unlinked |
| India | Parvati Agencies Private Limited | Gujarat, Plot No 35B Al Asara Esta, Ahmedabad, Daskroi, Nr Rv Denim Narol | S2 | Private Parvati Infra Limited | PLOT NO B3/37B AL ASARA ESTA, AHMEDABAD, Gujarat, DASKROI | 88 | unlinked |
| US | Phipps, Bell & Houser, LLC | 204 Center Street, Star, NC | S2 | Inc Phipps Bell & Houser | #217 CENTER STREET, STAR, NC | 88 | unlinked |
| US | Davis Income LLC | 950 Edgar Drive, Unit 32, Charleston, IL | S2 | Davis Income Ltd | 963 EDGAR DRIVE, IL, CHARLESTON | 88 | unlinked |
| US | Alana Chatterton Environmental LLC | 826 Washington Street, Unit 201, Findlay, OH | S3 | Alana Chtetrtn-Environmental Co | 833 Washington St, Unit 201, Findlay, Ohio | 86 | unlinked |
| US | Isner Stores Inc | 7 Greenway, Unit Apartment 20, Brookhaven, NY | S2 | Isner Stores Group | 10 GREENWAY, PO BOX 2723, BROOKHAVEN, NY | 86 | unlinked |
| US | Celie Rabinowitz Allied View | 3624 212th Street, Bothell, WA | S2 | Celie Rabsocitz Allied View Inc | 3635 212TH STREET, BOTHELL, WA | 85 | unlinked |
| US | Bowens Apex Valley | 9319 Britton Street, Fl 0, Wichita, KS | S2 | Bowens Aocex Valley Central | 9321-9325 BRITTON STREET, WICHITA, KS | 84 | unlinked |
| US | Doria's Ice Cream | 108 Easton Court, Garner, NC | S3 | Doria'S Ice Cerome Midtown | 00119 Easton Court, PO Box 3028, Garner, North Carolina | 79 | unlinked |
| India | Sathe Minerals | 18, Girnar Chapel Lane, Santacruz (West), Mumbai, Maharashtra | S3 | Sathe Mines Co | 19, Santacruz (West), Mumbai, MH | 79 | unlinked |
| India | Radial Rural Limited | Modern Colony, House No. 1379, Haryana, Yamuna Nagar, Jagadhri | S3 | Radial Peak (Limited) | House No. 1383, Yamuna Nagar, Jagadhri, HR | 78 | unlinked |
| US | Anvil Corp | 13415 Desert Primrose Lane, Victorville, CA | S3 | Anvil Group | 13417-B Desert Primrose Lane, <NULL>, Victorville, California | 76 | unlinked |
| US | Freeman Resources LLC | 34965 Roberts Road, Eastlake, OH | S2 | Freeman Roesubces Holdings | 34969. ROBERTS ROAD, EASTLAKE, OH | 72 | unlinked |

### What separates them

**True matches:** the table below is computed on a sample of 300k links.

| | India→S2 | India→S3 | US→S2 | US→S3 |
|---|---:|---:|---:|---:|
| target name in Indic script | 23.6% | 13.3% | 0 | 0 |
| normalised name exactly equal (Latin-script targets; lower-cased, accents and legal words stripped) | 56.0% | 51.4% | 55.0% | 50.7% |
| name token_set_ratio ≥ 90 (Latin) | 79.5% | 77.5% | 80.7% | 81.2% |
| name token_set_ratio < 60 (Latin): renamed or very noisy | 5.8% | 7.8% | 5.0% | 5.1% |
| target address empty | 3.9% | 4.1% | 4.9% | 4.6% |
| leading house number equal (when both present) | 90.5% | 91.6% | 83.7% | 85.1% |
| house number differs only by truncation or prefix (`3424`→`424`, `136`→`36`, `203`→`0203`) | 4.6% | 4.0% | 9.1% | 8.3% |
| S1 city string appears in target address | 63.0% | 83.6% | 76.6% | 75.3% |

**Name-noise types seen in true matches:**
- typos and character swaps (`Vlley`, `Xnsliola`, `Hea1thcare`, `Parns`);
- dropped or added generic words (`Care`, `Services`, `Center`, `Company`);
- dropped, doubled or moved legal suffixes (`Inc Inc`, `Limited Limited`, `[Inc]`, `(CORP.)`, `LLC Moncada …`);
- random accents on US names (`Báin`, `Áuclair`);
- transliteration into Devanagari, Bengali, Malayalam, Odia and other Indic scripts;
- domain forms (`canaraagro.com`, `WESTQUALITYCIONCOM`), handles (`@SWEETHAIR`) and initialisms (`DCH`, `FG`);
- a completely different synthetic brand name (`Dovaflux`, `Fluxveo`, `Gildfaye`, `Onyxlum`, `Ectofluxdelta`) at the same address.

**Address-noise types seen in true matches:**
- component reordering (`MINNEAPOLIS, 424 HUMBOLDT AVE, MN`, `TX, Brownsville, …`);
- abbreviation of street words (Avenue→AVE, Drive→DR/DRVE);
- state code ↔ full name, and Indian state names in Devanagari (`महाराष्ट्र`, `उत्तर प्रदेश`);
- a neighbouring city or a county name substituted (Weddington→Matthews, Irving→Coppell, Smyth County→Marion);
- dropped house number or street;
- number ranges (`3200-3202`, `5856-5860`);
- `# Apt 140`, `Unit UNIT 1029`, and null tokens.

**How the hard negatives (decoys) differ:** the name is very close (often a token_set_ratio of 100) but differs in one of these ways:
1. **An extra discriminating word** (`Pune Bit Industries`, `IN Bright Tankers Downtown LLC`, `Cratus Services Industries Corp`).
2. **A different legal form or type** (LLC↔Corp, LLC↔Ltd, Private↔Public Limited, Corp↔Group).
3. **The same street with a house or unit number shifted by a small amount** (425→430, 613→617, 15297→15298, `Office No 154`→`158`, `320/6`→`320/11`).

A token-set ratio is blind to both (1) and (2), so extra-token and legal-form mismatch features plus exact house-number comparison are essential.

---

## 4. Pattern catalogue

Percentages below are *sample* rates computed on up to 200k records per (split, source, country). Train and test US/India agree to about 1 percentage point, so the test columns are shown only where France matters.

### 4.1 Countries and regions
- **US:** all states appear. The top states are TX 8.7%, NY 6.7%, NC 6.2%, OH 5.5%, IL 4.9%, TN, VA, MA, AZ, IN, WA and MD. S1 and S2 end the address in a 2-letter code (86% and 84%). **S3 spells out the full state name** (`Texas`, `North Carolina`); only 3–4% of S3 addresses end in a code.
- **India:** the top states are Maharashtra 18%, Delhi 12%, Uttar Pradesh 7%, Karnataka 7%, Tamil Nadu 6%, West Bengal, Gujarat, Telangana, Haryana, Kerala, Rajasthan and Bihar.
  - S1 uses full English state names.
  - S2 uses full names, plus Devanagari or other Indic state names in about 14% of records (`महाराष्ट्र`, `दिल्ली`, `পশ্চিমবঙ্গ`, `കേരളം`).
  - **S3 uses state codes** (`MH`, `DL`, `UP`, `KA`, `TN`, `WB`, `GJ`, `TG`, `HR`, `OD`, `RJ`, `PB`) in 57% of records, plus Indic names.
  - Old and new place names co-exist: Bangalore/Bengaluru, Calcutta/Kolkata, Gurgaon/Gurugram, Orissa/Odisha, Hissar/Hisar, Keralam, and Andhra Pradesh↔Telangana for Hyderabad.
- **France (test only):** only 3 regions appear, Hauts-de-France (34%), Nouvelle-Aquitaine (29%) and Pays de la Loire (24%). The main cities are Lille, Roubaix, Tourcoing, Dunkerque, Calais, Bordeaux, Pessac, Nantes and Saint-Nazaire.
  - S1 writes `street, city, Region`.
  - S2 and S3 use the **department** instead (`Nord`, `Gironde`, `Loire-Atlantique`) or drop the region altogether (33–35% have ≤2 address parts). S2 upper-cases the city (`BORDEAUX`).
  - The French names include SARL 35%, SAS/SASU 24%, EURL, SA 5–9%, SCI 3–4%, `Ets`/`Établissements`, `& Fils`, `Frères`, `Cie`, and many associations (`Club`, `Amicale`, `Comité`, `Association sportive`, `École primaire`, `Amis de …`).

### 4.2 Legal suffixes (% of names containing one)

| | trS1 US | trS2 US | trS3 US | trS1 IN | trS2 IN | trS3 IN | teS1 FR | teS2 FR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LLC | 28.6 | 19.8 | 20.2 | – | – | – | – | – |
| Inc | 18.0 | 14.2 | 13.8 | – | – | – | – | – |
| Corp/Corporation | 2.7 | 5.4 | 5.0 | 1.5 | 2.0 | 1.9 | – | – |
| PC/PLLC | 5.3 | 2.9 | 3.2 | – | – | – | – | – |
| Ltd/Limited | 0.2 | 3.0 | 2.8 | 75.7 | 41.5 | 47.9 | – | – |
| Pvt/Private | – | – | – | 62.8 | 35.8 | 40.7 | – | – |
| LLP | 0.1 | 0.1 | 0 | 4.4 | 3.7 | 4.1 | – | – |
| Hindi legal words (प्राइवेट/लिमिटेड/एलएलपी) | – | – | – | – | 10.1 | 5.5 | – | – |
| SARL/EURL · SAS/SASU · SA · SCI | – | – | – | – | – | – | 34.9 · 24.4 · 4.9 · 3.2 | 26.8 · 21.1 · 8.9 · 3.9 |
| any legal token | 53.6 | 47.0 | 46.4 | 84.4 | 57.3 | 64.0 | 67.4 | 53.3 |
| legal token at the **start** of the name | 0 | 2.7 | 2.5 | 0 | 1.4 | 1.5 | 0 | 4.0 |

- S2 and S3 add spurious `Co`, `Ltd`, `Corp`, `Group`, `Holdings` and `Partners` (they are more frequent there than in S1).
- They drop suffixes (India `Pvt Ltd` falls from 63% in S1 to 21–25%) and move suffixes to the front.
- They also introduce `Public Limited` (about 3.4% of India S2/S3 names), which is a decoy signal.
- Variants include `L.L.C.`, `Pvt.`, `Ltd.`, `P.C.`, `Inc.`, `(CORP.)`, `[Inc]` and `-Inc`.

### 4.3 Abbreviations
- **US street types:** S1 uses full words (`Street/Road/Avenue/Drive`, 85%) and almost never abbreviates (3%). S2 and S3 abbreviate in 44–46% of records (`ST, RD, AVE, DR, BLVD, LN, CT, CIR, PKWY, HWY`), sometimes misspelled (`DRVE`). There are ordinal and number word swaps (`8th`↔`EIGHTH`).
- **US units:** S1 writes `Unit 11`, `Unit UNIT 136` or `Unit Apt 140` (a doubled keyword in 1.9% of records). S3 writes `# 401`, `Unit UNIT 1029` or `Fl. Floor 1`. S2 drops units entirely (0%). `PO Box` appears only in S2/S3 (1.6%), apparently as injected noise.
- **India:** the address vocabulary is `H.No`, `HN`, `House No.`, `Plot No`, `P.No`, `Flat No`, `Shop No`, `KH NO`/`Khasra`, `Door No`, `Sy No`/`Sr No`, `Bldg`, `Fl`, `Gr Floor`, `1St/2Nd Floor`, `Dist`, `Tal`/`Taluka`, `Tq`, `PO`, `Vill`/`Vii.`, `C/O`, `W/O`, `S/O`, and `Sector-19D`. Honorifics appear in names (`Sri`, `Shri`, `Shree`, `Smt`, `M/s`, `Dr`, `Mr`, 5–6% in S2/S3 against 0.9% in S1).
- **France:** S1 uses `Rue`, `Avenue`, `Boulevard`, `Place`, `Chemin`, `Allée`, `Impasse`, `Quai` and `Route`, but also `AV`, e.g. `3319 AV de la Liberation`. **S2 abbreviates** (`R.`, `AV`, `BD`, `CH`, `RTE`, `IMP`, 10.3%) and upper-cases. S2 and S3 also garble the street type (`COUR2`, `Rue Icmre`).
- **Names:** `&` ↔ `and` (the US has 4% `and`), `Co`/`Company`, `Corp`/`Corporation`, `Pvt`/`Private`, `Ltd`/`Limited`, `Intl`, and initialisms.

### 4.4 Address styles
- **Landmarks (India only):** `Near`, `Nr`, `Opp.`, `Opposite`, `Behind`, `Beside` and `Next to` appear in 13.4% of S1 and 9–11% of S2/S3 (`Near Nava Ranchana School`, `Nr Cama Hotel`, `Opp. State Bank Colony`). Landmarks are often dropped or kept as the only locator.
- **Postcodes are almost absent.** A 5-digit ZIP-like token appears in about 10% of US addresses in every source, but most of these are actually 5-digit house numbers. A 6-digit PIN appears in only 0.3% of India S1 and 1.1% of S2/S3 records, and a French code postal in 0.4–0.5%. **Postcodes cannot be used as a blocking key**; use city/state plus street tokens and house numbers instead.
- **House numbers:** 86% of S1 US and France addresses start with a number, against 75–77% of S2/S3 US addresses and 68–69% of S2/S3 France addresses. Only 22–26% of India addresses start with a number, because India uses `Plot No`, `Flat No` and so on. Numbers can be truncated (`3424`→`424`), zero-padded (`0203`), ranged (`320-322`), suffixed (`719A`) or perturbed (±1–15 in decoys).
- **Component order:** S1 is usually `street, city, state`, but 6.4% of S1 US records put the state code first (`IA, Iowa City, 1064 Newton Rd`). India S1 is sometimes reordered too (`Delhi, New Delhi, 5, …`), and S2/S3 reorder freely. Treat addresses as bags of components.
- **Component counts:** US has about 3 comma parts, India about 5–6 in S1 and about 4.8 in S2/S3, and France about 3 in S1 and about 2.6 in S2/S3. India addresses are long and repetitive (`Mumbai, Mumbai City`, `Hisar, Hissar`, `Lucknow, Uttar Pradesh, Lucknow`).
- **Glued tokens:** `Colonycharbagh`, `Consultantsprivate`, `Roadbangalore`, `ALUDHIANA` (a stray letter glued to the city).

### 4.5 Casing

| | S1 | S2 | S3 |
|---|---|---|---|
| name ALL UPPER | 0% | 14–22% | 2.5–5.7% |
| name all lower | 0% | 4–7% | 5–7% |
| address ALL UPPER | 0% | **US 90%**, India 24%, France 29% | 0% |

S1 is always Title Case. S3 mixes cases, e.g. `Infra ARYA Consultantsprivate PVT LTD`.

### 4.6 Punctuation and junk
- **S1 is clean:** a leading-junk prefix appears in only 0.1–0.2% of records.
- **S2 and S3 add noise:**
  - leading junk such as `--`, `***`, `<<`, `...`, `@`, `#` (1.3–2.3%);
  - bracket tags such as `[Service]` or `[Inc]` (2.5–3.2%);
  - double spaces (9–12%);
  - `.com`/`.in` domain names (3–4%);
  - hyphen joins (`Clinic-Inc`, `Sky-Hospitality`).
- **Aliases appear only in S3** (1.5–3.1%): `F/K/A`, `formerly`, `t/a`, `d/b/a`, `C/O`, e.g. `Beloarcgild formerly MB Milk Private Limited`, `Orbivera t/a Mangal Foundation`. The part after the marker is usually the real name.
- US names contain commas in about 12% of records (`Mcmahon, Mcmillon and Miles LLC`, `Strickland, Jemima E., MD LLC`). Person credentials (`MD`, `DDS`, `DO`, `CPA`, `DPM`) appear in about 1.2–1.5% of names.

### 4.7 Transliteration and scripts
- **India names:** 13.3% of S2 and 7.5% of S3 names are in Devanagari, and 10.1% of S2 and 5.6% of S3 are in other Indic scripts (Bengali, Malayalam, Odia, Tamil, Telugu, Gujarati, …). S1 is always in Latin script.
  - The Indic names are phonetic transliterations of the English name, including legal words: `राम मार्केटिंग प्राइवेट लिमिटेड` = "Ram Marketing Private Limited", `ജൈൻ എന്റർപ്രൈസസ് ലിമിറ്റഡ്` = "Jain Enterprises Limited".
  - A rule-based Indic→Latin romaniser (for example, the Apache-licensed `indic-transliteration`), or character n-grams learned from training pairs, is needed.
- **India addresses:** 13–14% contain Devanagari and 9–10% contain other Indic scripts, mostly the state name and sometimes whole components.
- **Accented Latin:**
  - France S1 names have genuine accents in 15.7% of records (`Établissements`, `Frères`).
  - France S2/S3 names have accents in 24% of records, as a mix of genuine and injected ones (`Àmicale`, `Çentre`, `Sóciety`).
  - **US S2/S3 names have injected accents in about 6.5% of records**, against 0% in S1.
  - Stripping diacritics with NFKD normalisation is safe for all countries.
- There are spelling variants such as `Ecole`↔`École` and `Pharmacie`, and Indian place-name variants (§4.1).

### 4.8 Missing fields per source

| | S1 | S2 | S3 |
|---|---|---|---|
| empty name | 0 | 0 | 0 |
| empty address | 0 | 2.3–3.7% | 2.4–3.5% |
| null token inside address | 0 | 3–4% (US and India; not France) | 3–4% |
| no house number | 14% US; most of India | about 23% US | about 23% US |
| state dropped / ≤2 parts | 0 | 3% US/IN, 35% FR | 3% US/IN, 33% FR |
| unit/suite | kept (13.5% US) | always dropped | partly kept (6.2%) |

---

## 5. Submission format

### 5.1 `matching_results.tsv` (scored; uploaded to the portal)
- UTF-8 text, **tab-separated**. The header must be exactly `source1_entity_id<TAB>matched_entity_ids`. The validator lower-cases and strips header cells, but write it exactly anyway.
- There must be **exactly one row for every id in `test_source1.tsv`**. A missing S1 row, a duplicate S1 row, or an S1 id not in the test set causes rejection.
- The second column is a comma-separated list **with no spaces and no quoting**, and it is **empty for no match**, so the line is `S1-xxx<TAB>` followed by a newline.
- Only `S2-`/`S3-` ids are allowed. An `S1-` id (self-match) or any other prefix causes rejection. There must be no duplicate id within a list.
- The validator splits on `,` without stripping, so `S2-1, S3-2` would produce the id `" S3-2"` and be rejected for having the wrong prefix. Use `\n` line endings, not `\r\n`: a trailing `\r` would end up glued to the last id.
- Ids that do not exist in the test S2/S3 files are not rejected by the scorer; they just lower the score. The validator checks for them only with `--check-ids`.

### 5.2 `candidate_pairs.tsv` (not scored; required in the final zip)
- The header is `source1_entity_id<TAB>candidate_entity_ids`, and the file follows the same rules as 5.1: one row per test S1, empty if there are no candidates, S2/S3 ids only, no duplicates.
- It must be the **last-stage candidate set that the matching model actually scores**, not an early blocking pass.
- **Every matched id should be a subset of that S1's candidates.** The validator only *warns* when it isn't, but the organisers use the file to audit the recall ceiling and reduction ratio.

### 5.3 Validator

Run it from `student_resource/`:

```bash
python3 utils/validate_submission.py --matching ../output/matching_results.tsv \
    --candidate ../output/candidate_pairs.tsv --test-dir dataset/test [--check-ids]
```

It prints `PASS` with exit code 0, or a numbered list of issues with exit code 1. Warnings never fail the run. It detects comma-separated files, and non-UTF-8 files are an error. `--check-ids` loads all test S2/S3 ids into memory, which takes several GB. The validator never computes a score.

### 5.4 Final zip

```
<team>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}
├── code/business_entity_resolution/{src/, README.md, requirements.txt}   # must regenerate both outputs from the data
└── Documentation_template.md (filled; .md or .pdf)
```

The model must be MIT or Apache-2.0 licensed and have **≤8B parameters**. No external data, APIs, geocoding or business-registry lookups are allowed; doing so means disqualification.

### 5.5 `Documentation_template.md`: required sections
1. **Executive Summary:** 2–3 sentences.
2. **Methodology:** 2.1 Problem Analysis (EDA insights) and 2.2 Solution Strategy (approach type and core innovation).
3. **Candidate Generation (Blocking):** blocking keys, the total number of candidate pairs, and how true matches were kept.
4. **Matching Model:** name, address and other features; model type; threshold selection method.
5. **Results & Error Analysis:** macro F0.5 on validation, common false positives, common false negatives.
6. **Conclusion.**
- **Appendix A:** code structure and entry points to reproduce both output files.
- **Appendix B:** additional results.

The header records the team name, members and date. Sections may be adapted, and there is no page limit.

---

## 6. Metric implications (for later)
- F0.5 is computed **per S1 and then macro-averaged**. For a singleton, any prediction scores 0 and an empty prediction scores 1. For a non-singleton, an empty prediction scores 0.
  - For an S1 with true size k, predicting j correct matches and nothing else gives F0.5 = 1.25·j / (0.25·k + j) (for k = 4: j = 1 → 0.625, j = 2 → 0.833, j = 3 → 0.9375).
  - One false positive on top of all k correct matches gives 1.25k / (1.25k + 1) (0.83 for k = 4).
  - So **adding a low-confidence link hurts more than omitting it**, but predicting at least one confident link is worth a lot.
- The partition property (§2) allows each S2/S3 record to be assigned to its single best S1, which removes cross-S1 conflicts.
- Validation should hold out whole S1 clusters together with all their S2/S3 records, **plus a proportional share of unlinked S2/S3 records**, so that the decoy density matches the test. Because the test has more S2/S3 per S1, consider stress-testing with extra decoys.
- France has no labels. Its normalisation (department ↔ region, `R.`/`AV`/`BD`, SARL/SAS/EURL, accents) must be written from first principles and checked by eyeballing test samples.

---

## 7. Baseline findings (2026-09-26, train DEV; the full report is in `reports/baselines_dev.md`)
- **All-empty** scores 0.0559, which equals the singleton share.
- **Exact normalised name + same city** scores 0.5453. Its link precision is 0.67 (India 0.48, US 0.83), link recall is 0.38, and 23% of S1s get an empty prediction.
- The exact-name join alone gives 4.9M pairs for 110k S1s. Requiring the same city cuts this to 216k.
- Where the false positives of this baseline come from:

  | | India | US |
  |---|---:|---:|
  | TP | 47,079 | 96,986 |
  | FP: record of another S1 (outside DEV) | 37,899 | 3,393 |
  | FP: record of another DEV S1 | 2,009 | 197 |
  | FP: unlinked (decoy) | 11,466 | 16,484 |

  **In India, name + city is not identifying.** Many distinct S1 businesses share a normalised name in the same city (for example, generic "Sri Balaji Enterprises"-style names once legal words are removed). Street-level address evidence, and letting S1s compete for records, are required. **In the US, most false positives are decoys.** Removing legal-form tokens from the name key merges legal-form decoys, so legal form should become a feature rather than being discarded.

---

## 8. Parsing, geography and DEV findings (2026-09-26; reports: `reports/parse_dev.md`, `reports/baselines_dev*.md`)

**State parsing** (`ber.geo`)
- State is found for 100% of S1 records and 96.5% of train S2/S3 records. The rest are essentially the empty addresses.
- In test, 4.5% of S2/S3 states are inferred from the city via S1 data, mostly France, where S2/S3 drop the region.
- On true pairs, S1 and target states agree for 99.93% of US pairs and 98.67% of India pairs.
  - **Telangana ↔ Andhra Pradesh accounts for about 90% of the India disagreement.** S2/S3 still place Hyderabad in AP (pre-2014), so treat the two as one state group.
  - US reversed addresses ("DC, WASHINGTON", "NY, Alabama") need the 2-letter code to win over a state *name*.
  - With both fixes, agreement is 99.85% for India.
- Indic state names are often *translations*, not transliterations: পশ্চিমবঙ্গ = Paschimbanga = West Bengal, കേരളം = Keralam. They need an alias list.
- City (the component next to the state) is the same for S1 and target in 68.5% of India pairs and 79.8% of US pairs. Indian S1 addresses put the district next to the state ("Rourkela, Sundargarh, Orissa").
- **Bug found and fixed:** `strip_accents` removed every Unicode combining mark, including the Indic virama and nukta, which corrupted Indic text (प्राइवेट → पराइवेट). It now strips only U+0300–U+036F.

**Field coverage** (DEV, % of records with the field filled)
- India S1: locality 88%, premise number 80%, street 34%, building 18%, landmark 16%, floor 11%.
- India S2/S3: locality 77–80%, premise number 81–83%, street 20–26%, building 16–20%, landmark 9–11%.
- US: street 93–97% and premise number 82–100% in every source. Unit is filled for 7.6% of S1, 6.8% of S2 and 15.6% of S3.
- PIN codes are almost never present (below 1%).

**Discrimination on exact-name + same-city pairs**

This is where name + city fails. India pairs are 43k TP, 19k FP_other (another S1's record) and 9k decoys.

| India | TP | FP_other | FP_decoy |
|---|---:|---:|---:|
| premise_no agree / conflict | 63.8 / 12.7 | **0.3 / 74.8** | 7.4 / 72.6 |
| locality_key agree / conflict | 70.0 / 7.7 | **2.2 / 64.1** | 29.2 / 43.9 |
| street_key agree | 26.9 | 0.2 | 10.8 |
| legal form conflict | **0.0** | 12.9 | 23.7 |

| US | TP | FP_other | FP_decoy |
|---|---:|---:|---:|
| premise_no agree / conflict | 68.3 / 14.9 | 0.0 / 82.5 | 2.4 / **89.7** |
| street_key agree | 84.7 | 0.1 | **77.6** |
| legal form conflict | 0.9 | 20.0 | **49.5** |

- **Distinct Indian businesses that share a name and city are separated by premise number and locality.** The US decoys sit on the same street with a different house number and/or a conflicting legal form.
- **Used as hard filters, these fields lower F0.5.** Requiring no premise conflict takes precision from 0.80 to 0.94 (India needing one address agreement: 0.61 → 0.97), but 12.7–14.9% of true pairs "conflict" on premise number (truncation 3424→424, padding, ranges). That empties many S1 predictions, and DEV F0.5 falls from 0.598 to 0.561.
- **Premise numbers must be compared softly** (suffix/prefix, numeric distance, typed vs untyped) inside a learned model, not as hard rules.

**Legal form**
- Keeping legal form separate and rejecting pairs with a conflicting form adds about +0.02 macro F0.5 on both DEVs. Most of the gain is on singletons (0.665 → 0.800) and US precision (0.805 → 0.875).
- True pairs almost never conflict (0% India, 0.9% US). `Pvt Ltd` vs `Ltd` is a `subset` relation, not a conflict.

**Geographic DEV vs the old random DEV** (same baseline code)
- The overall score is 0.5762 on the geographic DEV (0.5701 weighted) against 0.5432 on the random DEV.
- The whole gap is in the **India metro tier** (0.527 vs 0.438). The single drawn metro, Jaipur, is among the easiest (0.530), while Mumbai scores 0.390 and the Delhi districts 0.35–0.39.
- The std across the 21 Indian metros is 0.06. A label-free proxy, the city's S1 name-collision rate, barely predicts it (r = −0.26), so it can't be used to pick a representative metro.
- A/B deltas agree across the two DEVs: the legal filter gains +0.0198 on the random DEV and +0.0201 weighted on the geographic DEV.

---

## 9. Blocking findings (2026-09-26; reports: `reports/blocking_*.md`, `blocking_tune_train_dev.md`, `blocking_missed_*.tsv`)

**Results at the defaults** (K name 10 / addr 10 / phon 3 / placeless 2 / alias 1):

| | pair recall | India | US | candidates per S1 | full-set cover India / US |
|---|---|---|---|---|---|
| DEV | 0.9819 | 0.9729 | 0.9861 | 53.8 | 91.8% / 95.4% |
| full train | 0.9798 | 0.9703 | 0.9861 | 62.0 | 91.2% / 95.4% |

**Blockers**
- **Unique catches on full train:** address TF-IDF 3.9% of true pairs, name TF-IDF 0.74%, `place_premise` 0.70%, phonetic + place 0.21%, rare token 0.10%. The remaining blockers each add less than 0.1%.
- The India exact keys `in_loc_name` and `in_loc_premise` are almost entirely redundant with the other blockers.
- `place_premise` ((place, premise number) + loose name filter) is the key India blocker: dropping it costs −1.8% India recall on DEV.
- The synthetic noise puts typos on one word of "surname + business word" names ("Ballard Atnioiy"). Character n-grams then rank a typo'd rare word below intact common ones. Exact rare-token keys recover these.

**Remaining misses** (full train, 156k pairs)
- 22% are empty target addresses where the similar name is ranked out.
- 20% are similar name + shared place, ranked out or capped.
- 13% are similar address, ranked out.
- 10% are Indian-script names with noisy addresses.
- 10% are unrelated (renamed) names whose address is ranked out.
- Most are ranking problems (K), not missing keys. India ≥ 0.98 was not reached even at 142 candidates per S1 on DEV.

**Size shift on test**
- Candidates per S1 on test DEV: US 45 (train 48), India 86 (train 67), **France 117**.
- France's `place_premise` alone gives about 32 candidates per S1, because French names are generic ("club", "amicale", "école") and the token-set filter is loose.
