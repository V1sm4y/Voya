# OOF error analysis (decision {'t_keep': 0.6, 'ratio': 0.4, 't_null': 0.6, 'assign': True, 'cap': 11, 'cv_macro_f05': 0.9711526236034284, 'cv_by_country': {'India': 0.9651, 'US': 0.9752}})

CV macro F0.5 at this setting: **0.9712**

oracle_gain = macro F0.5 if exactly this category's errors were fixed, minus the current score.

| category           |   pairs |   India |    US |   oracle_gain |
|:-------------------|--------:|--------:|------:|--------------:|
| blocking miss      |  142564 |   85899 | 56665 |       0.00689 |
| decoy false merge  |   56710 |   27780 | 28930 |       0.00689 |
| empty-address miss |  138063 |   46115 | 91948 |       0.00579 |
| other low-p miss   |   62695 |   33065 | 29630 |       0.00324 |
| wrong-S1 merge     |   28501 |   10026 | 18475 |       0.00319 |
| generic-name merge |   14194 |    7228 |  6966 |       0.00155 |
| lost to assignment |   22258 |    9041 | 13217 |       0.00098 |
| address-only miss  |   16012 |    6922 |  9090 |       0.00068 |
| Indic-script miss  |    8975 |    8975 |     0 |       0.00052 |

### examples: decoy false merge

|     p | s1_name                                    | s1_addr                                                                                                                         | t_name                                 | t_addr                                                                                                              |
|------:|:-------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------|:--------------------------------------------------------------------------------------------------------------------|
| 0.992 | Celka's Prime Data Partners                | 4242 216th Place, Issaquah, WA                                                                                                  | Ce1ka's Prime Data Partners Pamtners   | #4249 216TH PLACE, ISSAQUAH, WA                                                                                     |
| 0.644 | Maloney's Massage                          | 890 Ravensbury Street, Westlake Village, CA                                                                                     | Maloney's Massage (Co)                 | 89 RAVENSBURY STREET, WESTLAKE VILLAGE, CA                                                                          |
| 0.645 | Womens Health Clinic                       | 561 Main Street, City Of Waupun, WI                                                                                             | Womens Health Clinic Ltd               | ##574 MAIN ST, WAUPUN DCP, WI                                                                                       |
| 0.748 | Mumbai Chits Limited                       | 1302, Riddhi Garden, L'Out Riddhi Garden Tower A K V Marg, Goregaon West, Mumbai, Mumbai City, Maharashtra                      | Mumbai [Promotion]                     | 1304, Riddhi Garden, L'out Riddhi Garden Tower A K V Marg, Goregaon West, Mumbai City, Mumbai, MH                   |
| 0.825 | Alpha Trading Private Limited              | Andhra Pradesh, Villa A130, Guntur, Rain Tree Park, Dwaraka Krishna Township Opp Nagarjuna University Maingate, Nambu, Ru, Kaza | ఆల్ఫా కన్సల్టింగ్ ప్రైవేట్ లిమిటెడ్    | Villa A139, Rain Tree Park, Dwaraka Krishna Township Opp Nagarjuna University Maingate, Nambu, Ru, Kaza, Guntur, AP |
| 0.888 | Kushinagar Exports (India) Private Limited | 149-150 Shanti Niketan, Vijay Nagar Behind Bombay Hospital, Indore, Madhya Pradesh                                              | Kushinagar Service (India) Private Ltd | Door No 49-155 Shanti Niketan, Vijay Nagar Behind Bombay Hospital, Indore, मध्य प्रदेश                              |
| 0.656 | Natecor                                    | 6777 St Rte 46, Cortland, OH                                                                                                    | Natecor Inc                            | 6779-6781 SAINT RTE 46, MECCA TWP-EAST, OH                                                                          |
| 0.808 | Galarza and Wallace, Inc                   | 11 Moxie Cove Road, Bristol, ME                                                                                                 | Galarza and Wallace, [Holdings]        | 12 MOXIE CVE RD, PMB 9655, ROUND POND TOWNSHIP, ME                                                                  |

### examples: empty-address miss

|     p | s1_name                               | s1_addr                                                                            | t_name                               | t_addr   |
|------:|:--------------------------------------|:-----------------------------------------------------------------------------------|:-------------------------------------|:---------|
| 0.471 | Silver & Davis Quasaredge Inc         | 4205 Satter Drive, Marion County, OR                                               | Silver & Davis Quasaredge Inc        |          |
| 0.184 | Aishani Traders                       | 20 Balaji Nagar Extn, Rakkiyapalayam, Avinashi Tk, Tirupur, Coimbatore, Tamil Nadu | Shri Aishani Trares #59010           |          |
| 0.158 | Blair Copley Center                   | 509 Graceview Drive, Saint Joseph, MN                                              | BLAIR COPLEY CENTER                  |          |
| 0.544 | Agneta Stevens Synchrony Inc          | 1824 Cotton Farm Trail, Leander, TX                                                | Agneta Stevens Synchrony             |          |
| 0.098 | Foster Fuel LLC                       | 201 Keeneland Drive, Oxford, NC                                                    | Foster Fuel                          |          |
| 0.58  | Dailey, Ross and Liriano Software Inc | 901 Burgs Park Drive, City Of Shell Lake, WI                                       | Dailey, Ross and Liiano Software Inc |          |
| 0.472 | Pinnacle Cargo Digital LLC            | 2501 Hancock Drive, Fairfield, CA                                                  | Pinnacle Cargo Digital               |          |
| 0.548 | Tiffani's Book Store Clinic           | 1996 300, Frankfort, IN                                                            | The Tiffani's Book-Store Clinic      |          |

### examples: other low-p miss

|     p | s1_name                             | s1_addr                                                                                                                | t_name                           | t_addr                                                                                |
|------:|:------------------------------------|:-----------------------------------------------------------------------------------------------------------------------|:---------------------------------|:--------------------------------------------------------------------------------------|
| 0.569 | Neeraj Public School Limited        | C/O. Surendrasinh Karansinh Gohil, Nr Darbargadh Chora, Amargadh, Shihor, Bhavnagar, Gujarat                           | Neeraj School Public Limited     | C/O. SURENDRASINH KARANSINH GOHIL, NR DARBARGADH CHORA, AMARGADH, SHIHOR, Gujarat     |
| 0.482 | Supreme Consultancy Private Limited | P.No-31, Shankarpuri Colony, Kamta, Lucknow, Uttar Pradesh                                                             | Supreme Consultancy Private      | P.no-32, Lucknow, UP                                                                  |
| 0.435 | Medford Rate LLC                    | 1052 1, Fairfield, IL                                                                                                  | Medford Medford Rs LLC           | 1052 1, Fairfield, Illinois                                                           |
| 0.241 | Amar Infrastructure (India) Ltd     | B-4/234 Safdarjang Enclave, New Delhi, South Delhi, Delhi                                                              | Amar Infrastructure Ltd Services | B-1-4/234 Safdarjang Enclave, South Delhi, DL                                         |
| 0.483 | White Engineering Limited           | H No: 5-5-35/45/1, Prashanth Nagar Kukatpally, Hyderabad, Telangana                                                    | White Engineering                | Hyderabad, H No: 4-5-5-35/45/1, Hyderabad, Prashanth Nagar Kukatpally, Andhra Pradesh |
| 0.349 | Raj Technology Private Limited      | Noida, Gautam Buddha Nagar, Noida, R-003 Sector 78, Uttar Pradesh                                                      | Raj Technology Private (Limited) | UP, Dadri, No. 300 R-003 Sector 78                                                    |
| 0.06  | Teri's Associates Care              | 922 Blackhawk Drive, Spokane, WA                                                                                       | Teri's Associates Care LLC       | 930 BLACKHAWK DR, SPOKANE, WA                                                         |
| 0.003 | All Healthcare Private Limited      | Mumbai City, Flat No. 3, Mumbai, 2Nd Floor, 118 Thomas Villa Chs, Opp. Cine Max, Sion Circle, Sion (East), Maharashtra | All Healthcare                   | D/1, Mumbai, Mumbai City, महाराष्ट्र                                                  |
