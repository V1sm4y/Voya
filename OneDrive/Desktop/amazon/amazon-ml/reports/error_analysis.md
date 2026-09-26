# OOF error analysis (decision {'t_keep': 0.7, 'ratio': 0.8, 't_null': 0.7, 'assign': True, 'cap': 11, 'cv_macro_f05': 0.9741057723544665, 'cv_by_country': {'India': 0.9672, 'US': 0.9787}})

CV macro F0.5 at this setting: **0.9741**

oracle_gain = macro F0.5 if exactly this category's errors were fixed, minus the current score.

| category           |   pairs |   India |    US |   oracle_gain |
|:-------------------|--------:|--------:|------:|--------------:|
| blocking miss      |  142564 |   85899 | 56665 |       0.00701 |
| other low-p miss   |  117438 |   59277 | 58161 |       0.00568 |
| empty-address miss |  133867 |   46429 | 87438 |       0.00568 |
| decoy false merge  |   33255 |   15683 | 17572 |       0.00437 |
| wrong-S1 merge     |   12874 |    6184 |  6690 |       0.00151 |
| address-only miss  |   25345 |   11212 | 14133 |       0.00103 |
| Indic-script miss  |   14664 |   14664 |     0 |       0.0008  |
| generic-name merge |    5270 |    2950 |  2320 |       0.00064 |
| lost to assignment |    6098 |    3432 |  2666 |       0.00034 |

### examples: other low-p miss

|     p | s1_name                         | s1_addr                                                                                                                               | t_name                           | t_addr                                                                                                                      |
|------:|:--------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------|:----------------------------------------------------------------------------------------------------------------------------|
| 0.58  | Metropolitan Council            | 22-24 78 Street, East Elmhurst, NY                                                                                                    | Metropolitan Council             | New York, East Elmhurst, 1835 78 St                                                                                         |
| 0.01  | Ram Media Pvt Ltd               | C/O Shri Pandurang Sukdeorao Akhare, K. Sarap, Knh Rl 549, K. Sarap, Akola, Maharashtra                                               | Ram Media                        | No. 181 C/o Shri Pandurang Sukdeorao Akhare, Akola, MH                                                                      |
| 0.781 | Anil State LLP                  | 10-50-19/3, Soudamani Complex, Opp Hsbc Call Center Near Hsbc Hitech Centre, Siripuram, Visakhapatnam, Vishakhapatnam, Andhra Pradesh | Anil Softe LLP                   | A-10-50-19/3, Soudamani Complex, Opp Hsbc Call Center Near Hsbc Hitech Centre, Siripuram, Visakhapatnam, Vishakhapatnam, AP |
| 0.326 | Tripura (India) International   | No.40 2Nd Floor 6Th Main, Punya Bumi Layout, Kalkere, Bangalore North, Bangalore, Karnataka                                           | Tripura (tfai) International     | NO.38 2ND FLOOR 6TH MAIN, PUNYA BUMI LAYOUT, KALKERE, BANGALORE NORTH, Karnataka                                            |
| 0.289 | Happy Cleaning Service          | 1008 Mildred Drive, Alcoa, TN                                                                                                         | Happy Cleaning Service Inc       | 1009 Mildred Drive, Alcoa, Tennessee                                                                                        |
| 0.727 | Shelton and Williams Roofing    | 29720 Jeans Road, Unit Space 9, Veneta, OR                                                                                            | Shelton and Williams Roofing Inc | 29721 Jeans Road, Unit Space 9, Veneta, Oregon                                                                              |
| 0.001 | Delhi Solutions Private Limited | 5Th Floor, North West, Hb Twin Tower, Plot No. A 2, 3 And 4, Netaji Subhash Place, Wazirpur District, Center, Delhi                   | Delhi Solutions Private          | No 4-5Th Floor, North West, Delhi, DL                                                                                       |
| 0.171 | Star Properties Private Limited | Jaipur, P No. 58 Saini Colony, Rajasthan, Kartarpura Mahesh Nagar Jaipur                                                              | Star Properties Limited [Center] | ###848 P No. 58 Saini Colony, Kartarpura Mahesh Nagar Jaipur, Jaipur, RJ                                                    |

### examples: empty-address miss

|     p | s1_name                            | s1_addr                                                                                                            | t_name                         | t_addr   |
|------:|:-----------------------------------|:-------------------------------------------------------------------------------------------------------------------|:-------------------------------|:---------|
| 0.029 | Gold Consulting Pvt Ltd            | 476, Expression Satyasha, Front Of Yamaha Showroom Bailey Road, Rukanpura, Patna, Bihar                            | Smt Gold Cónsulting Pvt        |          |
| 0.738 | Exclusive Brothers Private Limited | 1218, Build-A/60, Grd Flr, M.I.G. Bldg, Azad Nagar No-2, Veera Desai Rd, Andheri, Mumbai, Mumbai City, Maharashtra | Exclusive Brothers Private Ltd |          |
| 0.721 | Community Coalition III            | 804 Devin Court, Union, KY                                                                                         | COMMUNITY COALITION III        |          |
| 0.036 | Empire Equity Partners             | 5122 Forest Avenue, Kansas City, KS                                                                                | Empire Equity Potanes          |          |
| 0.391 | Maness, Lesher and Rayas Allegro   | 99 Maple Street, Bellingham, MA                                                                                    | Maness, Lesher-&               |          |
| 0.77  | Kestiq Electronic L.L.C.           | 4404 Mt Baldy Lane, Spokane, WA                                                                                    | Kestiq Electronic Llc          |          |
| 0.454 | BDY Cohen PLLC                     | 17340 Densmore Avenue, Shoreline, WA                                                                               | Bdy Bdy Cohen Pllc             |          |
| 0.026 | Bharat Solutions Pvt Ltd           | P-92 Lgfchitranjan Park, New Delhi, South Delhi, Delhi                                                             | Bharat Solutions Pvt           |          |

### examples: decoy false merge

|     p | s1_name                          | s1_addr                                    | t_name                           | t_addr                                           |
|------:|:---------------------------------|:-------------------------------------------|:---------------------------------|:-------------------------------------------------|
| 0.823 | Harper Enterprises               | 3400 Barger Drive, VA, Fairfax County      | Harper Enterprises Llc           | #3405 BARGER DR, FALLS CHURCH, VA                |
| 0.858 | Gianina's Coastal Textiles       | 701 Apache Drive, Wallburg, NC             | Gianina's Coastal Textiles LLC   | ##712 Apache Dr, Winston-Salem, North Carolina   |
| 0.801 | Lopez, Ryann N., CPA, M.D., P.C. | 3016 Oak Landing Drive, Charlotte, NC      | n., ryann lopez, cpl, m.d., p.c. | 019 Oak Landing Drive, Charlotte, North Carolina |
| 0.745 | Dedyne Trg of Fort Wayne         | 7416 Alleghany Court, Fort Wayne, IN       | Dedyne of Trg Fort Wayne Corp    | ##7423 ALLEGHANY CT, FORT WAYNE, IN              |
| 0.855 | T 3 Secure Krakacquisition, Inc  | 242 Erickson Avenue, Rockingham County, VA | Secure Krakacquisition Inc.      |                                                  |
| 0.92  | Addia Cressman, O.D.             | 251 Logan Street, Bessemer City, NC        | Addia Cressman, O.D. Inc         | #272 LOGAN STREET, BESSEMER CITY, NC             |
| 0.858 | ZMX Unified Beijing PC           | 16768 Dovie Lane, Vance, AL                | ZMX Unified Beijing [Partners]   | Alabama, 16769 Dovie Lane, Vance                 |
| 0.886 | Medrano & Reynolds               | 736 Jackson Street, Nelsonville, OH        | Medrano & Reynolds Ltd           | #743 Jackson Street, Nelsonville, Ohio           |
