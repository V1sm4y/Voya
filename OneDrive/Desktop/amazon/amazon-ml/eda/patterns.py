import pandas as pd, numpy as np, sys, re
SP=sys.argv[1] if len(sys.argv)>1 else str(__import__('pathlib').Path(__file__).resolve().parents[1]/'data'/'cache'); N=200000
pd.set_option('display.width',250); pd.set_option('display.max_columns',30); pd.set_option('display.max_rows',200)
fr=[]
for split in ['train','test']:
  for s in [1,2,3]:
    d=pd.read_parquet(f'{SP}/{split}_source{s}.parquet')
    d=d.groupby('country',group_keys=False).apply(lambda g:g.sample(min(len(g),N),random_state=0))
    d['src']=f'{split[:2]}S{s}'; fr.append(d)
D=pd.concat(fr,ignore_index=True); D['key']=D.src+'/'+D.country.str[:2]
n,a=D.business_name,D.business_address
def rate(mask): return (mask.groupby(D.key).mean()*100).round(1)
def table(dct,title):
    print(f'\n=== {title} (% of records)'); print(pd.DataFrame({k:rate(v) for k,v in dct.items()}).T.to_string())
table({
 'name ALL UPPER': n.str.fullmatch(r'[^a-z]*[A-Z][^a-z]*'),
 'name all lower': n.str.fullmatch(r'[^A-Z]*[a-z][^A-Z]*'),
 'addr ALL UPPER': a.str.fullmatch(r'[^a-z]*[A-Z][^a-z]*'),
 'addr all lower': a.str.fullmatch(r'[^A-Z]*[a-z][^A-Z]*'),
 'name non-ASCII': n.str.contains(r'[^\x00-\x7f]'),
 'name Devanagari': n.str.contains(r'[ऀ-ॿ]'),
 'name other Indic': n.str.contains(r'[ঀ-෿]'),
 'name accented Latin': n.str.contains(r'[À-ɏ]'),
 'addr Devanagari': a.str.contains(r'[ऀ-ॿ]'),
 'addr other Indic': a.str.contains(r'[ঀ-෿]'),
 'addr accented Latin': a.str.contains(r'[À-ɏ]'),
},'Casing & scripts')
table({
 'lead junk (--,***,<<,[..],@,#)': n.str.match(r'^\s*(--|\*+|<<|>>|\[|@|#|~|\.\.)'),
 'bracket tag [..]': n.str.contains(r'\[[^\]]*\]'),
 'double space': n.str.contains('  '),
 '& in name': n.str.contains('&'),
 '" and " in name': n.str.contains(r'\band\b',case=False),
 'comma in name': n.str.contains(','),
 'period in name': n.str.contains(r'\.'),
 'domain (.com/.in/.net)': n.str.contains(r'\.(com|in|net|org|co|fr)\b',case=False),
 'aka (dba/fka/t-a/formerly/c-o)': n.str.contains(r'\b(d/?b/?a|f/?k/?a|a/?k/?a|t/a|formerly|trading as|c/o)\b',case=False),
 'honorific (Sri/Shri/Smt/M/s)': n.str.contains(r'\b(sri|shri|shree|smt|m/s)\b',case=False),
 'person creds (MD/DDS/DO/CPA)': n.str.contains(r'\b(MD|DDS|DO|CPA|DMD|PhD|Esq)\b'),
},'Name punctuation / noise')
suf={'LLC':r'\bL\.?L\.?C\b','Inc':r'\binc(orporated)?\b','Corp/Corporation':r'\bcorp(oration)?\b','Co/Company':r'\b(co|company)\b','Ltd/Limited':r'\b(ltd|limited)\b',
 'Pvt/Private':r'\b(pvt|private)\b','Pvt Ltd combo':r'\b(pvt|private)\.?\s*(ltd|limited)\b','LLP':r'\bL\.?L\.?P\b','LP':r'\bL\.?P\b','PC/PLLC':r'\b(p\.?c|pllc)\b',
 'SARL/EURL':r'\b(s\.?a\.?r\.?l|eurl)\b','SAS/SASU':r'\bs\.?a\.?s\.?u?\b','SA':r'\bs\.?a\b','SCI':r'\bsci\b','Hindi legal (प्राइवेट/लिमिटेड/एलएलपी)':r'(प्राइवेट|लिमिटेड|एलएलपी)','any legal suffix':r'\b(llc|inc|corp|corporation|ltd|limited|pvt|private|llp|lp|pc|pllc|sarl|sas|sasu|sa|eurl|sci|co|company|incorporated)\b'}
table({k:n.str.contains(v,case=False) for k,v in suf.items()},'Legal suffix presence')
leg=r'^\s*(llc|inc|corp|ltd|pvt|llp|sarl|sas|sa|sci|co)\b'
table({'legal token at START of name':n.str.contains(leg,case=False)},'Legal token position')
table({
 'US ZIP 5-digit (\\b\\d{5}\\b)': a.str.contains(r'\b\d{5}(-\d{4})?\b') & D.country.eq('US'),
 'IN PIN 6-digit': a.str.contains(r'\b\d{3}\s?\d{3}\b') & D.country.eq('India'),
 'FR code postal 5-digit': a.str.contains(r'\b\d{5}\b') & D.country.eq('France'),
 'starts with house number': a.str.match(r'^\s*#?\d'),
 'contains any digit': a.str.contains(r'\d'),
 'empty address': a.str.strip().eq(''),
 'NULL token (null/<NULL>/N/A)': a.str.contains(r'(?:^|,\s*)(?:<?null>?|none|n/?a|nan)\s*(?:,|$)',case=False),
 'Unit/Apt/Suite/Ste': a.str.contains(r'\b(unit|apt|suite|ste|#)\b',case=False),
 'Unit UNIT dup': a.str.contains(r'Unit (UNIT|APT|STE|SUITE)'),
 'PO Box': a.str.contains(r'\bp\.?\s?o\.?\s?box\b',case=False),
 'street full (Street/Road/Avenue/Drive)': a.str.contains(r'\b(street|road|avenue|drive|boulevard|lane|court|place)\b',case=False),
 'street abbr (St/Rd/Ave/Dr/Blvd/Ln/Ct)': a.str.contains(r'\b(st|rd|ave|dr|blvd|ln|ct|pl|hwy|pkwy)\b\.?',case=False),
 'landmark Near/Nr/Opp/Behind/Beside': a.str.contains(r'\b(near|nr|opp|opposite|behind|beside|next to|adj)\b\.?',case=False),
 'IN house/plot/flat/shop/khasra': a.str.contains(r'\b(h\.?\s?no|house no|plot|flat|shop|kh\.?\s?no|khasra|door no|d\.?\s?no|sy\.?\s?no|survey)\b',case=False),
 'IN floor (1st Floor/Gr Floor)': a.str.contains(r'\b(floor|flr|fl)\b',case=False),
 'IN Dist/Tal/Tq/PO/Village': a.str.contains(r'\b(dist|district|tal|taluka|tq|tehsil|p\.?o|village|vill|post)\b\.?',case=False),
 'FR rue/av/bd/chemin/place/allee': a.str.contains(r'\b(rue|r\.|av|avenue|bd|boulevard|chemin|ch|place|pl|all[ée]e|impasse|imp|quai|route|rte)\b',case=False) & D.country.eq('France'),
 'FR abbreviated (R./AV/BD/CH/RTE/IMP)': a.str.contains(r'\b(R\.|AV|BD|CH|RTE|IMP|PL|ALL)\b') & D.country.eq('France'),
},'Address components')
# number of comma parts, and last part style (state code vs full name)
D['nparts']=a.map(lambda x:len([p for p in x.split(',') if p.strip()]))
print('\n=== comma-separated address parts: mean / pct with <=2 parts'); print(pd.DataFrame({'mean':D.nparts.groupby(D.key).mean().round(2),'<=2 parts %':(D.nparts<=2).groupby(D.key).mean().mul(100).round(1)}).T.to_string())
D['last']=a.map(lambda x:([p.strip() for p in x.split(',') if p.strip()] or [''])[-1])
D['first']=a.map(lambda x:([p.strip() for p in x.split(',') if p.strip()] or [''])[0])
print('\n=== last address component = 2-letter code %:'); print(rate(D['last'].str.fullmatch(r'[A-Z]{2}')).to_string())
print('\n=== first address component = 2-letter code or state-like word (reordering) %:'); print(rate(D['first'].str.fullmatch(r'[A-Z]{2}')).to_string())
for k in sorted(D.key.unique()):
    vc=D.loc[D.key==k,'last'].value_counts().head(12)
    print(f'\n--- top last-components {k}:', ', '.join(f'{i} ({v/ (D.key==k).sum()*100:.1f}%)' for i,v in vc.items()))
# name tokens top
for c in ['US','India','France']:
    for src in ['trS1','trS2','trS3','teS1','teS2','teS3']:
        sub=D[(D.country==c)&(D.src==src)]
        if not len(sub): continue
        t=sub.business_name.str.lower().str.findall(r'[a-zÀ-ɏ]+[.]?').explode().value_counts().head(30)
        print(f'\n--- top name tokens {src}/{c}:',' '.join(f'{i}:{v}' for i,v in t.items()))
