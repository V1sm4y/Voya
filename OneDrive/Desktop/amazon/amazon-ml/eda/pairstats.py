import pandas as pd, numpy as np, sys, re, unicodedata
from rapidfuzz import fuzz
SP=sys.argv[1] if len(sys.argv)>1 else str(__import__('pathlib').Path(__file__).resolve().parents[1]/'data'/'cache')
s1=pd.read_parquet(f'{SP}/train_source1.parquet')
oth=pd.concat([pd.read_parquet(f'{SP}/train_source2.parquet'),pd.read_parquet(f'{SP}/train_source3.parquet')],ignore_index=True)
gt=pd.read_parquet(f'{SP}/eda_gt_parsed.parquet')
ex=gt[['source1_entity_id','ids']].explode('ids').dropna().sample(300000,random_state=0)
p=ex.merge(s1,left_on='source1_entity_id',right_on='entity_id').merge(oth,left_on='ids',right_on='entity_id',suffixes=('_1','_2'))
LEG=r'\b(llc|l\.l\.c\.?|inc\.?|incorporated|corp\.?|corporation|co\.?|company|ltd\.?|limited|pvt\.?|private|llp|lp|pc|pllc|sarl|sas|sasu|sa|eurl|sci|public)\b'
def norm(s):
    s=unicodedata.normalize('NFKD',s.lower()); s=''.join(c for c in s if not unicodedata.combining(c))
    s=re.sub(LEG,' ',s); s=re.sub(r'[^a-z0-9ऀ-෿ ]',' ',s); return ' '.join(s.split())
p['n1']=p.business_name_1.map(norm); p['n2']=p.business_name_2.map(norm)
p['latin2']=~p.business_name_2.str.contains(r'[ऀ-෿]')
p['tsr']=[fuzz.token_set_ratio(a,b) for a,b in zip(p.n1,p.n2)]
p['exact']=p.n1==p.n2
num=lambda s:(re.findall(r'\d+',s) or [None])[0]
def housenum(a):
    m=re.search(r'(?:^|,\s*)#?\s*(\d+)[A-Za-z]?\s+[A-Za-z]',a); return m.group(1) if m else None
p['h1']=p.business_address_1.map(housenum); p['h2']=p.business_address_2.map(housenum)
p['a2empty']=p.business_address_2.str.strip()==''
def city1(a):
    parts=[x.strip() for x in a.split(',') if x.strip()]; return parts[-2].lower() if len(parts)>=3 else None
p['city']=p.business_address_1.map(city1)
p['city_in2']=[(c is not None) and (c in b.lower()) for c,b in zip(p.city,p.business_address_2)]
p['src']=p.ids.str[:2]
R=[]
for (c,s),g in p.groupby(['country_1','src']):
    lat=g[g.latin2]
    hh=g[g.h1.notna()&g.h2.notna()]
    R.append({'country':c,'src':s,'n':len(g),
      'target name non-Latin script %':100*(~g.latin2).mean(),
      'norm-name exact % (Latin only)':100*lat.exact.mean(),
      'token_set>=90 % (Latin)':100*(lat.tsr>=90).mean(),
      'token_set<60 % (Latin; renamed/very noisy)':100*(lat.tsr<60).mean(),
      'target addr empty %':100*g.a2empty.mean(),
      'house# equal % (both present)':100*(hh.h1==hh.h2).mean(),
      'house# truncated/suffix %':100*((hh.h1!=hh.h2)&[b.endswith(a) or a.endswith(b) for a,b in zip(hh.h1,hh.h2)]).mean(),
      'S1 city string found in target addr %':100*g.city_in2.mean()})
print(pd.DataFrame(R).round(1).T.to_string())
print('\ntoken_set_ratio quantiles (Latin targets):',p[p.latin2].tsr.quantile([.01,.05,.1,.25,.5]).to_dict())
print('\nexamples of Latin targets with token_set<50 (possible renames):')
print(p[p.latin2&(p.tsr<50)].sample(15,random_state=2)[['business_name_1','business_address_1','business_name_2','business_address_2']].to_string(index=False))
# siblings: pairwise similarity among matches from same source for same S1
sib=gt[gt.n2>=2].sample(20000,random_state=1)
names=dict(zip(oth.entity_id,oth.business_name)); addrs=dict(zip(oth.entity_id,oth.business_address))
sims=[];asims=[]
for l in sib.ids:
    s2=[i for i in l if i.startswith('S2-')][:2]
    sims.append(fuzz.token_set_ratio(norm(names[s2[0]]),norm(names[s2[1]]))); asims.append(fuzz.token_set_ratio(addrs[s2[0]].lower(),addrs[s2[1]].lower()))
print('\nsame-source sibling (two S2 matches of one S1): name tsr median %.0f, p10 %.0f; addr tsr median %.0f, p10 %.0f'%(np.median(sims),np.percentile(sims,10),np.median(asims),np.percentile(asims,10)))
