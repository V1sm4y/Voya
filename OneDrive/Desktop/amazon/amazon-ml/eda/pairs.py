import pandas as pd, numpy as np, sys, re
from rapidfuzz import fuzz
SP=sys.argv[1] if len(sys.argv)>1 else str(__import__('pathlib').Path(__file__).resolve().parents[1]/'data'/'cache'); rng=np.random.default_rng(42)
s1=pd.read_parquet(f'{SP}/train_source1.parquet').set_index('entity_id')
oth=pd.concat([pd.read_parquet(f'{SP}/train_source2.parquet'),pd.read_parquet(f'{SP}/train_source3.parquet')],ignore_index=True)
gt=pd.read_parquet(f'{SP}/eda_gt_parsed.parquet')
ex=gt[['source1_entity_id','ids']].explode('ids').dropna()
owner=dict(zip(ex.ids,ex.source1_entity_id))
oth_i=oth.set_index('entity_id')
def md(s): return str(s).replace('|','\\|')
# --- 30 true pairs
samp=ex.sample(30,random_state=1)
rows=[]
for a,b in zip(samp.source1_entity_id,samp.ids):
    x=s1.loc[a]; y=oth_i.loc[b]
    rows.append(f"| {x.country} | {md(x.business_name)} | {md(x.business_address)} | {b[:2]} | {md(y.business_name)} | {md(y.business_address)} |")
out=['### 30 random TRUE match pairs','','| C | S1 name | S1 address | Src | Match name | Match address |','|---|---|---|---|---|---|']+rows
# --- hard negatives
STOP=set('the and of & inc llc ltd limited private pvt corp corporation co company group services service associates india llp pc sarl sas sa enterprises solutions international holdings center centre trust foundation society de la le les du des et'.split())
tok=lambda s:[t for t in re.findall(r'[a-z0-9]+',s.lower()) if len(t)>=3 and t not in STOP]
cs=s1.sample(4000,random_state=3).reset_index()
cs['city']=cs.business_address.map(lambda a:[p.strip() for p in a.split(',')][-2].lower() if a.count(',')>=2 else '')
cs['toks']=cs.business_name.map(tok)
T=set(t for l in cs.toks for t in l)
oth['toks']=oth.business_name.map(lambda s:[t for t in tok(s) if t in T])
e=oth[['entity_id','toks']].explode('toks').dropna()
df_=e.toks.value_counts(); e=e[e.toks.map(df_)<=3000]
print('postings kept',len(e),flush=True)
c=cs[['entity_id','toks']].explode('toks').dropna().rename(columns={'entity_id':'s1'})
pairs=c.merge(e,on='toks').drop_duplicates(['s1','entity_id'])
pairs=pairs[pairs.entity_id.map(owner.get)!=pairs.s1]
pairs=pairs.merge(cs[['entity_id','business_name','business_address','country','city']].rename(columns={'entity_id':'s1'}),on='s1').merge(oth[['entity_id','business_name','business_address','country']],on='entity_id',suffixes=('','_t'))
pairs=pairs[(pairs.country==pairs.country_t)&(pairs.city.str.len()>=3)]
pairs=pairs[[c_ in a.lower() for c_,a in zip(pairs.city,pairs.business_address_t)]]
pairs['nsim']=[fuzz.token_set_ratio(a.lower(),b.lower()) for a,b in zip(pairs.business_name,pairs.business_name_t)]
pairs=pairs[pairs.nsim>=70].sort_values('nsim',ascending=False).drop_duplicates('s1')
print('hard-negative S1s found (of 4000 sampled):',len(pairs), ' nsim quantiles', pairs.nsim.quantile([.1,.5,.9]).to_dict())
# how often does the hard neg belong to another S1 vs unlinked
pairs['neg_owner']=pairs.entity_id.map(lambda i:'other S1' if i in owner else 'unlinked')
print(pairs.neg_owner.value_counts().to_dict())
h=pairs.sample(min(30,len(pairs)),random_state=5).sort_values('nsim',ascending=False)
out+=['','### 30 HARD NON-matches (same country, S1 city appears in candidate address, shared distinctive name token, token_set_ratio ≥ 70)','','| C | S1 name | S1 address | Src | Non-match name | Non-match address | name sim | belongs to |','|---|---|---|---|---|---|---|---|']
for r in h.itertuples():
    out.append(f"| {r.country} | {md(r.business_name)} | {md(r.business_address)} | {r.entity_id[:2]} | {md(r.business_name_t)} | {md(r.business_address_t)} | {r.nsim:.0f} | {r.neg_owner} |")
open(str(__import__('pathlib').Path(__file__).parent/'pairs.md'),'w').write('\n'.join(out)+'\n')
print('\n'.join(out))
