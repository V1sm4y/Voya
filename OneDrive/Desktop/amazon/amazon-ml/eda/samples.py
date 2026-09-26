import pandas as pd, sys
SP=sys.argv[1] if len(sys.argv)>1 else str(__import__('pathlib').Path(__file__).resolve().parents[1]/'data'/'cache'); out=[]
md=lambda s:str(s).replace('|','\\|') or '*(empty)*'
for split in ['train','test']:
  for s in [1,2,3]:
    df=pd.read_parquet(f'{SP}/{split}_source{s}.parquet').sample(10,random_state=7)
    out+=[f'#### {split}_source{s}.tsv','','| entity_id | business_name | business_address | country |','|---|---|---|---|']
    out+=[f'| {r.entity_id} | {md(r.business_name)} | {md(r.business_address)} | {r.country} |' for r in df.itertuples()]+['']
open(str(__import__('pathlib').Path(__file__).parent/'samples.md'),'w').write('\n'.join(out))
