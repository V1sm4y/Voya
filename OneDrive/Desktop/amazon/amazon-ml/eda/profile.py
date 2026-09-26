import pandas as pd, sys
SP=sys.argv[1] if len(sys.argv)>1 else str(__import__('pathlib').Path(__file__).resolve().parents[1]/'data'/'cache')
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',90); pd.set_option('display.max_columns',10)
for split in ['train','test']:
  for s in [1,2,3]:
    name=f'{split}_source{s}'
    df=pd.read_parquet(f'{SP}/{name}.parquet')
    print(f'\n######## {name}: {len(df):,} rows; unique ids {df.entity_id.nunique():,}; id prefix ok {df.entity_id.str.startswith(f"S{s}-").mean():.4f}')
    for c in ['business_name','business_address','country']:
        v=df[c].str.strip()
        placeholder=v.str.lower().isin(['nan','none','null','n/a','na','-','--','.','0','unknown'])
        print(f'  {c}: empty {(v=="").mean()*100:.2f}%  placeholder {placeholder.mean()*100:.3f}%  len mean {v.str.len().mean():.1f} p50 {v.str.len().median():.0f} max {v.str.len().max()}')
    print('  country counts:', df.country.value_counts(dropna=False).to_dict())
    print('  empty addr by country:', (df.business_address.str.strip()=='').groupby(df.country).mean().round(4).to_dict())
    print('  empty name by country:', (df.business_name.str.strip()=='').groupby(df.country).mean().round(4).to_dict())
    print('  dup (name,addr,country) rows:', df.duplicated(['business_name','business_address','country']).sum())
    print(df.sample(10,random_state=7).to_string(index=False))
