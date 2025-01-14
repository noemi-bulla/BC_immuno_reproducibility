import os
import pandas as pd
import numpy as np
from plotting_utils._plotting import *

#Path
path_main = '/Users/ieo7295/Desktop/BC_immuno_reproducibility'
path_data = os.path.join(path_main, 'data','summary_bulk_070125') 
path_results = os.path.join(path_main, 'results', 'clonal')

# Read prevalences
df = pd.read_csv(os.path.join(path_data, 'bulk_GBC_reference.csv'), index_col=0)
common = pd.read_csv(os.path.join(path_data, 'common.csv'), index_col=0)

# Reformat
tests = [ df['sample'].str.contains('IME_CTRL'), df['sample'].str.contains('IME_dep'), 
         df['sample'].str.contains('IMT_CTRL'), df['sample'].str.contains('IMT_CTLA4'), df['sample'].str.contains('IMT_COMBO')] 
df['origin'] = np.select(tests, ['IME_CTRL','IME_dep','IMT_CTRL','IMT_CTLA4','IMT_COMBO'], default='ref')




#IME_CTRL vs IME_dep rel_fre, cum_freq of common_cl
for s in df['origin'].unique():
    df_ = df.query('origin == @s')
    c = df_.index
    if s == 'IME_CTRL':
        ctrl_cl= set(c)
    elif s == 'IME_dep':
        dep_cl = set(c)

common_c = ctrl_cl & dep_cl
un_ctrl = ctrl_cl - dep_cl
un_dep = dep_cl - ctrl_cl

print(f"n common clones between IME_CTRL and IME_dep: {len(common_c)}")
print(f"n unique clones in IME_CTRL: {len(un_ctrl)}")
print(f"n unique clones in IME_dep: {len(un_dep)}")
df_c_ctrl= df.query('origin == "IME_CTRL"').loc[df.query('origin == "IME_CTRL"').index.isin(common_c)]
df_c_dep= df.query('origin == "IME_dep"').loc[df.query('origin == "IME_dep"').index.isin(common_c)]
print(df_c_ctrl)
print(df_c_dep)




#n clones
df_ = (
    df.groupby('sample')
    .apply(lambda x: x.index.unique().size)
    .sort_values(ascending=False)
    .to_frame('n_clones')
    .reset_index()
)
#df_.to_csv(os.path.join(path_results, "n_clones.csv"))



#Shannon entropies
SH = []
for s in df['sample'].unique():
    df_ = df.query('sample==@s')
    x = df_['read_count']/df_['read_count'].sum()
    SH.append(-np.sum(np.log10(x) * x))
df_ = (pd.Series(SH, index=df['sample'].unique())
       .to_frame('SH')
       .sort_values(by='SH', ascending=False)
       .reset_index().rename(columns={'index':'sample'})
       .merge(df[['sample','origin']], on='sample')
       .drop_duplicates()
       .set_index('sample'))

#df_.to_csv(os.path.join(path_results, "shannon_entropy.csv"))




#cumulative clone percentage (clone prevalence), all samples
results=[]
for s in df['sample'].unique():
    df_ = df.query('sample==@s')
    z = (df_['read_count'] / df_['read_count'].sum())
    x = (df_['read_count'] / df_['read_count'].sum()).cumsum()
    origin = df.query('sample==@s')['origin'].unique()[0]
    results.append(pd.DataFrame({
       'sample': s, 
       'relative_freq': z,
       'cumulative_freq': x
    }))
final_df = pd.concat(results)

final_df.to_csv(os.path.join(path_results, 'cumulative_frequencies.csv'))