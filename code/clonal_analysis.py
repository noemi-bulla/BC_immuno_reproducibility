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

#IME_CTRL vs IME_dep common_clones
df_common= df.groupby('origin')
ctrl_cl = df_common.get_group('IME_CTRL').index
dep_cl= df_common.get_group('IME_dep').index

common_c = ctrl_cl.intersection(dep_cl)
un_ctrl = ctrl_cl.difference(dep_cl)  
un_dep = dep_cl.difference(ctrl_cl) 



#GBC,sample,origin,freq,cum_freq
df_freq=(df.reset_index().rename(columns={'index':'GBC'})  
        .groupby('sample')
        .apply(lambda x: x.assign(
        freq=x['read_count'] / x['read_count'].sum(),    
        cum_freq=(x['read_count'] / x['read_count'].sum()).cumsum()
    ))
    .reset_index(drop=True)
)


#sample, Shannon entropy, origin, n_clones
SH = []
for s in df_freq['sample'].unique():
    df_ = df_freq.query('sample==@s')
    x = df_['freq']
    SH.append(-np.sum( np.log10(x) * x ))

df_sample = (
    pd.Series(SH, index=df_freq['sample'].unique())
    .to_frame('SH')
    .sort_values(by='SH', ascending=False)
    .reset_index().rename(columns={'index':'sample'})
    .merge(df_freq[['sample', 'origin']], on='sample')
    .drop_duplicates()
    .set_index('sample')
    .assign(
        n_clones=lambda df_: df_.index.map(
            lambda s: df_freq[df_freq['sample'] == s].index.nunique()
        ))
)





#Pivot_table:  
df_freq_wide = (df_freq.pivot(index='GBC', columns='sample', values='freq'))
unique_clones = df_freq_wide[df_freq_wide.notnull().sum(axis=1) == 1].index.tolist()  #unique_clones one entry != 0
common_clones = df_freq_wide[df_freq_wide.drop(columns=['ref_4T1_GBC']).notnull().all(axis=1)].index.tolist()   #common clones all vs all (except ref)
common_clones_ime = df_freq_wide.loc[
                df_freq_wide.filter(like='IME_CTRL').notnull().any(axis=1) & df_freq_wide.filter(like='IME_dep').notnull().any(axis=1)    #common clones IME_CTRL vs IME_dep
].index.tolist()
common_clones_imt= df_freq_wide.loc[
                df_freq_wide.filter(like='IMT_CTRL').notnull().any(axis=1) & df_freq_wide.filter(like='IMT_COMBO').notnull().any(axis=1)  #common clones IMT_CTRL vs IMT_COMBO
].index.tolist()





#GBC, n_sample_ime_ctrl, n_sample_ime_dep, n_sample_imt_ctrl, n_sample_imt_combo,n_sample_tot,mean_freq_ime_ctrl
df_clone=(df_freq_wide.reset_index()
    .assign(
    n_sample_ime_ctrl = lambda x: x.filter(like='IME_CTRL').notnull().sum(axis=1),          
    n_sample_ime_dep = lambda x: x.filter(like='IME_dep').notnull().sum(axis=1),
    n_sample_imt_ctrl = lambda x: x.filter(like='IMT_CTRL').notnull().sum(axis=1),
    n_sample_imt_combo = lambda x: x.filter(like='IMT_COMBO').notnull().sum(axis=1),
    n_sample_tot = lambda x: x.filter(like='IM').notnull().sum(axis=1) + x.filter(like='ref').notnull().sum(axis=1),
    mean_freq_ime_ctrl = lambda x: x.filter(like='IME_CTRL').apply(lambda row: row.dropna().mean(), axis=1),
    mean_freq_ime_dep = lambda x: x.filter(like='IME_dep').apply(lambda row: row.dropna().mean(), axis=1)

)[['GBC','n_sample_ime_ctrl','n_sample_ime_dep','n_sample_imt_ctrl','n_sample_imt_combo','n_sample_tot','mean_freq_ime_ctrl','mean_freq_ime_dep']])














