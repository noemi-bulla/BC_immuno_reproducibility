import os
import random
import pickle
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


#bubble plot
categories = [
    'IMT_CTLA4_2','IMT_COMBO_5','IMT_COMBO_4','IMT_COMBO_3','IMT_COMBO_2','IMT_CTRL_4', 'IMT_CTRL_3', 'IMT_CTRL_2', 'IMT_CTRL_1', 'IME_dep_8', 'IME_dep_7',
    'IME_dep_6', 'IME_dep_5', 'IME_dep_4', 'IME_dep_3', 'IME_dep_2', 'IME_dep_1',
    'IME_CTRL_8', 'IME_CTRL_7', 'IME_CTRL_6',
    'IME_CTRL_5', 'IME_CTRL_4', 'IME_CTRL_3', 'IME_CTRL_2', 'IME_CTRL_1', 'ref_4T1_GBC'
]
df_freq['sample'] = pd.Categorical(df_freq['sample'], categories=categories)
df_freq.sort_values(by=['sample'], inplace=True)
#Random colors for clones
# clones = df_freq['GBC'].unique()
# random.seed(1235)
# clones_colors = { 
#     clone : color for clone, color in \
#     zip(
#         clones, 
#         list(
#             ''.join( ['#'] + [random.choice('ABCDEF0123456789') for i in range(6)] )  \
#             for _ in range(clones.size)
#         )
#     )
# }
# with open(os.path.join(path_data, 'clones_colors_sc.pickle'), 'wb') as f:
#     pickle.dump(clones_colors, f)

with open(os.path.join(path_data, 'clones_colors_sc.pickle'), 'rb') as f:
    clones_colors = pickle.load(f)

df_freq['area_plot'] = df_freq['freq'] * (3000-5) + 5

fig, ax = plt.subplots(figsize=(6, 6))
scatter(df_freq, 'GBC', 'sample', by='GBC', c=clones_colors, s='area_plot', a=0.5, ax=ax)
format_ax(ax, title='Clones by sample', xlabel='Clones', xticks='')
fig.tight_layout()
fig.savefig(os.path.join(path_results, 'bubble_plot.png'), dpi=300)




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



#bar plot n_clones by sample 
order=['ref','IME_CTRL','IME_dep','IMT_CTRL','IMT_COMBO','IMT_CTLA4']
sorted_samples = sorted(
    [s for c in order for s in df_sample.index if c in s],
    key=lambda x: (
        next((order.index(c) for c in order if c in x), len(order)), 
        int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else float('inf')  
    )
)

df_sample_sorted = df_sample.loc[sorted_samples]
fig, ax = plt.subplots(figsize=(10,4.5))
bar(df_sample_sorted, 'n_clones', 'sample', s=.70, c='k', a=.7, ax=ax)
format_ax(ax=ax, title='n clones by sample', ylabel='n_clones', xticks=df_sample_sorted.index, rotx=90)
ax.spines[['left', 'top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(path_results, f'n_clones.png'), dpi=500)



#box,strip n_clones by condition
fig, ax = plt.subplots(figsize=(4,4))
box(df_sample, x='origin', y='n_clones', ax=ax, with_stats=True, 
    pairs=[['IME_CTRL', 'IME_dep'], ['IMT_CTRL', 'IMT_COMBO'], ['IMT_COMBO', 'IMT_CTLA4']], 
    order=['ref','IME_CTRL','IME_dep','IMT_CTRL','IMT_COMBO','IMT_CTLA4']
)
strip(df_sample, x='origin', y='n_clones', ax=ax, order=['ref','IME_CTRL','IME_dep','IMT_CTRL','IMT_COMBO','IMT_CTLA4'], c='k')
format_ax(ax=ax, title='n_clones by condition', ylabel='n_clones', rotx=90, reduce_spines=True)
fig.tight_layout()
fig.savefig(os.path.join(path_results, f'n_clones_condition.png'), dpi=300)



#box,strip SH by condition
fig, ax = plt.subplots(figsize=(4,4))
box(df_sample, x='origin', y='SH', ax=ax, with_stats=True, 
    pairs=[['IME_CTRL', 'IME_dep'], ['IMT_CTRL', 'IMT_COMBO'], ['IMT_COMBO', 'IMT_CTLA4']], 
    order=['ref','IME_CTRL','IME_dep','IMT_CTRL','IMT_COMBO','IMT_CTLA4']
)
strip(df_sample, x='origin', y='SH', ax=ax, order=['ref','IME_CTRL','IME_dep','IMT_CTRL','IMT_COMBO','IMT_CTLA4'], c='k')
format_ax(ax=ax, title='Shannon Entropy samples', ylabel='SH', rotx=90, reduce_spines=True)
fig.tight_layout()
fig.savefig(os.path.join(path_results, f'SH.png'), dpi=300)




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




#Heatmap common_clones
order=['ref','IME_CTRL','IME_dep','IMT_CTRL','IMT_COMBO','IMT_CTLA4']
ordered_samples = sorted(
    [s for c in order for s in common.index if c in s],
    key=lambda x: (
        next((order.index(c) for c in order if c in x), len(order)),
        int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else float('inf')
    )
)
df_reordered = common.loc[ordered_samples, ordered_samples]
fig, ax = plt.subplots(figsize=(10,8))
plot_heatmap(df_reordered, ax=ax, annot=True, title='n common clones', x_names_size=8, y_names_size=8, annot_size=5.5)
fig.tight_layout()
fig.savefig(os.path.join(path_results, f'common.png'), dpi=300)













