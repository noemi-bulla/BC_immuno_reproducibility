import os
import random
import pickle
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from plotting_utils._plotting import *
matplotlib.use('macOSX')

#Path
path_main = '/Users/ieo7295/Desktop/BC_immuno_reproducibility'
path_data = os.path.join(path_main, 'data','summary_bulk_240125_nospikeins') 
path_data_rt=os.path.join(path_main,'data','summary_bulk_rt_050225')
path_results = os.path.join(path_main, 'results', 'clonal_rt_050225')

# Read prevalences
df = pd.read_csv(os.path.join(path_data, 'bulk_GBC_reference.csv'), index_col=0)
df_rt = pd.read_csv(os.path.join(path_data_rt,'bulk_GBC_reference.csv'), index_col=0)
common = pd.read_csv(os.path.join(path_data, 'common.csv'), index_col=0)
common_rt = pd.read_csv(os.path.join(path_data_rt,'common.csv'),index_col=0)

#Merge dataframe keep IME 
df=df[df['sample'].str.contains('IME')]
df_m = pd.concat([df, df_rt], axis=0, ignore_index=False)

tests = [ df_m['sample'].str.contains('IME_CTRL'), df_m['sample'].str.contains('IME_dep'), 
         df_m['sample'].str.contains('IME_RT_'), df_m['sample'].str.contains('IME_RTdep')] 
df_m['origin'] = np.select(tests, ['IME_CTRL','IME_dep','IME_RT','IME_RTdep'], default='ref')


#GBC,sample,read_count,origin,freq,cum_freq
df_freq=(df_m.reset_index().rename(columns={'index':'GBC'})  
        .groupby('sample')
        .apply(lambda x: x.assign(
        freq=x['read_count'] / x['read_count'].sum(),    
        cum_freq=(x['read_count'] / x['read_count'].sum()).cumsum()
    ))
    .reset_index(drop=True)
)

df_freq.to_csv(os.path.join(path_results,'rel_freq.csv'))

#bubble plot
categories = [
    'IME_RT_8','IME_RT_7','IME_RT_6','IME_RT_4','IME_RT_3','IME_RT_2','IME_RT_1','IME_RTdep_4', 'IME_RTdep_3', 'IME_RTdep_2','IME_RTdep_1','IME_dep_8', 'IME_dep_7',
    'IME_dep_6','IME_dep_5','IME_dep_4', 'IME_dep_3', 'IME_dep_2', 'IME_dep_1',
    'IME_CTRL_8','IME_CTRL_7', 'IME_CTRL_6',
    'IME_CTRL_5','IME_CTRL_4', 'IME_CTRL_3', 'IME_CTRL_2', 'IME_CTRL_1'
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
ax.text(.3, .23, f'n clones total: {df_freq["GBC"].unique().size}', transform=ax.transAxes)
fig.tight_layout()
fig.savefig(os.path.join(path_results,'bubble_plot.png'),dpi=300)


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
order=['IME_RT_','IME_RTdep','IME_CTRL','IME_dep']
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
    pairs=[['IME_dep','IME_RT'], ['IME_RTdep', 'IME_dep'], ['IME_RT', 'IME_RTdep'],['IME_CTRL','IME_RT']], 
    order=['IME_RT','IME_RTdep','IME_CTRL','IME_dep']
)
strip(df_sample, x='origin', y='n_clones', ax=ax, order=['IME_RT','IME_RTdep','IME_CTRL','IME_dep'], c='k')
ax.set_yscale('log', base=2)
y_min, y_max = ax.get_ylim()
ticks = [2**i for i in range(int(np.log2(y_min)), int(np.log2(y_max)) + 1)]
ax.set_yticks(ticks)
ax.set_yticklabels([str(tick) for tick in ticks])
format_ax(ax=ax, title='n_clones by condition', ylabel='n_clones', rotx=90, reduce_spines=True)
fig.tight_layout()
fig.savefig(os.path.join(path_results, f'n_clones_condition.png'), dpi=300)



#box,strip SH by condition
fig, ax = plt.subplots(figsize=(4,4))
box(df_sample, x='origin', y='SH', ax=ax, with_stats=True, 
    pairs=[['IME_dep','IME_RT'], ['IME_RTdep', 'IME_dep'], ['IME_RT', 'IME_RTdep'],['IME_CTRL','IME_RT']], 
    order=['IME_RT','IME_RTdep','IME_CTRL','IME_dep']
)
strip(df_sample, x='origin', y='SH', ax=ax, order=['IME_RT','IME_RTdep','IME_CTRL','IME_dep'], c='k')
format_ax(ax=ax, title='Shannon Entropy samples', ylabel='SH', rotx=90, reduce_spines=True)
fig.tight_layout()
fig.savefig(os.path.join(path_results, f'SH.png'), dpi=300)