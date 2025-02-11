import os
import random
import pickle
import pandas as pd
import numpy as np
import re
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

#df for comparison of IME_dep, IME_RT, IME_RTdep
selected_samples=['IME_dep_1','IME_dep_2','IME_dep_3','IME_dep_4','IME_dep_5','IME_dep_6','IME_dep_7',
                  'IME_dep_8','IME_RT_1','IME_RT_1','IME_RT_2','IME_RT_3','IME_RT_4','IME_RT_6',
                  'IME_RT_7','IME_RT_8','IME_RTdep_1','IME_RTdep_2','IME_RTdep_3','IME_RTdep_4']

tests = [df_m['sample'].str.contains('IME_dep'), 
         df_m['sample'].str.contains('IME_RT_'), df_m['sample'].str.contains('IME_RTdep')] 
df_m['origin'] = np.select(tests, ['IME_dep','IME_RT','IME_RTdep'], default='ref')
df_m=df_m[df_m['sample'].isin(selected_samples)]

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
    'IME_RT_8','IME_RT_7','IME_RT_6','IME_RT_4','IME_RT_3','IME_RT_2','IME_RT_1','IME_RTdep_4','IME_RTdep_3',
    'IME_RTdep_2','IME_RTdep_1','IME_dep_8','IME_dep_7',
    'IME_dep_6','IME_dep_5','IME_dep_4','IME_dep_3','IME_dep_2','IME_dep_1',
    'IME_CTRL_8','IME_CTRL_7','IME_CTRL_6',
    'IME_CTRL_5','IME_CTRL_4','IME_CTRL_3','IME_CTRL_2','IME_CTRL_1'
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
plt.show()


#bubble plot 20%,50%,70% 
#n of clones with freq>10% common in 20%,50%,70% of samples of 1 condition
#clones freq > 10%
threshold = 0.1
criteria = [0.2, 0.5, 0.7]
categories = [
    'IME_RT_8','IME_RT_7','IME_RT_6','IME_RT_4','IME_RT_3','IME_RT_2','IME_RT_1','IME_RTdep_4','IME_RTdep_3',
    'IME_RTdep_2','IME_RTdep_1','IME_dep_8','IME_dep_7',
    'IME_dep_6','IME_dep_5','IME_dep_4','IME_dep_3','IME_dep_2','IME_dep_1',
    'IME_CTRL_8','IME_CTRL_7','IME_CTRL_6',
    'IME_CTRL_5','IME_CTRL_4','IME_CTRL_3','IME_CTRL_2','IME_CTRL_1'
]

df_freq['above_threshold'] = df_freq['freq'] > threshold
clone_summary = df_freq[df_freq['above_threshold']].groupby(['GBC', 'origin'])['sample'].nunique().reset_index(name='num_samples_above_threshold')

total_samples = df_freq.groupby('origin')['sample'].nunique()
clone_summary['percent_samples'] = clone_summary.apply(
    lambda row: row['num_samples_above_threshold'] / total_samples[row['origin']], axis=1
)
selected_clones = {
    criterion: clone_summary[clone_summary['percent_samples'] >= criterion]['GBC'].unique()
    for criterion in criteria
}

df_freq['sample'] = pd.Categorical(df_freq['sample'], categories=categories)
df_freq.sort_values(by=['sample'], inplace=True)

for criterion, clones in selected_clones.items():
    df_freq_filtered = df_freq[df_freq['GBC'].isin(clones)] 
    df_freq_filtered['area_plot'] = df_freq_filtered['freq'] * (3000 - 5) + 5  
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter(df_freq_filtered, 'GBC', 'sample', by='GBC', c=clones_colors, s='area_plot', a=0.5, ax=ax)
    fig.show()
    format_ax(ax, xlabel='Clones', xticks='')
    fig.tight_layout()
    fig.savefig(os.path.join(path_results, f'bubble_plot_{int(criterion*100)}.png'), dpi=300)


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
format_ax(ax=ax, ylabel='n_clones', rotx=90, reduce_spines=True)
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

#heatmap of common clones RT
df=df_m.reset_index().rename(columns={'index':'GBC'})
d = {sample: set(df[df['sample'] == sample]['GBC']) for sample in df['sample'].unique()}

n=len(d)
C=np.zeros((n,n))

sample=list(d.keys())
for i,x in enumerate(sample):
    for j,y in enumerate(sample):
        if i >= j:
            common_clones = len(d[x] & d[y])
            C[i,j] = C[j,i] = common_clones

df_cc=pd.DataFrame(C, index=sample, columns=sample) 

order=['IME_RT','IME_RTdep','IME_CTRL','IME_dep']
prefix_pattern = re.compile(r'^(IME_RT|IME_RTdep|IME_CTRL|IME_dep)_')

ordered_samples = sorted(
    {s for s in df_cc.index if prefix_pattern.match(s)},
    key=lambda x: (
        order.index(prefix_pattern.match(x).group(1)),
        int(re.search(r'_(\d+)$', x).group(1)) if re.search(r'_(\d+)$', x) else float('inf') 
    )
)
df_reordered = df_cc.loc[ordered_samples, ordered_samples]
vmin, vmax= 0, 300
fig, ax = plt.subplots(figsize=(10,8))
plot_heatmap(df_reordered, ax=ax, annot=True, title='n common clones', x_names_size=8, y_names_size=8, annot_size=5.5, cb=False)
sns.heatmap(data=df_reordered, ax=ax, robust=True, cmap="mako", vmin=vmin, vmax=vmax, fmt='.2f',cbar_kws={'fraction':0.05, 'aspect':35, 'pad': 0.02})
fig.tight_layout()
fig.savefig(os.path.join(path_results, f'common_rt.png'), dpi=300)
plt.show()


#heatmap jaccard INDEX
order=['IME_RT','IME_RTdep','IME_CTRL','IME_dep']
prefix_pattern = re.compile(r'^(IME_RT|IME_RTdep|IME_CTRL|IME_dep)_')

ordered_samples = sorted(
    {s for s in df_cc.index if prefix_pattern.match(s)},
    key=lambda x: (
        order.index(prefix_pattern.match(x).group(1)),
        int(re.search(r'_(\d+)$', x).group(1)) if re.search(r'_(\d+)$', x) else float('inf') 
    )
)
common_c = df_cc.loc[ordered_samples, ordered_samples]
set_sizes = np.diag(common_c.values)
JI = np.zeros_like(common_c.values, dtype=float)

for i, row in common_c.iterrows():
    for j, val in row.items():
        union_size = set_sizes[common_c.index.get_loc(i)] + set_sizes[common_c.columns.get_loc(j)] - val
        JI[common_c.index.get_loc(i), common_c.columns.get_loc(j)] = val / union_size if union_size != 0 else 0

JI_df = pd.DataFrame(JI, index=common_c.index, columns=common_c.columns)
order_clustering = leaves_list(linkage(JI_df.values))

vmin, vmax= 0, 0.35
fig, ax = plt.subplots(figsize=(10, 10))
plot_heatmap(JI_df.values[np.ix_(order_clustering, order_clustering)], palette='mako', ax=ax,
             x_names=JI_df.index[order_clustering], y_names=JI_df.index[order_clustering], annot=True, 
             annot_size=8, label='Jaccard Index', shrink=1, cb=False)
sns.heatmap(JI_df.values[np.ix_(order_clustering, order_clustering)], ax=ax, xticklabels=JI_df.index[order_clustering], yticklabels=JI_df.index[order_clustering],
                        robust=True, cmap="mako", vmin=vmin, vmax=vmax, fmt='.2f',cbar_kws={'fraction':0.05, 'aspect':35, 'pad': 0.02})
fig.tight_layout()
plt.savefig(os.path.join(path_results, f'heatmap_Jaccard.png'), dpi= 300)
plt.show()