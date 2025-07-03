import os
import random
import pickle
import re
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict,Iterable,Any
from scipy.cluster.hierarchy import leaves_list, linkage
import plotting_utils as plu 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('macOSX')

#Path
path_main = '/Users/ieo7295/Desktop/BC_immuno_reproducibility'
path_data = os.path.join(path_main, 'data','summary_020725_bulk_met/summary') 
path_results = os.path.join(path_main, 'results', 'clonal_020725_met')

# Read prevalences
df = pd.read_csv(os.path.join(path_data, 'bulk_GBC_reference.csv'), index_col=0)
common = pd.read_csv(os.path.join(path_data, 'common.csv'), index_col=0)

# Reformat
df['sample'] = df['sample'].apply(lambda s: re.sub(r'^(IME_[A-Za-z]+)_(\d+)_met$', r'\1_met_\2', s))
common.index = common.index.map(lambda s: re.sub(r'^(IME_[A-Za-z]+)_(\d+)_met$', r'\1_met_\2', s))
common.columns = common.columns.map(lambda s: re.sub(r'^(IME_[A-Za-z]+)_(\d+)_met$', r'\1_met_\2', s))
tests = [
    df['sample'].str.contains(r'^IME_CTRL(_\d+)?$', regex=True),
    df['sample'].str.contains(r'^IME_dep(_\d+)?$', regex=True),
    df['sample'].str.contains(r'^IME_NSG(_\d+)?$', regex=True),
    df['sample'].str.contains(r'^IME_CTRL_met(_\d+)?$', regex=True),
    df['sample'].str.contains(r'^IME_dep_met(_\d+)?$', regex=True),
    df['sample'].str.contains(r'^IME_NSG_met(_\d+)?$', regex=True)
]
df['origin'] = np.select(tests, ['IME_CTRL','IME_dep','IME_NSG','IME_CTRL_met','IME_dep_met','IME_NSG_met'], default='ref')

#IME_CTRL vs IME_dep common_clones
df_common= df.groupby('origin')
ctrl_cl = df_common.get_group('IME_CTRL').index
dep_cl= df_common.get_group('IME_dep').index

common_c = ctrl_cl.intersection(dep_cl)
un_ctrl = ctrl_cl.difference(dep_cl)  
un_dep = dep_cl.difference(ctrl_cl) 


#Bar plot function 
def bar_new(
    df: pd.DataFrame, 
    x: str, 
    y: str, 
    by: str = None, 
    color: str = None,
    edgecolor: str = 'k',
    categorical_cmap: str | Dict[str, Any] = 'tab10', 
    x_order: Iterable[str] = None,
    by_order: Iterable[str] = None,
    width: float = 0.8,  
    linewidth: float = 0.5,
    alpha: float = 0.8, 
    ax: matplotlib.axes.Axes = None, 
) -> matplotlib.axes.Axes:
    """
    Basic bar plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if by is None:
        sns.barplot(
            data=df, x=x, y=y, ax=ax, 
            order=x_order, 
            color=color,
            alpha=alpha, edgecolor=edgecolor, linewidth=linewidth
        )
    else:
        if by not in df.columns:
            raise KeyError(f'{by} not in df.columns!')
        if not pd.api.types.is_categorical_dtype(df[by]) and not pd.api.types.is_string_dtype(df[by]):
            raise ValueError(f'{by} must be categorical or string!')

        if isinstance(categorical_cmap, str):
            palette = sns.color_palette(categorical_cmap, df[by].nunique())
            cmap = dict(zip(df[by].unique(), palette))
        else:
            cmap = categorical_cmap
        
        assert all(x in cmap for x in df[by].unique())

        sns.barplot(
            data=df, x=x, y=y, ax=ax,
            order=x_order,
            hue=by, hue_order=by_order, palette=cmap,
            alpha=alpha, edgecolor=edgecolor, linewidth=linewidth
        )
        ax.get_legend().remove()

    return ax



#GBC,sample,read_count,origin,freq,cum_freq
df_freq=(df.reset_index().rename(columns={'index':'GBC'})  
        .groupby('sample')
        .apply(lambda x: x.assign(
        freq=x['read_count'] / x['read_count'].sum(),    
        cum_freq=(x['read_count'] / x['read_count'].sum()).cumsum()
    ))
    .reset_index(drop=True)
)

df_freq.to_csv(os.path.join(path_results,'rel_freq.csv'))

categories= [
    'IME_CTRL_9', 'IME_CTRL_10', 'IME_CTRL_11', 'IME_CTRL_12', 
    'IME_CTRL_13', 'IME_CTRL_14', 'IME_CTRL_15', 'IME_CTRL_16',
    'IME_dep_9', 'IME_dep_10', 'IME_dep_11', 'IME_dep_12', 
    'IME_dep_13', 'IME_dep_14', 'IME_dep_15', 'IME_dep_16',
    'IME_NSG_1', 'IME_NSG_2', 'IME_NSG_3', 'IME_NSG_4',
    'IME_NSG_5', 'IME_NSG_6', 'IME_NSG_7', 'IME_NSG_8',
    'IME_CTRL_met_9', 'IME_CTRL_met_10', 'IME_CTRL_met_11', 'IME_CTRL_met_12',
    'IME_CTRL_met_13', 'IME_CTRL_met_14', 'IME_CTRL_met_15', 'IME_CTRL_met_16',
    'IME_dep_met_9', 'IME_dep_met_10', 'IME_dep_met_11', 'IME_dep_met_14',
    'IME_dep_met_15', 'IME_dep_met_16','IME_NSG_met_1', 'IME_NSG_met_2',
    'IME_NSG_met_3', 'IME_NSG_met_4','IME_NSG_met_5', 'IME_NSG_met_6', 'IME_NSG_met_7', 'IME_NSG_met_8'
]

categories_bubble= [
    'IME_CTRL_9', 'IME_CTRL_10', 'IME_CTRL_11', 'IME_CTRL_12', 
    'IME_CTRL_13', 'IME_CTRL_14', 'IME_CTRL_15', 'IME_CTRL_16',
    'IME_dep_9', 'IME_dep_10', 'IME_dep_11', 'IME_dep_12', 
    'IME_dep_13', 'IME_dep_14', 'IME_dep_15', 'IME_dep_16',
    'IME_NSG_1', 'IME_NSG_2', 'IME_NSG_3', 'IME_NSG_4',
    'IME_NSG_5', 'IME_NSG_6', 'IME_NSG_7', 'IME_NSG_8',
    'IME_CTRL_met_9', 'IME_CTRL_met_10', 'IME_CTRL_met_11', 'IME_CTRL_met_12',
    'IME_CTRL_met_13', 'IME_CTRL_met_14', 'IME_CTRL_met_15', 'IME_CTRL_met_16',
    'IME_dep_met_9', 'IME_dep_met_10', 'IME_dep_met_11', 'IME_dep_met_14',
    'IME_dep_met_15', 'IME_dep_met_16','IME_NSG_met_1', 'IME_NSG_met_2',
    'IME_NSG_met_3', 'IME_NSG_met_4','IME_NSG_met_5', 'IME_NSG_met_6', 'IME_NSG_met_7', 'IME_NSG_met_8'
][::-1]


#bubble plot

df_freq['sample'] = pd.Categorical(df_freq['sample'], categories=categories_bubble)
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
# order=['IME_NSG_met','IME_dep_met','IME_CTRL_met','IME_NSG','IME_dep','IME_CTRL']
# unique_samples = df_freq['sample'].unique()

# sorted_samples = sorted(
#     unique_samples,
#     key=lambda x: (
#         next((order.index(c) for c in order if c in x), len(order)),
#         int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else float('inf')
#     )
# )
# df_freq['sample'] = pd.Categorical(df_freq['sample'], categories=sorted_samples, ordered=True)
# df_freq_sorted = df_freq.sort_values('sample').reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8.5, 8.5))
plu.scatter(df_freq, 'GBC', 'sample', by='GBC', color=clones_colors, size='area_plot',alpha=0.5, ax=ax)
plu.format_ax(ax, title='Clones by sample', xlabel='Clones', xticks='')
#ax.text(.3, .23, f'n clones total: {df_freq_sorted["GBC"].unique().size}', transform=ax.transAxes)
fig.tight_layout()
fig.savefig(os.path.join(path_results,'bubble_plot.png'),dpi=300)

#bubble plot 20%,50%,70% 
#n of clones with freq>10% common in 20%,50%,70% of samples of 1 condition
#clones freq > 10%
threshold = 0.05
criteria = [0.2, 0.5, 0.7]

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

for criterion, clones in selected_clones.items():
    df_freq_filtered = df_freq[df_freq['GBC'].isin(clones)] 
    #df_freq_filtered = df_freq_filtered[df_freq_filtered['sample'] != 'ref_4T1_GBC']
    df_freq_filtered['area_plot'] = df_freq_filtered['freq'] * (3000 - 5) + 5  
    order=['IME_NSG_met','IME_dep_met','IME_CTRL_met','IME_NSG','IME_dep','IME_CTRL']
    unique_samples = df_freq_filtered['sample'].unique()

    sorted_samples = sorted(
        unique_samples,
        key=lambda x: (
        next((order.index(c) for c in order if c in x), len(order)),
        int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else float('inf')
        )
    )
    df_freq_filtered['sample'] = pd.Categorical(df_freq_filtered['sample'], categories=sorted_samples, ordered=True)
    df_freq_sorted = df_freq_filtered.sort_values('sample').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    plu.scatter(df_freq_sorted, 'GBC', 'sample', by='GBC', color=clones_colors, size='area_plot', alpha=0.5, ax=ax)
    plu.format_ax(ax, xlabel='Clones', xticks='')
    fig.tight_layout()
    fig.savefig(os.path.join(path_results, f'bubble_plot_{int(criterion*100)}.png'), dpi=300)

#Bubble Plot of clones freq > 10%
threshold = 0.60
df_freq_filtered = df_freq[df_freq['cum_freq'] >= threshold]
df_freq_filtered['sample'] = pd.Categorical(df_freq_filtered['sample'], categories=categories_bubble)
df_freq_filtered.sort_values(by=['sample'], inplace=True)

# Compute bubble size
df_freq_filtered['area_plot'] = df_freq_filtered['freq'] * (3000 - 5) + 5  

# Generate Bubble Plot
fig, ax = plt.subplots(figsize=(15, 6))
plu.scatter(df_freq_filtered, 'GBC', 'sample', by='GBC', color=clones_colors, size='area_plot', alpha=0.5, ax=ax)
plu.format_ax(ax, xlabel='Clones', xticks='')

# Save figure
fig.tight_layout()
plt.show()
fig.savefig(os.path.join(path_results, 'bubble_plot_90_cumulative_dep.png'), dpi=300)


# Cumulative clone percentage, all samples
colors = plu.create_palette(df, 'origin', plu.ten_godisnot)

fig, ax = plt.subplots(figsize=(4.5,4.5))
for s in df['sample'].unique():
    df_ = df.query('sample==@s')
    x = (df_['read_count'] / df_['read_count'].sum()).cumsum()
    origin = df.query('sample==@s')['origin'].unique()[0]
    ax.plot(range(len(x)), x, c=colors[origin], linewidth=2.5)

ax.set(title='Clone prevalences', xlabel='Ranked clones', ylabel='Cumulative frequence')
plu.add_legend(ax=ax, colors=colors, bbox_to_anchor=(1,0), loc='lower right', ticks_size=8, label_size=10, artists_size=8)
fig.tight_layout()
fig.savefig(os.path.join(path_results, f'cum_percentages.png'), dpi=300)



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
order=['IME_CTRL','IME_dep','IME_NSG','IME_CTRL_met','IME_dep_met','IME_NSG_met']
# sorted_samples = sorted(
#     [s for c in order for s in df_sample.index if c in s],
#     key=lambda x: (
#         next((order.index(c) for c in order if c in x), len(order)), 
#         int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else float('inf')  
#     )
# )

df_sample_sorted = df_sample.loc[categories]
df_sample_sorted=df_sample_sorted.reset_index()

fig, ax = plt.subplots(figsize=(10,4.5))
bar_new(df_sample_sorted, y= 'n_clones', x='sample', color='k', x_order=categories,alpha=.7, ax=ax)

for i, row in df_sample_sorted.iterrows():
    ax.text(i, row['n_clones'], str(row['n_clones']), ha='center', va='bottom', fontsize=8)

plu.format_ax(ax=ax, title='n clones by sample', ylabel='n_clones', xticks=df_sample_sorted['sample'], rotx=90)
ax.spines[['left', 'top', 'right']].set_visible(False)
ax.invert_yaxis() 
fig.tight_layout()
fig.savefig(os.path.join(path_results, f'n_clones.png'), dpi=500)


#box,strip n_clones by condition
fig, ax = plt.subplots(figsize=(8,6))
plu.box(df_sample_sorted, x='origin', y='n_clones', ax=ax, add_stats=True,
    pairs=[['IME_CTRL', 'IME_CTRL_met'], ['IME_dep', 'IME_dep_met'], ['IME_NSG', 'IME_NSG_met']]
)
plu.strip(df_sample_sorted, x='origin', y='n_clones', ax=ax,color='k')
#ax.set_yscale('log', base=2)
#y_min, y_max = ax.get_ylim()
#ticks = [2**i for i in range(int(np.log2(y_min)), int(np.log2(y_max)) + 1)]
#ax.set_yticks(ticks)
#ax.set_yticklabels([str(tick) for tick in ticks])
plu.format_ax(ax=ax, title='n_clones by condition', ylabel='n_clones', rotx=90, reduced_spines=True)
fig.tight_layout()
fig.savefig(os.path.join(path_results, f'n_clones_condition.png'), dpi=300)


#box,strip SH by condition
fig, ax = plt.subplots(figsize=(8,6))
plu.box(df_sample_sorted, x='origin', y='SH', ax=ax, add_stats=True, 
    pairs=[['IME_CTRL', 'IME_CTRL_met'], ['IME_dep', 'IME_dep_met'], ['IME_NSG', 'IME_NSG_met']]
)
plu.strip(df_sample_sorted, x='origin', y='SH', ax=ax, color='k') #order=['ref','IME_CTRL','IME_dep','IMT_CTRL','IMT_COMBO','IMT_CTLA4']
plu.format_ax(ax=ax, title='Shannon Entropy samples', ylabel='SH', rotx=90, reduced_spines=True)
fig.tight_layout()
fig.savefig(os.path.join(path_results, f'SH.png'), dpi=300)




#Pivot_table:  
df_freq_wide = (df_freq.pivot(index='GBC', columns='sample', values='freq'))
unique_clones = df_freq_wide[df_freq_wide.notnull().sum(axis=1) == 1].index.tolist()  #unique_clones one entry != 0
common_clones = df_freq_wide.loc[
    df_freq_wide.drop(columns=['ref_4T1_GBC']).notnull().all(axis=1)
].index.tolist()   #common clones all vs all (except ref)
common_clones_ime = df_freq_wide.loc[
                df_freq_wide.filter(like='IME_CTRL').notnull().any(axis=1) & 
                df_freq_wide.filter(like='IME_dep').notnull().any(axis=1)    #common clones IME_CTRL vs IME_dep
].index.tolist()
common_clones_imt= df_freq_wide.loc[
                df_freq_wide.filter(like='IMT_CTRL').notnull().any(axis=1) & 
                df_freq_wide.filter(like='IMT_COMBO').notnull().any(axis=1)  #common clones IMT_CTRL vs IMT_COMBO
].index.tolist()

len(unique_clones)



#GBC, n_sample_ref,n_sample_ime_ctrl, n_sample_ime_dep, n_sample_imt_ctrl, n_sample_imt_combo,n_sample_tot,mean_freq_ime_ctrl
df_clone=(df_freq_wide.reset_index()
    .assign(
    n_sample_ref = lambda x: x.filter(like='ref_4T1_GBC').notnull().sum(axis=1),
    n_sample_ime_ctrl = lambda x: x.filter(like='IME_CTRL').notnull().sum(axis=1),          
    n_sample_ime_dep = lambda x: x.filter(like='IME_dep').notnull().sum(axis=1),
    n_sample_imt_ctrl = lambda x: x.filter(like='IMT_CTRL').notnull().sum(axis=1),
    n_sample_imt_combo = lambda x: x.filter(like='IMT_COMBO').notnull().sum(axis=1),
    n_sample_tot = lambda x: x.filter(like='IM').notnull().sum(axis=1) + x.filter(like='ref_4T1_GBC').notnull().sum(axis=1),
    mean_freq_ime_ctrl = lambda x: x.filter(like='IME_CTRL').apply(lambda row: row.dropna().mean(), axis=1),
    mean_freq_ime_dep = lambda x: x.filter(like='IME_dep').apply(lambda row: row.dropna().mean(), axis=1),
    mean_freq_imt_ctrl = lambda x: x.filter(like='IMT_CTRL').apply(lambda row: row.dropna().mean(), axis=1),
    mean_freq_imt_combo = lambda x: x.filter(like='IMT_COMBO').apply(lambda row: row.dropna().mean(), axis=1)

)[['GBC','n_sample_ref','n_sample_ime_ctrl','n_sample_ime_dep','n_sample_imt_ctrl','n_sample_imt_combo','n_sample_tot','mean_freq_ime_ctrl','mean_freq_ime_dep','mean_freq_imt_ctrl','mean_freq_imt_combo']])
df_clone.to_csv(os.path.join(path_results,'clones_statistic.csv'))



#Heatmap common_clones
# order=['IME_CTRL','IME_dep','IME_NSG','IME_CTRL_met','IME_dep_met','IME_NSG_met'] #'IME_CTRL_met','IME_dep_met','IME_NSG_met'
# prefix_pattern = re.compile(r'^(IME_CTRL|IME_dep|IME_NSG|IME_CTRL_met|IME_dep_met|IME_NSG_met)') #IME_CTRL_met|IME_dep_met|IME_NSG_met

# ordered_samples = sorted(
#     {s for s in common.index if prefix_pattern.match(s)},
#     key=lambda x: (
#         order.index(prefix_pattern.match(x).group(1)),
#         int(re.search(r'_(\d+)$', x).group(1)) if re.search(r'_(\d+)$', x) else float('inf') 
#     )
# )


df_reordered = common.reindex(index=categories, columns=categories)
fig, ax = plt.subplots(figsize=(16,16))
vmin, vmax= 0, 300
plu.plot_heatmap(df_reordered, ax=ax, annot=True, title='n common clones', fmt='.1f',x_names_size=8, y_names_size=8, annot_size=5.5, cb=False)
sns.heatmap(data=df_reordered, ax=ax, robust=True, cmap="mako", vmin=vmin, vmax=vmax, fmt='.1f',cbar_kws={'fraction':0.05, 'aspect':35, 'pad': 0.02})
fig.tight_layout()
fig.savefig(os.path.join(path_results, f'common.png'), dpi=300)

#heatmap jaccard INDEX
order=['IME_CTRL','IME_dep','IME_NSG']
prefix_pattern = re.compile(r'^(IME_CTRL|IME_dep|IME_NSG)') #IME_CTRL_met|IME_dep_met|IME_NSG_met

ordered_samples = sorted(
    {s for s in common.index if prefix_pattern.match(s)},
    key=lambda x: (
        order.index(prefix_pattern.match(x).group(1)),
        int(re.search(r'_(\d+)$', x).group(1)) if re.search(r'_(\d+)$', x) else float('inf') 
    )
)
common_c = common.loc[ordered_samples, ordered_samples]
set_sizes = np.diag(common_c.values)
JI = np.zeros_like(common_c.values, dtype=float)

for i, row in common_c.iterrows():
    for j, val in row.items():
        union_size = set_sizes[common_c.index.get_loc(i)] + set_sizes[common_c.columns.get_loc(j)] - val
        JI[common_c.index.get_loc(i), common_c.columns.get_loc(j)] = val / union_size if union_size != 0 else 0

JI_df = pd.DataFrame(JI, index=common_c.index, columns=common_c.columns)
order_clustering = leaves_list(linkage(JI_df.values))

vmin, vmax= 0, 0.45
fig, ax = plt.subplots(figsize=(16, 16))
plu.plot_heatmap(JI_df.iloc[order_clustering, order_clustering], palette='mako', ax=ax,   #JI_df.values[np.ix_(order_clustering, order_clustering)]
             x_names=JI_df.index[order_clustering], y_names=JI_df.index[order_clustering], annot=True, 
             annot_size=8, label='Jaccard Index', fmt='.2f',shrink=1, cb=False)
sns.heatmap(JI_df.iloc[order_clustering, order_clustering], ax=ax, xticklabels=JI_df.index[order_clustering], yticklabels=JI_df.index[order_clustering],
                        robust=True, cmap="mako", vmin=vmin, vmax=vmax, fmt='.2f',cbar_kws={'fraction':0.05, 'aspect':35, 'pad': 0.02})
fig.tight_layout()
plt.savefig(os.path.join(path_results, f'heatmap_Jaccard.png'), dpi= 300)