import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotting_utils as plu
import pickle
import random
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#paths
path_file = "/Users/ieo7295/Desktop/ilaCatch/results/barcode_collation"
path_results = "/Users/ieo7295/Desktop/ilaCatch/results"
path_data = "/Users/ieo7295/Desktop/ilaCatch/results"  # Assuming data files will be saved here

#Apply nf-lenti filtering for min_n_reads = 1000 

df = pd.read_csv(os.path.join(path_file, "CaTCHseq_collation.txt"), sep="\t")

# Transform from wide to long format
df_long = pd.melt(df, 
                  id_vars=['Barcode', 'bc_id'], 
                  value_vars=['CaTCH_1', 'CaTCH_2', 'CaTCH_3'],
                  var_name='sample', 
                  value_name='read_count')

# Remove rows with NaN read counts
df_long = df_long.dropna(subset=['read_count'])

# Rename columns to match the original format
df_long = df_long.rename(columns={'Barcode': 'GBC'})

#apply filtering
min_n_reads = 1000
df_long = df_long[df_long['read_count'] >= min_n_reads]

# Calculate frequencies
df_freq = (df_long.groupby('sample')
           .apply(lambda x: x.assign(
               freq=x['read_count'] / x['read_count'].sum(),    
               cum_freq=(x['read_count'] / x['read_count'].sum()).cumsum()
           ))
           .reset_index(drop=True)
)

# Add origin column based on sample names (simplified for CaTCH data)
df_freq['origin'] = df_freq['sample']  # For CaTCH data, sample and origin are the same

df_freq.to_csv(os.path.join(path_results,'rel_freq.csv'))
print("Available samples:", df_freq['sample'].unique())

# Updated categories for CaTCH samples
categories = ['CaTCH_1', 'CaTCH_2', 'CaTCH_3']
categories_bubble = categories[::-1]


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
        n_barcodes=lambda df_: df_.index.map(
            lambda s: df_freq[df_freq['sample'] == s].index.nunique()
        ))
)

#bar plot n_clones by sample 
order=['CaTCH_1','CaTCH_2','CaTCH_3']
# sorted_samples = sorted(
#     [s for c in order for s in df_sample.index if c in s],
#     key=lambda x: (
#         next((order.index(c) for c in order if c in x), len(order)), 
#         int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else float('inf')  
#     )
# )

df_sample_sorted = df_sample.loc[categories]
df_sample_sorted=df_sample_sorted.reset_index()

fig, ax = plt.subplots(figsize=(10.5, 4.5))
plu.bar(
    df=df_sample_sorted,
    x='sample',
    y='n_barcodes',
    color='k',
    categorical_cmap=None,   # this is REQUIRED
    x_order=order,
    alpha=0.7,
    ax=ax
)

for i, row in df_sample_sorted.iterrows():
    ax.text(i, row['n_barcodes'], str(row['n_barcodes']), ha='center', va='bottom', fontsize=8)

plu.format_ax(ax=ax, title='n barcodes by sample', ylabel='n barcodes', xticks=df_sample_sorted['sample'], rotx=90)
ax.spines[['left', 'top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(path_results, f'n_barcodes_filtered.png'), dpi=500)


#box,strip SH by condition

fig, ax = plt.subplots(figsize=(8,6))
plu.box(df_sample_sorted, x='origin', y='SH', ax=ax, add_stats=True,
    pairs=[['CaTCH_1','CaTCH_2'], ['CaTCH_3', 'CaTCH_1']]
,x_order=order)
plu.strip(df_sample_sorted, x='origin', y='SH', ax=ax, color='k') #order=['ref','IME_CTRL','IME_dep','IMT_CTRL','IMT_COMBO','IMT_CTLA4']
plu.format_ax(ax=ax, title='Shannon Entropy', ylabel='SH', rotx=90, reduced_spines=True)
fig.tight_layout()
fig.savefig(os.path.join(path_results, f'SH_filtered.png'), dpi=300)

# Cumulative clone percentage, all samples
colors = plu.create_palette(df_freq, 'origin', plu.ten_godisnot)

fig, ax = plt.subplots(figsize=(4.5,4.5))
for s in df_freq['sample'].unique():
    df_ = df_freq.query('sample==@s')
    x = (df_['read_count'] / df_['read_count'].sum()).cumsum()
    origin = df_freq.query('sample==@s')['origin'].unique()[0]
    ax.plot(range(len(x)), x, c=colors[origin], linewidth=2.5)

ax.set(title='Clone prevalences', xlabel='Ranked clones', ylabel='Cumulative frequence')
plu.add_legend(ax=ax, colors=colors, bbox_to_anchor=(1,0), loc='lower right', ticks_size=8, label_size=10, artists_size=8)
fig.tight_layout()
fig.savefig(os.path.join(path_results, f'cum_percentages_filtered.png'), dpi=300)


#bubble plot filtered

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
