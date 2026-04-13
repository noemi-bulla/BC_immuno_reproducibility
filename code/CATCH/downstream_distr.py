'''
script to analyze CaTCH bulk data
'''

import os
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotting_utils as plu
from itertools import product
from utils.plotting import *
import pickle

#paths
path_file = "/Users/ieo7295/Desktop/BC_immuno_reproducibility/data/catch_110426/barcode_collation"
path_results = "/Users/ieo7295/Desktop/BC_immuno_reproducibility/results/catch_110426"

# Load data
df = pd.read_csv(os.path.join(path_file, "CaTCHseq_collation.txt"), sep="\t")
df_long = pd.melt(df, 
                  id_vars=['Barcode', 'bc_id'], 
                  value_vars=['REF_A1','REF_B3','A1_2_PT','A1_2_LUNG','A1_2_CTC','A1_3_PT','A1_5_PT','A1_5_LUNG','B3_1_PT','B3_1_LUNG','B3_1_CTC',
                              'B3_4_PT','B3_4_LUNG','B3_4_CTC'],
                  var_name='sample', 
                  value_name='read_count')

# swap 'B3_4_CTC' and 'B3_1_CTC' sample names
df_long['sample'] = df_long['sample'].replace({'B3_4_CTC': 'B3_1_CTC', 'B3_1_CTC': 'B3_4_CTC'})


df_long = df_long.dropna(subset=['read_count'])
df_long = df_long.rename(columns={'Barcode': 'GBC'})

# Assign origin based on sample patterns
tests = [
    df_long['sample'] == 'REF_A1',
    df_long['sample'] == 'REF_B3',
    df_long['sample'].str.contains(r'^A1_\d+_PT$', regex=True),
    df_long['sample'].str.contains(r'^A1_\d+_LUNG$', regex=True),
    df_long['sample'].str.contains(r'^A1_\d+_CTC$', regex=True),
    df_long['sample'].str.contains(r'^B3_\d+_PT$', regex=True),
    df_long['sample'].str.contains(r'^B3_\d+_LUNG$', regex=True),
    df_long['sample'].str.contains(r'^B3_\d+_CTC$', regex=True)
]
df_long['origin'] = np.select(tests, ['REF_A1', 'REF_B3', 'A1_PT', 'A1_LUNG', 'A1_CTC', 'B3_PT', 'B3_LUNG', 'B3_CTC'], default='other')



#Filter
min_n_reads = 60
df_long = df_long[df_long['read_count'] > min_n_reads]

for x in df_long['GBC']:
    if 'N' in x:
        print(x)

df_freq = (df_long.groupby('sample')
           .apply(lambda x: x.assign(
               freq=x['read_count'] / x['read_count'].sum(),    
               cum_freq=(x['read_count'] / x['read_count'].sum()).cumsum()
           ))
           .reset_index(drop=True)
)
df_freq.to_csv(os.path.join(path_results,'rel_freq.csv'))


categories = ['REF_A1','REF_B3','A1_2_PT','A1_2_LUNG','A1_2_CTC','A1_3_PT','A1_5_PT','A1_5_LUNG','B3_1_PT','B3_1_LUNG','B3_1_CTC',
                              'B3_4_PT','B3_4_LUNG','B3_4_CTC']
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
        n_clones=lambda df_: df_.index.map(
            lambda s: df_freq[df_freq['sample'] == s].index.nunique()
        ))
)

#bar plot n_clones by sample 
order=['REF_A1','REF_B3','A1_2_PT','A1_2_LUNG','A1_2_CTC','A1_3_PT','A1_5_PT','A1_5_LUNG','B3_1_PT','B3_1_LUNG','B3_1_CTC',
      'B3_4_PT','B3_4_LUNG','B3_4_CTC']
# sorted_samples = sorted(
#     [s for c in order for s in df_sample.index if c in s],
#     key=lambda x: (
#         next((order.index(c) for c in order if c in x), len(order)), 
#         int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else float('inf')  
#     )
# )

df_sample_sorted = df_sample.loc[categories]
df_sample_sorted=df_sample_sorted.reset_index()
df_sample_sorted.to_csv(os.path.join(path_results,'sample_summary.csv'), index=False)

fig, ax = plt.subplots(figsize=(10.5, 4.5))
plu.bar(
    df=df_sample_sorted,
    x='sample',
    y='n_clones',
    color='k',
    x_order=order,
    categorical_cmap=None,   
    alpha=0.7,
    ax=ax
)

for i, row in df_sample_sorted.iterrows():
    ax.text(i, row['n_clones'], str(row['n_clones']), ha='center', va='bottom', fontsize=8)

plu.format_ax(ax=ax, title='n clones by sample', ylabel='n clones', xticks=df_sample_sorted['sample'], rotx=90)
ax.spines[['left', 'top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(path_results, f'n_clones_filtered_60thr.png'), dpi=500)


#box,strip SH by condition
fig, ax = plt.subplots(figsize=(8,6))
plu.box(df_sample_sorted, x='origin', y='SH', ax=ax, add_stats=True,
    pairs=[['A1_PT','A1_LUNG'],['A1_PT','A1_CTC'],['A1_LUNG','A1_CTC'],['B3_PT','B3_LUNG'],['B3_PT','B3_CTC'],['B3_LUNG','B3_CTC']])
plu.strip(df_sample_sorted, x='origin', y='SH', ax=ax, color='k') #order=['ref','IME_CTRL','IME_dep','IMT_CTRL','IMT_COMBO','IMT_CTLA4']
plu.format_ax(ax=ax, title='Shannon Entropy', ylabel='SH', rotx=90, reduced_spines=True)
fig.tight_layout()
fig.savefig(os.path.join(path_results, f'SH_filtered_60thr.png'), dpi=300)

# Cumulative clone percentage, all samples
colors = plu.create_palette(df_freq, 'origin', plu.ten_godisnot)

fig, ax = plt.subplots(figsize=(4.5,4.5))
for s in df_freq['sample'].unique():
    df_ = df_freq.query('sample==@s')
    x = (df_['read_count'] / df_['read_count'].sum()).cumsum()
    origin = df_freq.query('sample==@s')['origin'].unique()[0]
    ax.plot(range(len(x)), x, c=colors[origin], linewidth=2.5)

ax.set(title='Clone prevalences', xlabel='Ranked clones', ylabel='Cumulative frequence')
ax.set_xlim(0, 1000)
plu.add_legend(ax=ax, colors=colors, bbox_to_anchor=(1,0), loc='lower right', ticks_size=8, label_size=10, artists_size=8)
fig.tight_layout()
fig.savefig(os.path.join(path_results, f'cum_percentages_filtered_60thr.png'), dpi=300)

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
# with open(os.path.join(path_file, 'clones_colors_sc.pickle'), 'wb') as f:
#     pickle.dump(clones_colors, f)

with open(os.path.join(path_file, 'clones_colors_sc.pickle'), 'rb') as f:
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
fig.tight_layout()
fig.savefig(os.path.join(path_results,'bubble_plot.png'),dpi=300)

#
freq_inputs = (
    df_freq[df_freq['sample'].isin(['CaTCH_10k','CaTCH_15k','CaTCH_20k'])]
    .pivot_table(index='GBC', columns='sample', values='freq', fill_value=0)
    .reset_index()
)

freq_inputs.to_csv(
    os.path.join(path_results, 'barcode_freq_10k_15k_20k.csv'),
    index=False
)

# Get top 10 barcodes for each sample
top_10k = freq_inputs.nlargest(10, 'CaTCH_10k')['GBC']
top_15k = freq_inputs.nlargest(10, 'CaTCH_15k')['GBC']
top_20k = freq_inputs.nlargest(10, 'CaTCH_20k')['GBC']
top_barcodes = pd.unique(pd.concat([top_10k, top_15k, top_20k]))
top = freq_inputs[freq_inputs['GBC'].isin(top_barcodes)].set_index('GBC')

#Viz
plt.figure(figsize=(12, 8))
sns.heatmap(
    top[['CaTCH_10k','CaTCH_15k','CaTCH_20k']],
    cmap='viridis'
)
plt.subplots_adjust(left=0.6)
plt.title('Top barcode frequencies in CaTCH 10k, 15k, and 20k samples')
plt.savefig(os.path.join(path_results, 'top_barcode_frequencies.png'), dpi=300)


#Circled pack plot 

# Extract sample ID (e.g., A1_2, A1_5, B3_1, B3_4) and tissue type
df_freq_plot = df_freq.copy()
parts = df_freq_plot['sample'].str.split('_', expand=True)
df_freq_plot['sample_id'] = parts[0] + '_' + parts[1]  # e.g., A1_2, B3_1
df_freq_plot['tissue'] = parts[2]  # e.g., PT, LUNG, CTC

# Filter out reference samples
df_freq_plot = df_freq_plot[~df_freq_plot['sample_id'].isin(['A1', 'B3', 'REF_A1', 'REF_B3'])]

# Define row order (sample IDs) and column order (tissues)
sample_ids = ['A1_2', 'A1_5', 'B3_1', 'B3_4']
tissues = sorted(df_freq_plot['tissue'].unique())[::-1]

# Create figure with 4 rows (sample_ids) and variable columns (tissues)
fig, axs = plt.subplots(len(sample_ids), len(tissues), figsize=(11, 12))

for row, sample_id in enumerate(sample_ids):
    for col, tissue in enumerate(tissues):
        ax = axs[row, col]
        
        # Get all clones that survived the 60-read filter
        df_all = df_freq_plot.query('sample_id==@sample_id and tissue==@tissue')
        n_all_clones = df_all['GBC'].nunique()
        
        df_ = df_all.query('freq >= 0.01').set_index('GBC')
        n_filtered_clones = len(df_)
        
        packed_circle_plot(
            df_, covariate='freq', ax=ax, color=clones_colors, annotate=True, t_cov=.05,
            alpha=.65, linewidth=2.5, fontsize=8, fontcolor='k', fontweight='medium'
        )
        # Title shows: filtered clones / total clones (after 60-read filter)
        ax.set(title=f'{sample_id}_{tissue}\nShown: {n_filtered_clones}/{n_all_clones}')

fig.tight_layout()
fig.savefig(os.path.join(path_results, 'circle_plot.png'), dpi=1000)


#heatmap of common clones
df = df_freq.copy()
d = {sample: set(df[df['sample'] == sample]['GBC']) for sample in df['sample'].unique()}

n=len(d)
C=np.zeros((n,n), dtype=int)

sample=list(d.keys())
for i,x in enumerate(sample):
    for j,y in enumerate(sample):
        if i >= j:
            common_clones = len(d[x] & d[y])
            C[i,j] = C[j,i] = common_clones

df_cc=pd.DataFrame(C, index=sample, columns=sample) 

# Sort by categories order
ordered_samples = sorted(df_cc.index, key=lambda x: categories.index(x) if x in categories else len(categories))
df_reordered = df_cc.loc[ordered_samples, ordered_samples]
vmin, vmax= 0, 1000
fig, ax = plt.subplots(figsize=(10,8))
plot_heatmap(df_reordered, ax=ax, annot=True,fmt='d', title='n common clones', x_names_size=8, y_names_size=8, annot_size=9,cb=False)
plu.format_ax(ax=ax, xlabel_size=9, ylabel_size=9)
fig.tight_layout()
fig.savefig(os.path.join(path_results, f'common.png'), dpi=300)


