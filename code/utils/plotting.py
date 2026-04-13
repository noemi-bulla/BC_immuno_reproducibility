"""
Plotting functions.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotting_utils as plu
import numpy as np 
import pandas as pd 
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import textalloc as ta
from matplotlib.lines import Line2D 
from statannotations.Annotator import Annotator 
from sklearn.metrics.pairwise import pairwise_distances
from scipy.cluster.hierarchy import linkage, leaves_list
from typing import Dict, Iterable, Any, Tuple, List
plt.style.use('default')
from circlify import _bubbles, circlify, Circle

##

def packed_circle_plot(
    df, covariate=None, ax=None, color='b', cmap=None, alpha=.5, linewidth=1.2,
    t_cov=.01, annotate=False, fontsize=6, ascending=False, fontcolor='white', 
    fontweight='normal'
    ):

    """
    Circle plot. Packed.
    """
    df = df.sort_values(covariate, ascending=False)
    circles = circlify(
        df[covariate].to_list(),
        show_enclosure=True, 
        target_enclosure=Circle(x=0, y=0, r=1)
    )
    lim = max(
        max(
            abs(c.x) + c.r,
            abs(c.y) + c.r,
        )
        for c in circles
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    
    if isinstance(color, str) and not color in df.columns:
        colors = { k : color for k in df.index }
    elif isinstance(color, str) and color in df.columns:
        c_cont = plu.create_palette(
            df.sort_values(color, ascending=True),
            color, cmap
        )
        colors = {}
        for name in df.index:
            colors[name] = c_cont[df.loc[name, color]]
    else:
        assert isinstance(color, dict)
        colors = color
        print('Try to use custom colors...')

    for name, circle in zip(df.index[::-1], circles): # Don't know why, but it reverses...
        x, y, r = circle
        ax.add_patch(
            plt.Circle((x, y), r*0.95, alpha=alpha, linewidth=linewidth, 
                fill=True, edgecolor=colors[name], facecolor=colors[name])
        )
        if annotate:
            cov = df.loc[name, covariate]
            if cov > t_cov:
                n = name if len(name)<=5 else name[:5]
                ax.annotate(
                    f'{n}: {df.loc[name, covariate]:.3f}', 
                    (x,y), 
                    va='center', ha='center', 
                    fontweight=fontweight, fontsize=fontsize, color=fontcolor, 
                )

    ax.axis('off')
    
    return ax


##

def _reorder(X, metric='euclidean', method='average', n_jobs=-1):
    """
    Reorder rows of an array X.
    """
    D = pairwise_distances(X, metric=metric, n_jobs=n_jobs)
    order = leaves_list(linkage(D, method=method))

    return order

##


def plot_heatmap(
    df: pd.DataFrame, 
    palette: str = 'mako', 
    ax: matplotlib.axes.Axes = None, 
    title: str = None, 
    x_names: bool = True, y_names: bool = True, 
    x_names_size: float = 7, y_names_size: float = 7, 
    xlabel: Iterable[Any] = None, ylabel: Iterable[Any] = None, 
    annot: bool = False, annot_size: float = 5, 
    label: str = None, shrink: float = 1.0, cb: bool = True, 
    vmin: float = None, vmax: float = None, 
    cluster_rows: bool = False, 
    cluster_cols: bool = False, 
    fmt: str = ".0f",
    outside_linewidth: float = 1, linewidths: float = 0, 
    linecolor: Any = 'white'
    ) -> matplotlib.axes.Axes:
    """
    Simple heatmap.
    """
    
    # Re-order rows and cols
    row_order = _reorder(df.values) if cluster_rows else range(df.shape[0])
    col_order =  _reorder(df.values.T) if cluster_cols else range(df.shape[1])   
    df_plot = df.iloc[row_order, col_order].copy()

    # Main plot
    ax = sns.heatmap(
        data=df_plot, 
        ax=ax, 
        robust=True, 
        cmap=palette, 
        annot=annot, 
        xticklabels=x_names, 
        yticklabels=y_names, 
        fmt=fmt, 
        annot_kws={'size':annot_size}, 
        cbar=cb,
        cbar_kws={'fraction':0.05, 'aspect':35, 'pad': 0.02, 'shrink':shrink, 'label':label},
        vmin=vmin, 
        vmax=vmax, 
        linewidths=linewidths, 
        linecolor=linecolor
    )
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=x_names_size)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=y_names_size)

    # Prettify spines
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(outside_linewidth)

    return ax

##