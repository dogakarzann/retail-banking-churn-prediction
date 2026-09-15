"""
EDA Utility Functions
=====================
Reusable helper functions for exploratory data analysis.
Provides plotting and statistical analysis tools.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import sys
from warnings import filterwarnings

filterwarnings('ignore')

# ──────────────────────────── Color Palette ────────────────────────────
PALETTE = ['#003366', '#C8102E', '#006699', '#009966', '#F2A900', '#4B4B4B']
DEFAULT_COLOR = '#003366'


# ══════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════

def _create_subplots(num_features, num_cols=3, figsize=(20, 15)):
    """Create a figure with subplots grid and return (fig, flat_axes)."""
    num_rows = math.ceil(num_features / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Always return a flat, iterable array regardless of grid dimensions
    if num_features == 1:
        axes_flat = np.array([axes]) if not isinstance(axes, np.ndarray) else axes.flatten()
    else:
        axes_flat = axes.flatten()

    return fig, axes_flat


def _cleanup_subplots(fig, axes_flat, num_used):
    """Remove unused subplots and apply tight layout."""
    for j in range(num_used, len(axes_flat)):
        fig.delaxes(axes_flat[j])
    plt.tight_layout()
    plt.show()


def _format_title(feature_name):
    """Convert snake_case feature name to Title Case."""
    return feature_name.replace('_', ' ').title()


# ══════════════════════════════════════════════════════════════════════
# MAIN PLOTTING FUNCTION
# ══════════════════════════════════════════════════════════════════════

def analysis_plots(data, features, histplot=True, barplot=False, mean=None,
                   text_y=0.5, outliers=False, boxplot=False, boxplot_x=None,
                   kde=False, hue=None, color=DEFAULT_COLOR, figsize=(20, 15)):
    """
    Unified function for EDA visualizations.

    Generates histograms, horizontal bar plots, or boxplots for a list of features.
    Only ONE plot type should be active at a time (controlled by boolean flags).

    Plot types (pick one):
        histplot=True (default) : Distribution histograms, optionally with KDE overlay.
        barplot=True            : Horizontal bar chart showing proportions or means.
        outliers=True           : Univariate boxplots to spot outliers.
        boxplot=True            : Bivariate boxplots comparing distributions across groups.

    Args:
        data        : DataFrame to visualize.
        features    : List of column names.
        histplot    : If True, draw histograms (default).
        barplot     : If True, draw horizontal bar plots.
        mean        : Column name whose mean is shown per category (barplot mode).
        text_y      : Horizontal offset for bar labels.
        outliers    : If True, draw univariate boxplots (outlier detection).
        boxplot     : If True, draw bivariate boxplots (requires boxplot_x).
        boxplot_x   : Grouping variable for bivariate boxplots.
        kde         : If True, overlay KDE on histograms.
        hue         : Grouping variable for color encoding.
        color       : Single color for plots without hue.
        figsize     : Figure size tuple.
    """
    if not features:
        print("No features provided.")
        return

    fig, axes_flat = _create_subplots(len(features), figsize=figsize)

    for i, feature in enumerate(features):
        ax = axes_flat[i]

        # ── Barplot Mode ──
        if barplot:
            _draw_barplot(ax, data, feature, hue=hue, mean=mean,
                          text_y=text_y, color=color)

        # ── Univariate Boxplot (Outlier) Mode ──
        elif outliers:
            sns.boxplot(data=data, y=feature, ax=ax, color=color)

        # ── Bivariate Boxplot Mode ──
        elif boxplot:
            sns.boxplot(data=data, x=boxplot_x, y=feature, ax=ax,
                        palette=PALETTE[:2], hue=boxplot_x, legend=False)

        # ── Histogram Mode (default) ──
        else:
            sns.histplot(data=data, x=feature, kde=kde, ax=ax,
                         color=color, hue=hue)

        ax.set_title(_format_title(feature), fontweight='bold')
        ax.set_xlabel('')

    _cleanup_subplots(fig, axes_flat, len(features))


def _draw_barplot(ax, data, feature, hue=None, mean=None, text_y=0.5, color=DEFAULT_COLOR):
    """Internal: draw a single horizontal bar subplot."""
    if mean:
        # Mean aggregation mode
        grouped = data.groupby(feature)[[mean]].mean().round(2).reset_index()
        values = grouped[mean]
        labels = grouped[feature].astype(str)
        fmt = '{:.1f}'
    elif hue:
        # Churn rate (proportion) mode
        grouped = data.groupby(feature)[[hue]].mean().reset_index().rename(columns={hue: 'pct'})
        grouped['pct'] *= 100
        values = grouped['pct']
        labels = grouped[feature].astype(str)
        fmt = '{:.1f}%'
    else:
        # Simple count proportion mode
        grouped = (data[feature].value_counts(normalize=True) * 100).reset_index()
        grouped.columns = [feature, 'pct']
        values = grouped['pct']
        labels = grouped[feature].astype(str)
        fmt = '{:.1f}%'

    ax.barh(y=labels, width=values, color=color)

    # Add value labels
    for idx, val in enumerate(values):
        ax.text(val + text_y, idx, fmt.format(val), va='center', fontsize=12)

    # Clean styling
    ax.get_xaxis().set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)


# ══════════════════════════════════════════════════════════════════════
# OUTLIER ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def check_outliers(data, features):
    """
    Detect outliers using the IQR method.

    For each feature, calculates Q1, Q3, IQR and flags values
    outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] as outliers.

    Args:
        data     : DataFrame to analyze.
        features : List of numerical column names.

    Returns:
        tuple: (outlier_indexes, outlier_counts, total_outliers)
            - outlier_indexes (dict): {feature: [list of outlier row indices]}
            - outlier_counts  (dict): {feature: count}
            - total_outliers  (int) : Sum of all outlier counts.
    """
    outlier_counts = {}
    outlier_indexes = {}
    total_outliers = 0

    print("─" * 50)
    print("  OUTLIER ANALYSIS (IQR Method)")
    print("─" * 50)

    for feature in features:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        mask = (data[feature] < lower) | (data[feature] > upper)
        idxs = data[mask].index.tolist()
        count = len(idxs)

        outlier_indexes[feature] = idxs
        outlier_counts[feature] = count
        total_outliers += count

        pct = round(count / len(data) * 100, 2)
        status = f"{count} ({pct}%)" if count > 0 else "None"
        print(f"  {feature:<35} → {status}")

    print("─" * 50)
    print(f"  Total: {total_outliers} outliers")
    print("─" * 50)

    return outlier_indexes, outlier_counts, total_outliers


# ══════════════════════════════════════════════════════════════════════
# CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def plot_correlation_matrix(data, features=None, figsize=(16, 12)):
    """
    Plot a lower-triangle correlation heatmap.

    Args:
        data     : DataFrame (or subset).
        features : List of columns to include. If None, uses all numeric columns.
        figsize  : Figure size tuple.
    """
    subset = data[features] if features else data.select_dtypes('number')
    corr = subset.corr()

    # Lower triangle mask
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=figsize)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5)
    plt.title("Correlation Matrix", fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()


def show_target_correlation(data, target, features=None):
    """
    Print correlations of all numeric features with the target, sorted by strength.

    Args:
        data     : DataFrame.
        target   : Target column name (e.g. 'churn_flag').
        features : Optional list to restrict columns.
    """
    subset = data[features + [target]] if features else data.select_dtypes('number')
    corr = subset.corr()[target].drop(target).sort_values(ascending=False)

    print("─" * 50)
    print(f"  Correlation with '{target}'")
    print("─" * 50)
    for feat, val in corr.items():
        direction = "↑" if val > 0 else "↓"
        print(f"  {feat:<35} {direction} {val:+.4f}")
    print("─" * 50)

    return corr


# ══════════════════════════════════════════════════════════════════════
# SCATTER PLOT ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def scatter_plots(data, pairs, hue=None, figsize=(20, 5)):
    """
    Draw scatter plots for given (x, y) feature pairs.

    Args:
        data   : DataFrame.
        pairs  : List of (x_col, y_col) tuples.
        hue    : Grouping variable for color.
        figsize: Figure size tuple.
    """
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axes = [axes]

    for ax, (x, y) in zip(axes, pairs):
        sns.scatterplot(data=data, x=x, y=y, hue=hue, ax=ax,
                        palette=PALETTE[:2] if hue else None, alpha=0.6)
        ax.set_title(f"{_format_title(x)} vs {_format_title(y)}", fontweight='bold')
        ax.set_xlabel(_format_title(x))
        ax.set_ylabel(_format_title(y))

    plt.tight_layout()
    plt.show()
