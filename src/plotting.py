"""Plot helpers.

House rules (PDF section 7.3), applied by every function here:
  * every plot has a title and labelled axes, and a legend where relevant
  * colourblind-friendly palette
  * tight_layout / bbox_inches='tight' so labels are never clipped
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # scripts must not need a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
from sklearn.tree import plot_tree

from . import config

sns.set_theme(style="whitegrid", palette=config.PALETTE)
plt.rcParams["figure.dpi"] = config.FIG_DPI
plt.rcParams["savefig.bbox"] = "tight"


def savefig(fig, name: str, subdir: str = "") -> Path:
    config.ensure_dirs()
    out = config.FIGURES_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path.relative_to(config.PROJECT_ROOT)}")
    return path


# --------------------------------------------------------------------------
# EDA
# --------------------------------------------------------------------------
def distribution_grid(df: pd.DataFrame, title: str, name: str,
                      subdir: str = "", ncols: int = 3):
    """One distribution plot per feature: histogram+KDE for numerics,
    count plot for categoricals."""
    cols = list(df.columns)
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
            sns.histplot(s.dropna(), kde=True, ax=ax, color=sns.color_palette(config.PALETTE)[0])
            ax.set_ylabel("Count")
        else:
            order = sorted(s.dropna().unique(), key=str)
            sns.countplot(x=s.astype(str), ax=ax, order=[str(o) for o in order],
                          hue=s.astype(str), legend=False, palette=config.PALETTE)
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=30)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel(col)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return savefig(fig, name, subdir)


def correlation_heatmap(df: pd.DataFrame, title: str, name: str, subdir: str = ""):
    num = df.select_dtypes(include=[np.number])
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(1.1 * len(num.columns) + 3,
                                    0.9 * len(num.columns) + 2.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0,
                square=True, linewidths=.5, ax=ax,
                cbar_kws={"label": "Pearson r"})
    ax.set_title(title)
    fig.tight_layout()
    return savefig(fig, name, subdir)


def target_distribution(y: pd.Series, title: str, name: str, subdir: str = "",
                        discrete: bool = False, class_names: dict | None = None):
    fig, ax = plt.subplots(figsize=(6, 4))
    if discrete:
        vc = y.value_counts().sort_index()
        labels = [class_names.get(k, str(k)) if class_names else str(k) for k in vc.index]
        sns.barplot(x=labels, y=vc.values, ax=ax, hue=labels, legend=False,
                    palette=config.PALETTE)
        for i, v in enumerate(vc.values):
            ax.text(i, v, f"{v}\n({100*v/vc.sum():.2f}%)", ha="center", va="bottom",
                    fontsize=9)
        ax.set_ylabel("Count")
        ax.set_xlabel("Class")
        ax.margins(y=0.15)
    else:
        sns.histplot(y, kde=True, ax=ax, color=sns.color_palette(config.PALETTE)[0])
        ax.set_xlabel(y.name)
        ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    return savefig(fig, name, subdir)


def feature_target_scatter(df: pd.DataFrame, features: list[str], target: str,
                           name: str, subdir: str = "", jitter: float = 0.0,
                           title: str | None = None):
    """Scatter of feature vs target. `jitter` > 0 spreads a discrete/binary
    target so the points do not collapse into unreadable lines."""
    fig, axes = plt.subplots(1, len(features), figsize=(5.2 * len(features), 4))
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(config.RANDOM_STATE)
    for ax, feat in zip(axes, features):
        y = df[target].astype(float).to_numpy()
        if jitter:
            y = y + rng.uniform(-jitter, jitter, size=len(y))
        ax.scatter(df[feat], y, s=8, alpha=0.25,
                   color=sns.color_palette(config.PALETTE)[0],
                   label=f"{feat} vs {target}")
        ax.set_xlabel(feat)
        ax.set_ylabel(target + (" (jittered)" if jitter else ""))
        ax.set_title(f"{feat} vs {target}", fontsize=10)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(title or f"Feature-target relationships ({target})", fontsize=13)
    fig.tight_layout()
    return savefig(fig, name, subdir)


def grouped_box(df: pd.DataFrame, features: list[str], target: str,
                name: str, subdir: str = "", class_names: dict | None = None):
    """Companion plot for a binary target: distribution of each feature per class."""
    fig, axes = plt.subplots(1, len(features), figsize=(4.6 * len(features), 4))
    axes = np.atleast_1d(axes)
    lab = df[target].map(class_names) if class_names else df[target].astype(str)
    for ax, feat in zip(axes, features):
        sns.boxplot(x=lab, y=df[feat], ax=ax, hue=lab, legend=False,
                    palette=config.PALETTE)
        ax.set_xlabel(target)
        ax.set_ylabel(feat)
        ax.set_title(f"{feat} by {target}", fontsize=10)
    fig.suptitle(f"Feature distributions by class ({target})", fontsize=13)
    fig.tight_layout()
    return savefig(fig, name, subdir)


# --------------------------------------------------------------------------
# Regression diagnostics
# --------------------------------------------------------------------------
def residual_plot(y_true, y_pred, model_name: str, name: str, subdir: str = ""):
    resid = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y_pred, resid, s=10, alpha=0.3,
                    color=sns.color_palette(config.PALETTE)[0], label="Residual")
    axes[0].axhline(0, color=sns.color_palette(config.PALETTE)[3], ls="--",
                    label="Zero error")
    axes[0].set_xlabel("Predicted value")
    axes[0].set_ylabel("Residual (actual - predicted)")
    axes[0].set_title("Residuals vs predicted")
    axes[0].legend()
    sns.histplot(resid, kde=True, ax=axes[1],
                 color=sns.color_palette(config.PALETTE)[0])
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual distribution")
    fig.suptitle(f"Residual diagnostics - {model_name}", fontsize=13)
    fig.tight_layout()
    return savefig(fig, name, subdir)


def predicted_vs_actual(y_true, y_pred, model_name: str, name: str, subdir: str = ""):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(y_true, y_pred, s=10, alpha=0.3,
               color=sns.color_palette(config.PALETTE)[0], label="Test observations")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], ls="--", color=sns.color_palette(config.PALETTE)[3],
            label="Perfect prediction (y = x)")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Predicted vs actual - {model_name}")
    ax.legend()
    fig.tight_layout()
    return savefig(fig, name, subdir)


def importance_plot(values, names, title: str, name: str, subdir: str = "",
                    top: int = 20, xlabel: str = "Importance"):
    s = pd.Series(np.asarray(values).ravel(), index=list(names))
    s = s.reindex(s.abs().sort_values(ascending=False).index)[:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(s) + 2))
    colors = [sns.color_palette(config.PALETTE)[0] if v >= 0
              else sns.color_palette(config.PALETTE)[3] for v in s.values]
    ax.barh(s.index, s.values, color=colors)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Feature")
    ax.set_title(title)
    ax.axvline(0, color="grey", lw=0.8)
    fig.tight_layout()
    return savefig(fig, name, subdir)


def model_comparison_bar(table: pd.DataFrame, metric: str, title: str,
                         name: str, subdir: str = ""):
    d = table.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(d) + 2))
    ax.barh(d["Model"], d[metric], color=sns.color_palette(config.PALETTE)[0],
            label=metric)
    ax.set_xlabel(metric)
    ax.set_ylabel("Model")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return savefig(fig, name, subdir)


# --------------------------------------------------------------------------
# Classification diagnostics
# --------------------------------------------------------------------------
def confusion_grid(cms: dict, class_names: list[str], title: str, name: str,
                   subdir: str = "", ncols: int = 3):
    items = list(cms.items())
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (mname, cm) in zip(axes, items):
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
            ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(mname, fontsize=9)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.tick_params(axis="x", rotation=20)
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return savefig(fig, name, subdir)


def roc_curves(models: dict, X_test, y_test, title: str, name: str, subdir: str = ""):
    from .evaluation import _scores_for_auc
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for mname, model in models.items():
        score, how = _scores_for_auc(model, X_test)
        if score is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, score)
        ax.plot(fpr, tpr, lw=1.4, label=mname)
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return savefig(fig, name, subdir)


def decision_tree_figure(tree_model, feature_names, class_names, name: str,
                         subdir: str = "", max_depth: int = 3):
    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(tree_model, feature_names=list(feature_names),
              class_names=[str(c) for c in class_names], filled=True, rounded=True,
              max_depth=max_depth, fontsize=8, ax=ax, proportion=True)
    ax.set_title(f"Decision Tree structure (display truncated to depth {max_depth})")
    fig.tight_layout()
    return savefig(fig, name, subdir)
