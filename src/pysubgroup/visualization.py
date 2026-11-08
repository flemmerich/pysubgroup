"""Static visualizations of subgroup discovery results (requires matplotlib).

Every function takes the ``(quality, Rule)`` pairs returned by a search (plain
``Rule``s work too), the DataFrame, and the target fitted on it, and draws onto
``ax`` (a new figure if omitted). Subgroups are numbered ``#1, #2, ...`` in the
order given. The visualizations follow the subgroup discovery literature:

- ``plot_subgroup_bars``: positives right and negatives left of a common axis,
  dataset first, subgroups sorted by target share (Kralj, Lavrač, Zupan: Subgroup
  visualization, SiKDD 2005).
- ``plot_spineplot``: stacked bars whose width is the subgroup size and whose
  lower part is the target share, with the dataset share as reference (stacked
  bar visualization of VIKAMINE; Atzmueller, Puppe: Semi-automatic visual subgroup
  mining using VIKAMINE, J.UCS 11(11), 2005).
- ``plot_roc_space``: subgroups in ROC space (false vs. true positive rate) or in
  coverage space (absolute counts) with the ROC convex hull (Lavrač, Kavšek,
  Flach, Todorovski: Subgroup discovery with CN2-SD, JMLR 5, 2004) and isometrics
  of a quality function (Fürnkranz, Flach: ROC 'n' rule learning, MLJ 58, 2005).
- ``plot_size_quality``: subgroup size against quality or target statistic, as in
  Slice Finder's size/effect-size view (Chung et al., ICDE 2019) and pysubgroup 0.x.
- ``plot_target_distributions``: numeric target distributions of the dataset and
  each subgroup (distribution rules; Jorge, Pereira, Azevedo: Visual interactive
  subgroup discovery with numerical properties of interest, DS 2006).
- ``plot_attribute_distribution``: positives above and negatives below the axis
  over a numeric attribute, for the dataset and subgroups (Gamberger, Lavrač,
  Wettschereck: Subgroup visualization, IDAMAP 2002); counts use histogram bins,
  so attributes in the subgroup description need no manual correction.

Provenance: the designs come from the cited papers; pysubgroup 0.x's
``visualization.py`` has related plots (``plot_npspace``, ``similarity_sgs``,
``supportSetVisualization``). No code from it or other libraries is reused.
- ``plot_model_scatter``: two numeric EMM model attributes with a subgroup
  highlighted against its complement and their regression lines (Duivesteijn,
  Feelders, Knobbe: Exceptional Model Mining, DMKD 30, 2016, Figs. 3-4).
- ``plot_performance_curves``: ROC or precision-recall curves of a soft
  classifier on the dataset and on each subgroup, as in SubROC (Siegl et al.,
  https://doi.org/10.48550/arXiv.2505.11283), which compares subgroup curves
  with the dataset's.
- ``plot_overlap`` and ``plot_similarity``: covered rows per subgroup and Jaccard
  similarity, ordered by complete-linkage clustering (VIKAMINE cluster view).
- ``plot_similarity_dendrogram``: hierarchical clustering of the subgroups by
  the Jaccard distance of their covered rows, the clustering behind the
  cluster view.
- ``plot_specialization_graph``: subgroups by description length with edges from
  generalizations to specializations (VIKAMINE overview visualization).
- ``plot_null_distribution``: each subgroup's statistic against its
  randomization null distribution, with the empirical p-value (the test of
  SubROC, Siegl et al., Section 3.3; see ``significance``). It resembles
  pysubgroup 0.x's ``plot_qualities_on_sample_distribution``, which draws given
  qualities over one histogram of sampled qualities.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .rules import Condition, Rule
from .significance import _p_values, randomization_null_distribution
from .targets import (
    BaseTarget,
    LabelledTarget,
    QualityFunction,
    SoftClassifierTarget,
    StandardQF,
)
from .targets.base import orient
from .targets.soft_classifier import _threshold_counts, pr_auc_score, roc_auc_score
from .utils import result_pairs, rule_label, subgroup_masks

# Validated reference palette: positives/negatives use categorical slots 1/2
# (CVD-safe pair); other subgroup series continue in fixed slot order.
_THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "positive": "#2a78d6", "negative": "#eb6834",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                   "#4a3aa7", "#e34948"],
        "ramp": ["#fcfcfb", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"],
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "positive": "#3987e5", "negative": "#d95926",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300",
                   "#9085e9", "#e66767"],
        "ramp": ["#1a1a19", "#184f95", "#2a78d6", "#6da7ec", "#b7d3f6"],
    },
}


# Shared helpers


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on the environment
        message = "visualizations require matplotlib: pip install pysubgroup[plot]"
        raise ImportError(message) from error
    return plt


def _theme(theme: str) -> dict[str, Any]:
    if theme not in _THEMES:
        raise ValueError(f"theme must be one of {sorted(_THEMES)}")
    return _THEMES[theme]


def _axes(ax, theme: str, figsize: tuple[float, float], grid: str | None = "both"):
    """Return themed axes: recessive hairline grid and axes, muted ticks."""
    colors = _theme(theme)
    if ax is None:
        _, ax = _pyplot().subplots(figsize=figsize, layout="constrained")
    ax.figure.set_facecolor(colors["surface"])
    ax.set_facecolor(colors["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(colors["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=colors["axis"], labelcolor=colors["secondary"], length=3)
    ax.xaxis.label.set_color(colors["secondary"])
    ax.yaxis.label.set_color(colors["secondary"])
    ax.title.set_color(colors["ink"])
    if grid:
        ax.grid(True, axis=grid, color=colors["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
    return ax, colors


def _legend(ax, colors: dict[str, Any], **kwargs) -> None:
    legend = ax.legend(frameon=False, labelcolor=colors["ink"], **kwargs)
    legend.set_zorder(5)


def _normalize(results: Iterable) -> list[tuple[float | None, Rule]]:
    normalized = result_pairs(results)
    if not normalized:
        raise ValueError("no subgroups to visualize")
    return normalized


def condition_label(condition: Condition) -> str:
    """Readable condition, e.g. ``age ∈ (30, ∞]`` or ``sex=f`` (``str(condition)``)."""
    return str(condition)


def _check_target(target: BaseTarget, data: pd.DataFrame, needs: str | None = None) -> None:
    if getattr(target, "n_samples_", None) != len(data):
        raise ValueError("target must be fitted on the visualized data")
    if needs == "positives" and not isinstance(target, LabelledTarget):
        raise TypeError("this visualization requires a binary or soft-classifier target")
    if needs == "values" and not hasattr(target, "values_"):
        raise TypeError("this visualization requires a numeric target")


def _counts(masks: np.ndarray, target: BaseTarget) -> tuple[np.ndarray, np.ndarray]:
    sizes = masks.sum(axis=1)
    positives = (masks & target.positives_).sum(axis=1)
    return sizes, positives


def _index_labels(count: int) -> list[str]:
    return [f"#{index + 1}" for index in range(count)]


def _abs_formatter():
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda value, _: f"{abs(value):g}")


def _linkage(masks: np.ndarray, method: str = "complete") -> np.ndarray:
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    distances = np.nan_to_num(pdist(masks, metric="jaccard"), nan=0.0)
    return linkage(distances, method=method)


def _cluster_order(masks: np.ndarray, order: str) -> np.ndarray:
    if order not in ("cluster", "given"):
        raise ValueError("order must be 'cluster' or 'given'")
    if order == "given" or len(masks) < 3:
        return np.arange(len(masks))
    from scipy.cluster.hierarchy import leaves_list

    return leaves_list(_linkage(masks))


# Subgroup parameters (size, target share)


def plot_subgroup_bars(
    results: Iterable,
    data: pd.DataFrame,
    target: BaseTarget,
    *,
    sort: bool = True,
    ax=None,
    theme: str = "light",
):
    """Positives right and negatives left of zero; dataset row first (Kralj et al. 2005)."""
    _check_target(target, data, "positives")
    entries = _normalize(results)
    sizes, positives = _counts(subgroup_masks(entries, data), target)
    shares = positives / np.maximum(sizes, 1)
    order = np.argsort(-shares, kind="stable") if sort else np.arange(len(entries))
    ax, colors = _axes(ax, theme, (8, 0.45 * len(entries) + 1.6), grid="x")

    dataset_positives = int(target.positives_.sum())
    dataset_negatives = target.n_samples_ - dataset_positives
    pos = np.concatenate([[dataset_positives], positives[order]])
    neg = np.concatenate([[dataset_negatives], (sizes - positives)[order]])
    labels = ["dataset"] + [
        f"#{index + 1}  {rule_label(entries[index][1])}" for index in order
    ]
    y = np.arange(len(labels))
    style = {"height": 0.62, "edgecolor": colors["surface"], "linewidth": 1.0}
    ax.barh(y, pos, color=colors["positive"], label="positives", **style)
    ax.barh(y, -neg, color=colors["negative"], label="negatives", **style)
    ax.axvline(0, color=colors["axis"], linewidth=0.8)
    ax.axhline(0.5, color=colors["axis"], linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(_abs_formatter())
    ax.set_xlabel("rows: ← negatives | positives →")
    _legend(ax, colors, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=2)
    return ax


def plot_spineplot(
    results: Iterable,
    data: pd.DataFrame,
    target: BaseTarget,
    *,
    ax=None,
    theme: str = "light",
):
    """Width = subgroup size, lower part = target share; dataset share as reference line.

    The stacked bar visualization of VIKAMINE (Atzmueller and Puppe 2005).
    """
    _check_target(target, data, "positives")
    entries = _normalize(results)
    sizes, positives = _counts(subgroup_masks(entries, data), target)
    shares = positives / np.maximum(sizes, 1)
    widths = sizes / target.n_samples_
    gap = 0.012 * widths.sum()
    lefts = np.concatenate([[0.0], np.cumsum(widths + gap)[:-1]])
    ax, colors = _axes(ax, theme, (max(6, 0.5 * len(entries) + 3), 4), grid="y")

    style = {"align": "edge", "edgecolor": colors["surface"], "linewidth": 1.0}
    ax.bar(lefts, shares, widths, color=colors["positive"], label="positives", **style)
    ax.bar(lefts, 1 - shares, widths, bottom=shares, color=colors["negative"],
           label="negatives", **style)
    dataset_share = target.positives_.mean()
    ax.axhline(dataset_share, color=colors["ink"], linewidth=1.0,
               label=f"dataset target share {dataset_share:.3g}")
    ax.set_xticks(lefts + widths / 2, _index_labels(len(entries)))
    ax.set_xlim(-gap, lefts[-1] + widths[-1] + gap)
    ax.set_ylim(0, 1)
    ax.set_xlabel("subgroups (width ∝ size)")
    ax.set_ylabel("share of subgroup rows")
    _legend(ax, colors, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3)
    return ax


def plot_roc_space(
    results: Iterable,
    data: pd.DataFrame,
    target: BaseTarget,
    *,
    counts: bool = False,
    isometrics: StandardQF | None = None,
    levels: int = 10,
    hull: bool = True,
    annotate: bool = True,
    ax=None,
    theme: str = "light",
):
    """Subgroups in ROC space, or coverage space with ``counts=True``.

    ``isometrics`` draws contour lines of a ``StandardQF`` (e.g. ``WRAccQF()``);
    ``hull`` draws the ROC convex hull of the subgroups.
    """
    _check_target(target, data, "positives")
    entries = _normalize(results)
    sizes, tp = _counts(subgroup_masks(entries, data), target)
    fp = sizes - tp
    total_positives = int(target.positives_.sum())
    total_negatives = target.n_samples_ - total_positives
    x_max, y_max = (total_negatives, total_positives) if counts else (1.0, 1.0)
    x = fp if counts else fp / max(total_negatives, 1)
    y = tp if counts else tp / max(total_positives, 1)
    ax, colors = _axes(ax, theme, (5.5, 5.5))

    if isometrics is not None:
        if not isinstance(isometrics, StandardQF):
            raise TypeError("isometrics require a StandardQF (e.g. WRAccQF())")
        grid_fp, grid_tp = np.meshgrid(
            np.linspace(0, total_negatives, 201), np.linspace(0, total_positives, 201)
        )
        size = grid_fp + grid_tp
        with np.errstate(invalid="ignore", divide="ignore"):
            quality = orient((size / target.n_samples_) ** isometrics.a * (
                grid_tp / size - total_positives / target.n_samples_
            ), isometrics.direction)
        scale_x = 1 if counts else max(total_negatives, 1)
        scale_y = 1 if counts else max(total_positives, 1)
        contours = ax.contour(grid_fp / scale_x, grid_tp / scale_y, quality, levels=levels,
                              colors=colors["grid"], linewidths=0.8,
                              negative_linestyles="solid")
        ax.clabel(contours, fontsize=7, colors=colors["muted"], fmt="%.2g")
    ax.plot([0, x_max], [0, y_max], color=colors["axis"], linewidth=0.8)
    if hull:
        points = sorted(set(zip(x.tolist(), y.tolist(), strict=True)) | {(0, 0), (x_max, y_max)})
        upper: list[tuple[float, float]] = []
        for point in points:
            while len(upper) >= 2 and (
                (upper[-1][0] - upper[-2][0]) * (point[1] - upper[-2][1])
                - (upper[-1][1] - upper[-2][1]) * (point[0] - upper[-2][0])
            ) >= 0:
                upper.pop()
            upper.append(point)
        ax.plot(*zip(*upper, strict=True), color=colors["secondary"], linewidth=1.0,
                label="convex hull")
    ax.scatter(x, y, s=64, color=colors["positive"], edgecolors=colors["surface"],
               linewidths=1.5, zorder=3, label="subgroups")
    if annotate:
        for index, (px, py) in enumerate(zip(x, y, strict=True)):
            ax.annotate(f"#{index + 1}", (px, py), xytext=(5, 4), textcoords="offset points",
                        fontsize=8, color=colors["secondary"])
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_aspect("equal" if not counts else "auto")
    ax.set_xlabel("negatives covered" if counts else "false positive rate")
    ax.set_ylabel("positives covered" if counts else "true positive rate")
    _legend(ax, colors, loc="lower right")
    return ax


def plot_size_quality(
    results: Iterable,
    data: pd.DataFrame,
    target: BaseTarget,
    *,
    y: str = "quality",
    annotate: bool = True,
    ax=None,
    theme: str = "light",
):
    """Subgroup size against ``"quality"``, ``"target_share"``, or ``"mean"``.

    The latter two show the dataset value as a reference line.
    """
    _check_target(target, data)
    entries = _normalize(results)
    masks = subgroup_masks(entries, data)
    sizes = masks.sum(axis=1)
    reference = None
    if y == "quality":
        if any(quality is None for quality, _ in entries):
            raise ValueError("y='quality' requires (quality, Rule) results")
        values = np.array([quality for quality, _ in entries], dtype=float)
    elif y == "target_share":
        _check_target(target, data, "positives")
        values = (masks & target.positives_).sum(axis=1) / np.maximum(sizes, 1)
        reference = target.positives_.mean()
    elif y == "mean":
        _check_target(target, data, "values")
        values = np.array([target.values_[mask].mean() for mask in masks])
        reference = target.values_.mean()
    else:
        raise ValueError("y must be 'quality', 'target_share', or 'mean'")
    ax, colors = _axes(ax, theme, (6.5, 4.5))
    if reference is not None:
        ax.axhline(reference, color=colors["ink"], linewidth=1.0,
                   label=f"dataset {y.replace('_', ' ')} {reference:.3g}")
    ax.scatter(sizes, values, s=64, color=colors["positive"], edgecolors=colors["surface"],
               linewidths=1.5, zorder=3)
    if annotate:
        for index, (px, py) in enumerate(zip(sizes, values, strict=True)):
            ax.annotate(f"#{index + 1}", (px, py), xytext=(5, 4), textcoords="offset points",
                        fontsize=8, color=colors["secondary"])
    ax.set_xlim(0, target.n_samples_)
    ax.set_xlabel("subgroup size (rows)")
    ax.set_ylabel(y.replace("_", " "))
    if reference is not None:
        _legend(ax, colors, loc="best")
    return ax


# Distributions


def plot_target_distributions(
    results: Iterable,
    data: pd.DataFrame,
    target: BaseTarget,
    *,
    ax=None,
    theme: str = "light",
):
    """Box plots of a numeric target for the dataset (top) and each subgroup."""
    _check_target(target, data, "values")
    entries = _normalize(results)
    masks = subgroup_masks(entries, data)
    samples = [target.values_] + [target.values_[mask] for mask in masks]
    labels = ["dataset"] + [f"#{i + 1}  {rule_label(rule)}" for i, (_, rule) in enumerate(entries)]
    ax, colors = _axes(ax, theme, (8, 0.5 * len(labels) + 1.5), grid="x")
    import matplotlib

    horizontal = (
        {"orientation": "horizontal"}
        if tuple(int(part) for part in matplotlib.__version__.split(".")[:2]) >= (3, 10)
        else {"vert": False}
    )
    boxes = ax.boxplot(
        samples, widths=0.55, patch_artist=True, **horizontal,
        medianprops={"color": colors["surface"], "linewidth": 1.5},
        whiskerprops={"color": colors["muted"], "linewidth": 1.0},
        capprops={"color": colors["muted"], "linewidth": 1.0},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": colors["muted"],
                    "markeredgecolor": "none"},
    )
    for index, box in enumerate(boxes["boxes"]):
        box.set_facecolor(colors["muted"] if index == 0 else colors["positive"])
        box.set_edgecolor(colors["surface"])
    ax.axvline(target.values_.mean(), color=colors["ink"], linewidth=1.0,
               label=f"dataset mean {target.values_.mean():.6g}")
    ax.axhline(1.5, color=colors["axis"], linewidth=0.8)
    ax.set_yticks(range(1, len(labels) + 1), labels)
    ax.invert_yaxis()
    ax.set_xlabel("target value")
    _legend(ax, colors, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return ax


def plot_attribute_distribution(
    results: Iterable,
    data: pd.DataFrame,
    target: BaseTarget,
    attribute: str,
    *,
    bins: int = 20,
    ax=None,
    theme: str = "light",
):
    """Positives above and negatives below zero over a numeric ``attribute``.

    The dataset is the shaded reference; each subgroup (at most six) is a line
    (Gamberger, Lavrač, and Wettschereck 2002). Subgroup colors skip the
    positive/negative hues, which mean classes in the other visualizations.
    """
    _check_target(target, data, "positives")
    entries = _normalize(results)
    series = _theme(theme)["series"][2:]
    if len(entries) > len(series):
        raise ValueError(f"at most {len(series)} subgroups can be compared")
    values = data[attribute]
    if not pd.api.types.is_numeric_dtype(values.dtype):
        raise TypeError("attribute must be numeric")
    values = values.to_numpy(dtype=float, na_value=np.nan)
    present = ~np.isnan(values)
    edges = np.histogram_bin_edges(values[present], bins=bins)
    masks = subgroup_masks(entries, data)
    ax, colors = _axes(ax, theme, (8, 4.5))

    def histogram(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos, _ = np.histogram(values[rows & present & target.positives_], edges)
        neg, _ = np.histogram(values[rows & present & ~target.positives_], edges)
        return np.append(pos, pos[-1]), np.append(neg, neg[-1])

    pos, neg = histogram(np.ones(len(data), dtype=bool))
    ax.fill_between(edges, pos, step="post", color=colors["muted"], alpha=0.25, linewidth=0,
                    label="dataset")
    ax.fill_between(edges, -neg, step="post", color=colors["muted"], alpha=0.25, linewidth=0)
    for index, ((_, rule), mask) in enumerate(zip(entries, masks, strict=True)):
        pos, neg = histogram(mask)
        color = series[index]
        ax.step(edges, pos, where="post", color=color, linewidth=1.5,
                label=f"#{index + 1}  {rule_label(rule)}")
        ax.step(edges, -neg, where="post", color=color, linewidth=1.5)
    ax.axhline(0, color=colors["axis"], linewidth=0.8)
    ax.yaxis.set_major_formatter(_abs_formatter())
    ax.set_xlabel(attribute)
    ax.set_ylabel("rows: negatives ↓ | positives ↑")
    _legend(ax, colors, loc="upper left", bbox_to_anchor=(1.01, 1))
    return ax


# Exceptional model mining


def _jitter(values: np.ndarray, jitter: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform offsets of up to ``jitter`` times the smallest gap between distinct values."""
    if jitter < 0:
        raise ValueError("jitter must not be negative")
    distinct = np.unique(values)
    gap = np.diff(distinct).min() if len(distinct) > 1 else 0.0
    return rng.uniform(-jitter * gap, jitter * gap, len(values))


def plot_model_scatter(
    results: Iterable,
    data: pd.DataFrame,
    target: BaseTarget,
    *,
    subgroup: int = 1,
    lines: bool = True,
    jitter: float = 0.3,
    ax=None,
    theme: str = "light",
):
    """Scatter plot of two numeric model attributes with the subgroup ``#subgroup`` highlighted.

    For EMM correlation and regression targets (``pysubgroup.emm``): the
    complement's rows are blue rings, the subgroup's rows red rings, and
    ``lines`` adds the least-squares line of each, labelled with the
    correlation and the fitted line, as in the scatter plots of Duivesteijn,
    Feelders, and Knobbe (DMKD 2016, Figs. 3, 4, and 7), which show subgroup and
    complement side by side. With lines, the rings are fainter so that the
    lines stand out.

    ``jitter`` moves each point by a random offset of up to ``jitter`` times the
    smallest gap between distinct values of each attribute, so that rows with
    equal values (e.g. whole numbers) do not hide each other; 0 draws the exact
    values. The offsets are drawn with a fixed seed; correlations and lines use
    the exact values.
    """
    from matplotlib import patheffects

    from .emm.base import NumericModelTarget
    from .emm.correlation import correlations
    from .emm.regression import simple_regression

    _check_target(target, data)
    if not isinstance(target, NumericModelTarget) or len(target.attributes_) != 2:
        raise TypeError("this visualization requires an EMM correlation or regression target")
    entries = _normalize(results)
    if not 1 <= subgroup <= len(entries):
        raise ValueError(f"subgroup must be between 1 and {len(entries)}")
    mask = subgroup_masks(entries, data)[subgroup - 1]
    x, y = target.values_[:, 0], target.values_[:, 1]
    rng = np.random.default_rng(0)
    shown_x, shown_y = (values + _jitter(values, jitter, rng) for values in (x, y))
    ax, colors = _axes(ax, theme, (6.5, 5))
    # Faint points when lines are drawn, so that the lines stand out.
    point_alpha = 0.12 if lines else 0.6
    groups = [(~mask, "complement", colors["positive"]),
              (mask, f"#{subgroup}  {rule_label(entries[subgroup - 1][1])}", colors["series"][7])]
    for rows, label, color in groups:
        moments = target.moments(rows)
        r = correlations(target, *moments)[0]
        size = int(np.count_nonzero(rows))
        ax.scatter(shown_x[rows], shown_y[rows], s=18, facecolors="none", edgecolors=color,
                   alpha=point_alpha, linewidths=0.7,
                   label=f"{label} (n = {size}, r = {r:.3f})")
        if lines and size > 2:
            slope, intercept, _ = simple_regression(target, *moments)
            if np.isfinite(slope[0]):
                ends = np.array([x[rows].min(), x[rows].max()])
                ax.plot(ends, intercept[0] + slope[0] * ends, color=color, linewidth=2.4,
                        zorder=4, label=f"y = {intercept[0]:.4g} + {slope[0]:.4g} x",
                        path_effects=[patheffects.withStroke(linewidth=4.5,
                                                             foreground=colors["surface"])])
    ax.set_xlabel(str(target.attributes_[0]))
    ax.set_ylabel(str(target.attributes_[1]))
    _legend(ax, colors, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    for handle in ax.get_legend().legend_handles:
        handle.set_alpha(1)
    return ax


def _curve(positives: np.ndarray, scores: np.ndarray, curve: str):
    """Points of the ROC or precision-recall curve and its area (nan if undefined)."""
    true_positives, false_positives = _threshold_counts(positives, scores)
    n_positives = true_positives[-1] if len(true_positives) else 0
    n_negatives = false_positives[-1] if len(false_positives) else 0
    if curve == "roc":
        if not n_positives or not n_negatives:
            return None
        x = np.concatenate(([0.0], false_positives / n_negatives))
        y = np.concatenate(([0.0], true_positives / n_positives))
        return x, y, roc_auc_score(positives, scores)
    if not n_positives:
        return None
    x = np.concatenate(([0.0], true_positives / n_positives))
    y = np.concatenate(([1.0], true_positives / (true_positives + false_positives)))
    return x, y, pr_auc_score(positives, scores)


def plot_performance_curves(
    results: Iterable,
    data: pd.DataFrame,
    target: BaseTarget,
    *,
    curve: str = "roc",
    ax=None,
    theme: str = "light",
):
    """ROC (``curve="roc"``) or precision-recall (``"pr"``) curves of the dataset and subgroups.

    For soft-classifier targets: the dataset's curve is drawn in black, each
    subgroup's in its own color, and the legend gives the areas under the
    curves (the measures of ``ROCAUCQF`` and ``PRAUCQF``). Subgroups whose
    curve is undefined (a ROC curve without positives or negatives, a
    precision-recall curve without positives) are listed without a curve. Ties
    in the scores are drawn as diagonal steps, matching the areas.
    """
    if curve not in ("roc", "pr"):
        raise ValueError("curve must be 'roc' or 'pr'")
    _check_target(target, data)
    if not isinstance(target, SoftClassifierTarget):
        raise TypeError("this visualization requires a soft-classifier target")
    entries = _normalize(results)
    masks = subgroup_masks(entries, data)
    ax, colors = _axes(ax, theme, (6, 5.5))
    if curve == "roc":
        ax.plot([0, 1], [0, 1], color=colors["axis"], linewidth=0.8)
    else:
        share = target.n_positives_ / target.n_samples_
        ax.plot([0, 1], [share, share], color=colors["axis"], linewidth=0.8)
    area = "ROC AUC" if curve == "roc" else "PR AUC"
    series = colors["series"]
    curves = [("dataset", np.ones(target.n_samples_, dtype=bool), colors["ink"], 2.0)]
    curves += [(f"#{index + 1}", mask, series[index % len(series)], 1.4)
               for index, mask in enumerate(masks)]
    for label, mask, color, width in curves:
        points = _curve(*target._subgroup(mask), curve)
        if points is None:
            ax.plot([], [], color=color, linewidth=width, label=f"{label} (undefined)")
            continue
        x, y, value = points
        ax.plot(x, y, color=color, linewidth=width,
                label=f"{label} ({area} {value:.3f}, n = {int(mask.sum())})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("false positive rate" if curve == "roc" else "recall")
    ax.set_ylabel("true positive rate" if curve == "roc" else "precision")
    _legend(ax, colors, loc="upper left", bbox_to_anchor=(1.02, 1))
    return ax


# Relations between subgroups


def plot_overlap(
    results: Iterable,
    data: pd.DataFrame,
    target: BaseTarget,
    *,
    order: str = "cluster",
    ax=None,
    theme: str = "light",
):
    """Covered rows per subgroup, colored by class (VIKAMINE cluster view).

    Only rows covered by some subgroup are shown; rows and subgroups are ordered
    so that overlapping subgroups form blocks.
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    _check_target(target, data)
    entries = _normalize(results)
    masks = subgroup_masks(entries, data)
    rows = _cluster_order(masks, order)
    covered = np.flatnonzero(masks.any(axis=0))
    ordered = masks[rows][:, covered]
    binary = isinstance(target, LabelledTarget)
    classes = target.positives_[covered] if binary else np.zeros(len(covered), dtype=bool)
    columns = np.lexsort((~classes, *(~ordered[::-1])))
    ax, colors = _axes(ax, theme, (9, 0.4 * len(entries) + 1.5), grid=None)
    matrix = ordered[:, columns] * np.where(classes[columns], 2, 1)
    palette = [colors["surface"], colors["negative"], colors["positive"]]
    if not binary:
        palette[1] = colors["positive"]
    ax.imshow(matrix, aspect="auto", interpolation="nearest", vmin=0, vmax=2,
              cmap=ListedColormap(palette))
    ax.set_yticks(range(len(rows)), [f"#{i + 1}  {rule_label(entries[i][1])}" for i in rows])
    ax.set_xticks([])
    ax.set_xlabel(f"covered rows ({len(covered)} of {len(data)})")
    if binary:
        handles = [Patch(color=colors["positive"], label="positive row"),
                   Patch(color=colors["negative"], label="negative row")]
        ax.legend(handles=handles, frameon=False, labelcolor=colors["ink"], ncols=2,
                  loc="upper center", bbox_to_anchor=(0.5, -0.08))
    return ax


def plot_similarity(
    results: Iterable,
    data: pd.DataFrame,
    *,
    order: str = "cluster",
    ax=None,
    theme: str = "light",
):
    """Jaccard similarity of the subgroups' covered rows."""
    from matplotlib.colors import LinearSegmentedColormap

    entries = _normalize(results)
    masks = subgroup_masks(entries, data)
    rows = _cluster_order(masks, order)
    ordered = masks[rows].astype(float)
    intersection = ordered @ ordered.T
    sizes = ordered.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - intersection
    similarity = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
    size = 0.4 * len(entries) + 2.5
    ax, colors = _axes(ax, theme, (size + 3, size), grid=None)
    cmap = LinearSegmentedColormap.from_list("sequential", colors["ramp"])
    image = ax.imshow(similarity, vmin=0, vmax=1, cmap=cmap, interpolation="nearest",
                      aspect="auto")
    labels = [f"#{i + 1}" for i in rows]
    ax.set_xticks(range(len(rows)), labels)
    ax.set_yticks(range(len(rows)), [f"#{i + 1}  {rule_label(entries[i][1])}" for i in rows])
    colorbar = ax.figure.colorbar(image, ax=ax, shrink=0.8)
    colorbar.set_label("Jaccard similarity", color=colors["secondary"])
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=colors["axis"], labelcolor=colors["secondary"])
    return ax


def plot_similarity_dendrogram(
    results: Iterable,
    data: pd.DataFrame,
    *,
    method: str = "complete",
    ax=None,
    theme: str = "light",
):
    """Dendrogram of the subgroups clustered by the Jaccard distance of their rows.

    ``method`` is the linkage of ``scipy.cluster.hierarchy.linkage`` (default
    complete linkage, as for the row order of ``plot_overlap`` and
    ``plot_similarity``). Subgroups joined at distance 0 cover the same rows.
    """
    from scipy.cluster.hierarchy import dendrogram

    entries = _normalize(results)
    if len(entries) < 2:
        raise ValueError("a dendrogram needs at least two subgroups")
    masks = subgroup_masks(entries, data)
    ax, colors = _axes(ax, theme, (9, 0.4 * len(entries) + 1.5), grid="x")
    labels = [f"#{i + 1}  {rule_label(rule)}" for i, (_, rule) in enumerate(entries)]
    dendrogram(_linkage(masks, method), orientation="right", labels=labels, ax=ax,
               link_color_func=lambda _: colors["secondary"])
    for line in ax.collections:
        line.set_linewidth(1.2)
    ax.tick_params(axis="y", length=0, labelcolor=colors["ink"])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Jaccard distance of covered rows")
    return ax


def plot_specialization_graph(
    results: Iterable,
    data: pd.DataFrame,
    target: BaseTarget,
    *,
    ax=None,
    theme: str = "light",
):
    """Subgroups by description length; edges lead to direct specializations.

    Each node is a bar split into positives and negatives (width ∝ size); the
    shade of the positive part shows quality, darker is better (VIKAMINE).
    """
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    _check_target(target, data)
    entries = _normalize(results)
    masks = subgroup_masks(entries, data)
    sizes = masks.sum(axis=1)
    labelled = isinstance(target, LabelledTarget)
    positives = (masks & target.positives_).sum(axis=1) if labelled else sizes
    descriptions = [frozenset(rule.conditions) for _, rule in entries]
    depths = np.array([len(description) for description in descriptions])
    qualities = [quality for quality, _ in entries]
    has_quality = all(quality is not None for quality in qualities)

    # Place each depth level on its own row, best quality (or given order) first.
    positions = {}
    for depth in np.unique(depths):
        level = [i for i in range(len(entries)) if depths[i] == depth]
        if has_quality:
            level.sort(key=lambda i: -qualities[i])
        for slot, index in enumerate(level):
            positions[index] = ((slot + 0.5) / len(level), -float(depth))
    ax, colors = _axes(ax, theme, (9, 1.6 * len(np.unique(depths)) + 1), grid=None)

    for child, child_description in enumerate(descriptions):
        parents = [p for p, d in enumerate(descriptions) if d < child_description]
        direct = [
            p for p in parents
            if not any(descriptions[p] < descriptions[q] < child_description for q in parents)
        ]
        for parent in direct:
            (x0, y0), (x1, y1) = positions[parent], positions[child]
            ax.plot([x0, x1], [y0 - 0.18, y1 + 0.18], color=colors["axis"], linewidth=0.8,
                    zorder=1)

    cmap = LinearSegmentedColormap.from_list("sequential", colors["ramp"][1:])
    norm = Normalize(min(qualities), max(qualities)) if has_quality else None
    max_width = 0.8 / max(np.bincount(depths).max(), 1)
    for index, (x, y) in positions.items():
        width = max_width * sizes[index] / max(sizes.max(), 1)
        positive_width = width * positives[index] / max(sizes[index], 1)
        positive_color = cmap(norm(qualities[index])) if norm else colors["positive"]
        left = x - width / 2
        ax.barh(y, positive_width, left=left, height=0.28, color=positive_color, zorder=2,
                edgecolor=colors["surface"], linewidth=1.0)
        ax.barh(y, width - positive_width, left=left + positive_width, height=0.28,
                color=colors["negative"], zorder=2, edgecolor=colors["surface"], linewidth=1.0)
        ax.text(x, y - 0.24, f"#{index + 1}", ha="center", va="top", fontsize=8,
                color=colors["secondary"])
    levels = [int(depth) for depth in np.unique(depths)]
    ax.set_yticks([-float(d) for d in levels], [f"{d} condition{'s' * (d != 1)}" for d in levels])
    ax.set_ylim(-max(levels) - 0.6, -min(levels) + 0.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    if norm:
        mappable = _pyplot().cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = ax.figure.colorbar(mappable, ax=ax, shrink=0.8)
        colorbar.set_label("quality (positive part)", color=colors["secondary"])
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(colors=colors["axis"], labelcolor=colors["secondary"])
    return ax


# Statistical validation


def plot_null_distribution(
    results: Iterable,
    data: pd.DataFrame,
    target: BaseTarget,
    qf: QualityFunction,
    *,
    n_samples: int = 1000,
    stratify_target: bool = False,
    stratify=None,
    random_state: int | np.random.Generator | None = None,
    plus_one: bool = False,
    ax=None,
    theme: str = "light",
):
    """Each subgroup's statistic (dot) over its randomization null distribution.

    The null distribution of a subgroup is the statistic of ``n_samples``
    random subsets of ``data`` of the subgroup's size and strata, drawn by
    ``significance.randomization_null_distribution`` (the arguments are those
    of ``randomization_p_values``). The box spans the quartiles and the
    whiskers the 2.5th to 97.5th percentiles; the label on the right is the
    uncorrected empirical p-value. ``data`` and ``target`` are usually the
    validation set. Subgroups with an undefined statistic show no box.
    """
    _check_target(target, data)
    entries = _normalize(results)
    observed, samples = randomization_null_distribution(
        entries, data, target, qf, n_samples=n_samples, stratify_target=stratify_target,
        stratify=stratify, random_state=random_state,
    )
    p_values = _p_values(observed, samples, plus_one)
    ax, colors = _axes(ax, theme, (9, 0.45 * len(entries) + 1.6), grid="x")
    y = -np.arange(len(entries), dtype=float)
    for row, values in zip(y, samples, strict=True):
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        low, q1, median, q3, high = np.percentile(values, [2.5, 25, 50, 75, 97.5])
        ax.plot([low, high], [row, row], color=colors["muted"], linewidth=1.2, zorder=2)
        ax.barh(row, q3 - q1, left=q1, height=0.5, color=colors["grid"],
                edgecolor=colors["muted"], linewidth=0.8, zorder=2)
        ax.plot([median, median], [row - 0.25, row + 0.25], color=colors["secondary"],
                linewidth=1.2, zorder=3)
    defined = np.isfinite(observed)
    ax.scatter(observed[defined], y[defined], s=48, color=colors["positive"],
               edgecolor=colors["surface"], linewidth=2, zorder=4, label="subgroup")
    ax.plot([], [], color=colors["muted"], linewidth=6, alpha=0.5,
            label="null distribution (2.5-25-50-75-97.5 %)")
    transform = ax.get_yaxis_transform()
    for row, p_value in zip(y, p_values, strict=True):
        ax.text(1.01, row, f"p = {p_value:.3g}", transform=transform, va="center",
                fontsize=8, color=colors["secondary"])
    ax.set_yticks(y, [f"#{i + 1}  {rule_label(rule)}" for i, (_, rule) in enumerate(entries)])
    ax.tick_params(axis="y", length=0, labelcolor=colors["ink"])
    ax.set_ylim(y[-1] - 0.7, 0.7)
    ax.margins(x=0.04)
    ax.set_xlabel(f"{type(qf).__name__} on random subsets of equal size")
    from matplotlib.transforms import offset_copy

    below_label = offset_copy(ax.transAxes, fig=ax.figure, y=-32, units="points")
    _legend(ax, colors, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 0),
            bbox_transform=below_label)
    return ax
