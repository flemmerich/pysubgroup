"""Correlation model: the correlation coefficient of two numeric targets.

From W. Duivesteijn, A. Feelders, and A. Knobbe, "Exceptional Model Mining",
Data Mining and Knowledge Discovery 30 (2016), Sect. 5.1 (first in Leman,
Feelders, and Knobbe, ECML PKDD 2008, Sect. 3.1). With sample correlations
``r_G`` of the subgroup and ``r_C`` of its complement:

- ``phi_scd`` (``CorrelationSignificanceQF``): one minus the two-sided p-value
  of ``z* = (z'_G - z'_C) / sqrt(1 / (n - 3) + 1 / (n_C - 3))`` for the Fisher
  transforms ``z' = artanh(r)``, a standard normal statistic under
  ``rho_G = rho_C`` for bivariate normal data;
- ``phi_abs = |r_G - r_C|`` and ``phi_ent = phi_ef * phi_abs``
  (``CorrelationDifferenceQF``).

Written from the papers without reusing code. Deviations: groups in which a
target is constant, and the empty subgroup and complement, score ``nan``; so do
groups with at most three rows for ``phi_scd``. ``phi_abs`` and ``phi_ent`` can
also compare with the whole dataset (``reference="dataset"``, DMKD
Sect. 3.2.2); ``phi_scd`` tests two independent samples and compares with the
complement only. On the Windsor housing data, ``r_G`` and ``r_C`` of the
paper's best subgroup ``D1`` agree with DMKD Table 1, but ``phi_scd(D1)`` is
0.99989 instead of the reported 0.9993; the cause is unknown.
"""

from __future__ import annotations

import numpy as np

from .base import (
    MomentQF,
    NumericModelTarget,
    check_reference,
    check_score,
    entropy_weight,
    normal_score,
)


class CorrelationTarget(NumericModelTarget):
    """Two numeric model attributes ``x`` and ``y`` and their correlation."""

    n_model_attributes = 2


def correlations(target: NumericModelTarget, sizes, sums, products) -> np.ndarray:
    """Sample correlations of the first two model attributes, ``nan`` for constant groups."""
    scatter = target.scatter(sizes, sums, products)
    constant = target.constant(sizes, scatter)[..., :2].any(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = scatter[..., 0, 1] / np.sqrt(scatter[..., 0, 0] * scatter[..., 1, 1])
    return np.where(constant | (np.asarray(sizes) < 2), np.nan, np.clip(r, -1.0, 1.0))


class CorrelationSignificanceQF(MomentQF):
    """``phi_scd``: significance of the correlation difference (DMKD Sect. 5.1).

    ``score="1-p"`` (default) gives the papers' one minus the two-sided p-value,
    ``score="statistic"`` the order-equivalent ``|z*|``. The papers consider
    the normal approximation accurate when both groups have more than 25 rows,
    a constraint for a ``MinSizeFilter``; groups of at most three rows score
    ``nan``. Perfectly correlated groups have an infinite ``|z*|``.
    """

    target_type = CorrelationTarget

    def __init__(self, score: str = "1-p"):
        self.score = check_score(score)

    def _scores(self, sizes, sums, products, target) -> np.ndarray:
        complement = target.complement(sizes, sums, products)
        r_subgroup = correlations(target, sizes, sums, products)
        r_complement = correlations(target, *complement)
        with np.errstate(divide="ignore", invalid="ignore"):
            difference = np.arctanh(r_subgroup) - np.arctanh(r_complement)
            z = difference / np.sqrt(1 / (sizes - 3) + 1 / (complement[0] - 3))
        z = np.where((sizes > 3) & (complement[0] > 3), z, np.nan)
        return normal_score(z, self.score)


class CorrelationDifferenceQF(MomentQF):
    """``phi_abs = |r_G - r_ref|``, or ``phi_ent = phi_ef * phi_abs`` with ``entropy=True``.

    ``reference="complement"`` (default, as in the papers) compares with the
    complement, ``"dataset"`` with the whole dataset (DMKD Sect. 3.2.2).
    """

    target_type = CorrelationTarget

    def __init__(self, reference: str = "complement", entropy: bool = False):
        self.reference = check_reference(reference)
        self.entropy = bool(entropy)

    def _scores(self, sizes, sums, products, target) -> np.ndarray:
        r_subgroup = correlations(target, sizes, sums, products)
        if self.reference == "complement":
            r_reference = correlations(target, *target.complement(sizes, sums, products))
        else:
            r_reference = correlations(target, *(np.asarray(t)[None] for t in target.totals_))
        quality = np.abs(r_subgroup - r_reference)
        if self.reference == "dataset":
            quality = np.where(sizes == target.n_samples_, 0.0, quality)
        if self.entropy:
            quality = entropy_weight(sizes, target.n_samples_) * quality
        return np.where(sizes > 0, quality, np.nan)
