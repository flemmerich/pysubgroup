"""Optional numba kernels (``pip install pysubgroup[numba]``).

Callers keep a NumPy path and use a kernel only if ``enabled()``; both paths
compute the same counts, and float sums agree up to rounding. Set the
environment variable ``PYSUBGROUP_DISABLE_NUMBA=1`` to force the NumPy paths.
Compiled kernels are cached on disk (``cache=True``); ``NUMBA_CACHE_DIR``
redirects the cache, e.g. away from network or synchronized drives.
"""

from __future__ import annotations

import os

import numpy as np

try:
    import numba
    from numba.cpython.unsafe.numbers import leading_zeros, trailing_zeros
except ImportError:  # pragma: no cover - depends on the installation
    numba = None

AVAILABLE = numba is not None

# Prefix scores of ``ordered_prefix_scan``, from the sum ``s`` of ``k`` rows out of ``n``.
PREFIX_MEAN, PREFIX_VARIANCE_REDUCTION, PREFIX_MANN_WHITNEY, PREFIX_RANK_AUC = range(4)
_DISABLED = os.environ.get("PYSUBGROUP_DISABLE_NUMBA", "").lower() in ("1", "true", "yes")
ENABLED = AVAILABLE and not _DISABLED


def enabled() -> bool:
    """Whether numba kernels are installed and not disabled."""
    return ENABLED


def pack64(masks: np.ndarray) -> np.ndarray:
    """Boolean masks (last axis = rows) as uint64 words; bit ``i % 64`` of word ``i // 64``."""
    packed = np.packbits(masks, axis=-1, bitorder="little")
    words = np.zeros((*packed.shape[:-1], -(-packed.shape[-1] // 8) * 8), dtype=np.uint8)
    words[..., : packed.shape[-1]] = packed
    return words.view("<u8").astype(np.uint64, copy=False)


if AVAILABLE:
    # Reassociation lets LLVM vectorize the float sums. Branch-free masking keeps
    # random masks fast; it is exact because target values are finite.
    _FASTMATH = {"reassoc", "nsz"}

    @numba.njit(cache=True, nogil=True, fastmath=_FASTMATH)
    def masked_sum(values, mask):
        """Size and value sum of the rows in ``mask``, without copying them."""
        size = 0
        total = 0.0
        for i in range(values.shape[0]):
            size += mask[i]
            total += values[i] * mask[i]
        return size, total

    @numba.njit(cache=True, nogil=True, fastmath=_FASTMATH)
    def masked_squared_deviation(values, mask, center):
        """Sum of ``(value - center) ** 2`` over the rows in ``mask``."""
        total = 0.0
        for i in range(values.shape[0]):
            deviation = (values[i] - center) * mask[i]
            total += deviation * deviation
        return total

    @numba.njit(cache=True, nogil=True, fastmath=_FASTMATH)
    def masked_excess(values, mask, threshold):
        """Size, and count and summed excess ``value - threshold`` of values above it.

        Statistics are over the rows in ``mask``.
        """
        size = 0
        n_above = 0
        excess = 0.0
        for i in range(values.shape[0]):
            size += mask[i]
            difference = values[i] - threshold
            above = (difference > 0) & mask[i]
            n_above += above
            excess += difference * above
        return size, n_above, excess

    @numba.njit(cache=True, nogil=True, fastmath=_FASTMATH)
    def candidate_sums(masks, candidates, weighted):
        """Sum of ``weighted`` over the rows of ``masks[c]`` for every candidate ``c``.

        Rows outside the parent subgroup have weight 0, so no child mask is built.
        """
        sums = np.zeros(candidates.shape[0])
        for j in range(candidates.shape[0]):
            row = masks[candidates[j]]
            total = 0.0
            for i in range(weighted.shape[0]):
                total += weighted[i] * row[i]
            sums[j] = total
        return sums

    @numba.njit(cache=True, nogil=True)
    def candidate_maxima(masks, candidates, parent, values):
        """Largest value in ``masks[c] & parent`` for every candidate ``c`` (``-inf`` if empty)."""
        maxima = np.full(candidates.shape[0], -np.inf)
        for j in range(candidates.shape[0]):
            row = masks[candidates[j]]
            best = -np.inf
            for i in range(values.shape[0]):
                if row[i] and parent[i] and values[i] > best:
                    best = values[i]
            maxima[j] = best
        return maxima

    @numba.njit(cache=True, nogil=True, inline="always")
    def _next_bit(word, reverse):
        """Position of the lowest set bit of a nonzero word (the highest, if ``reverse``)."""
        if reverse:
            return np.int64(63) - np.int64(leading_zeros(word))
        return np.int64(trailing_zeros(word))

    @numba.njit(cache=True, nogil=True)
    def ordered_mean_scan(parent, condition, values, weights, centroid, max_size, threshold,
                          reverse, check):
        """NumBSD's ``computeRefinement`` for ``n ** a * (mean - centroid)``.

        ``weights[k]`` is ``k ** a`` for every size ``k`` up to the dataset size.
        Rows are sorted by decreasing value (``values``) and packed into
        ``parent`` and ``condition`` with ``pack64``. Rows of ``parent &
        condition`` are read in that order (or, with ``reverse``, in the opposite
        order on the negated values), tracking the size, sum, and the best
        quality of a prefix: the ordering-based bound. With ``check``, after
        every word the fast bound of Theorem 9 (``max_size`` bounds the size)
        is checked; once the prefixes read so far and the fast bound are both
        at most ``threshold``, the scan stops. Returns ``(size, sum, best,
        stopped)``. Lemmerich, Atzmueller and Puppe (DMKD 2016), Algorithm 1.
        """
        n_words = parent.shape[0]
        size = 0
        total = 0.0
        best = -np.inf
        last = 0.0
        for k in range(n_words):
            w = n_words - 1 - k if reverse else k
            word = parent[w] & condition[w]
            if word == 0:
                continue
            while word:
                bit = _next_bit(word, reverse)
                word ^= np.uint64(1) << np.uint64(bit)
                value = values[w * 64 + bit]
                if reverse:
                    value = -value
                size += 1
                total += value
                last = value
                quality = weights[size] * (total / size - centroid)
                if quality > best:
                    best = quality
            if check and best <= threshold:
                remaining = (total + (max_size - size) * last) / max_size - centroid
                if weights[max_size] * remaining <= threshold:
                    return size, total, best, True
        return size, total, best, False

    @numba.njit(cache=True, nogil=True)
    def _prefix_score(kind, s, k, n, weight, centroid):
        if kind == PREFIX_MEAN:
            return 0.0 if k == n else weight * (s / k - centroid)
        if kind == PREFIX_VARIANCE_REDUCTION:
            return 0.0 if k >= n else s * s / weight
        if kind == PREFIX_MANN_WHITNEY:
            return 0.0 if k >= n else s / weight
        return 0.5 if k >= n else 0.5 + s / weight

    @numba.njit(cache=True, nogil=True)
    def ordered_prefix_scan(parent, condition, values, kind, weights, centroid, n, reverse):
        """One pass of NumBSD's ``computeRefinement`` for a prefix-scored measure.

        ``values`` (a per-row array in target order, e.g. deviations) are read
        for the rows of ``parent & condition`` (packed with ``pack64``) in
        order, or, with ``reverse``, in the opposite order and negated. Every
        prefix of ``k`` rows with sum ``s`` is scored by ``kind``:
        ``weights[k] * (s / k - centroid)`` (mean), ``s**2 / weights[k]``
        (variance reduction), ``s / weights[k]`` (Mann-Whitney) or ``0.5 + s /
        weights[k]`` (rank AUC), with the size factors precomputed per ``k``.
        Returns ``(size, best prefix score, score of all rows)``; the best score
        is the ordering-based bound of Lemmerich, Atzmueller and Puppe (DMKD
        2016), Definition 1.
        """
        n_words = parent.shape[0]
        size = 0
        total = 0.0
        best = -np.inf
        score = np.nan
        for k in range(n_words):
            w = n_words - 1 - k if reverse else k
            word = parent[w] & condition[w]
            while word:
                bit = _next_bit(word, reverse)
                word ^= np.uint64(1) << np.uint64(bit)
                value = values[w * 64 + bit]
                size += 1
                total += -value if reverse else value
                score = _prefix_score(kind, total, size, n, weights[size], centroid)
                if score > best:
                    best = score
        return size, best, score
