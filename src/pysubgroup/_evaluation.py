"""Evaluation of the children of a search node: one subgroup at a time or batched.

Searches ask ``_Search.expand`` for the refinements of one subgroup by a list
of candidate conditions and replay their decisions child by child, in their own
order. Two implementations answer:

- ``ScalarChildren`` evaluates each child on demand with the quality
  function's ``evaluate`` and ``optimistic_estimate`` on its Boolean row mask,
  as the searches did before batching.
- ``BatchedChildren`` computes the statistics of all children at once
  (``ChildStatistics``: sizes, counts, sums, and maxima over the children's
  rows) and passes them to the quality function's ``evaluate_many`` and
  ``optimistic_estimate_many`` and the filters' ``is_satisfied_many``.

Qualities and estimates do not depend on the search state, so replaying the
precomputed values in the original order takes the same decisions as the
scalar path. Integer statistics are exact; float sums agree up to rounding.

``Backend`` holds the condition masks on the device: NumPy arrays on the CPU,
or CuPy arrays on a CUDA GPU (``pip install pysubgroup[gpu]``). Counts are
popcounts of bit-packed masks (64 rows per word). On the CPU, sums and maxima
use the numba kernels when available; otherwise, and on the GPU, they reduce
blocks of child masks. Only the statistics of each node, one value per child,
return to the host.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import _numba

# Children per block, chosen so that a block of child masks (and its float
# temporaries) stays below about 2**24 elements.
_BLOCK_ELEMENTS = 1 << 24
# Value columns up to which the numba kernel sums column by column; wider
# arrays use a matrix product.
_KERNEL_COLUMNS = 16
DEVICES = ("cpu", "cuda")


def _cupy():
    try:
        import cupy
    except ImportError as error:  # pragma: no cover - depends on the installation
        raise ImportError(
            "device='cuda' requires CuPy: pip install pysubgroup[gpu] "
            "(or the cupy wheel matching your CUDA version)"
        ) from error
    return cupy


class Backend:
    """Condition masks (conditions x rows) on the CPU (NumPy) or a CUDA GPU (CuPy).

    Masks are kept both as Booleans (for sums and maxima) and as uint64 words
    (``_numba.pack64``) for counts.
    """

    def __init__(self, masks: np.ndarray, device: str = "cpu"):
        if device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {device!r}")
        self.device = device
        self.xp = np if device == "cpu" else _cupy()
        self.n_rows = masks.shape[1]
        self.masks = np.ascontiguousarray(masks) if device == "cpu" else self.xp.asarray(masks)
        self.words = self.xp.asarray(_numba.pack64(masks))
        self.use_kernels = device == "cpu" and _numba.enabled()
        # Device copies of target arrays, keyed by kind and id; the host array is
        # kept alive so that its id is not reused.
        self._arrays: dict[tuple, tuple[np.ndarray, object]] = {}

    def _cached(self, kind: str, array: np.ndarray, convert):
        cached = self._arrays.get((kind, id(array)))
        if cached is None or cached[0] is not array:
            cached = self._arrays[(kind, id(array))] = (array, convert(array))
        return cached[1]

    def asarray(self, array: np.ndarray):
        """The device copy of a long-lived host array (such as a target's values)."""
        if self.xp is np:
            return array
        return self._cached("array", array, self.xp.asarray)

    def packed(self, rows: np.ndarray):
        """The packed device copy of a long-lived host row mask."""
        return self._cached("packed", rows, lambda r: self.xp.asarray(_numba.pack64(r)))

    def pack(self, mask):
        """Pack a device row mask into uint64 words."""
        if self.xp is np:
            return _numba.pack64(mask)
        packed = self.xp.packbits(mask, bitorder="little")
        words = self.xp.zeros(-(-len(packed) // 8) * 8, dtype=np.uint8)
        words[: len(packed)] = packed
        return words.view(np.uint64)

    def to_host(self, array) -> np.ndarray:
        return array if self.xp is np else self.xp.asnumpy(array)

    def mask(self, indices: Sequence[int]):
        """Row mask of the conjunction of the conditions ``indices`` (all rows if empty)."""
        if not len(indices):
            return self.xp.ones(self.n_rows, dtype=bool)
        return self.masks[list(indices)].all(axis=0)

    def child_mask(self, parent, index: int):
        return self.masks[index] if parent is None else parent & self.masks[index]


def _selection(candidates: np.ndarray, xp):
    """A slice for consecutive candidates (a view, no copy), otherwise an index array."""
    if len(candidates) and candidates[-1] - candidates[0] == len(candidates) - 1 and (
        np.all(np.diff(candidates) == 1)
    ):
        return slice(int(candidates[0]), int(candidates[0]) + len(candidates))
    return xp.asarray(candidates)


class ChildStatistics:
    """Statistics of the children ``parent & condition`` for all candidate conditions.

    Every method returns a host NumPy array with one entry per candidate (and
    one column per value column for two-dimensional ``values``). Arguments are
    host arrays of the fitted target; results are cached per argument, so
    quality functions and filters may request the same statistic repeatedly.
    """

    def __init__(self, backend: Backend, parent, candidates: Sequence[int]):
        self.backend = backend
        self.parent = parent
        self.candidates = np.asarray(candidates, dtype=np.intp)
        self.n_children = len(self.candidates)
        self._parent_words = None
        self._host_parent = None
        self._results: dict[tuple, tuple[object, np.ndarray]] = {}

    def _blocks(self, row_length: int):
        step = max(1, _BLOCK_ELEMENTS // max(1, row_length))
        xp = self.backend.xp
        for start in range(0, self.n_children, step):
            yield _selection(self.candidates[start:start + step], xp)

    def _cache(self, key: tuple, argument, compute) -> np.ndarray:
        cached = self._results.get(key)
        if cached is not None and cached[0] is argument:
            return cached[1]
        result = compute()
        self._results[key] = (argument, result)
        return result

    def _concatenate(self, parts: list, shape: tuple) -> np.ndarray:
        if not parts:
            return np.zeros(shape)
        return self.backend.to_host(self.backend.xp.concatenate(parts))

    def _counts(self, rows: np.ndarray | None) -> np.ndarray:
        backend, xp = self.backend, self.backend.xp
        if self.parent is not None and self._parent_words is None:
            self._parent_words = backend.pack(self.parent)
        extra = self._parent_words
        if rows is not None:
            words = backend.packed(rows)
            extra = words if extra is None else extra & words
        parts = []
        for selection in self._blocks(8 * backend.words.shape[1]):
            words = backend.words[selection]
            if extra is not None:
                words = words & extra
            parts.append(xp.bitwise_count(words).sum(axis=1, dtype=np.int64))
        return self._concatenate(parts, (0,)).astype(np.int64)

    @property
    def sizes(self) -> np.ndarray:
        """Number of rows of each child."""
        return self._cache(("sizes",), None, lambda: self._counts(None))

    def count(self, rows: np.ndarray) -> np.ndarray:
        """Number of rows of each child inside the Boolean row mask ``rows``."""
        return self._cache(("count", id(rows)), rows, lambda: self._counts(rows))

    def _children(self):
        """Blocks of child masks (children x rows) on the device."""
        masks = self.backend.masks
        for selection in self._blocks(self.backend.n_rows):
            selected = masks[selection]
            yield selected if self.parent is None else selected & self.parent

    def _weighted(self, values: np.ndarray) -> np.ndarray:
        """Host values with weight 0 outside the parent, for the numba kernels."""
        if self.parent is None:
            return values
        parent = self.parent if values.ndim == 1 else self.parent[:, None]
        return np.where(parent, values, 0.0)

    def sum(self, values: np.ndarray) -> np.ndarray:
        """Sum of ``values`` (rows, or rows x columns) over the rows of each child."""
        return self._cache(("sum", id(values)), values, lambda: self._sum(values))

    def _sum(self, values: np.ndarray) -> np.ndarray:
        backend = self.backend
        columns = 1 if values.ndim == 1 else values.shape[1]
        if backend.use_kernels and columns <= _KERNEL_COLUMNS:
            weighted = self._weighted(values)
            if values.ndim == 1:
                return _numba.candidate_sums(backend.masks, self.candidates, weighted)
            sums = [
                _numba.candidate_sums(
                    backend.masks, self.candidates, np.ascontiguousarray(weighted[:, k])
                )
                for k in range(columns)
            ]
            return np.stack(sums, axis=1).reshape(self.n_children, columns)
        device = backend.asarray(values)
        xp = backend.xp
        if values.ndim == 1:
            parts = [xp.where(children, device, 0.0).sum(axis=1) for children in self._children()]
        else:
            parts = [children.astype(np.float64) @ device for children in self._children()]
        return self._concatenate(parts, (0,) if values.ndim == 1 else (0, columns))

    def maximum(self, values: np.ndarray) -> np.ndarray:
        """Largest of ``values`` (one per row) in each child; ``-inf`` for empty children."""
        return self._cache(("maximum", id(values)), values, lambda: self._maximum(values))

    def _maximum(self, values: np.ndarray) -> np.ndarray:
        backend = self.backend
        if backend.use_kernels:
            if self._host_parent is None:
                self._host_parent = (
                    np.ones(backend.n_rows, dtype=bool) if self.parent is None else self.parent
                )
            return _numba.candidate_maxima(
                backend.masks, self.candidates, self._host_parent, values
            )
        device, xp = backend.asarray(values), backend.xp
        parts = [xp.where(children, device, -np.inf).max(axis=1) for children in self._children()]
        return self._concatenate(parts, (0,))


class ScalarChildren:
    """Children evaluated one at a time, on demand, from their host row masks."""

    def __init__(self, search, parent: tuple[int, ...], parent_mask, candidates: Sequence[int]):
        self.search = search
        self.parent = parent
        self.parent_mask = parent_mask
        self.candidates = candidates
        self._last: tuple[int, np.ndarray] | None = None

    def indices(self, i: int) -> tuple[int, ...]:
        return (*self.parent, self.candidates[i])

    def mask(self, i: int) -> np.ndarray:
        # Searches process children in turn, so only the latest mask is kept.
        if self._last is None or self._last[0] != i:
            condition = self.search.masks[self.candidates[i]]
            mask = condition if self.parent_mask is None else self.parent_mask & condition
            self._last = (i, mask)
        return self._last[1]

    def admissible(self, i: int) -> bool:
        return self.search.admissible(self.mask(i), self.indices(i))

    def quality(self, i: int) -> float:
        return self.search.evaluate(self.mask(i), self.indices(i))

    def estimate(self, i: int) -> float:
        return self.search.optimistic_estimate(self.mask(i))

    def passes_result_filters(self, i: int, indices: tuple[int, ...]) -> bool:
        return self.search._passes(self.search.result_filters, self.mask(i), indices)


class BatchedChildren:
    """Children evaluated together from their ``ChildStatistics``."""

    def __init__(self, search, parent: tuple[int, ...], parent_mask, candidates: Sequence[int]):
        self.search = search
        self.parent = parent
        self.parent_mask = parent_mask
        self.candidates = candidates
        self.statistics = ChildStatistics(search.backend, parent_mask, candidates)
        self._admissible = None
        self._quality = None
        self._estimate = None
        self._result_filters = None

    def indices(self, i: int) -> tuple[int, ...]:
        return (*self.parent, self.candidates[i])

    def mask(self, i: int):
        """The child's row mask on the device."""
        return self.search.backend.child_mask(self.parent_mask, self.candidates[i])

    def admissible(self, i: int) -> bool:
        if self._admissible is None:
            task, statistics = self.search.task, self.statistics
            admissible = statistics.sizes > 0
            for f in self.search.pruning_filters:
                admissible &= f.is_satisfied_many(statistics, task.target)
            self._admissible = admissible
        return bool(self._admissible[i])

    def quality(self, i: int) -> float:
        if self._quality is None:
            task = self.search.task
            with np.errstate(divide="ignore", invalid="ignore"):
                if self.search.is_area:
                    quality = task.qf.evaluate_many(
                        self.statistics, task.target, depth=len(self.parent) + 1
                    )
                else:
                    quality = task.qf.evaluate_many(self.statistics, task.target)
            self._quality = np.asarray(quality, dtype=float)
        return float(self._quality[i])

    def estimate(self, i: int) -> float:
        if self._estimate is None:
            task = self.search.task
            if self.search.is_bounded:
                with np.errstate(divide="ignore", invalid="ignore"):
                    estimate = task.qf.optimistic_estimate_many(self.statistics, task.target)
                self._estimate = np.asarray(estimate, dtype=float)
            else:
                self._estimate = np.full(self.statistics.n_children, float("inf"))
        return float(self._estimate[i])

    def passes_result_filters(self, i: int, indices: tuple[int, ...]) -> bool:
        filters = self.search.result_filters
        if not filters:
            return True
        if all(f.batchable for f in filters):
            if self._result_filters is None:
                task = self.search.task
                passes = np.ones(self.statistics.n_children, dtype=bool)
                for f in filters:
                    passes &= f.is_satisfied_many(self.statistics, task.target)
                self._result_filters = passes
            return bool(self._result_filters[i])
        mask = self.search.backend.to_host(self.mask(i))
        return self.search._passes(filters, mask, indices)
