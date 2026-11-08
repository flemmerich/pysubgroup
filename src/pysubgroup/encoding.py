"""Conditions-first encoding with interchangeable coverage representations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.preprocessing import KBinsDiscretizer

from .rules import Condition


def _distinct(series: pd.Series) -> list:
    """Distinct values in first-seen order, as the Python scalars that iteration yields.

    Hashing happens in pandas; if it rejects a value, all values are returned.
    """
    try:
        return list(series.drop_duplicates())
    except TypeError:
        return list(series)


def equals(series: pd.Series, value) -> pd.Series:
    """``series == value`` per row; a list-like ``value`` (a tuple) is one value, not a sequence."""
    if pd.api.types.is_list_like(value):
        return series.isin([value])
    return series.eq(value)


def _frame(X, expected: list | None = None) -> pd.DataFrame:
    if sparse.issparse(X):
        raise TypeError("Sparse input is not supported")
    if isinstance(X, pd.DataFrame):
        if X.columns.has_duplicates:
            raise ValueError("X must not contain duplicate feature names")
        frame = X.copy(deep=False)
        if expected is not None:
            missing = [name for name in expected if name not in frame.columns]
            unexpected = [name for name in frame.columns if name not in expected]
            if missing:
                raise ValueError(f"X is missing features: {missing}")
            if unexpected:
                raise ValueError(f"X has unexpected features: {unexpected}")
            frame = frame.loc[:, expected]
    else:
        values = np.asarray(X)
        if values.ndim != 2:
            raise ValueError(
                "X must be a two-dimensional array or DataFrame. Reshape your data first."
            )
        if expected is not None and values.shape[1] != len(expected):
            raise ValueError(f"X has {values.shape[1]} features; expected {len(expected)}")
        frame = pd.DataFrame(
            values,
            columns=expected if expected is not None else [f"x{i}" for i in range(values.shape[1])],
        )
    if not len(frame):
        raise ValueError(f"Found array with 0 sample(s) (shape={frame.shape})")
    if not frame.shape[1]:
        raise ValueError(f"0 feature(s) (shape={frame.shape}) while a minimum of 1 is required.")
    temporal = [
        column
        for column in frame.columns
        if pd.api.types.is_datetime64_any_dtype(frame[column].dtype)
        or pd.api.types.is_timedelta64_dtype(frame[column].dtype)
    ]
    if temporal:
        frame = frame.copy(deep=False)
        for column in temporal:
            series = frame[column]
            if pd.api.types.is_datetime64_any_dtype(series.dtype):
                if getattr(series.dt, "tz", None) is not None:
                    series = series.dt.tz_convert("UTC").dt.tz_localize(None)
                frame[column] = (
                    (series - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
                ).astype(float)
            else:
                frame[column] = (series / pd.Timedelta(seconds=1)).astype(float)
    numeric = frame.select_dtypes(include=["number"])
    if numeric.size and np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("X must not contain infinity")
    for column in frame.columns:
        dtype = frame[column].dtype
        # String columns hold no numbers; other columns (object, categorical) may.
        if pd.api.types.is_numeric_dtype(dtype) or isinstance(dtype, pd.StringDtype):
            continue
        for value in _distinct(frame[column].dropna()):
            if isinstance(value, (int, float, np.number)) and not np.isfinite(value):
                raise ValueError("X must not contain infinity")
    return frame


def prepare_frame(X) -> pd.DataFrame:
    """Return ``X`` as conditions see it: a validated DataFrame.

    Datetime columns become seconds since the epoch (UTC) and timedelta columns
    become seconds, so other tools can evaluate conditions on identical values.
    """
    return _frame(X)


def _column_index(feature: str | int, columns: list) -> int:
    if isinstance(feature, int):
        if feature < 0 or feature >= len(columns):
            raise ValueError(f"unknown feature position: {feature}")
        return feature
    if feature not in columns:
        raise ValueError(f"unknown feature name: {feature}")
    return columns.index(feature)


def _bin_thresholds(series: pd.Series, n_bins: int, discretizer: KBinsDiscretizer | None):
    if discretizer is not None:
        fitted = clone(discretizer)
        values = series.dropna().to_numpy(dtype=float).reshape(-1, 1)
        fitted.fit(values)
        return np.asarray(fitted.bin_edges_[0][1:-1])
    if pd.api.types.is_integer_dtype(series.dtype):
        distinct = np.sort(series.dropna().unique())
        positions = np.ceil(np.arange(1, n_bins) * len(distinct) / n_bins).astype(int) - 1
        return distinct[np.unique(positions)]
    values = series.dropna().to_numpy(dtype=float)
    return np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1]))


def generate_conditions(
    X,
    *,
    discretizer: KBinsDiscretizer | None = None,
    per_feature_discretizers: Mapping[str | int, KBinsDiscretizer] | None = None,
    n_bins: int | None = 10,
    numeric_candidates: str = "bins",
    max_categories: int | None = 100,
) -> list[Condition]:
    """Generate editable, raw-data predicates from a training dataset.

    A supplied KBinsDiscretizer is cloned and fitted per eligible numeric feature.
    Low-cardinality numeric columns use equality, as in the original encoder.
    Features above ``max_categories`` are skipped rather than truncated.
    """
    if n_bins is not None and (
        isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 2
    ):
        raise ValueError("n_bins must be at least 2 or None")
    if max_categories is not None and (
        isinstance(max_categories, bool)
        or not isinstance(max_categories, int)
        or max_categories < 1
    ):
        raise ValueError("max_categories must be positive or None")
    if numeric_candidates not in {"bins", "thresholds"}:
        raise ValueError("numeric_candidates must be 'bins' or 'thresholds'")
    if discretizer is not None and not isinstance(discretizer, KBinsDiscretizer):
        raise TypeError("discretizer must be a KBinsDiscretizer")
    frame = _frame(X)
    columns = list(frame.columns)
    overrides: dict[int, KBinsDiscretizer] = {}
    for feature, override in (per_feature_discretizers or {}).items():
        if not isinstance(override, KBinsDiscretizer):
            raise TypeError("per-feature discretizers must be KBinsDiscretizer instances")
        index = _column_index(feature, columns)
        if index in overrides:
            raise ValueError(f"duplicate discretizer override for column {index}")
        overrides[index] = override

    conditions: list[Condition] = []
    for index, name in enumerate(columns):
        series = frame.iloc[:, index]
        feature = name if isinstance(name, str) else index
        nonmissing = series.dropna()
        numeric = pd.api.types.is_numeric_dtype(series.dtype)
        if numeric:
            n_distinct = nonmissing.nunique()
        else:
            # Every unhashable value of the column is among its distinct values.
            distinct = _distinct(nonmissing)
            for value in distinct:
                try:
                    hash(value)
                except TypeError as error:
                    raise TypeError("argument must be a string or number") from error
            n_distinct = len(distinct)
        chosen = overrides.get(index, discretizer)
        eligible_bins = chosen.n_bins if chosen is not None else n_bins
        if eligible_bins is not None and not isinstance(eligible_bins, int):
            raise ValueError("KBinsDiscretizer.n_bins must be an integer for one feature")
        if numeric and eligible_bins is not None and n_distinct > eligible_bins:
            thresholds = _bin_thresholds(series, eligible_bins, chosen)
            if numeric_candidates == "bins":
                bounds = (None, *thresholds, None)
                conditions.extend(
                    Condition(feature, "interval", lower=lower, upper=upper)
                    for lower, upper in zip(bounds[:-1], bounds[1:], strict=True)
                    if lower is not None or upper is not None
                )
            else:
                for threshold in thresholds:
                    conditions.extend(
                        (Condition(feature, "le", threshold), Condition(feature, "gt", threshold))
                    )
        elif max_categories is None or n_distinct <= max_categories:
            # First-seen order and Python scalar types, as iterating the column gives.
            if numeric:
                distinct = _distinct(nonmissing)
            conditions.extend(Condition(feature, "eq", value) for value in distinct)
        if series.isna().any():
            conditions.append(Condition(feature, "is_missing"))
    return conditions


class Coverage:
    """Boolean condition coverage over rows, independent of physical storage."""

    def __init__(self, storage: np.ndarray, n_rows: int, n_conditions: int, representation: str):
        self.storage = storage
        self.shape = (n_rows, n_conditions)
        self.representation = representation

    def __len__(self) -> int:
        return self.shape[0]

    def _packed_intersection(self, indices: tuple[int, ...]) -> np.ndarray:
        n_bytes = self.storage.shape[1]
        if not indices:
            return np.packbits(np.ones(self.shape[0], dtype=bool), bitorder="little")
        if any(index < 0 or index >= self.shape[1] for index in indices):
            raise IndexError("condition index out of range")
        return np.bitwise_and.reduce(self.storage[list(indices)], axis=0).reshape(n_bytes)

    def intersect(self, indices: Iterable[int] = ()) -> np.ndarray:
        """Return a new Boolean row mask covered by all selected conditions."""
        indices = tuple(indices)
        if self.representation == "dense":
            if any(index < 0 or index >= self.shape[1] for index in indices):
                raise IndexError("condition index out of range")
            if not indices:
                return np.ones(self.shape[0], dtype=bool)
            mask = self.storage[indices[0]].copy()
            for index in indices[1:]:
                mask &= self.storage[index]
            return mask
        return np.unpackbits(self._packed_intersection(indices), bitorder="little")[
            : self.shape[0]
        ].astype(bool)

    def condition_mask(self, index: int) -> np.ndarray:
        """Return the row mask of one condition (a read-only view for dense storage)."""
        if self.representation == "dense":
            if index < 0 or index >= self.shape[1]:
                raise IndexError("condition index out of range")
            return self.storage[index]
        return self.intersect((index,))

    def count(self, indices: Iterable[int] = (), rows: np.ndarray | None = None) -> int:
        """Count matching rows, optionally restricted by a Boolean row mask."""
        indices = tuple(indices)
        if rows is not None:
            rows = np.asarray(rows)
            if rows.shape != (self.shape[0],) or rows.dtype != bool:
                raise ValueError("rows must be a Boolean mask with one entry per row")
        if self.representation == "dense":
            if len(indices) == 1:
                covered = self.condition_mask(indices[0])
            else:
                covered = self.intersect(indices)
            return int(np.count_nonzero(covered if rows is None else covered & rows))
        covered = self._packed_intersection(indices)
        if rows is not None:
            covered &= np.packbits(rows, bitorder="little")
        return int(np.bitwise_count(covered).sum())

    def condition_counts(self, rows: np.ndarray | None = None) -> np.ndarray:
        """Count the matching rows of every condition, optionally within a row mask."""
        if rows is not None:
            rows = np.asarray(rows)
            if rows.shape != (self.shape[0],) or rows.dtype != bool:
                raise ValueError("rows must be a Boolean mask with one entry per row")
        if self.representation == "dense":
            selected = self.storage if rows is None else self.storage[:, np.flatnonzero(rows)]
            return np.count_nonzero(selected, axis=1)
        words = self.storage
        if rows is not None:
            words = words & np.packbits(rows, bitorder="little")
        return np.bitwise_count(words).sum(axis=1, dtype=np.int64)


class Encoder:
    """Evaluate a frozen list of raw-data conditions using a chosen backend.

    Lists retain order and duplicates. Sets receive a deterministic order.
    Changes to the caller's collection after fitting do not affect this encoder.
    """

    def __init__(self, conditions: Iterable[Condition], *, representation: str = "dense"):
        if representation not in {"dense", "packed"}:
            raise ValueError("representation must be 'dense' or 'packed'")
        self.conditions = conditions
        self.representation = representation

    def fit_transform(self, X) -> Coverage:
        frame = _frame(X)
        self.feature_names_in_ = list(frame.columns)
        supplied = tuple(self.conditions)
        if not all(isinstance(condition, Condition) for condition in supplied):
            raise TypeError("conditions must contain Condition instances")
        if isinstance(self.conditions, (set, frozenset)):
            conditions = sorted(
                supplied,
                key=lambda condition: (
                    _column_index(condition.feature, self.feature_names_in_),
                    condition.operator,
                    repr(condition.value),
                    repr(condition.lower),
                    repr(condition.upper),
                ),
            )
        else:
            conditions = list(supplied)
        for condition in conditions:
            column = frame.iloc[:, _column_index(condition.feature, self.feature_names_in_)]
            if condition.operator in {"le", "gt", "interval"} and not pd.api.types.is_numeric_dtype(
                column.dtype
            ):
                raise ValueError(f"numeric predicate requires numeric feature: {condition.feature}")
        self.conditions_ = tuple(conditions)
        return self._evaluate(frame)

    def transform(self, X) -> Coverage:
        if not hasattr(self, "conditions_"):
            raise RuntimeError("Encoder must be fitted before transform")
        return self._evaluate(_frame(X, expected=self.feature_names_in_))

    def _evaluate(self, frame: pd.DataFrame) -> Coverage:
        n_rows = len(frame)
        n_conditions = len(self.conditions_)
        if self.representation == "dense":
            # Condition-major, so each condition's row mask is contiguous.
            storage = np.empty((n_conditions, n_rows), dtype=bool)
            for index, condition in enumerate(self.conditions_):
                storage[index] = self._evaluate_condition(frame, condition)
            storage.flags.writeable = False
        else:
            storage = np.empty((n_conditions, (n_rows + 7) // 8), dtype=np.uint8)
            for index, condition in enumerate(self.conditions_):
                storage[index] = np.packbits(
                    self._evaluate_condition(frame, condition), bitorder="little"
                )
        return Coverage(storage, n_rows, n_conditions, self.representation)

    def _evaluate_condition(self, frame: pd.DataFrame, condition: Condition) -> np.ndarray:
        series = frame.iloc[:, _column_index(condition.feature, self.feature_names_in_)]
        if condition.operator == "is_missing":
            return series.isna().to_numpy(dtype=bool)
        if condition.operator == "eq":
            matched = equals(series, condition.value)
        elif condition.operator == "ne":
            matched = ~equals(series, condition.value) & series.notna()
        elif condition.operator == "in":
            matched = series.isin(list(condition.value)) & series.notna()
        elif condition.operator == "le":
            matched = series.le(condition.value)
        elif condition.operator == "gt":
            matched = series.gt(condition.value)
        else:
            matched = pd.Series(True, index=series.index)
            if condition.lower is not None:
                matched &= series.gt(condition.lower)
            if condition.upper is not None:
                matched &= series.le(condition.upper)
        return matched.fillna(False).to_numpy(dtype=bool)
