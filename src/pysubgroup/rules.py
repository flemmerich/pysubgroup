"""Immutable, publicly accessible rule-model objects."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from numbers import Integral, Real
from typing import Any


def _pretty(self: Any, printer: Any, cycle: bool) -> None:
    """IPython display hook: show the readable ``str`` form, also inside displayed lists."""
    printer.text(str(self))


@dataclass(frozen=True, slots=True)
class Condition:
    """A predicate on a raw feature, independent of an encoding backend.

    ``interval`` denotes (lower, upper], with ``None`` for an open end. ``ne``
    matches present values other than ``value``; ``in`` matches any value of the
    non-empty tuple ``value``. Missing values only match ``is_missing``. Integer
    features refer to column positions; strings refer to column names.
    """

    feature: str | int
    operator: str = "eq"
    value: Any = None
    lower: Real | None = None
    upper: Real | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.feature, (str, int)) or isinstance(self.feature, bool):
            raise TypeError("feature must be a column name or integer position")
        if self.operator not in {"eq", "ne", "in", "le", "gt", "interval", "is_missing"}:
            raise ValueError(f"unsupported condition operator: {self.operator}")
        if self.operator in {"eq", "ne"}:
            if self.value is None:
                raise ValueError(
                    f"{self.operator} requires a value; use is_missing for missing values"
                )
            try:
                hash(self.value)
            except TypeError as error:
                raise TypeError(f"{self.operator} value must be hashable") from error
        elif self.operator == "in":
            if not isinstance(self.value, tuple) or not self.value:
                raise ValueError("in requires a non-empty tuple of values")
            if any(value is None for value in self.value):
                raise ValueError("in values must not be None; use is_missing for missing values")
            try:
                hash(self.value)
            except TypeError as error:
                raise TypeError("in values must be hashable") from error
        elif self.operator in {"le", "gt"}:
            if (
                isinstance(self.value, bool)
                or not isinstance(self.value, Real)
                or not isfinite(self.value)
            ):
                raise ValueError("numeric bounds must be finite real numbers")
        elif self.operator == "interval":
            if self.lower is None and self.upper is None:
                raise ValueError("interval needs at least one bound")
            for bound in (self.lower, self.upper):
                if bound is not None and (
                    isinstance(bound, bool) or not isinstance(bound, Real) or not isfinite(bound)
                ):
                    raise ValueError("numeric bounds must be finite real numbers")
            if self.lower is not None and self.upper is not None and self.lower >= self.upper:
                raise ValueError("interval lower bound must be smaller than upper bound")
        if self.operator != "interval" and (self.lower is not None or self.upper is not None):
            raise ValueError("bounds are only valid for interval conditions")
        if self.operator == "is_missing" and self.value is not None:
            raise ValueError("is_missing does not take a value")

    def __str__(self) -> str:
        feature = self.feature
        if self.operator == "eq":
            return f"{feature}={self.value}"
        if self.operator == "ne":
            return f"{feature}≠{self.value}"
        if self.operator == "in":
            return f"{feature} ∈ {{{', '.join(map(str, self.value))}}}"
        if self.operator == "is_missing":
            return f"{feature} is missing"
        if self.operator == "le":
            return f"{feature} ≤ {_number(self.value)}"
        if self.operator == "gt":
            return f"{feature} > {_number(self.value)}"
        lower = "-∞" if self.lower is None else _number(self.lower)
        upper = "∞" if self.upper is None else _number(self.upper)
        return f"{feature} ∈ ({lower}, {upper}]"

    def __repr__(self) -> str:
        value = tuple(map(_plain, self.value)) if self.operator == "in" else _plain(self.value)
        arguments = [repr(self.feature), repr(self.operator)]
        if value is not None:
            arguments.append(repr(value))
        arguments += [
            f"{name}={_plain(bound)!r}"
            for name, bound in (("lower", self.lower), ("upper", self.upper))
            if bound is not None
        ]
        return f"Condition({', '.join(arguments)})"

    _repr_pretty_ = _pretty

    def to_dict(self) -> dict[str, Any]:
        result = {"feature": self.feature, "operator": self.operator}
        if self.operator in {"eq", "ne", "le", "gt"}:
            result["value"] = _json_value(self.value)
        if self.operator == "in":
            result["value"] = [_json_value(value) for value in self.value]
        if self.operator == "interval":
            result["lower"] = _json_value(self.lower)
            result["upper"] = _json_value(self.upper)
            result["lower_inclusive"] = False
            result["upper_inclusive"] = True
        return result


@dataclass(frozen=True, slots=True)
class Rule:
    """A conjunction with an optional class consequent.

    An omitted consequent inherits the enclosing ruleset's positive class.
    """

    conditions: tuple[Condition, ...] = ()
    consequent: Any = None

    def __len__(self) -> int:
        return len(self.conditions)

    def __str__(self) -> str:
        antecedent = (
            "[" + " ∧ ".join(map(str, self.conditions)) + "]" if self.conditions else "[True]"
        )
        return f"{antecedent} -> {self.consequent}" if self.consequent is not None else antecedent

    def __repr__(self) -> str:
        return _call_repr("Rule", self.conditions, consequent=self.consequent)

    _repr_pretty_ = _pretty

    def resolved_consequent(self, positive_class: Any = None) -> Any:
        """Return the explicit consequent or the supplied positive class."""
        return self.consequent if self.consequent is not None else positive_class

    def to_dict(self, positive_class: Any = None) -> dict[str, Any]:
        return {
            "conditions": [condition.to_dict() for condition in self.conditions],
            "consequent": _json_value(self.resolved_consequent(positive_class)),
        }


def _number(value: Real) -> str:
    """A numeric bound as text, e.g. ``30`` rather than ``30.0``."""
    return str(value) if isinstance(value, Integral) else f"{value:.15g}"


def _plain(value: Any) -> Any:
    """A NumPy scalar as the matching Python value, so reprs show ``31``, not ``np.int64(31)``."""
    return value.item() if hasattr(value, "item") and hasattr(value, "dtype") else value


def _call_repr(name: str, items: tuple, **keywords: Any) -> str:
    """A constructor call that omits the empty tuple and ``None`` keywords."""
    arguments = [repr(items)] if items else []
    arguments += [
        f"{key}={_plain(value)!r}" for key, value in keywords.items() if value is not None
    ]
    return f"{name}({', '.join(arguments)})"


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


@dataclass(frozen=True, slots=True)
class Ruleset:
    """A collection of rules with optional positive and default classes.

    Explicit rule consequents can refer to different classes. Binary learners
    produce rules with inherited positive consequents and a negative default.
    """

    rules: tuple[Rule, ...] = ()
    positive_class: Any = None
    default_class: Any = None

    def __post_init__(self) -> None:
        if self.positive_class is not None:
            object.__setattr__(
                self,
                "rules",
                tuple(
                    replace(rule, consequent=self.positive_class)
                    if rule.consequent is None
                    else rule
                    for rule in self.rules
                ),
            )

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self):
        return iter(self.rules)

    def __str__(self) -> str:
        return " ∨ ".join(map(str, self.rules)) if self.rules else "[]"

    def __repr__(self) -> str:
        return _call_repr(
            "Ruleset", self.rules,
            positive_class=self.positive_class, default_class=self.default_class,
        )

    _repr_pretty_ = _pretty

    def get_selected_features(self) -> list[str | int]:
        return list(
            dict.fromkeys(
                condition.feature for rule in self.rules for condition in rule.conditions
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rules": [rule.to_dict(self.positive_class) for rule in self.rules],
            "semantics": (
                "dnf_default"
                if all(rule.resolved_consequent(self.positive_class) == self.positive_class
                       for rule in self.rules)
                else "ordered_rules_with_default"
            ),
            "positive_class": _json_value(self.positive_class),
            "default_class": _json_value(self.default_class),
        }
