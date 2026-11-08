import json

import numpy as np
import pytest

from pysubgroup import Condition, Rule, Ruleset
from pysubgroup.rules import Condition as PublicCondition
from pysubgroup.rules import Rule as PublicRule
from pysubgroup.rules import Ruleset as PublicRuleset


def test_rule_model_is_structured_and_readable():
    condition = Condition("weather", "eq", "sunny")
    rule = Rule((condition,))
    ruleset = Ruleset((rule,))

    assert str(condition) == "weather=sunny"
    assert str(rule) == "[weather=sunny]"
    assert len(rule) == len(rule.conditions) == 1
    assert not hasattr(rule, "conds")
    assert str(ruleset) == "[weather=sunny]"
    assert ruleset.get_selected_features() == ["weather"]
    assert ruleset.to_dict()["semantics"] == "dnf_default"


def test_string_forms_use_mathematical_notation():
    age = Condition("age", "interval", lower=30.0, upper=40)
    sex = Condition("sex", "eq", "f")

    assert str(Condition("age", "le", np.float64(30.0))) == "age ≤ 30"
    assert str(Condition("age", "gt", 2.5)) == "age > 2.5"
    assert str(Condition("age", "interval", upper=30)) == "age ∈ (-∞, 30]"
    assert str(Condition("age", "interval", lower=1234567.5)) == "age ∈ (1234567.5, ∞]"
    assert str(Condition("sex", "ne", "f")) == "sex≠f"
    assert str(Condition("age", "is_missing")) == "age is missing"
    assert str(Rule((age, sex), consequent="yes")) == "[age ∈ (30, 40] ∧ sex=f] -> yes"
    assert str(Ruleset((Rule((age,)), Rule((sex,))))) == "[age ∈ (30, 40]] ∨ [sex=f]"


def test_reprs_are_compact_constructor_calls():
    age = Condition("age", "interval", lower=np.int64(31), upper=np.float64(46.5))
    color = Condition("color", "in", (np.str_("red"), "green"))
    rule = Rule((age, color), consequent=np.int64(1))
    ruleset = Ruleset((Rule((age,)),), positive_class=1, default_class=0)

    assert repr(age) == "Condition('age', 'interval', lower=31, upper=46.5)"
    assert repr(Condition("age", "interval", upper=31)) == "Condition('age', 'interval', upper=31)"
    assert repr(Condition("x", "eq", "a")) == "Condition('x', 'eq', 'a')"
    assert repr(Condition("x", "is_missing")) == "Condition('x', 'is_missing')"
    assert repr(color) == "Condition('color', 'in', ('red', 'green'))"
    assert repr([Condition("x", "le", 2)]) == "[Condition('x', 'le', 2)]"
    assert repr(Rule()) == "Rule()"
    assert repr(Ruleset()) == "Ruleset()"
    assert repr(rule) == (
        "Rule((Condition('age', 'interval', lower=31, upper=46.5), "
        "Condition('color', 'in', ('red', 'green'))), consequent=1)"
    )
    assert repr(ruleset) == (
        "Ruleset((Rule((Condition('age', 'interval', lower=31, upper=46.5),), consequent=1),), "
        "positive_class=1, default_class=0)"
    )
    for item in (age, color, rule, ruleset):
        assert eval(repr(item)) == item


def test_ipython_displays_the_readable_form():
    pretty = pytest.importorskip("IPython.lib.pretty").pretty
    age = Condition("age", "interval", upper=np.int64(31))
    sex = Condition("sex", "eq", "f")

    assert pretty([age, sex]) == "[age ∈ (-∞, 31], sex=f]"
    assert pretty(Rule((age, sex))) == "[age ∈ (-∞, 31] ∧ sex=f]"
    assert pretty(Ruleset((Rule((sex,)),))) == "[sex=f]"


def test_empty_rules_have_explicit_representations():
    assert str(Rule()) == "[True]"
    assert str(Ruleset()) == "[]"


def test_public_rules_module_and_multiclass_consequents():
    assert (PublicCondition, PublicRule, PublicRuleset) == (Condition, Rule, Ruleset)
    first = Rule((Condition("weather", "eq", "sunny"),), consequent="play")
    second = Rule((Condition("weather", "eq", "rainy"),), consequent="stay")
    inherited = Rule((Condition("weather", "eq", "cloudy"),))
    ruleset = Ruleset((first, second, inherited), positive_class="maybe", default_class="other")

    assert first.consequent == "play"
    assert str(first) == "[weather=sunny] -> play"
    assert first != Rule(first.conditions, consequent="stay")
    assert inherited.consequent is None
    assert inherited.resolved_consequent(ruleset.positive_class) == "maybe"
    assert ruleset.rules[2].consequent == "maybe"
    exported = ruleset.to_dict()
    assert [rule["consequent"] for rule in exported["rules"]] == ["play", "stay", "maybe"]
    assert exported["default_class"] == "other"
    assert exported["semantics"] == "ordered_rules_with_default"
    json.dumps(exported)


def test_numpy_scalar_consequents_are_serializable():
    rule = Rule(consequent=np.int64(2))
    assert rule.to_dict()["consequent"] == 2
