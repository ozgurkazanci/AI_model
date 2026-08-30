"""The optimizer must actually search.

BayesianOptimizer.optimize previously returned each parameter's LOWER BOUND with
best_score 0.0 and converged=True, having called eval_fn exactly zero times. A
search that reports success without searching is worse than one that fails,
because the caller sizes a circuit from the answer. CmaesOptimizer did the same
with converged=False.

The design document calls this layer the most-skipped and most-consequential
part of an analog flow -- "the model picks the topology, the optimizer sizes the
devices; projects that skip this hit a wall on analog". It was skipped.

test_eval_fn_is_actually_called is the regression that matters: it fails against
the old implementation for the simplest possible reason.
"""
from __future__ import annotations

import math
import tempfile

import pytest

from asic_ai.optimizer import get_optimizer
from asic_ai.optimizer.base import OptimizationObjective, OptParam
from asic_ai.optimizer.bayesian import BayesianOptimizer
from asic_ai.optimizer.circuit import (
    build_eval_fn, format_spice, optimize_sizing, substitute,
    template_placeholders,
)
from asic_ai.optimizer.cmaes import CmaesOptimizer
from asic_ai.optimizer.scipy_opt import ScipyOptimizer

try:
    from asic_ai.adapters.ngspice_shared import find_ngspice_dll
    HAS_NGSPICE = find_ngspice_dll() is not None
except Exception:
    HAS_NGSPICE = False

skip_no_ngspice = pytest.mark.skipif(not HAS_NGSPICE, reason="ngspice DLL not found")


def _quadratic(target: float):
    """Maximised at `target`. Records every call so the search can be audited."""
    calls: list[dict] = []

    def f(params):
        calls.append(dict(params))
        return -((params["x"] - target) ** 2)
    return f, calls


def _obj(lo=0.0, hi=10.0, **kw):
    return OptimizationObjective(
        parameters=[OptParam(name="x", min_val=lo, max_val=hi, **kw)],
        objectives=[])


# ------------------------------------------------------------ it searches ---

def test_eval_fn_is_actually_called():
    """The original defect, stated as plainly as possible."""
    f, calls = _quadratic(3.0)
    ScipyOptimizer(seed=0).optimize(_obj(), f, max_iterations=25)
    assert len(calls) > 0, "the optimizer never evaluated anything"


def test_finds_a_known_optimum():
    f, _ = _quadratic(3.0)
    result = ScipyOptimizer(seed=0).optimize(_obj(), f, max_iterations=120)
    assert result.best_params["x"] == pytest.approx(3.0, abs=0.02)
    assert result.best_score > -1e-3


def test_does_not_just_return_the_lower_bound():
    """The exact old answer: min_val for every parameter."""
    f, _ = _quadratic(7.5)
    result = ScipyOptimizer(seed=0).optimize(_obj(lo=1.0, hi=10.0), f,
                                             max_iterations=120)
    assert result.best_params["x"] != pytest.approx(1.0, abs=0.05)
    assert result.best_params["x"] == pytest.approx(7.5, abs=0.05)


def test_score_is_a_real_evaluation_not_zero():
    f, _ = _quadratic(3.0)
    result = ScipyOptimizer(seed=0).optimize(_obj(), f, max_iterations=60)
    assert result.best_score != 0.0
    assert result.best_score == pytest.approx(
        -((result.best_params["x"] - 3.0) ** 2), abs=1e-9)


@pytest.mark.parametrize("budget", [10, 40, 150])
def test_evaluation_budget_is_a_hard_cap(budget):
    """Every evaluation is a SPICE run; an overrun is an overnight job."""
    f, calls = _quadratic(3.0)
    result = ScipyOptimizer(seed=0).optimize(_obj(), f, max_iterations=budget)
    assert len(calls) <= budget
    assert result.iterations == len(calls)


def test_log_scale_finds_an_optimum_spanning_decades():
    """Device widths span decades; a linear search wastes its samples up top."""
    calls: list[dict] = []

    def f(params):
        calls.append(dict(params))
        return -(math.log10(params["W"] / 42e-6) ** 2)

    obj = OptimizationObjective(
        parameters=[OptParam(name="W", min_val=1e-7, max_val=1e-3, log_scale=True)],
        objectives=[])
    result = ScipyOptimizer(seed=1).optimize(obj, f, max_iterations=150)
    assert result.best_params["W"] == pytest.approx(42e-6, rel=0.02)


def test_fixed_parameters_are_held():
    seen: list[dict] = []

    def f(params):
        seen.append(dict(params))
        return -((params["x"] - 3.0) ** 2)

    obj = OptimizationObjective(
        parameters=[OptParam(name="x", min_val=0, max_val=10),
                    OptParam(name="L", min_val=1e-7, max_val=1e-5,
                             initial=1.8e-7, fixed=True)],
        objectives=[])
    ScipyOptimizer(seed=0).optimize(obj, f, max_iterations=40)
    assert seen and all(s["L"] == pytest.approx(1.8e-7) for s in seen)


def test_a_failing_evaluation_does_not_stop_the_search():
    """Non-convergence is a property of a bad design, not a crash."""
    calls: list[dict] = []

    def f(params):
        calls.append(dict(params))
        if params["x"] < 5.0:
            raise RuntimeError("simulation did not converge")
        return -((params["x"] - 7.0) ** 2)

    result = ScipyOptimizer(seed=0).optimize(_obj(), f, max_iterations=120)
    assert len(calls) > 10
    assert result.best_params["x"] == pytest.approx(7.0, abs=0.1)


def test_history_records_every_evaluation():
    f, calls = _quadratic(3.0)
    result = ScipyOptimizer(seed=0).optimize(_obj(), f, max_iterations=30)
    assert len(result.history) == len(calls)
    assert all("params" in h for h in result.history)


def test_suggest_next_stays_inside_the_bounds():
    opt = ScipyOptimizer(seed=0)
    for _ in range(20):
        assert 0.0 <= opt.suggest_next(_obj(), [])["x"] <= 10.0


def test_repeated_suggestions_differ():
    """A caller asking for a batch must not get the same point N times."""
    opt = ScipyOptimizer(seed=0)
    history = [{"params": {"x": 3.0}, "score": -0.1}]
    picks = [opt.suggest_next(_obj(), history)["x"] for _ in range(10)]
    assert len(set(picks)) > 5, f"suggestions barely vary: {picks}"


def test_suggest_next_clusters_around_the_best_point():
    """It is a local perturbation, so it must actually be local."""
    opt = ScipyOptimizer(seed=0)
    history = [{"params": {"x": 3.0}, "score": -0.1},
               {"params": {"x": 9.0}, "score": -36.0},
               {"params": {"x": 8.0}, "score": -25.0},
               {"params": {"x": 0.5}, "score": -6.2},
               {"params": {"x": 6.0}, "score": -9.0},
               {"params": {"x": 1.0}, "score": -4.0}]
    picks = [opt.suggest_next(_obj(), history)["x"] for _ in range(40)]
    nearer_to_best = sum(abs(v - 3.0) < abs(v - 9.0) for v in picks)
    assert nearer_to_best >= 34, f"{nearer_to_best}/40 near the best point"


# ----------------------------------------------------------- honest deps ----

def test_cmaes_refuses_rather_than_fabricating():
    with pytest.raises(ImportError, match="cma"):
        CmaesOptimizer().optimize(_obj(), lambda p: 0.0, 10)


def test_bayesian_says_it_is_not_bayesian_and_still_searches():
    b = BayesianOptimizer(seed=0)
    assert b.is_bayesian is False, "BoTorch is not installed on this machine"
    f, calls = _quadratic(3.0)
    result = b.optimize(_obj(), f, max_iterations=80)
    assert len(calls) > 0
    assert result.best_params["x"] == pytest.approx(3.0, abs=0.05)


def test_default_factory_returns_something_that_works():
    f, calls = _quadratic(3.0)
    get_optimizer().optimize(_obj(), f, max_iterations=40)
    assert len(calls) > 0


# -------------------------------------------------------------- template ----

@pytest.mark.parametrize("value,expected", [
    (4.2e-5, "42u"), (1.8e-7, "180n"), (1e3, "1k"), (2.2e4, "22k"),
    (1e-12, "1p"), (0.0, "0"),
])
def test_spice_formatting(value, expected):
    assert format_spice(value) == expected


def test_non_finite_size_is_refused():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            format_spice(bad)


def test_placeholders_are_found_in_order():
    assert template_placeholders("M1 d g s b n W={W1} L={L1} m=1\n* {W1}") == ["W1", "L1"]


def test_substitution_fills_every_placeholder():
    out = substitute("R2 out 0 {R2}\nM1 d g 0 0 n W={W1}", {"R2": 1e4, "W1": 4.2e-5})
    assert "10k" in out and "42u" in out and "{" not in out


def test_missing_placeholder_raises_rather_than_reaching_the_simulator():
    """An unsubstituted {W1} is a parse error the optimizer would read as a
    bad design and search away from."""
    with pytest.raises(KeyError, match="W1"):
        substitute("M1 d g s b n W={W1}", {})


def test_variable_not_in_the_template_is_refused():
    with pytest.raises(ValueError, match="does not appear"):
        optimize_sizing("R2 out 0 {R2}\n", [{"name": "W1", "min": 1, "max": 2}],
                        {"vout": {"target": 0.9, "unit": "V"}}, adapter=None)


def test_template_placeholder_with_no_variable_is_refused():
    with pytest.raises(ValueError, match="no variable"):
        optimize_sizing("R2 out 0 {R2}\nR3 out 0 {R3}\n",
                        [{"name": "R2", "min": 1, "max": 2}],
                        {"vout": {"target": 0.9, "unit": "V"}}, adapter=None)


# ------------------------------------------------------------ end to end ----

@skip_no_ngspice
def test_sizes_a_real_circuit_against_real_simulation():
    """The whole point: search device values against measured specs.

    A 1.8 V source and a 10k top resistor. The optimizer must discover that R2
    = 10k puts the output at 0.9 V, from simulation alone.
    """
    from asic_ai.adapters import get_adapter

    template = ("* divider\n"
                "V1 vdd 0 DC 1.8\n"
                "R1 vdd out 10k\n"
                "R2 out 0 {R2}\n"
                ".op\n.end\n")
    specs = {"vout": {"target": 0.9, "unit": "V"},
             "idd": {"max": 120, "unit": "uA"}}
    adapter = get_adapter("ngspice_shared", binary_path="",
                          work_dir=tempfile.mkdtemp())

    result = optimize_sizing(template, [{"name": "R2", "min": 1e3, "max": 1e6}],
                             specs, adapter, max_iterations=40, analyses=("dc",))

    assert result.iterations > 0, "no simulation was run"
    r2 = result.best_params["R2"]
    assert r2 == pytest.approx(10e3, rel=0.05), f"analytic optimum is 10k, got {r2:.4g}"

    # Check the answer against the physics rather than against the optimizer.
    vout = 1.8 * r2 / (10e3 + r2)
    idd_ua = 1.8 / (10e3 + r2) * 1e6
    assert vout == pytest.approx(0.9, abs=0.01)
    assert idd_ua < 120


@skip_no_ngspice
def test_eval_fn_records_measurements_per_candidate():
    from asic_ai.adapters import get_adapter

    seen: list[dict] = []
    template = "* d\nV1 vdd 0 DC 1.8\nR1 vdd out 10k\nR2 out 0 {R2}\n.op\n.end\n"
    adapter = get_adapter("ngspice_shared", binary_path="",
                          work_dir=tempfile.mkdtemp())
    f = build_eval_fn(template, adapter, {"vout": {"target": 0.9, "unit": "V"}},
                      analyses=("dc",), on_step=seen.append)

    score = f({"R2": 10e3})
    assert seen and "measured" in seen[0]
    assert seen[0]["measured"]["vout"] == pytest.approx(0.9, abs=1e-6)
    assert score > 0.0
