"""Memory and strategy must use the arguments they are given.

Both modules claimed behaviour they did not have -- the same shape as the
adapter returning zeros and the optimizer never evaluating, but harder to see,
because a plausible list of trajectories looks exactly like a correct one.

    query_similar(spec, k)   "Find similar past designs"
                             returned past_trajectories[-k:] and never read
                             `spec`. On a long run the most RECENT designs are
                             the least likely to be relevant.
    get_context_for_task()   promised PDK info, similar designs and topology
                             hints; returned an f-string with none of them.
    prune(max_tokens)        "Keep context within budget"; body was `pass`.
    build_prompt(...)        "ensuring token limits are respected"; counted
                             nothing.
    analyze_failure(results, spec, history)
                             ignored `results` and `spec` and branched on
                             len(history): an episode making steady progress was
                             told to abandon its topology at step five, and an
                             episode repeating one identical failing call was
                             told to carry on until then.
    suggest_topology(spec)   promised a RANKED list; returned the same three
                             names for every analog task.

Each test below names the argument that used to be ignored.
"""
from __future__ import annotations

import pytest

from asic_ai.agent.loop import EvalTask
from asic_ai.agent.memory import ContextBuilder, DesignMemory, spec_similarity
from asic_ai.agent.strategy import RetryStrategy, StrategyManager, TopologySelector

OTA = {"dc_gain": {"min": 60, "unit": "dB"}, "ugb": {"min": 30, "unit": "MHz"}}
OTA_NEAR = {"dc_gain": {"min": 62, "unit": "dB"}, "ugb": {"min": 35, "unit": "MHz"}}
OTA_FAR = {"dc_gain": {"min": 60, "unit": "dB"}, "ugb": {"min": 30000, "unit": "MHz"}}
LDO = {"dropout": {"max": 200, "unit": "mV"}, "psrr": {"min": 40, "unit": "dB"}}


class _Traj:
    def __init__(self, task_id, specs, success=True, final_score=0.5):
        self.task_id = task_id
        self.metadata = {"specs": specs}
        self.success = success
        self.final_score = final_score

    def __str__(self):  # cheap token cost for prune()
        return f"{self.task_id}:{self.metadata}"


# ----------------------------------------------------------- similarity ----

def test_near_identical_specs_are_similar():
    assert spec_similarity(OTA, OTA_NEAR) > 0.9


def test_unrelated_specs_score_zero_not_somewhat_similar():
    """Sharing no spec name means 'not comparable', which is 0.0, not 0.3."""
    assert spec_similarity(OTA, LDO) == 0.0


def test_distance_is_judged_on_a_log_scale():
    """1000x on bandwidth is a different problem; a linear metric hides that."""
    near = spec_similarity(OTA, OTA_NEAR)
    far = spec_similarity(OTA, OTA_FAR)
    assert far < near
    assert far < 0.8


def test_similarity_is_symmetric():
    assert spec_similarity(OTA, LDO) == spec_similarity(LDO, OTA)
    assert spec_similarity(OTA, OTA_NEAR) == pytest.approx(
        spec_similarity(OTA_NEAR, OTA))


def test_empty_spec_is_not_similar_to_anything():
    assert spec_similarity({}, OTA) == 0.0
    assert spec_similarity(OTA, {}) == 0.0


# --------------------------------------------------------------- memory ----

@pytest.fixture
def memory():
    m = DesignMemory()
    m.store_trajectory(_Traj("ldo_a", LDO, True, 0.9))
    m.store_trajectory(_Traj("ota_a", OTA_NEAR, True, 0.8))
    m.store_trajectory(_Traj("ldo_b", LDO, False, 0.1))
    return m


def test_query_similar_reads_the_spec_rather_than_returning_the_last_k(memory):
    """The original defect, stated directly."""
    most_recent = [t.task_id for t in memory.past_trajectories[-2:]]
    assert most_recent == ["ota_a", "ldo_b"]

    got = [t.task_id for t in memory.query_similar(OTA, k=2)]
    assert got == ["ota_a"], (
        f"returned {got}; the LDO is not a comparable design and padding it in "
        "would put a misleading example in the model's context")


def test_an_unrelated_design_is_excluded_not_padded_in(memory):
    assert memory.query_similar(LDO, k=5)[0].task_id in ("ldo_a", "ldo_b")
    assert all(t.task_id.startswith("ldo") for t in memory.query_similar(LDO, k=5))


def test_results_are_ordered_most_similar_first():
    m = DesignMemory()
    m.store_trajectory(_Traj("far", OTA_FAR))
    m.store_trajectory(_Traj("near", OTA_NEAR))
    assert [t.task_id for t in m.query_similar(OTA, k=2)] == ["near", "far"]


def test_context_says_so_when_nothing_is_comparable():
    """An empty section leaves the model to infer why it is empty."""
    text = DesignMemory().get_context_for_task(EvalTask(spec=OTA))
    assert "None in memory" in text


def test_context_names_the_comparable_designs_and_their_outcome(memory):
    text = memory.get_context_for_task(EvalTask(spec=OTA))
    assert "ota_a" in text
    assert "met its specs" in text
    assert "ldo_a" not in text, "an unrelated design must not appear as comparable"


def test_prune_enforces_the_budget(memory):
    """The body used to be `pass`, so the budget was never enforced."""
    before = memory.token_cost()
    assert before > 0
    removed = memory.prune(max_tokens=before // 3)
    assert removed > 0
    assert memory.token_cost() <= before


def test_prune_drops_failures_before_successes(memory):
    memory.prune(max_tokens=1)
    kept = {t.task_id for t in memory.past_trajectories}
    assert "ldo_b" not in kept, "a failed attempt is the least useful to carry"


def test_prune_to_zero_empties_the_memory(memory):
    assert memory.prune(max_tokens=0) == 3
    assert memory.past_trajectories == []


# -------------------------------------------------------- context budget ---

def test_build_prompt_enforces_a_token_budget(memory):
    builder = ContextBuilder(memory)
    task = EvalTask(spec=OTA)
    system = "SYSTEM " * 200

    unbounded = builder.build_prompt(system, "pdk notes", task, {"netlist": "x"},
                                     ["h"] * 50)
    bounded = builder.build_prompt(system, "pdk notes", task, {"netlist": "x"},
                                   ["h"] * 50, max_tokens=len(system) // 4 + 20)
    assert len(bounded) < len(unbounded)
    assert bounded.startswith("SYSTEM"), "the system prompt is dropped last"


def test_build_prompt_without_a_budget_keeps_everything(memory):
    text = ContextBuilder(memory).build_prompt(
        "SYS", "pdk", EvalTask(spec=OTA), {"netlist": "x"}, ["one", "two"])
    assert "SYS" in text and "Current state" in text and "History" in text


# ------------------------------------------------------------- strategy ----

def test_a_repeated_identical_attempt_changes_topology():
    """The case the length rule missed: four steps of one failing call read as
    'the same approach' until step five."""
    history = [{"tool_calls": ["sim.ac"], "score": -0.5}] * 3
    assert StrategyManager().analyze_failure({}, {}, history) is RetryStrategy.CHANGE_TOPOLOGY


def test_varied_and_improving_keeps_the_same_approach():
    """The old rule abandoned the topology at step five regardless of progress."""
    history = [{"tool_calls": ["sim.dc"], "score": -0.5},
               {"tool_calls": ["sim.ac"], "score": -0.3},
               {"tool_calls": ["spec.check"], "score": -0.1}]
    assert StrategyManager().analyze_failure({}, {}, history) is RetryStrategy.SAME_APPROACH


def test_a_shortfall_too_large_to_size_away_changes_topology():
    """`current_results` and `spec` used to be ignored entirely."""
    strategy = StrategyManager().analyze_failure(
        {"dc_gain": 0.6}, {"dc_gain": {"min": 60}}, [])
    assert strategy is RetryStrategy.CHANGE_TOPOLOGY


def test_a_small_shortfall_adjusts_parameters_instead():
    strategy = StrategyManager().analyze_failure(
        {"dc_gain": 55.0}, {"dc_gain": {"min": 60}}, [])
    assert strategy is RetryStrategy.MODIFY_PARAMS


def test_no_progress_over_several_moves_escalates():
    history = [{"tool_calls": [f"t{i}"], "score": s}
               for i, s in enumerate((-0.2, -0.3, -0.25, -0.4))]
    assert StrategyManager().analyze_failure({}, {}, history) is RetryStrategy.ESCALATE


# ------------------------------------------------------------- topology ----

def test_topology_ranking_depends_on_the_specification():
    """It used to return the same three names for every analog task."""
    selector = TopologySelector()
    high_gain = selector.suggest_topology({"gain": {"min": 75}})
    fast_low_gain = selector.suggest_topology(
        {"gain": {"min": 25}, "ugb": {"min": 100e6}})
    assert high_gain != fast_low_gain
    assert high_gain[0] == "two_stage_ota", "only it reaches 75 dB"


def test_a_topology_that_cannot_reach_the_gain_ranks_last():
    ranked = TopologySelector().suggest_topology({"gain": {"min": 75}})
    assert ranked[-1] == "common_source"


def test_a_digital_specification_gets_digital_topologies():
    ranked = TopologySelector().suggest_topology({"clock_frequency": {"min": 100}})
    assert ranked == ["pipelined", "fsm", "combinational"]


def test_an_undiscriminating_spec_returns_the_conventional_order():
    """When nothing in the spec ranks them, say so by not pretending to rank."""
    ranked = TopologySelector().suggest_topology({})
    assert ranked == ["telescopic_cascode", "folded_cascode",
                      "two_stage_ota", "common_source"]
