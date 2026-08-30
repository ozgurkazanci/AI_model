"""What to do after a failed attempt.

The design document calls this the most valuable thing the agent learns:

    "The model learns three things at once: to call tools in the right format,
     to interpret simulator output, and WHAT TO DO AFTER A FAILURE -- the last
     being the most valuable."

`analyze_failure(current_results, spec, history)` previously ignored both
`current_results` and `spec` and decided purely on `len(history)`: three steps
meant "change the parameters", five meant "change the topology. So an episode
making steady progress was told to abandon its topology at step five, and an
episode repeating one identical failing call was told to keep going until then.
The decision had nothing to do with what had actually happened.

`suggest_topology` likewise read only `spec["type"]` and returned a fixed list
regardless of the specification it was handed, while its docstring promised a
RANKED list.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["RetryStrategy", "StrategyManager", "TopologySelector"]


class RetryStrategy(Enum):
    SAME_APPROACH = "SAME_APPROACH"
    MODIFY_PARAMS = "MODIFY_PARAMS"
    CHANGE_TOPOLOGY = "CHANGE_TOPOLOGY"
    ESCALATE = "ESCALATE"


def _score_of(entry: Any) -> Optional[float]:
    if isinstance(entry, dict):
        for key in ("score", "final_score", "reward"):
            v = entry.get(key)
            if isinstance(v, (int, float)):
                return float(v)
    v = getattr(entry, "score", None)
    return float(v) if isinstance(v, (int, float)) else None


def _signature(entry: Any) -> Optional[str]:
    """What was attempted, coarsely, so repetition can be recognised."""
    if isinstance(entry, dict):
        for key in ("tool_calls", "action", "tool", "name"):
            v = entry.get(key)
            if v:
                return str(v)
        if "error" in entry:
            return f"error:{entry['error']}"
    return None


def _shortfall(results: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, float]:
    """How far each measured spec is from its bound, in decades.

    Decades because a spec two orders of magnitude out is a topology problem
    and one 20 pct out is a sizing problem, and a linear distance cannot tell
    those apart across specs whose units differ by a million.
    """
    out: Dict[str, float] = {}
    for name, bound in (spec or {}).items():
        if not isinstance(bound, dict):
            continue
        value = (results or {}).get(name)
        if not isinstance(value, (int, float)):
            continue
        lo, hi = bound.get("min"), bound.get("max")
        target = bound.get("target")
        miss = 0.0
        if isinstance(lo, (int, float)) and value < lo:
            miss = (math.log10(lo / value) if value > 0 and lo > 0
                    else abs(lo - value) / (abs(lo) or 1.0))
        elif isinstance(hi, (int, float)) and value > hi:
            miss = (math.log10(value / hi) if value > 0 and hi > 0
                    else abs(value - hi) / (abs(hi) or 1.0))
        elif isinstance(target, (int, float)) and target != 0:
            rel = abs(value - target) / abs(target)
            miss = rel if rel > 0.02 else 0.0
        if miss > 0.0:
            out[name] = float(miss)
    return out


class StrategyManager:
    """Chooses the next move from what actually happened, not from step count."""

    # A spec missed by more than this many decades will not be closed by
    # resizing: 10x on gain or bandwidth is a different circuit.
    TOPOLOGY_SHORTFALL_DECADES = 1.0

    def __init__(self) -> None:
        self.error_history: List[str] = []

    def analyze_failure(self, current_results: Dict[str, Any],
                        spec: Dict[str, Any],
                        history: Sequence[Any],
                        max_repeats: int = 3) -> RetryStrategy:
        """Decide from the evidence: what was tried, and how far off it is.

        Order of checks, most decisive first:

        1. REPETITION. The same attempt made `max_repeats` times in a row has
           already shown it does not work; doing it again cannot inform
           anything. This is the case the old length-based rule missed
           entirely -- it would let an agent repeat one failing call for four
           steps and call that "the same approach".
        2. A SHORTFALL TOO LARGE TO SIZE AWAY. More than a decade off is a
           topology problem; no amount of device sizing closes 10x on gain.
        3. PROGRESS. If the score is improving, keep going. The old rule
           abandoned the topology at step five regardless.
        4. Otherwise adjust the parameters.
        """
        recent = [_signature(h) for h in list(history)[-max_repeats:]]
        if (len(recent) >= max_repeats and all(r is not None for r in recent)
                and len(set(recent)) == 1):
            self.error_history.append(str(recent[0]))
            return RetryStrategy.CHANGE_TOPOLOGY

        shortfall = _shortfall(current_results or {}, spec or {})
        if shortfall and max(shortfall.values()) > self.TOPOLOGY_SHORTFALL_DECADES:
            return RetryStrategy.CHANGE_TOPOLOGY

        scores = [s for s in (_score_of(h) for h in history) if s is not None]
        if len(scores) >= 2:
            if scores[-1] > scores[-2]:
                return RetryStrategy.SAME_APPROACH
            if len(scores) >= 4 and scores[-1] <= min(scores[-4:-1]):
                # Three moves without improving on the best of them.
                return RetryStrategy.ESCALATE

        return RetryStrategy.MODIFY_PARAMS


class TopologySelector:
    """Ranks candidate topologies against the specification it is given."""

    # (name, gain_db_capability, speed_capability, swing_capability)
    _ANALOG: List[Tuple[str, float, float, float]] = [
        ("telescopic_cascode", 70.0, 1.0, 0.3),
        ("folded_cascode", 65.0, 0.7, 0.6),
        ("two_stage_ota", 80.0, 0.4, 0.9),
        ("common_source", 30.0, 0.8, 0.7),
    ]

    def suggest_topology(self, spec: Dict[str, Any]) -> List[str]:
        """A list ranked by how well each topology fits THIS specification.

        The previous version returned the same three names for every analog
        task, so the "ranked" in its docstring was decoration. Ranking here is
        deliberately simple and readable rather than clever -- the numerical
        optimizer does the sizing, this only has to order the candidates.
        """
        spec = spec or {}
        task_type = str(spec.get("type", "")).lower()

        if task_type == "digital" or any(
                k in spec for k in ("clock_frequency", "data_width", "throughput")):
            return ["pipelined", "fsm", "combinational"]

        def want(*names: str) -> Optional[float]:
            for n in names:
                d = spec.get(n)
                if isinstance(d, dict):
                    for key in ("min", "target", "max"):
                        if isinstance(d.get(key), (int, float)):
                            return float(d[key])
                elif isinstance(d, (int, float)):
                    return float(d)
            return None

        gain = want("gain", "dc_gain", "gain_db")
        ugb = want("ugb", "gbw", "bw", "bandwidth")
        swing = want("output_swing", "vout")

        ranked: List[Tuple[float, str]] = []
        for name, gain_cap, speed_cap, swing_cap in self._ANALOG:
            score = 0.0
            if gain is not None:
                # Enough gain scores well; not enough is disqualifying.
                score += 2.0 if gain_cap >= gain else -3.0
            if ugb is not None:
                score += speed_cap * (2.0 if ugb >= 50e6 else 1.0)
            if swing is not None:
                score += swing_cap
            if gain is None and ugb is None and swing is None:
                score = 0.0
            ranked.append((score, name))

        if all(s == 0.0 for s, _ in ranked):
            # Nothing in the spec discriminates. Say so by returning the
            # conventional order rather than pretending it was ranked.
            return [name for name, *_ in self._ANALOG]

        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [name for _, name in ranked]
