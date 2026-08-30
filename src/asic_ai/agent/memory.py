"""Design memory: what the agent has tried before, and what to put in context.

The design document's Layer 4 -- the model does not memorise PDK data or past
designs, it queries them.

Every method here previously claimed something it did not do:

    query_similar(spec, k)      "Find similar past designs"
                                returned past_trajectories[-k:], never reading
                                `spec` at all. Similarity claimed, recency
                                delivered -- and on a long run the most recent
                                designs are the least likely to be relevant.
    get_context_for_task()      "including PDK info, similar designs, and
                                topology hints"
                                returned an f-string containing none of the three.
    prune(max_tokens)           "Keep context within budget"
                                body was `pass`, so the budget was never enforced.
    build_prompt(...)           "ensuring token limits are respected"
                                counted no tokens.

Same shape as the rest of this repo's placeholders: a plausible return value
instead of a failure. A caller cannot tell "no similar design exists" from
"similarity was never computed".
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from asic_ai.agent.loop import EvalTask, Trajectory

__all__ = ["DesignMemory", "ContextBuilder", "spec_similarity"]

# Roughly 4 characters per token. Only used when no tokenizer is supplied; a
# caller that needs an exact budget should pass one.
_CHARS_PER_TOKEN = 4


def _bounds(spec_def: Any) -> Dict[str, float]:
    if not isinstance(spec_def, dict):
        return {}
    return {k: float(v) for k, v in spec_def.items()
            if k in ("min", "max", "target") and isinstance(v, (int, float))}


def spec_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """How alike two spec blocks are, in [0, 1].

    Two parts, because both matter and neither alone is enough:

      - WHICH specs are named. An OTA task and an LDO task overlap barely at all,
        and no amount of numeric closeness makes them comparable. Jaccard over
        the spec names.
      - HOW CLOSE the shared bounds are, on a LOG scale. Circuit specs span
        orders of magnitude -- 60 dB against 62 dB is nearly the same problem,
        1 uA against 1 mA is not -- so a linear distance would be dominated by
        whichever spec happens to carry the largest number.

    Returns 0.0 when the two share no spec names, which is the honest answer:
    not "somewhat similar", but "not comparable".
    """
    names_a, names_b = set(a or {}), set(b or {})
    if not names_a or not names_b:
        return 0.0
    shared = names_a & names_b
    if not shared:
        return 0.0

    name_score = len(shared) / len(names_a | names_b)

    closeness: List[float] = []
    for name in shared:
        ba, bb = _bounds(a[name]), _bounds(b[name])
        for key in set(ba) & set(bb):
            va, vb = ba[key], bb[key]
            if va == vb:
                closeness.append(1.0)
                continue
            if va <= 0 or vb <= 0:
                # dB and temperature coefficients are legitimately negative or
                # zero; fall back to a linear ratio against the larger value.
                span = max(abs(va), abs(vb)) or 1.0
                closeness.append(max(0.0, 1.0 - abs(va - vb) / span))
                continue
            decades = abs(math.log10(va / vb))
            closeness.append(max(0.0, 1.0 - decades))  # 1 decade apart -> 0

    if not closeness:
        return name_score  # same specs named, no comparable numbers
    value_score = sum(closeness) / len(closeness)
    return 0.5 * name_score + 0.5 * value_score


def _spec_of(trajectory: Any) -> Dict[str, Any]:
    """The spec block a trajectory was run against, wherever it is stored."""
    meta = getattr(trajectory, "metadata", None)
    if isinstance(meta, dict):
        for key in ("specs", "spec", "task_specs"):
            if isinstance(meta.get(key), dict):
                return meta[key]
    for key in ("specs", "spec", "task_specs"):
        value = getattr(trajectory, key, None)
        if isinstance(value, dict):
            return value
    return {}


class DesignMemory:
    """Stores completed design trajectories and retrieves comparable ones."""

    def __init__(self) -> None:
        self.past_trajectories: List[Trajectory] = []

    def store_trajectory(self, trajectory: Trajectory) -> None:
        """Save a completed design."""
        self.past_trajectories.append(trajectory)

    def query_similar(self, spec: Dict[str, Any], k: int = 5,
                      min_similarity: float = 0.1) -> List[Trajectory]:
        """The k most similar past designs, most similar first.

        Trajectories scoring below `min_similarity` are excluded rather than
        padded in. Returning an unrelated design as "similar" is worse than
        returning nothing: it puts a misleading example in the model's context,
        and the model has no way to tell it was a filler.
        """
        scored = [(spec_similarity(spec, _spec_of(t)), t)
                  for t in self.past_trajectories]
        scored = [(s, t) for s, t in scored if s >= min_similarity]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [t for _, t in scored[:k]]

    def rank_similar(self, spec: Dict[str, Any],
                     k: int = 5) -> List[Tuple[float, Trajectory]]:
        """query_similar with the scores, for callers that want to show why."""
        scored = [(spec_similarity(spec, _spec_of(t)), t)
                  for t in self.past_trajectories]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:k]

    def get_context_for_task(self, task: EvalTask, k: int = 3,
                             pdk_context: str = "") -> str:
        """Context for a task: PDK notes, then comparable past designs.

        Says plainly when there is no comparable design, so the model is not
        left to infer that from an empty section.
        """
        parts: List[str] = []
        if pdk_context:
            parts.append("## PDK\n" + pdk_context.strip())

        similar = self.rank_similar(task.spec, k=k)
        similar = [(s, t) for s, t in similar if s >= 0.1]
        if similar:
            lines = ["## Comparable past designs"]
            for score, traj in similar:
                outcome = ("met its specs" if getattr(traj, "success", False)
                           else "did NOT meet its specs")
                lines.append(
                    f"- {getattr(traj, 'task_id', '?')} "
                    f"(similarity {score:.2f}, {outcome}, "
                    f"score {getattr(traj, 'final_score', 0.0):.3f})")
            parts.append("\n".join(lines))
        else:
            parts.append("## Comparable past designs\nNone in memory. "
                         "Size this design from the specification alone.")
        return "\n\n".join(parts)

    def prune(self, max_tokens: int, tokenizer: Any = None) -> int:
        """Drop the least useful trajectories until the budget is met.

        Returns how many were removed. Keeps SUCCESSFUL designs preferentially:
        a failed attempt is the least useful thing to carry into a new task's
        context, and the previous no-op meant the budget was never enforced at
        all.
        """
        if max_tokens <= 0:
            removed = len(self.past_trajectories)
            self.past_trajectories = []
            return removed

        def cost(t: Any) -> int:
            text = getattr(t, "model_dump_json", None)
            body = text() if callable(text) else str(t)
            if tokenizer is not None:
                return len(tokenizer(body)["input_ids"])
            return max(1, len(body) // _CHARS_PER_TOKEN)

        # Least useful first: failures before successes, then lowest score.
        order = sorted(
            range(len(self.past_trajectories)),
            key=lambda i: (bool(getattr(self.past_trajectories[i], "success", False)),
                           float(getattr(self.past_trajectories[i], "final_score", 0.0))))

        total = sum(cost(t) for t in self.past_trajectories)
        drop: set = set()
        for i in order:
            if total <= max_tokens:
                break
            total -= cost(self.past_trajectories[i])
            drop.add(i)

        if drop:
            self.past_trajectories = [t for i, t in enumerate(self.past_trajectories)
                                      if i not in drop]
        return len(drop)

    def token_cost(self, tokenizer: Any = None) -> int:
        """Current size of the memory, in tokens."""
        total = 0
        for t in self.past_trajectories:
            text = getattr(t, "model_dump_json", None)
            body = text() if callable(text) else str(t)
            total += (len(tokenizer(body)["input_ids"]) if tokenizer is not None
                      else max(1, len(body) // _CHARS_PER_TOKEN))
        return total


class ContextBuilder:
    """Assembles the prompt for the LLM."""

    def __init__(self, memory: DesignMemory):
        self.memory = memory

    def build_prompt(self, system_prompt: str, pdk_context: str, task: EvalTask,
                     current_state: Dict[str, Any], history: Sequence[Any],
                     max_tokens: Optional[int] = None,
                     tokenizer: Any = None) -> str:
        """Assemble the context, dropping the least important part first.

        Priority, highest first: system prompt, specification, current state,
        comparable designs, history. When a budget is given it is ENFORCED by
        dropping from the bottom -- the previous version documented a limit and
        counted nothing.

        `system_prompt` must be the output of
        asic_ai.data.format.build_system_message(). Anything else is
        training/serving prompt drift, which silently stops tool calling.
        """
        def count(text: str) -> int:
            if tokenizer is not None:
                return len(tokenizer(text)["input_ids"])
            return max(1, len(text) // _CHARS_PER_TOKEN)

        spec_block = f"## Task\n{task.spec}"
        state_block = f"## Current state\n{current_state}"
        memory_block = self.memory.get_context_for_task(task, pdk_context=pdk_context)
        history_block = ("## History\n" + "\n".join(str(h) for h in history)
                         if history else "")

        # Bottom of this list is dropped first.
        sections = [system_prompt, spec_block, state_block,
                    memory_block, history_block]
        sections = [s for s in sections if s]

        if max_tokens is None:
            return "\n\n".join(sections)

        while len(sections) > 1 and count("\n\n".join(sections)) > max_tokens:
            sections.pop()
        return "\n\n".join(sections)
