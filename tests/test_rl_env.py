"""Tests for RL environment and tokenizer extension."""
import json
import pytest

from asic_ai.training.rl_env import CircuitDesignEnv, DesignState, StepResult


SAMPLE_TASK = {
    "id": "test_ota",
    "description": "Test OTA",
    "category": "analog",
    "difficulty": "easy",
    "pdk": "sky130",
    "supply": 1.8,
    "specs": {"gain": {"min": 60, "unit": "dB"}},
}


def simple_reward(specs, results):
    return 0.5


class TestCircuitDesignEnv:
    def test_reset(self):
        env = CircuitDesignEnv(adapter=None, reward_fn=simple_reward)
        obs = env.reset(SAMPLE_TASK)
        assert "Test OTA" in obs
        assert env.state is not None
        assert env.state.step == 0

    def test_step_pdk_query(self):
        env = CircuitDesignEnv(adapter=None, reward_fn=simple_reward)
        env.reset(SAMPLE_TASK)
        result = env.step({"name": "pdk.list_devices", "arguments": {}})
        assert isinstance(result, StepResult)
        assert not result.done
        assert env.state.step == 1

    def test_step_device_query(self):
        env = CircuitDesignEnv(adapter=None, reward_fn=simple_reward)
        env.reset(SAMPLE_TASK)
        result = env.step({
            "name": "pdk.device_query",
            "arguments": {"model": "nfet_01v8", "W": 10e-6, "L": 180e-9, "VGS": 0.6, "VDS": 0.9},
        })
        data = json.loads(result.observation)
        assert "gm" in data
        assert "vth" in data

    def test_step_lint_check(self):
        env = CircuitDesignEnv(adapter=None, reward_fn=simple_reward)
        env.reset(SAMPLE_TASK)
        result = env.step({"name": "lint.check", "arguments": {"netlist": ".subckt test\n.ends"}})
        data = json.loads(result.observation)
        assert data["status"] == "ok"

    def test_step_spec_check(self):
        env = CircuitDesignEnv(adapter=None, reward_fn=simple_reward)
        env.reset(SAMPLE_TASK)
        result = env.step({
            "name": "spec.check",
            "arguments": {"results": {"gain": 65}, "specs": {"gain": {"min": 60}}},
        })
        data = json.loads(result.observation)
        assert "score" in data

    def test_max_steps_terminates(self):
        env = CircuitDesignEnv(adapter=None, reward_fn=simple_reward, max_steps=3)
        env.reset(SAMPLE_TASK)
        for i in range(3):
            result = env.step({"name": "pdk.list_devices", "arguments": {}})
        assert result.done

    def test_unknown_tool(self):
        env = CircuitDesignEnv(adapter=None, reward_fn=simple_reward)
        env.reset(SAMPLE_TASK)
        result = env.step({"name": "nonexistent.tool", "arguments": {}})
        assert "Unknown tool" in result.observation

    def test_episode_summary(self):
        env = CircuitDesignEnv(adapter=None, reward_fn=simple_reward)
        env.reset(SAMPLE_TASK)
        env.step({"name": "pdk.list_devices", "arguments": {}})
        summary = env.get_episode_summary()
        assert summary["task_id"] == "test_ota"
        assert summary["steps"] == 1

    def test_step_penalty_accumulates(self):
        env = CircuitDesignEnv(adapter=None, reward_fn=simple_reward, step_penalty=0.01)
        env.reset(SAMPLE_TASK)
        env.step({"name": "pdk.list_devices", "arguments": {}})
        env.step({"name": "pdk.list_devices", "arguments": {}})
        # Total reward should be negative (only step penalties, no sim bonus)
        assert env.state.total_reward < 0


class TestTokenizerExtensionScript:
    def test_get_new_tokens(self):
        from asic_ai.tokenizer.extend import get_new_tokens, TokenExtensionConfig
        tokens = get_new_tokens(TokenExtensionConfig())
        assert len(tokens) > 50
        assert "uA" in tokens
        assert "pF" in tokens
        assert ".subckt" in tokens

    def test_default_test_strings(self):
        from asic_ai.tokenizer.extend import DEFAULT_TEST_STRINGS
        assert len(DEFAULT_TEST_STRINGS) > 5
        assert any("gm" in s for s in DEFAULT_TEST_STRINGS)


class TestExplicitResultsHardening:
    """A dict-wrapped spec.check value must degrade, not detonate.

    The 824g eval passed {"dc_gain": {"value": 60, "unit": "dB"}} as results;
    RewardFunction then raised TypeError comparing dict to int, which scored
    as "no usable reward function". Reverting the numeric filter makes this
    test fail with that same TypeError-driven zero.
    """

    def _env(self):
        import tempfile
        from asic_ai.adapters.mock import MockSimulatorAdapter
        from asic_ai.adapters.base import AdapterConfig
        from asic_ai.reward.reward import RewardFunction
        from asic_ai.training.rl_env import CircuitDesignEnv

        task = {"id": "t", "specs": {"dc_gain": {"min": 40, "unit": "dB"}}}
        env = CircuitDesignEnv(
            MockSimulatorAdapter(AdapterConfig(binary_path="",
                                               work_dir=tempfile.mkdtemp())),
            RewardFunction.from_eval_task(task), max_steps=4)
        env.reset(task)
        return env

    def test_dict_wrapped_value_reports_unmeasurable(self):
        import json
        env = self._env()
        r = env.step({"name": "spec.check", "arguments": {
            "results": {"dc_gain": {"value": 60, "unit": "dB"}},
            "specs": {"dc_gain": {"min": 40, "unit": "dB"}}}})
        out = json.loads(r.observation)
        assert out["specs_measured"] == 0
        assert "number" in out["unmeasurable"]["dc_gain"]

    def test_bare_number_still_scores(self):
        import json
        env = self._env()
        r = env.step({"name": "spec.check", "arguments": {
            "results": {"dc_gain": 60.0},
            "specs": {"dc_gain": {"min": 40, "unit": "dB"}}}})
        out = json.loads(r.observation)
        assert out["specs_measured"] == 1
        assert out["score"] > 0
