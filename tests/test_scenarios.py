"""Unit tests for scenario registry."""

import os

import pytest

from doom_mcp.scenarios import get_scenario_config_path, list_scenarios


def test_list_scenarios():
    scenarios = list_scenarios()
    assert "basic" in scenarios
    assert "deadly_corridor" in scenarios
    assert len(scenarios) >= 9


def test_get_scenario_config_path():
    path = get_scenario_config_path("basic")
    assert path.endswith("basic.cfg")
    assert os.path.exists(path)


def test_get_scenario_config_path_case_insensitive():
    path = get_scenario_config_path("Basic")
    assert path.endswith("basic.cfg")


def test_unknown_scenario_raises():
    with pytest.raises(ValueError, match="Unknown scenario"):
        get_scenario_config_path("nonexistent")
