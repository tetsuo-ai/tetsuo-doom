"""Scenario registry mapping names to ViZDoom .cfg files."""

import os

import vizdoom as vzd

SCENARIOS: dict[str, dict] = {
    "basic": {
        "cfg": "basic.cfg",
    },
    "deadly_corridor": {
        "cfg": "deadly_corridor.cfg",
    },
    "defend_the_center": {
        "cfg": "defend_the_center.cfg",
    },
    "defend_the_line": {
        "cfg": "defend_the_line.cfg",
    },
    "health_gathering": {
        "cfg": "health_gathering.cfg",
    },
    "health_gathering_supreme": {
        "cfg": "health_gathering_supreme.cfg",
    },
    "my_way_home": {
        "cfg": "my_way_home.cfg",
    },
    "predict_position": {
        "cfg": "predict_position.cfg",
    },
    "deathmatch": {
        "cfg": "deathmatch.cfg",
    },
}


def get_scenario_config_path(scenario_name: str) -> str:
    """Get the full path to a scenario's .cfg file.

    Raises:
        ValueError: If the scenario name is not recognized.
    """
    scenario_name = scenario_name.lower()
    if scenario_name not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario {scenario_name!r}. "
            f"Available: {sorted(SCENARIOS.keys())}"
        )
    cfg_file = SCENARIOS[scenario_name]["cfg"]
    return os.path.join(vzd.scenarios_path, cfg_file)


def list_scenarios() -> list[str]:
    """Return sorted list of available scenario names."""
    return sorted(SCENARIOS.keys())
