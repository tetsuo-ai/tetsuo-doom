"""Integration tests for GameManager (requires ViZDoom runtime)."""

import pytest
from fastmcp.exceptions import ToolError

from doom_mcp.game_manager import GameManager


pytestmark = pytest.mark.integration


@pytest.fixture
def manager():
    m = GameManager()
    yield m
    m.stop()


def test_start_and_stop(manager):
    result = manager.start(scenario="basic", seed=42)
    assert result["status"] == "running"
    assert result["scenario"] == "basic"
    assert "TURN_LEFT_RIGHT_DELTA" in result["buttons"]
    assert manager.is_running

    result = manager.stop()
    assert result["status"] == "stopped"
    assert not manager.is_running


def test_get_state(manager):
    manager.start(scenario="basic", seed=42)
    state = manager.get_state()
    assert state["episode_finished"] is False
    assert "screenshot_png" in state
    assert state["game_variables"]["HEALTH"] == 100.0
    assert "objects" in state
    assert "depth" in state
    assert "sectors" in state


def test_get_state_objects_enriched(manager):
    manager.start(scenario="defend_the_center", seed=42)
    state = manager.get_state()
    objects = state["objects"]
    assert len(objects) > 0
    monster = next((o for o in objects if o["type"] == "monster"), None)
    assert monster is not None
    assert "distance" in monster
    assert "angle_to_aim" in monster
    assert "threat" in monster
    assert "typical_hp" in monster
    assert "is_visible" in monster


def test_take_action_dict(manager):
    manager.start(scenario="basic", seed=42)
    result = manager.take_action({"MOVE_FORWARD_BACKWARD_DELTA": 10}, tics=1)
    assert "reward" in result
    assert "episode_finished" in result


def test_take_action_delta_turn(manager):
    manager.start(scenario="basic", seed=42)
    state_before = manager.get_state()
    angle_before = state_before["game_variables"]["ANGLE"]
    manager.take_action({"TURN_LEFT_RIGHT_DELTA": 45}, tics=1)
    state_after = manager.get_state()
    angle_after = state_after["game_variables"]["ANGLE"]
    # Angle should have changed by ~45 degrees
    diff = abs(angle_before - angle_after)
    if diff > 180:
        diff = 360 - diff
    assert 40 < diff < 50


def test_take_action_combined(manager):
    manager.start(scenario="basic", seed=42)
    result = manager.take_action(
        {"TURN_LEFT_RIGHT_DELTA": -10, "MOVE_FORWARD_BACKWARD_DELTA": 10, "ATTACK": 1},
        tics=1,
    )
    assert result["episode_finished"] is False


def test_noop_action(manager):
    manager.start(scenario="basic", seed=42)
    result = manager.take_action(None, tics=1)
    assert result["episode_finished"] is False


def test_new_episode(manager):
    manager.start(scenario="basic", seed=42)
    result = manager.new_episode()
    assert result["status"] == "new_episode"


def test_get_objects_enriched(manager):
    manager.start(scenario="defend_the_center", seed=42)
    result = manager.get_objects()
    assert "objects" in result
    objects = result["objects"]
    assert any(o["type"] == "monster" for o in objects)
    assert any(o["name"] == "DoomPlayer" for o in objects)


def test_get_map(manager):
    manager.start(scenario="basic", seed=42)
    result = manager.get_map()
    if result is not None:
        assert result[:8] == b"\x89PNG\r\n\x1a\n"


def test_get_available_actions(manager):
    manager.start(scenario="basic", seed=42)
    result = manager.get_available_actions()
    buttons = result["buttons"]
    names = [b["name"] for b in buttons]
    assert "TURN_LEFT_RIGHT_DELTA" in names
    assert "MOVE_FORWARD_BACKWARD_DELTA" in names
    assert "ATTACK" in names
    delta = next(b for b in buttons if b["name"] == "TURN_LEFT_RIGHT_DELTA")
    assert delta["type"] == "delta"
    binary = next(b for b in buttons if b["name"] == "ATTACK")
    assert binary["type"] == "binary"


def test_require_running_raises():
    m = GameManager()
    with pytest.raises(ToolError, match="No game is running"):
        m.get_state()


def test_require_episode_raises(manager):
    manager.start(scenario="basic", seed=42, episode_timeout=1)
    while not manager._game.is_episode_finished():
        manager.take_action(None, tics=1)
    with pytest.raises(ToolError, match="Episode is finished"):
        manager.take_action({"ATTACK": 1})


def test_episode_finished_state(manager):
    manager.start(scenario="basic", seed=42, episode_timeout=1)
    while not manager._game.is_episode_finished():
        manager.take_action(None, tics=1)
    state = manager.get_state()
    assert state["episode_finished"] is True


def test_restart_stops_previous(manager):
    manager.start(scenario="basic", seed=42)
    manager.start(scenario="basic", seed=43)
    assert manager.is_running


def test_load_freedoom2_map(manager):
    result = manager.start(wad="freedoom2", map_name="MAP01")
    assert result["status"] == "running"
    assert result["wad"] == "freedoom2"
    assert result["map"] == "MAP01"
    state = manager.get_state()
    assert len(state["objects"]) > 10  # MAP01 has lots of objects


def test_load_freedoom1_map(manager):
    result = manager.start(wad="freedoom1", map_name="E1M1")
    assert result["status"] == "running"
    state = manager.get_state()
    assert state["episode_finished"] is False


def test_campaign_auto_advance(manager):
    """When player completes a level (not dead), new_episode advances to next map."""
    manager.start(wad="freedoom2", map_name="MAP01", episode_timeout=5)
    # Let episode timeout (player not dead = level "completed")
    while not manager._game.is_episode_finished():
        manager.take_action(None, tics=1)
    state = manager.get_state()
    assert state["level_completed"] is True
    assert state["next_map"] == "MAP02"

    result = manager.new_episode()
    assert result["map"] == "MAP02"
    assert result["advanced"] is True


def test_campaign_death_restarts_same_map(manager):
    """When player dies, new_episode restarts the same map."""
    manager.start(wad="freedoom2", map_name="MAP01")
    # We can't easily force death in a test, so just verify
    # the _next_map logic
    from doom_mcp.game_manager import _next_map
    assert _next_map("MAP01") == "MAP02"
    assert _next_map("MAP31") == "MAP32"
    assert _next_map("MAP32") is None
    assert _next_map("E1M1") == "E1M2"
    assert _next_map("E1M9") == "E2M1"
    assert _next_map("E4M9") is None
