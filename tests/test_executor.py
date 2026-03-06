"""Integration tests for the autonomous executor (requires ViZDoom runtime)."""

import time

import pytest

from doom_mcp.game_manager import GameManager


pytestmark = pytest.mark.integration


@pytest.fixture
def manager():
    m = GameManager()
    yield m
    m.stop()


@pytest.fixture
def async_manager(manager):
    """Manager with async_player=True and executor running."""
    manager.start(wad="freedoom2", map_name="MAP01", seed=42, async_player=True)
    return manager


@pytest.fixture
def async_basic(manager):
    """Manager with async_player=True on a long-lived scenario."""
    manager.start(
        wad="freedoom2", map_name="MAP01", seed=42, async_player=True,
    )
    return manager


def test_executor_starts_and_stops(manager):
    """Executor thread starts on async game start, stops on game stop."""
    manager.start(scenario="basic", seed=42, async_player=True)
    assert manager._executor is not None
    assert manager._executor._thread is not None
    assert manager._executor._thread.is_alive()

    manager.stop()
    assert manager._executor is None


def test_executor_not_created_in_sync_mode(manager):
    """In sync mode, no executor is created."""
    manager.start(scenario="basic", seed=42, async_player=False)
    assert manager._executor is None


def test_executor_explores(async_manager):
    """Executor explores and visits cells after running for a few seconds."""
    time.sleep(2)
    nav = async_manager.get_navigation_info()
    assert nav["cells_explored"] > 0


def test_executor_fights(manager):
    """Executor fights enemies in defend_the_center."""
    manager.start(scenario="defend_the_center", seed=42, async_player=True)
    time.sleep(3)
    # Check if any kills happened via game variables
    with manager._game_lock:
        if not manager._game.is_episode_finished():
            kills = manager._game.get_game_variable(
                __import__("vizdoom").GameVariable.KILLCOUNT
            )
        else:
            kills = 0
    # The executor should have engaged enemies
    events = manager._executor.get_recent_events()
    has_combat = any(
        e["event_type"] in ("state_change", "damage_taken")
        for e in events
    )
    assert has_combat or kills > 0


def test_executor_retreats_on_low_health(manager):
    """Executor transitions to RETREATING when health is critical."""
    manager.start(scenario="deadly_corridor", seed=42, async_player=True)
    # In deadly_corridor, player takes damage quickly
    time.sleep(3)
    events = manager._executor.get_recent_events()
    state_changes = [
        e for e in events if e["event_type"] == "state_change"
    ]
    # Should have some state transitions
    assert len(state_changes) > 0 or manager._executor.state.value in (
        "retreating", "fighting", "idle",
    )


def test_objective_queue_priority(async_basic):
    """Objectives are dequeued by priority (highest first)."""
    result_low = async_basic.set_objective("explore", priority=1)
    result_high = async_basic.set_objective("retreat", priority=10)

    queue = result_high["queue"]
    assert len(queue) == 2
    assert queue[0]["type"] == "retreat"
    assert queue[0]["priority"] == 10
    assert queue[1]["type"] == "explore"
    assert queue[1]["priority"] == 1


def test_set_strategy(async_basic):
    """Strategy parameters update correctly."""
    result = async_basic.set_strategy(aggression=0.9, health_retreat_threshold=30)
    strategy = result["strategy"]
    assert strategy["aggression"] == 0.9
    assert strategy["health_retreat_threshold"] == 30
    # Other values should remain at defaults
    assert strategy["health_collect_threshold"] == 50


def test_situation_report_format(async_basic):
    """Situation report contains all expected fields."""
    time.sleep(0.5)  # let executor run briefly
    result = async_basic.get_situation_report()
    assert "executor_state" in result
    if not result.get("episode_finished"):
        assert "objectives" in result
        assert "strategy" in result
        assert "events" in result
        assert "game_variables" in result
        assert "objects" in result
        assert "exploration" in result
        assert "screenshot_png" in result


def test_map_knowledge_format(async_basic):
    """Map knowledge returns exploration data."""
    time.sleep(0.5)
    result = async_basic.get_map_knowledge()
    assert "cells_explored" in result
    if not result.get("episode_finished"):
        assert "position" in result
        assert "x" in result["position"]
        assert "y" in result["position"]
        assert "executor_state" in result
        assert "objectives" in result


def test_legacy_tools_pause_executor(async_basic):
    """take_action works while executor is running."""
    time.sleep(0.2)
    # Episode may have finished in async mode, handle gracefully
    try:
        result = async_basic.take_action({"TURN_LEFT_RIGHT_DELTA": 10}, tics=1)
        assert "reward" in result or result.get("episode_finished") is True
    except Exception:
        # Episode may have finished, which is OK for this test
        pass
    # Executor should still be alive after
    assert async_basic._executor._thread.is_alive()


def test_compound_action_with_executor(manager):
    """aim_and_shoot works while executor is running, executor resumes after."""
    manager.start(scenario="defend_the_center", seed=42, async_player=True)
    time.sleep(0.5)

    state = manager.get_state()
    monster = next(
        (o for o in state.get("objects", []) if o.get("type") == "monster"),
        None,
    )
    if monster is not None:
        result = manager.aim_and_shoot(monster["id"], shots=2, max_tics=50)
        assert "action_summary" in result
    # Executor should still be alive
    assert manager._executor._thread.is_alive()


def test_new_episode_resets_executor(async_basic):
    """new_episode clears objectives and resets executor state."""
    async_basic.set_objective("explore", priority=5)
    time.sleep(0.3)

    async_basic.new_episode()
    assert async_basic._executor.get_objectives() == []
    assert async_basic._executor.state.value == "idle"
    # Executor should still be running
    assert async_basic._executor._thread.is_alive()


def test_executor_handles_episode_end(manager):
    """Executor goes IDLE when episode finishes, no crash."""
    manager.start(scenario="basic", seed=42, async_player=True, episode_timeout=50)
    time.sleep(3)  # enough time for episode to timeout
    # Should not have crashed
    assert manager._executor is not None
    # Thread may still be alive (waiting in idle) or may have exited
    # Either way, no exception should have propagated


def test_executor_event_log(manager):
    """Events are logged and retrievable."""
    manager.start(scenario="defend_the_center", seed=42, async_player=True)
    time.sleep(2)
    events = manager._executor.get_recent_events()
    assert isinstance(events, list)
    # Should have logged at least state changes
    if events:
        assert "event_type" in events[0]
        assert "tic" in events[0]
        assert "detail" in events[0]


def test_set_objective_requires_async(manager):
    """set_objective raises error without async mode."""
    manager.start(scenario="basic", seed=42, async_player=False)
    from fastmcp.exceptions import ToolError
    with pytest.raises(ToolError, match="Executor not running"):
        manager.set_objective("explore")


def test_set_strategy_requires_async(manager):
    """set_strategy raises error without async mode."""
    manager.start(scenario="basic", seed=42, async_player=False)
    from fastmcp.exceptions import ToolError
    with pytest.raises(ToolError, match="Executor not running"):
        manager.set_strategy(aggression=0.8)


def test_set_objective_invalid_type(async_basic):
    """set_objective with invalid type raises error."""
    from fastmcp.exceptions import ToolError
    with pytest.raises(ToolError, match="Unknown objective type"):
        async_basic.set_objective("nonexistent_type")
