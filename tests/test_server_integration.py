"""Mock-based tests for server tool error handling."""

import pytest
from fastmcp.exceptions import ToolError

from doom_mcp import server


@pytest.fixture(autouse=True)
def reset_manager():
    """Ensure manager is stopped between tests."""
    yield
    server.manager.stop()


def test_start_game_unknown_scenario():
    with pytest.raises(ToolError, match="Unknown scenario"):
        server.start_game(scenario="nonexistent")


def test_start_game_unknown_button():
    with pytest.raises(ToolError, match="Unknown button"):
        server.start_game(buttons=["FAKE_BUTTON"])


def test_start_game_unknown_variable():
    with pytest.raises(ToolError, match="Unknown game variable"):
        server.start_game(variables=["FAKE_VAR"])


def test_start_game_bad_resolution():
    with pytest.raises(ToolError, match="Unknown resolution"):
        server.start_game(screen_resolution="RES_9999X9999")


def test_get_state_no_game():
    with pytest.raises(ToolError, match="No game is running"):
        server.get_state()


def test_take_action_no_game():
    with pytest.raises(ToolError, match="No game is running"):
        server.take_action(actions={"ATTACK": 1})


def test_stop_game_idempotent():
    result = server.stop_game()
    assert result["status"] == "stopped"
    result = server.stop_game()
    assert result["status"] == "stopped"
