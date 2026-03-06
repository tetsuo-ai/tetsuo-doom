"""Unit tests for action name mapping."""

import pytest
import vizdoom as vzd

from doom_mcp.actions import BUTTON_NAMES, names_to_action_list


SAMPLE_BUTTONS = [
    vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD,
    vzd.Button.TURN_LEFT,
    vzd.Button.ATTACK,
]


def test_button_names_populated():
    assert len(BUTTON_NAMES) > 0
    assert "MOVE_FORWARD" in BUTTON_NAMES
    assert "ATTACK" in BUTTON_NAMES
    assert "TURN_LEFT_RIGHT_DELTA" in BUTTON_NAMES


def test_single_action():
    result = names_to_action_list(["ATTACK"], SAMPLE_BUTTONS)
    assert result == [0, 0, 0, 1]


def test_multiple_actions():
    result = names_to_action_list(["MOVE_FORWARD", "ATTACK"], SAMPLE_BUTTONS)
    assert result == [1, 0, 0, 1]


def test_empty_actions_noop():
    result = names_to_action_list([], SAMPLE_BUTTONS)
    assert result == [0, 0, 0, 0]


def test_case_insensitive():
    result = names_to_action_list(["move_forward"], SAMPLE_BUTTONS)
    assert result == [1, 0, 0, 0]


def test_unknown_button_raises():
    with pytest.raises(ValueError, match="Unknown button"):
        names_to_action_list(["NONEXISTENT"], SAMPLE_BUTTONS)


def test_unconfigured_button_raises():
    with pytest.raises(ValueError, match="not configured"):
        names_to_action_list(["JUMP"], SAMPLE_BUTTONS)
