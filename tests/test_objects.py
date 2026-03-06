"""Unit tests for object info database."""

from doom_mcp.objects import get_object_info, OBJECT_INFO


def test_known_monster():
    info = get_object_info("Demon")
    assert info["type"] == "monster"
    assert info["threat"] == "medium"
    assert info["typical_hp"] == 150


def test_known_item():
    info = get_object_info("Medikit")
    assert info["type"] == "item"
    assert info["threat"] == "none"


def test_known_projectile():
    info = get_object_info("DoomImpBall")
    assert info["type"] == "projectile"


def test_player():
    info = get_object_info("DoomPlayer")
    assert info["type"] == "player"


def test_unknown_returns_fallback():
    info = get_object_info("SomethingWeird")
    assert info["type"] == "unknown"
    assert "SomethingWeird" in info["description"]


def test_vizdoom_custom_monster():
    info = get_object_info("MarineChainsawVzd")
    assert info["type"] == "monster"
    assert info["attack"] == "melee"


def test_poison_is_dangerous():
    info = get_object_info("Poison")
    assert info["threat"] == "low"
    assert "AVOID" in info["description"]


def test_all_monsters_have_hp():
    for name, info in OBJECT_INFO.items():
        if info["type"] == "monster":
            assert info["typical_hp"] > 0, f"{name} should have HP"
