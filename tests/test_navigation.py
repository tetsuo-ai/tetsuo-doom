"""Unit tests for NavigationMemory (no ViZDoom required)."""

from doom_mcp.navigation import NavigationMemory, _CELL_SIZE


def test_nav_memory_tracks_cells():
    nav = NavigationMemory()
    nav.update(0, 0, 90)
    nav.update(_CELL_SIZE * 2, 0, 90)
    nav.update(_CELL_SIZE * 2, _CELL_SIZE * 3, 90)
    summary = nav.get_exploration_summary(_CELL_SIZE * 2, _CELL_SIZE * 3, 90)
    assert summary["cells_explored"] >= 3


def test_nav_memory_reset():
    nav = NavigationMemory()
    nav.update(0, 0, 0)
    nav.update(500, 500, 0)
    assert nav.get_exploration_summary(500, 500, 0)["cells_explored"] >= 2
    nav.reset()
    summary = nav.get_exploration_summary(0, 0, 0)
    assert summary["cells_explored"] == 0
    assert summary["keys_found"] == []
    assert summary["breadcrumbs"] == 0


def test_nav_memory_key_tracking():
    nav = NavigationMemory()
    key_obj = {
        "id": 42,
        "name": "BlueCard",
        "type": "key",
        "position_x": 100,
        "position_y": 200,
        "distance": 50,
        "is_visible": True,
    }
    # First update: key visible
    nav.update(80, 180, 0, objects=[key_obj])
    summary = nav.get_exploration_summary(80, 180, 0)
    assert len(summary["known_key_locations"]) == 1
    assert summary["known_key_locations"][0]["name"] == "BlueCard"

    # Second update: key gone (player picked it up - was within range)
    nav.update(100, 200, 0, objects=[])
    summary = nav.get_exploration_summary(100, 200, 0)
    assert len(summary["keys_found"]) == 1
    assert summary["keys_found"][0]["name"] == "BlueCard"
    assert len(summary["known_key_locations"]) == 0


def test_nav_memory_key_not_picked_up_if_far():
    nav = NavigationMemory()
    key_obj = {
        "id": 42,
        "name": "RedCard",
        "type": "key",
        "position_x": 1000,
        "position_y": 1000,
        "distance": 500,
        "is_visible": True,
    }
    nav.update(0, 0, 0, objects=[key_obj])
    # Key disappears but player is far away
    nav.update(0, 0, 0, objects=[])
    summary = nav.get_exploration_summary(0, 0, 0)
    assert len(summary["keys_found"]) == 0


def test_nav_memory_breadcrumb_spacing():
    nav = NavigationMemory()
    # Small movements should not create multiple breadcrumbs
    nav.update(0, 0, 0)
    nav.update(10, 10, 0)
    nav.update(20, 20, 0)
    summary = nav.get_exploration_summary(20, 20, 0)
    assert summary["breadcrumbs"] == 1  # only initial breadcrumb

    # Large movement should add a breadcrumb
    nav.update(200, 200, 0)
    summary = nav.get_exploration_summary(200, 200, 0)
    assert summary["breadcrumbs"] == 2


def test_nav_memory_exploration_directions():
    nav = NavigationMemory()
    # Visit cells going east
    for i in range(5):
        nav.update(i * _CELL_SIZE + 10, 10, 0)

    cx = 4 * _CELL_SIZE + 10
    summary = nav.get_exploration_summary(cx, 10, 0)
    assert "east" in summary["unexplored_directions"]
    assert "north" in summary["unexplored_directions"]
    assert "south" in summary["unexplored_directions"]
    # West should be explored (we came from there)
    assert "west" in summary["explored_directions"]


def test_nav_memory_suggested_direction():
    nav = NavigationMemory()
    # Visit only current cell
    nav.update(0, 0, 0)  # facing east (angle 0)
    summary = nav.get_exploration_summary(0, 0, 0)
    # All directions unexplored, but east (angle 0) most aligned with facing
    assert summary["suggested_direction"] == "east"


def test_nav_memory_door_detection():
    nav = NavigationMemory()
    # Simulate a door sector: small sector with low ceiling gap
    door_sector = {
        "floor_height": 0,
        "ceiling_height": 4,  # gap = 4 < threshold of 8
        "lines": [
            {"x1": 100, "y1": 100, "x2": 200, "y2": 100},
            {"x1": 200, "y1": 100, "x2": 200, "y2": 120},
            {"x1": 200, "y1": 120, "x2": 100, "y2": 120},
            {"x1": 100, "y1": 120, "x2": 100, "y2": 100},
        ],
    }
    # Non-door sector: normal room
    room_sector = {
        "floor_height": 0,
        "ceiling_height": 128,
        "lines": [
            {"x1": 0, "y1": 0, "x2": 500, "y2": 0},
            {"x1": 500, "y1": 0, "x2": 500, "y2": 500},
        ],
    }
    nav.update(150, 110, 0, sectors=[door_sector, room_sector])
    summary = nav.get_exploration_summary(150, 110, 0)
    assert summary["total_doors_found"] == 1
    assert len(summary["nearby_doors"]) == 1


def test_nav_memory_door_dedup():
    nav = NavigationMemory()
    door1 = {
        "floor_height": 0,
        "ceiling_height": 4,
        "lines": [
            {"x1": 100, "y1": 100, "x2": 150, "y2": 100},
            {"x1": 150, "y1": 100, "x2": 150, "y2": 120},
        ],
    }
    door2 = {
        "floor_height": 0,
        "ceiling_height": 4,
        "lines": [
            {"x1": 110, "y1": 105, "x2": 160, "y2": 105},
            {"x1": 160, "y1": 105, "x2": 160, "y2": 125},
        ],
    }
    nav.update(100, 100, 0, sectors=[door1, door2])
    summary = nav.get_exploration_summary(100, 100, 0)
    # Should deduplicate since they're close together
    assert summary["total_doors_found"] == 1
