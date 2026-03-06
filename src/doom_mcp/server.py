"""MCP server exposing ViZDoom as tools for AI agents."""

import atexit

from fastmcp import FastMCP
from fastmcp.utilities.types import Image

from .game_manager import GameManager

mcp = FastMCP("doom")
manager = GameManager()
atexit.register(manager.stop)


@mcp.tool
def start_game(
    scenario: str = "basic",
    wad: str | None = None,
    map_name: str | None = None,
    difficulty: int = 3,
    buttons: list[str] | None = None,
    variables: list[str] | None = None,
    screen_resolution: str = "RES_320X240",
    episode_timeout: int | None = None,
    render_hud: bool = False,
    window_visible: bool = False,
    seed: int | None = None,
) -> dict:
    """Start a new Doom game.

    Use scenario for built-in mini-scenarios, or wad+map_name for full campaign.

    Args:
        scenario: Built-in scenario name (basic, deadly_corridor, defend_the_center,
            defend_the_line, health_gathering, health_gathering_supreme,
            my_way_home, predict_position, deathmatch). Ignored if wad is set.
        wad: WAD file to load. Use "freedoom1" (Doom 1 maps E1M1-E4M9),
            "freedoom2" (Doom 2 maps MAP01-MAP32), or an absolute path.
        map_name: Map to load (e.g. "MAP01", "E1M1"). Required when using wad.
        difficulty: Doom skill level 1-5 (1=easiest, 5=nightmare).
        buttons: Button names to enable. Defaults to delta aim + movement + combat.
        variables: Game variable names to track. Defaults to comprehensive set.
        screen_resolution: ViZDoom resolution enum name (e.g. RES_640X480).
        episode_timeout: Max tics per episode. None uses scenario/map default.
        render_hud: Whether to render the HUD overlay.
        window_visible: Open a game window so you can watch. Requires a display.
        seed: Random seed for reproducibility.
    """
    return manager.start(
        scenario=scenario,
        wad=wad,
        map_name=map_name,
        difficulty=difficulty,
        buttons=buttons,
        variables=variables,
        screen_resolution=screen_resolution,
        episode_timeout=episode_timeout,
        render_hud=render_hud,
        window_visible=window_visible,
        seed=seed,
    )


@mcp.tool
def get_state() -> list:
    """Get the current game state: screenshot + full structured data.

    Returns a screenshot image and game state dict containing:
    - game_variables: health, armor, position, angle, ammo, weapons, kills, etc.
    - objects: all entities with distance, angle_to_aim, type, threat level, HP
    - sectors: map geometry (wall lines, floor/ceiling heights)
    - depth: spatial awareness (min/mean distance per screen region)
    - episode_finished, tic, total_reward

    Each object includes computed fields:
    - distance: units from player
    - angle_to_aim: degrees to turn to face it (positive=right, negative=left)
      Pass this value directly to TURN_LEFT_RIGHT_DELTA to aim at the object.
    - type/threat/attack_type/typical_hp/description: from enemy database
    - is_visible: whether currently on screen
    """
    state = manager.get_state()
    screenshot_png = state.pop("screenshot_png", None)

    if screenshot_png is not None:
        return [Image(data=screenshot_png, format="png"), state]
    return [state]


@mcp.tool
def take_action(actions: dict[str, float] | None = None, tics: int = 1) -> list:
    """Execute an action and return the resulting game state.

    Args:
        actions: Dict mapping button names to values.
            Binary buttons: 1 to press (e.g. {"MOVE_FORWARD": 1, "ATTACK": 1}).
            Delta buttons: degrees (e.g. {"TURN_LEFT_RIGHT_DELTA": -15.5}).
                TURN_LEFT_RIGHT_DELTA: positive=turn right, negative=turn left.
                LOOK_UP_DOWN_DELTA: positive=look down, negative=look up.
            Omitted buttons default to 0. None or {} is a no-op.
            TIP: Each object's angle_to_aim can be passed directly to
            TURN_LEFT_RIGHT_DELTA to aim at that object.
        tics: Game tics to hold the action (default 1).
            WARNING: delta values are MULTIPLIED by tics.
            Use tics=1 for precise aiming.

    Returns screenshot + full state (same format as get_state).
    """
    result = manager.take_action(actions, tics)
    screenshot_png = result.pop("screenshot_png", None)

    if screenshot_png is not None:
        return [Image(data=screenshot_png, format="png"), result]
    return [result]


@mcp.tool
def get_objects() -> dict:
    """Get all objects in the game world with enriched info.

    Each object includes:
    - distance from player, angle_to_aim (degrees to turn to face it)
    - type (monster/item/projectile/weapon/player/decoration)
    - threat level, attack type, typical HP, description
    - is_visible (on screen), screen bounding box if visible
    - raw position, velocity, angle
    """
    return manager.get_objects()


@mcp.tool
def get_map() -> list:
    """Get the automap (top-down view) of the level."""
    map_png = manager.get_map()
    if map_png is not None:
        return [Image(data=map_png, format="png")]
    return [{"error": "Automap buffer not available"}]


@mcp.tool
def new_episode() -> dict:
    """Start a new episode in the current game.

    Resets the level while keeping the same configuration.
    """
    return manager.new_episode()


@mcp.tool
def get_available_actions() -> dict:
    """Get the list of available action buttons with types and usage.

    Returns button names, whether each is binary or delta,
    sign conventions for delta buttons, and a usage example.
    """
    return manager.get_available_actions()


@mcp.tool
def stop_game() -> dict:
    """Stop the current game and release resources."""
    return manager.stop()
