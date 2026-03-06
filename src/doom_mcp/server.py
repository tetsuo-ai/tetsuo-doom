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
    async_player: bool = False,
    ticrate: int | None = None,
    seed: int | None = None,
    recording_path: str | None = None,
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
        async_player: Run game in real-time (ASYNC_PLAYER mode). The game window
            shows smooth continuous gameplay instead of freezing between agent
            actions. The world keeps running between take_action calls. Use with
            window_visible=true to watch the agent play.
        ticrate: Game speed in tics/sec for async mode (default 35 = normal speed).
            Lower values slow the game down, higher speeds it up.
        seed: Random seed for reproducibility.
        recording_path: File path to record episode demo (.lmp). The demo file
            can be replayed in ViZDoom or Doom source ports.
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
        async_player=async_player,
        ticrate=ticrate,
        seed=seed,
        recording_path=recording_path,
    )


@mcp.tool
def get_state(
    include_sectors: bool = False,
    include_depth: bool = True,
):
    """Get the current game state: screenshot + full structured data.

    Returns a screenshot image and game state dict containing:
    - game_variables: health, armor, position, angle, ammo, weapons, kills, etc.
    - objects: all entities with distance, angle_to_aim, type, threat level, HP
    - depth: spatial awareness (min/mean distance per screen region)
    - episode_finished, tic, total_reward

    Each object includes computed fields:
    - distance: units from player
    - angle_to_aim: degrees to turn to face it (positive=right, negative=left)
      Pass this value directly to TURN_LEFT_RIGHT_DELTA to aim at the object.
    - type/threat/attack_type/typical_hp/description: from enemy database
    - is_visible: whether currently on screen

    Args:
        include_sectors: Include map geometry (wall lines, heights). Very large -
            only request when you need spatial/navigation data. Default false.
        include_depth: Include depth buffer stats per screen region. Default true.
    """
    state = manager.get_state(
        include_sectors=include_sectors,
        include_depth=include_depth,
    )
    screenshot_png = state.pop("screenshot_png", None)

    if screenshot_png is not None:
        return [Image(data=screenshot_png, format="png"), state]
    return [state]


@mcp.tool
def take_action(
    actions: dict[str, float] | None = None,
    tics: int = 1,
    include_sectors: bool = False,
    include_depth: bool = True,
):
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
        include_sectors: Include map geometry. Very large - default false.
        include_depth: Include depth buffer stats. Default true.

    Returns screenshot + full state (same format as get_state).
    """
    result = manager.take_action(
        actions, tics,
        include_sectors=include_sectors,
        include_depth=include_depth,
    )
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
def get_map():
    """Get the automap (top-down view) of the level."""
    map_png = manager.get_map()
    if map_png is not None:
        return [Image(data=map_png, format="png")]
    return [{"error": "Automap buffer not available"}]


@mcp.tool
def new_episode(recording_path: str | None = None) -> dict:
    """Start a new episode in the current game.

    Resets the level while keeping the same configuration.

    Args:
        recording_path: File path to record this episode's demo (.lmp).
            If not set, uses the recording_path from start_game (if any).
    """
    return manager.new_episode(recording_path=recording_path)


@mcp.tool
def get_available_actions() -> dict:
    """Get the list of available action buttons with types and usage.

    Returns button names, whether each is binary or delta,
    sign conventions for delta buttons, and a usage example.
    """
    return manager.get_available_actions()


@mcp.tool
def aim_and_shoot(
    object_id: int,
    shots: int = 3,
    max_tics: int = 100,
):
    """Aim at an enemy and fire multiple shots. Handles aiming, firing, and weapon cooldown automatically.

    This is a compound action - it runs many game tics internally in milliseconds,
    so the player doesn't stand idle between LLM decisions.

    Typical workflow: get_state -> find enemy -> aim_and_shoot(enemy_id) -> assess result.

    Args:
        object_id: Numeric ID of the target (from the objects list in game state).
        shots: Number of shots to fire (default 3). Stops early if target dies.
        max_tics: Maximum game tics before giving up (default 100).

    Returns screenshot + state with action_summary containing:
        shots_fired, hits_landed, kills, ammo_spent, target_name, stop_reason.
    Stop reasons: shots_complete, target_killed, target_lost, player_died,
        out_of_ammo, episode_finished, max_tics.
    """
    result = manager.aim_and_shoot(object_id, shots=shots, max_tics=max_tics)
    screenshot_png = result.pop("screenshot_png", None)
    if screenshot_png is not None:
        return [Image(data=screenshot_png, format="png"), result]
    return [result]


@mcp.tool
def move_to(
    object_id: int,
    max_tics: int = 140,
    use: bool = False,
    stop_on_enemy: bool = True,
):
    """Move toward an object by ID. Handles pathfinding, turning, and stuck recovery automatically.

    This is a compound action - it runs many game tics internally in milliseconds.

    Typical workflow: get_state -> find object -> move_to(object_id) -> assess result.

    Args:
        object_id: Numeric ID of the target (from the objects list in game state).
        max_tics: Maximum game tics before giving up (default 140).
        use: Press USE when arriving (for switches/doors). Default false.
        stop_on_enemy: Stop if a visible monster appears nearby. Default true.

    Returns screenshot + state with action_summary containing:
        distance_moved, distance_remaining, target_name, used_object,
        threat_object, stop_reason.
    Stop reasons: arrived, target_lost, enemy_nearby, stuck, player_died,
        episode_finished, max_tics.
    """
    result = manager.move_to(
        object_id, max_tics=max_tics, use=use, stop_on_enemy=stop_on_enemy,
    )
    screenshot_png = result.pop("screenshot_png", None)
    if screenshot_png is not None:
        return [Image(data=screenshot_png, format="png"), result]
    return [result]


@mcp.tool
def explore(
    max_tics: int = 200,
    stop_on_enemy: bool = True,
    stop_on_item: bool = False,
):
    """Explore the environment autonomously. Walks forward, avoids walls, scans for threats and items.

    This is a compound action - it runs many game tics internally in milliseconds.
    Uses depth buffer for wall avoidance and stuck detection for recovery.

    Typical gameplay loop: explore -> enemy_spotted -> aim_and_shoot -> explore -> item_found -> move_to.

    Args:
        max_tics: Maximum game tics to explore (default 200).
        stop_on_enemy: Stop when a visible monster is spotted nearby. Default true.
        stop_on_item: Stop when a new item/ammo is spotted. Default false.

    Returns screenshot + state with action_summary containing:
        distance_moved, direction_changes, enemies_seen[], items_seen[], stop_reason.
    Stop reasons: enemy_spotted, item_found, stuck, player_died,
        episode_finished, max_tics.
    """
    result = manager.explore(
        max_tics=max_tics, stop_on_enemy=stop_on_enemy, stop_on_item=stop_on_item,
    )
    screenshot_png = result.pop("screenshot_png", None)
    if screenshot_png is not None:
        return [Image(data=screenshot_png, format="png"), result]
    return [result]


@mcp.tool
def strafe_and_shoot(
    object_id: int,
    direction: str = "auto",
    shots: int = 5,
    max_tics: int = 100,
):
    """Strafe laterally while firing at an enemy. Better than aim_and_shoot against hitscan enemies.

    This is a compound action - it runs many game tics internally in milliseconds.
    The player dodges left/right while keeping the target in the crosshair and firing.

    Args:
        object_id: Numeric ID of the target (from the objects list in game state).
        direction: Strafe direction - "left", "right", or "auto" (alternates every ~15 tics).
        shots: Number of shots to fire (default 5).
        max_tics: Maximum game tics before giving up (default 100).

    Returns screenshot + state with action_summary containing:
        shots_fired, hits_landed, kills, ammo_spent, target_name, strafe_direction,
        damage_taken, stop_reason.
    """
    result = manager.strafe_and_shoot(
        object_id, direction=direction, shots=shots, max_tics=max_tics,
    )
    screenshot_png = result.pop("screenshot_png", None)
    if screenshot_png is not None:
        return [Image(data=screenshot_png, format="png"), result]
    return [result]


@mcp.tool
def retreat(
    tics: int = 35,
    backpedal: bool = False,
):
    """Retreat from the current position. Turn and run or backpedal.

    This is a compound action - it runs many game tics internally in milliseconds.

    Args:
        tics: Total game tics for the retreat (default 35, ~1 second).
        backpedal: If true, move backward while keeping current facing direction
            (slower but maintains line of sight). If false (default), turn 180
            degrees then sprint forward (faster escape).

    Returns screenshot + state with action_summary containing:
        distance_moved, mode ("backpedal" or "turn_and_run"), stop_reason.
    """
    result = manager.retreat(tics=tics, backpedal=backpedal)
    screenshot_png = result.pop("screenshot_png", None)
    if screenshot_png is not None:
        return [Image(data=screenshot_png, format="png"), result]
    return [result]


@mcp.tool
def get_threat_assessment() -> dict:
    """Analyze all visible threats and return prioritized tactical intelligence.

    No game tics are consumed. Call freely between actions to assess the situation.

    Returns:
        threat_level: Overall threat - "none", "low", "medium", "high", or "critical".
        threats: Sorted list of enemies with id, name, distance, angle_to_aim,
            attack_type, priority_rank, priority_score.
        incoming_projectiles: Active projectiles to dodge.
        tactical_advice: String list with prioritized recommendations.
        player_health, player_armor, selected_weapon_ammo.
    """
    return manager.get_threat_assessment()


@mcp.tool
def get_navigation_info() -> dict:
    """Get spatial navigation intelligence. Tracks exploration across calls.

    No game tics are consumed. Call to check exploration progress, find unexplored
    areas, locate keys, and detect nearby doors.

    Returns:
        cells_explored: Number of 128-unit grid cells visited.
        explored_directions / unexplored_directions: Cardinal directions from current cell.
        suggested_direction: Best unexplored direction aligned with player facing.
        keys_found: Keys picked up this episode.
        known_key_locations: Visible keys not yet picked up.
        nearby_doors: Detected doors within 512 units.
        total_doors_found: All doors detected this episode.
    """
    return manager.get_navigation_info()


@mcp.tool
def get_situation_report():
    """Get a situation report for directing the autonomous executor.

    Use this instead of get_state when the executor is running (async_player=True).
    Returns screenshot + compact summary of executor state, recent events,
    game variables, nearby objects, and exploration progress.

    The executor plays autonomously at 35 Hz. Use this tool every few seconds
    to monitor progress and decide whether to change objectives or strategy.

    Returns screenshot + dict with:
        executor_state: Current state (idle/exploring/fighting/collecting/retreating/moving_to).
        objectives: Current objective queue.
        strategy: Current strategy parameters.
        events: Recent events since last call (kills, damage, state changes, etc).
        game_variables: Health, ammo, position, etc.
        objects: Nearby filtered objects.
        exploration: Cells explored, directions, keys, doors.
    """
    result = manager.get_situation_report()
    screenshot_png = result.pop("screenshot_png", None)
    if screenshot_png is not None:
        return [Image(data=screenshot_png, format="png"), result]
    return [result]


@mcp.tool
def set_objective(
    objective_type: str,
    params: dict | None = None,
    priority: int = 0,
    timeout_tics: int = 0,
) -> dict:
    """Set an objective for the autonomous executor.

    The executor will work toward this objective while handling combat and
    navigation autonomously. Higher priority objectives are executed first.
    Multiple objectives can be queued.

    Requires async_player=True (game started with autonomous executor).

    Args:
        objective_type: What to do. One of:
            - "explore": Explore the map autonomously.
            - "kill": Kill a specific enemy (params: {"object_id": int}).
            - "move_to_pos": Move to coordinates (params: {"x": float, "y": float}).
            - "move_to_obj": Move to an object (params: {"object_id": int}).
            - "collect": Collect nearby items.
            - "use_object": Move to and use an object (params: {"object_id": int}).
            - "retreat": Fall back from current position.
            - "hold_position": Stay in place, fight if attacked.
        params: Parameters for the objective.
        priority: Higher = executed first. Default 0.
        timeout_tics: Auto-fail after this many tics. 0 = no timeout.

    Returns the updated objective queue.
    """
    return manager.set_objective(
        objective_type=objective_type,
        params=params,
        priority=priority,
        timeout_tics=timeout_tics,
    )


@mcp.tool
def set_strategy(
    aggression: float | None = None,
    health_retreat_threshold: int | None = None,
    health_collect_threshold: int | None = None,
    ammo_switch_threshold: int | None = None,
    engage_range: float | None = None,
    collect_range: float | None = None,
    prefer_cover: bool | None = None,
) -> dict:
    """Tune the autonomous executor's behavior.

    Requires async_player=True (game started with autonomous executor).

    Args:
        aggression: 0.0=passive (avoid fights), 1.0=aggressive (engage everything).
            Default 0.5. At low values, executor retreats from distant enemies.
        health_retreat_threshold: HP at which executor always retreats. Default 20.
        health_collect_threshold: HP at which executor seeks health items. Default 50.
        ammo_switch_threshold: Ammo count at which executor switches weapons. Default 5.
        engage_range: Max distance (map units) to engage enemies. Default 1500.
        collect_range: Max distance to collect items. Default 800.
        prefer_cover: Try to use cover during combat. Default false.

    Returns the updated strategy.
    """
    kwargs = {}
    if aggression is not None:
        kwargs["aggression"] = aggression
    if health_retreat_threshold is not None:
        kwargs["health_retreat_threshold"] = health_retreat_threshold
    if health_collect_threshold is not None:
        kwargs["health_collect_threshold"] = health_collect_threshold
    if ammo_switch_threshold is not None:
        kwargs["ammo_switch_threshold"] = ammo_switch_threshold
    if engage_range is not None:
        kwargs["engage_range"] = engage_range
    if collect_range is not None:
        kwargs["collect_range"] = collect_range
    if prefer_cover is not None:
        kwargs["prefer_cover"] = prefer_cover
    return manager.set_strategy(**kwargs)


@mcp.tool
def get_map_knowledge() -> dict:
    """Get accumulated map knowledge for strategic planning.

    Returns exploration data including position, cells explored, unexplored
    directions, known keys, doors, and current executor state/objectives.

    No game tics are consumed. Use for planning which areas to explore next.
    """
    return manager.get_map_knowledge()


@mcp.tool
def stop_game() -> dict:
    """Stop the current game and release resources."""
    return manager.stop()
