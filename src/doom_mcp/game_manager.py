"""Core game manager wrapping ViZDoom's DoomGame."""

import os

import vizdoom as vzd
from fastmcp.exceptions import ToolError

from .actions import BUTTON_NAMES, names_to_action_list
from .scenarios import get_scenario_config_path
from .state import (
    extract_depth_as_stats,
    extract_game_variables,
    extract_objects,
    extract_sectors,
    screen_buffer_to_png,
)

# Delta buttons for precise aiming + binary buttons for movement/actions.
# TURN_LEFT_RIGHT_DELTA: positive = turn right (degrees, multiplied by tics)
# LOOK_UP_DOWN_DELTA: positive = look down (degrees, multiplied by tics)
DEFAULT_BUTTONS = [
    # Delta buttons for precise control (value = amount per tic)
    "TURN_LEFT_RIGHT_DELTA",
    "LOOK_UP_DOWN_DELTA",
    "MOVE_FORWARD_BACKWARD_DELTA",
    "MOVE_LEFT_RIGHT_DELTA",
    # Binary buttons
    "ATTACK",
    "USE",
    "SPEED",
    "SELECT_NEXT_WEAPON",
    "SELECT_PREV_WEAPON",
    "JUMP",
    "CROUCH",
]

DEFAULT_VARIABLES = [
    # Vitals
    "HEALTH",
    "ARMOR",
    "DEAD",
    "ON_GROUND",
    # Position & orientation
    "POSITION_X",
    "POSITION_Y",
    "POSITION_Z",
    "ANGLE",
    "PITCH",
    "VELOCITY_X",
    "VELOCITY_Y",
    "VELOCITY_Z",
    # Combat
    "ATTACK_READY",
    "SELECTED_WEAPON",
    "SELECTED_WEAPON_AMMO",
    "AMMO0",
    "AMMO1",
    "AMMO2",
    "AMMO3",
    "AMMO4",
    "AMMO5",
    "AMMO6",
    # Weapons inventory
    "WEAPON0",
    "WEAPON1",
    "WEAPON2",
    "WEAPON3",
    "WEAPON4",
    "WEAPON5",
    "WEAPON6",
    "WEAPON7",
    # Stats
    "KILLCOUNT",
    "ITEMCOUNT",
    "SECRETCOUNT",
    "FRAGCOUNT",
    "DEATHCOUNT",
    "HITCOUNT",
    "HITS_TAKEN",
    "DAMAGECOUNT",
    "DAMAGE_TAKEN",
]

# Delta buttons accept numeric values, not just 0/1
DELTA_BUTTONS = {
    "TURN_LEFT_RIGHT_DELTA",
    "LOOK_UP_DOWN_DELTA",
    "MOVE_FORWARD_BACKWARD_DELTA",
    "MOVE_LEFT_RIGHT_DELTA",
    "MOVE_UP_DOWN_DELTA",
}


# Map progression order
_DOOM2_MAPS = [f"MAP{i:02d}" for i in range(1, 33)]
_DOOM1_MAPS = [f"E{e}M{m}" for e in range(1, 5) for m in range(1, 10)]


def _next_map(current: str) -> str | None:
    """Return the next map in progression, or None if at the end."""
    for map_list in (_DOOM2_MAPS, _DOOM1_MAPS):
        if current.upper() in map_list:
            idx = map_list.index(current.upper())
            if idx + 1 < len(map_list):
                return map_list[idx + 1]
            return None
    return None


class GameManager:
    """Manages a single ViZDoom game instance."""

    def __init__(self) -> None:
        self._game: vzd.DoomGame | None = None
        self._buttons: list[vzd.Button] = []
        self._variable_names: list[str] = []
        self._scenario_name: str = ""
        self._wad: str | None = None
        self._current_map: str | None = None

    @property
    def is_running(self) -> bool:
        return self._game is not None

    def _require_running(self) -> vzd.DoomGame:
        if self._game is None:
            raise ToolError(
                "No game is running. Call start_game first."
            )
        return self._game

    def _require_episode(self) -> vzd.DoomGame:
        game = self._require_running()
        if game.is_episode_finished():
            raise ToolError(
                "Episode is finished. Call new_episode to start a new one."
            )
        return game

    @staticmethod
    def _resolve_wad_path(wad: str) -> str:
        """Resolve a WAD name or path to a full file path."""
        # Shorthand names for bundled WADs
        bundled = {
            "freedoom1": os.path.join(os.path.dirname(vzd.__file__), "freedoom1.wad"),
            "freedoom2": os.path.join(os.path.dirname(vzd.__file__), "freedoom2.wad"),
        }
        if wad.lower() in bundled:
            return bundled[wad.lower()]
        if os.path.isabs(wad) and os.path.exists(wad):
            return wad
        raise ToolError(
            f"WAD not found: {wad!r}. Use 'freedoom1', 'freedoom2', "
            f"or an absolute path to a .wad file."
        )

    def _get_player_pos(self, game: vzd.DoomGame) -> tuple[float, float, float]:
        """Get player position and angle for computing relative object info."""
        px = game.get_game_variable(vzd.GameVariable.POSITION_X)
        py = game.get_game_variable(vzd.GameVariable.POSITION_Y)
        pa = game.get_game_variable(vzd.GameVariable.ANGLE)
        return px, py, pa

    def _extract_full_state(self, game: vzd.DoomGame) -> dict:
        """Extract complete game state: variables, objects, sectors, depth."""
        state = game.get_state()
        variables = extract_game_variables(game, self._variable_names)
        px, py, pa = self._get_player_pos(game)
        objects = extract_objects(state, player_x=px, player_y=py, player_angle=pa)
        sectors = extract_sectors(state)
        depth = extract_depth_as_stats(state.depth_buffer) if state.depth_buffer is not None else None
        screenshot_png = screen_buffer_to_png(state.screen_buffer)

        return {
            "episode_finished": False,
            "tic": state.tic,
            "game_variables": variables,
            "objects": objects,
            "sectors": sectors,
            "depth": depth,
            "total_reward": game.get_total_reward(),
            "screenshot_png": screenshot_png,
        }

    def _finished_state(self, game: vzd.DoomGame, reward: float | None = None) -> dict:
        """Return state dict for a finished episode."""
        dead = game.get_game_variable(vzd.GameVariable.DEAD)
        result = {
            "episode_finished": True,
            "player_dead": bool(dead),
            "level_completed": not bool(dead),
            "total_reward": game.get_total_reward(),
            "game_variables": {},
        }
        if reward is not None:
            result["reward"] = reward
        if self._current_map:
            result["map"] = self._current_map
            nxt = _next_map(self._current_map)
            if nxt and not dead:
                result["next_map"] = nxt
                result["hint"] = "Level completed! Call new_episode() to advance to next map."
            elif dead:
                result["hint"] = "You died. Call new_episode() to retry this map."
        return result

    def start(
        self,
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
        """Start a new game with the given configuration.

        Use scenario for ViZDoom's built-in scenarios, or wad+map_name
        for full Doom campaign maps. If wad is provided, scenario is ignored.

        Available WADs bundled with ViZDoom:
            "freedoom1" — Freedoom Phase 1 (Doom 1 format, maps E1M1-E4M9)
            "freedoom2" — Freedoom Phase 2 (Doom 2 format, maps MAP01-MAP32)
            Or provide an absolute path to any .wad file.
        """
        if self._game is not None:
            self.stop()

        button_names = buttons or DEFAULT_BUTTONS
        variable_names = variables or DEFAULT_VARIABLES

        # Validate button names
        for name in button_names:
            if name.upper() not in BUTTON_NAMES:
                raise ToolError(f"Unknown button {name!r}. Valid: {sorted(BUTTON_NAMES.keys())}")

        # Validate variable names
        for name in variable_names:
            if not hasattr(vzd.GameVariable, name.upper()):
                raise ToolError(
                    f"Unknown game variable {name!r}. "
                    f"Valid: {sorted(vzd.GameVariable.__members__.keys())}"
                )

        # Validate screen resolution
        try:
            resolution = getattr(vzd.ScreenResolution, screen_resolution)
        except AttributeError:
            raise ToolError(
                f"Unknown resolution {screen_resolution!r}. "
                f"Valid: {sorted(vzd.ScreenResolution.__members__.keys())}"
            )

        game = vzd.DoomGame()

        if wad is not None:
            # Custom WAD mode — load a full Doom campaign
            wad_path = self._resolve_wad_path(wad)
            game.set_doom_game_path(wad_path)
            if map_name:
                game.set_doom_map(map_name)
        else:
            # Built-in scenario mode
            try:
                cfg_path = get_scenario_config_path(scenario)
            except ValueError as e:
                raise ToolError(str(e))
            game.load_config(cfg_path)

        # Universal settings (override anything from .cfg)
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_screen_resolution(resolution)
        game.set_objects_info_enabled(True)
        game.set_labels_buffer_enabled(True)
        game.set_depth_buffer_enabled(True)
        game.set_sectors_info_enabled(True)
        game.set_automap_buffer_enabled(True)
        game.set_automap_mode(vzd.AutomapMode.OBJECTS_WITH_SIZE)
        game.set_automap_rotate(False)
        game.set_episode_start_time(14)
        game.set_window_visible(window_visible)
        game.set_mode(vzd.Mode.PLAYER)
        game.set_doom_skill(difficulty)
        game.set_render_hud(render_hud)

        if seed is not None:
            game.set_seed(seed)

        if episode_timeout is not None:
            game.set_episode_timeout(episode_timeout)

        # Configure buttons
        game.clear_available_buttons()
        resolved_buttons = []
        for name in button_names:
            btn = BUTTON_NAMES[name.upper()]
            game.add_available_button(btn)
            resolved_buttons.append(btn)

        # Configure game variables
        game.clear_available_game_variables()
        resolved_variable_names = []
        for name in variable_names:
            upper = name.upper()
            var = getattr(vzd.GameVariable, upper)
            game.add_available_game_variable(var)
            resolved_variable_names.append(upper)

        game.init()

        # Set map exit reward so completing a level gives positive feedback
        if wad is not None:
            game.set_map_exit_reward(100.0)

        game.init()

        self._game = game
        self._buttons = resolved_buttons
        self._variable_names = resolved_variable_names
        self._scenario_name = scenario if wad is None else f"{wad}:{map_name or 'default'}"
        self._wad = wad
        self._current_map = map_name

        result = {
            "status": "running",
            "buttons": [b.name for b in self._buttons],
            "variables": self._variable_names,
            "screen_resolution": screen_resolution,
        }
        if wad is not None:
            result["wad"] = wad
            result["map"] = map_name or "default"
        else:
            result["scenario"] = scenario
        return result

    def stop(self) -> dict:
        """Stop the current game."""
        if self._game is not None:
            self._game.close()
            self._game = None
            self._buttons = []
            self._variable_names = []
            self._scenario_name = ""
            self._wad = None
            self._current_map = None
        return {"status": "stopped"}

    def new_episode(self) -> dict:
        """Start a new episode. In campaign mode, auto-advances on level completion.

        If the player completed the level (not dead), advances to the next map.
        If the player died, restarts the same map.
        In scenario mode, just restarts the episode.
        """
        game = self._require_running()

        advanced = False
        if self._current_map and not game.get_game_variable(vzd.GameVariable.DEAD):
            nxt = _next_map(self._current_map)
            if nxt:
                game.set_doom_map(nxt)
                self._current_map = nxt
                advanced = True

        game.new_episode()

        result = {"status": "new_episode"}
        if self._current_map:
            result["map"] = self._current_map
            if advanced:
                result["advanced"] = True
        else:
            result["scenario"] = self._scenario_name
        return result

    def get_state(self) -> dict:
        """Get complete game state: variables, objects, sectors, depth, screenshot."""
        game = self._require_running()

        if game.is_episode_finished():
            return self._finished_state(game)

        return self._extract_full_state(game)

    def take_action(
        self,
        actions: dict[str, float] | None = None,
        tics: int = 1,
    ) -> dict:
        """Execute an action and return the full resulting state.

        Args:
            actions: Dict mapping button name to value.
                Binary buttons: 1 to press (e.g. {"MOVE_FORWARD": 1, "ATTACK": 1}).
                Delta buttons: degree value (e.g. {"TURN_LEFT_RIGHT_DELTA": -15.5}).
                    TURN_LEFT_RIGHT_DELTA: positive=right, negative=left.
                    LOOK_UP_DOWN_DELTA: positive=down, negative=up.
                Omitted buttons default to 0. None or {} is a no-op.
                NOTE: delta values are multiplied by tics. Use tics=1 for precise aim.
            tics: Number of game tics to hold the action (default 1).
        """
        game = self._require_episode()

        action_list = [0.0] * len(self._buttons)

        if actions:
            button_index = {b.name: i for i, b in enumerate(self._buttons)}
            for name, value in actions.items():
                upper = name.upper()
                if upper not in BUTTON_NAMES:
                    raise ToolError(
                        f"Unknown button {name!r}. "
                        f"Valid: {sorted(BUTTON_NAMES.keys())}"
                    )
                if upper not in button_index:
                    raise ToolError(
                        f"Button {name!r} is not configured. "
                        f"Configured: {sorted(button_index.keys())}"
                    )
                action_list[button_index[upper]] = float(value)

        reward = game.make_action(action_list, tics)

        if game.is_episode_finished():
            return self._finished_state(game, reward=reward)

        result = self._extract_full_state(game)
        result["reward"] = reward
        return result

    def get_objects(self) -> dict:
        """Get object and label info from the current state."""
        game = self._require_episode()
        state = game.get_state()
        px, py, pa = self._get_player_pos(game)
        return {
            "objects": extract_objects(state, player_x=px, player_y=py, player_angle=pa),
        }

    def get_map(self) -> bytes | None:
        """Get the automap buffer as PNG bytes."""
        game = self._require_episode()
        state = game.get_state()
        if state.automap_buffer is not None:
            return screen_buffer_to_png(state.automap_buffer)
        return None

    def get_available_actions(self) -> dict:
        """Return the configured buttons with usage info."""
        self._require_running()
        buttons = []
        for b in self._buttons:
            is_delta = b.name in DELTA_BUTTONS
            buttons.append({
                "name": b.name,
                "type": "delta" if is_delta else "binary",
                "description": (
                    "Numeric value in degrees (multiplied by tics)"
                    if is_delta else "1 to press, 0 to release"
                ),
            })
        return {
            "buttons": buttons,
            "usage": (
                "Pass a dict mapping button names to values. "
                "Example: take_action(actions={"
                '"TURN_LEFT_RIGHT_DELTA": -15, '
                '"MOVE_FORWARD": 1, '
                '"ATTACK": 1'
                "}, tics=1)"
            ),
            "delta_convention": {
                "TURN_LEFT_RIGHT_DELTA": "positive=right, negative=left (degrees)",
                "LOOK_UP_DOWN_DELTA": "positive=down, negative=up (degrees)",
                "MOVE_FORWARD_BACKWARD_DELTA": "positive=forward, negative=backward (speed units)",
                "MOVE_LEFT_RIGHT_DELTA": "positive=right, negative=left (speed units)",
            },
        }
