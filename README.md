# doom-mcp

<img width="1184" height="880" alt="image" src="https://github.com/user-attachments/assets/0ac87a79-6a2a-41e8-8233-e6ae9a7a7d53" />

An MCP server that lets AI agents play Doom. Exposes game control, state observation, and action execution as [Model Context Protocol](https://modelcontextprotocol.io) tools. Works best with [xAI Grok](https://console.x.ai) models and comes pre-installed with [AgenC](https://github.com/tetsuo-ai/AgenC).

Agents receive structured game state (objects, depth, sectors, variables) plus screenshots, then send named actions back - no raw button arrays or pixel parsing required.

## Features

- **Full game state as structured data** -object positions with distance/angle-to-aim, depth buffer stats, sector geometry, 39 tracked game variables (health, ammo, position, velocity, kills, etc.)
- **Screenshots as PNG images** -returned alongside structured state for visual grounding
- **Delta buttons for precise control** -continuous turn/move values in degrees, not just binary left/right
- **Enemy database** -70+ entity types with threat level, attack type, typical HP, and descriptions
- **Campaign mode** -play through all 32 Doom II maps (or 36 Doom I maps) with automatic level progression
- **9 built-in scenarios** -ViZDoom's training scenarios for focused skill development
- **Headless or visible** -run without a display for AI training, or open a window to watch

## Prerequisites

- Python 3.10+
- A working C++ compiler (ViZDoom builds native code)
- On Linux: `libgl1-mesa-dev` and related OpenGL packages

## Installation

```bash
git clone https://github.com/your-username/tetsuo-doom.git
cd tetsuo-doom
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Verify the installation:

```bash
python -c "import vizdoom; print(vizdoom.__version__)"
python -c "import fastmcp; print(fastmcp.__version__)"
```

## Quick Start

### With VS Code (Copilot)

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "doom": {
      "command": "${workspaceFolder}/.venv/bin/fastmcp",
      "args": ["run", "src/doom_mcp/server.py"]
    }
  }
}
```
### With Claude Code

The repo includes `.mcp.json` for automatic configuration. Claude Code will detect the server when you open the project.

### With Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "doom": {
      "command": "/path/to/tetsuo-doom/.venv/bin/fastmcp",
      "args": ["run", "src/doom_mcp/server.py"],
      "cwd": "/path/to/tetsuo-doom"
    }
  }
}
```


### Standalone

```bash
# stdio transport (for MCP clients)
fastmcp run src/doom_mcp/server.py

# SSE transport (for web clients)
fastmcp run src/doom_mcp/server.py --transport sse
```

## Tools

### `start_game`

Start a new Doom game. Use `scenario` for built-in training scenarios, or `wad` + `map_name` for full campaign maps.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scenario` | `string` | `"basic"` | Built-in scenario name (ignored if `wad` is set) |
| `wad` | `string` | `null` | WAD file: `"freedoom1"`, `"freedoom2"`, or absolute path |
| `map_name` | `string` | `null` | Map to load (e.g. `"MAP01"`, `"E1M1"`) |
| `difficulty` | `int` | `3` | Doom skill level 1-5 |
| `buttons` | `list[str]` | see below | Action buttons to enable |
| `variables` | `list[str]` | see below | Game variables to track |
| `screen_resolution` | `string` | `"RES_320X240"` | ViZDoom resolution enum |
| `episode_timeout` | `int` | `null` | Max tics per episode |
| `render_hud` | `bool` | `false` | Show HUD overlay |
| `window_visible` | `bool` | `false` | Open a game window |
| `seed` | `int` | `null` | Random seed |

### `get_state`

Returns a screenshot image and game state dict:

- `game_variables` -health, armor, position, angle, ammo, weapons, kills, etc.
- `objects` -all entities with computed `distance`, `angle_to_aim`, type, threat level, HP
- `sectors` -map geometry (wall lines, floor/ceiling heights)
- `depth` -min/mean distance per screen region (6 regions + crosshair)
- `episode_finished`, `tic`, `total_reward`

### `take_action`

Execute an action and return the resulting state (same format as `get_state`).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `actions` | `dict[str, float]` | `null` | Button name to value mapping |
| `tics` | `int` | `1` | Game tics to hold the action |

```jsonc
// Example: turn right 15 degrees, move forward, and shoot
{"actions": {"TURN_LEFT_RIGHT_DELTA": 15, "MOVE_FORWARD_BACKWARD_DELTA": 10, "ATTACK": 1}, "tics": 1}
```

### `get_objects`

Get all objects in the game world with enriched info: distance, angle to aim, type/threat/HP from the enemy database, visibility, and screen bounding box.

### `get_map`

Get the automap (top-down view) as a PNG image.

### `new_episode`

Start a new episode. In campaign mode, automatically advances to the next map on level completion, or restarts the same map on death.

### `get_available_actions`

Returns configured buttons with types (`delta` or `binary`), sign conventions, and usage examples.

### `stop_game`

Stop the current game and release resources.

## Action System

Actions use a dict mapping button names to values:

| Button | Type | Convention |
|--------|------|------------|
| `TURN_LEFT_RIGHT_DELTA` | delta | positive = right, negative = left (degrees) |
| `LOOK_UP_DOWN_DELTA` | delta | positive = down, negative = up (degrees) |
| `MOVE_FORWARD_BACKWARD_DELTA` | delta | positive = forward, negative = backward |
| `MOVE_LEFT_RIGHT_DELTA` | delta | positive = right, negative = left |
| `ATTACK` | binary | 1 = fire |
| `USE` | binary | 1 = open doors / activate switches |
| `SPEED` | binary | 1 = run |
| `SELECT_NEXT_WEAPON` | binary | 1 = cycle weapon forward |
| `SELECT_PREV_WEAPON` | binary | 1 = cycle weapon backward |
| `JUMP` | binary | 1 = jump |
| `CROUCH` | binary | 1 = crouch |

Each object in the game state includes an `angle_to_aim` field -pass it directly to `TURN_LEFT_RIGHT_DELTA` to face that object.

## Game Modes

### Training Scenarios

9 built-in ViZDoom scenarios for focused practice:

| Scenario | Description |
|----------|-------------|
| `basic` | Single room, single enemy -learn to shoot |
| `deadly_corridor` | Navigate a corridor with enemies on both sides |
| `defend_the_center` | Survive waves of enemies approaching from all directions |
| `defend_the_line` | Hold a defensive line against approaching enemies |
| `health_gathering` | Collect health packs to survive in a toxic environment |
| `health_gathering_supreme` | Harder version of health gathering |
| `my_way_home` | Navigate a maze to find the goal |
| `predict_position` | Track and shoot a moving target |
| `deathmatch` | Free-for-all deathmatch |

### Campaign Mode

Play through the full Doom game using the bundled Freedoom WADs:

```jsonc
// Doom II campaign (32 maps)
{"wad": "freedoom2", "map_name": "MAP01"}

// Doom I campaign (36 maps)
{"wad": "freedoom1", "map_name": "E1M1"}
```

When a level is completed, `new_episode()` automatically advances to the next map. On death, it restarts the current map. The state includes `level_completed`, `player_dead`, and `next_map` hints to guide the agent.

## Architecture

```
AI Agent <──MCP──> FastMCP Server <──Python API──> ViZDoom Engine
                        │
              ┌─────────┼─────────┐
              │         │         │
          server.py  game_manager  state.py
          (tools)    (lifecycle)   (extraction)
              │         │         │
          actions.py  scenarios.py  objects.py
          (buttons)   (configs)    (enemy DB)
```

- **`server.py`** -8 MCP tool definitions (thin wrappers)
- **`game_manager.py`** -game lifecycle, action execution, state management
- **`state.py`** -screenshot conversion, object enrichment, depth/sector extraction
- **`actions.py`** -button name-to-enum mapping
- **`scenarios.py`** -scenario registry
- **`objects.py`** -database of 70+ Doom entity types

## Development

### Running Tests

```bash
# Unit tests (no ViZDoom runtime needed)
pytest tests/ -k "not integration"

# Integration tests (requires ViZDoom)
pytest tests/ -m integration

# All tests
pytest tests/
```

### Project Structure

```
tetsuo-doom/
├── src/doom_mcp/
│   ├── __init__.py
│   ├── server.py          # MCP tool definitions
│   ├── game_manager.py    # Core game lifecycle
│   ├── actions.py         # Button mapping
│   ├── scenarios.py       # Scenario registry
│   ├── state.py           # State extraction
│   └── objects.py         # Enemy database
├── tests/
│   ├── test_actions.py
│   ├── test_state.py
│   ├── test_scenarios.py
│   ├── test_objects.py
│   ├── test_game_manager.py    # Integration
│   └── test_server_integration.py
├── pyproject.toml
└── .mcp.json              # MCP client config
```

## Known Limitations

- **Single client** -one game instance at a time (no multiplayer MCP sessions this will be added very soon)
- **Delta values scale with tics** -use `tics=1` for precise aiming
- **Freedoom assets** -bundled WADs use Freedoom (free replacement art/levels), not original id Software assets
- **Headless rendering** -requires OpenGL libraries even without a visible window

## License

MIT
