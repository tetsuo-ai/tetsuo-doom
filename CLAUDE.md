# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MCP (Model Context Protocol) server that lets AI agents play Doom via ViZDoom. The server wraps ViZDoom's Python API, exposing game control, state observation, and action execution as MCP tools. AI agents receive game screenshots as images and structured game state data, then send named actions back.

## Tech Stack

- **Python 3.10+**
- **fastmcp** — MCP server SDK (decorator-based tools, Image return type)
- **vizdoom 1.3.0** — Doom engine with AI research API
- **Pillow** — numpy screen buffer to PNG conversion
- **numpy** — buffer handling

## Architecture

```
AI Agent <--MCP (stdio/SSE)--> server.py (FastMCP tools)
                                    ↓ delegates to
                               game_manager.py (GameManager singleton)
                                    ↓ calls
                               ViZDoom DoomGame API
```

### Module Layout
```
src/doom_mcp/
  __init__.py          # Package init
  server.py            # FastMCP server + 8 tool definitions (thin wrappers)
  game_manager.py      # DoomGame wrapper with lifecycle guards
  actions.py           # Button name↔enum mapping, action list construction
  scenarios.py         # Scenario registry (maps names to .cfg files)
  state.py             # Screenshot conversion (numpy→PNG), state extraction
```

- **server.py**: Module-scope `GameManager` singleton + `atexit` cleanup. `mcp` variable for FastMCP auto-discovery.
- **game_manager.py**: Owns `DoomGame`, manages lifecycle. All pre-`init()` config explicit.
- **actions.py**: `names_to_action_list(["MOVE_FORWARD", "ATTACK"], buttons)` → `[0, 0, 1, 1]`
- **scenarios.py**: Maps scenario names to ViZDoom's built-in `.cfg` files via `game.load_config()`
- **state.py**: `screen_buffer_to_png()` and game variable/object extraction

### MCP Tools (8 total)
`start_game`, `get_state`, `take_action`, `get_objects`, `get_map`, `new_episode`, `get_available_actions`, `stop_game`

### Key Design Points
- ViZDoom pybind11 enums use `__members__` (not iterable with `for x in Enum`)
- `Image` import: `from fastmcp.utilities.types import Image`
- `ToolError` import: `from fastmcp.exceptions import ToolError`
- Binary buttons only (no delta/continuous) in v1
- Single game instance (no multi-client support)

## Key ViZDoom Constraints

- All configuration (`set_screen_resolution`, `set_available_buttons`, etc.) must happen **before** `init()`
- `get_state()` returns `None` when episode is finished — always check `is_episode_finished()` first
- Use `ScreenFormat.RGB24` for `(H,W,3)` numpy arrays (not default `CRCGCB` which is channels-first)
- Set `episode_start_time=14` to skip weapon-raise animation
- `living_reward` is multiplied by the number of tics in `make_action(action, tics)`
- `make_action([])` is a valid no-op
- Default resolution: `RES_320X240` (keeps base64 payload small)

## Development Commands

```bash
# Create venv and install (editable + dev deps)
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run unit tests only (no ViZDoom runtime needed)
pytest tests/ -k "not integration" -v

# Run integration tests only (requires ViZDoom)
pytest tests/ -m integration -v

# Run the MCP server (stdio transport for Claude Code)
fastmcp run src/doom_mcp/server.py

# Run with SSE transport (for web clients)
fastmcp run src/doom_mcp/server.py --transport sse

# Verify install
python -c "import doom_mcp; import vizdoom; import fastmcp; print('OK')"
```

## MCP Client Configuration

To use with Claude Code, add to MCP settings:
```json
{
  "mcpServers": {
    "doom": {
      "command": "fastmcp",
      "args": ["run", "src/doom_mcp/server.py"],
      "cwd": "/path/to/tetsuo-doom"
    }
  }
}
```
