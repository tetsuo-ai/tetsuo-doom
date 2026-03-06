"""Game state extraction and screenshot conversion."""

import io
import math

import numpy as np
from PIL import Image as PILImage

import vizdoom as vzd

from .objects import get_object_info


def screen_buffer_to_png(buffer: np.ndarray) -> bytes:
    """Convert a ViZDoom RGB24 screen buffer (H,W,3) to PNG bytes."""
    img = PILImage.fromarray(buffer, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _relative_angle(px: float, py: float, pangle: float, tx: float, ty: float) -> float:
    """Compute angle from player to target, relative to player's facing direction.

    Returns degrees in [-180, 180]. Negative = target is to the LEFT,
    positive = target is to the RIGHT. 0 = directly ahead.
    """
    dx = tx - px
    dy = ty - py
    target_angle = math.degrees(math.atan2(dy, dx))
    # ViZDoom angle: 0=east, increases counter-clockwise
    diff = (target_angle - pangle + 180) % 360 - 180
    # Invert so positive = right (matches TURN_LEFT_RIGHT_DELTA convention)
    return -diff


def extract_game_variables(
    game: vzd.DoomGame,
    variable_names: list[str],
) -> dict[str, float]:
    """Extract named game variables from the current state."""
    result = {}
    for name in variable_names:
        var = getattr(vzd.GameVariable, name)
        result[name] = game.get_game_variable(var)
    return result


def extract_objects(
    state: vzd.GameState,
    player_x: float = 0.0,
    player_y: float = 0.0,
    player_angle: float = 0.0,
) -> list[dict]:
    """Extract objects with computed distance, angle, and type info."""
    if not state or not state.objects:
        return []

    # Build label lookup for visibility info
    label_by_id = {}
    if state.labels:
        for label in state.labels:
            label_by_id[label.object_id] = label

    results = []
    for obj in state.objects:
        dx = obj.position_x - player_x
        dy = obj.position_y - player_y
        distance = math.hypot(dx, dy)
        rel_angle = _relative_angle(
            player_x, player_y, player_angle,
            obj.position_x, obj.position_y,
        )

        info = get_object_info(obj.name)
        label = label_by_id.get(obj.id)

        entry = {
            "id": obj.id,
            "name": obj.name,
            # Computed fields
            "distance": round(distance, 1),
            "angle_to_aim": round(rel_angle, 1),
            # Type info from database
            "type": info["type"],
            "threat": info["threat"],
            "attack_type": info["attack"],
            "typical_hp": info["typical_hp"],
            "description": info["description"],
            # Raw position
            "position_x": obj.position_x,
            "position_y": obj.position_y,
            "position_z": obj.position_z,
            "angle": obj.angle,
            "pitch": obj.pitch,
            "velocity_x": obj.velocity_x,
            "velocity_y": obj.velocity_y,
            "velocity_z": obj.velocity_z,
            # Visibility (from labels buffer)
            "is_visible": label is not None,
        }

        if label is not None:
            entry["screen_x"] = label.x
            entry["screen_y"] = label.y
            entry["screen_width"] = label.width
            entry["screen_height"] = label.height

        results.append(entry)

    return results


def extract_sectors(state: vzd.GameState) -> list[dict]:
    """Extract sector geometry (floors, ceilings, wall lines)."""
    if not state or not state.sectors:
        return []
    return [
        {
            "floor_height": sector.floor_height,
            "ceiling_height": sector.ceiling_height,
            "lines": [
                {
                    "x1": line.x1,
                    "y1": line.y1,
                    "x2": line.x2,
                    "y2": line.y2,
                    "is_blocking": line.is_blocking,
                }
                for line in sector.lines
            ],
        }
        for sector in state.sectors
    ]


def extract_depth_as_stats(depth_buffer: np.ndarray) -> dict:
    """Summarize depth buffer as region stats for spatial awareness."""
    h, w = depth_buffer.shape
    third_w = w // 3
    mid_h = h // 2

    regions = {
        "far_left": depth_buffer[0:mid_h, 0:third_w],
        "far_center": depth_buffer[0:mid_h, third_w:2*third_w],
        "far_right": depth_buffer[0:mid_h, 2*third_w:w],
        "near_left": depth_buffer[mid_h:h, 0:third_w],
        "near_center": depth_buffer[mid_h:h, third_w:2*third_w],
        "near_right": depth_buffer[mid_h:h, 2*third_w:w],
    }

    cx, cy = w // 2, h // 2
    margin_x, margin_y = max(w // 20, 1), max(h // 20, 1)
    crosshair = depth_buffer[cy-margin_y:cy+margin_y, cx-margin_x:cx+margin_x]

    result = {}
    for name, region in regions.items():
        result[name] = {
            "min_dist": float(np.min(region)),
            "mean_dist": float(np.mean(region)),
        }
    result["crosshair"] = {
        "min_dist": float(np.min(crosshair)),
        "mean_dist": float(np.mean(crosshair)),
    }
    return result
