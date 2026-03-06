"""Unit tests for state extraction utilities."""

import math

import numpy as np

from doom_mcp.state import screen_buffer_to_png, _relative_angle, extract_depth_as_stats


def test_screen_buffer_to_png_basic():
    buffer = np.zeros((240, 320, 3), dtype=np.uint8)
    png = screen_buffer_to_png(buffer)
    assert isinstance(png, bytes)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_screen_buffer_to_png_colored():
    buffer = np.full((100, 100, 3), 128, dtype=np.uint8)
    png = screen_buffer_to_png(buffer)
    assert isinstance(png, bytes)
    assert len(png) > 0


def test_screen_buffer_to_png_preserves_dimensions():
    from PIL import Image as PILImage
    import io

    buffer = np.zeros((480, 640, 3), dtype=np.uint8)
    png = screen_buffer_to_png(buffer)
    img = PILImage.open(io.BytesIO(png))
    assert img.size == (640, 480)


def test_relative_angle_ahead():
    # Target directly east, player facing east (angle=0)
    angle = _relative_angle(0, 0, 0, 100, 0)
    assert abs(angle) < 1.0


def test_relative_angle_left():
    # Target north, player facing east -> target is to the left
    angle = _relative_angle(0, 0, 0, 0, 100)
    assert angle < 0  # negative = left


def test_relative_angle_right():
    # Target south, player facing east -> target is to the right
    angle = _relative_angle(0, 0, 0, 0, -100)
    assert angle > 0  # positive = right


def test_relative_angle_behind():
    # Target west, player facing east
    angle = _relative_angle(0, 0, 0, -100, 0)
    assert abs(abs(angle) - 180) < 1.0


def test_depth_stats_structure():
    depth = np.random.randint(0, 256, (240, 320), dtype=np.uint8)
    stats = extract_depth_as_stats(depth)
    assert "crosshair" in stats
    assert "near_left" in stats
    assert "far_center" in stats
    for region in stats.values():
        assert "min_dist" in region
        assert "mean_dist" in region
