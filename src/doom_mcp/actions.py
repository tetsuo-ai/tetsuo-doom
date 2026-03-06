"""Button name to action list mapping for ViZDoom."""

import vizdoom as vzd

BUTTON_NAMES: dict[str, vzd.Button] = dict(vzd.Button.__members__)


def names_to_action_list(
    action_names: list[str],
    configured_buttons: list[vzd.Button],
) -> list[int]:
    """Convert human-readable button names to a ViZDoom action array.

    Args:
        action_names: Button names to activate (e.g. ["MOVE_FORWARD", "ATTACK"]).
            Case-insensitive. Empty list produces all-zeros (no-op).
        configured_buttons: The buttons configured on the game instance,
            in order. Only these buttons can be activated.

    Returns:
        List of 0s and 1s, one per configured button.

    Raises:
        ValueError: If a button name is not recognized or not configured.
    """
    button_set = {b.name: i for i, b in enumerate(configured_buttons)}
    action = [0] * len(configured_buttons)

    for name in action_names:
        upper = name.upper()
        if upper not in BUTTON_NAMES:
            raise ValueError(
                f"Unknown button {name!r}. "
                f"Valid buttons: {sorted(BUTTON_NAMES.keys())}"
            )
        if upper not in button_set:
            raise ValueError(
                f"Button {name!r} is not configured for this game. "
                f"Configured: {sorted(button_set.keys())}"
            )
        action[button_set[upper]] = 1

    return action
