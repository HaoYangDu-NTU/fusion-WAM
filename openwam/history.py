"""Shared episode-local history sampling and legacy config migration."""


def validate_history_num_tubelets(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("history_num_tubelets must be a non-negative integer")
    return value


def history_from_config(config, *, default=0):
    """Old saved configs retain their enabled/disabled history behavior."""
    def read(key):
        return config.get(key) if hasattr(config, "get") else getattr(config, key, None)

    value = read("history_num_tubelets")
    legacy = read("add_one_history_context")
    if legacy is not None and not isinstance(legacy, bool):
        raise ValueError("legacy add_one_history_context must be a boolean")
    if value is None:
        return int(legacy) if legacy is not None else default
    value = validate_history_num_tubelets(value)
    if legacy is not None and value != int(legacy):
        raise ValueError("Conflicting history_num_tubelets and legacy add_one_history_context")
    return value


def condition_frame_offsets(history_num_tubelets):
    """Chronological raw control-step offsets, including the current frame."""
    h = validate_history_num_tubelets(history_num_tubelets)
    return list(range(-4 * (2 * h - 1), 1, 4)) if h else [0]


def validate_condition_frame_offsets(offsets):
    offsets = tuple(offsets)
    if (not offsets or offsets[-1] != 0
            or any(isinstance(x, bool) or not isinstance(x, int) or x > 0 for x in offsets)
            or list(offsets) != sorted(set(offsets))):
        raise ValueError("clean_condition_frame_offsets must be increasing integer offsets ending at 0")
    return offsets


def select_condition_images(history, history_num_tubelets=0, *, frame_offsets=None):
    """Sample a per-step buffer, repeating episode frame 0 at the left boundary."""
    current = len(history) - 1
    offsets = condition_frame_offsets(history_num_tubelets) if frame_offsets is None else validate_condition_frame_offsets(frame_offsets)
    return [history[max(0, current + offset)] for offset in offsets]
