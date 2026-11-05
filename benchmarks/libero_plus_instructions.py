"""Optional filename-metadata removal, shared by Plus clients and cache export."""

import argparse
import re


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise argparse.ArgumentTypeError("Expected True or False")


def task_instruction(task, correct_instruction_suffix=False):
    """Preserve language perturbations; otherwise remove only variant metadata.

    Keep the original filename-derived LIBERO wording, rather than replacing
    it with potentially different BDDL wording. Never change task identity.
    """
    if not isinstance(correct_instruction_suffix, bool):
        raise TypeError("correct_instruction_suffix must be a boolean")
    if not correct_instruction_suffix or "_language_" in task.name:
        return task.language
    marker = re.search(r"_(?:(?:view|initstate|noise|table|tb|light|add)_\d|level\d|moved(?:_|$))", task.name)
    if marker is None:
        return task.language
    suffix = task.name[marker.start():].replace("_", " ")
    if not task.language.endswith(suffix):
        raise ValueError(f"Task language does not match its metadata suffix: {task.name}")
    return task.language[:-len(suffix)]
