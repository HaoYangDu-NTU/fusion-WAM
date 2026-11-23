"""Parse the released benchmark inventory and its USD layers without GPU rendering."""
import argparse
from collections import Counter
import json
from pathlib import Path

from pxr import Sdf, Usd


def check(root):
    metadata = root / "robolab/tasks/_metadata/task_metadata.json"
    tasks = json.loads(metadata.read_text())
    assert len(tasks) == len({task["task_name"] for task in tasks}) == 120
    scenes = set()
    for task in tasks:
        assert (root / "robolab/tasks" / task["filename"]).is_file()
        assert all(task["instruction_variants"][k].strip() for k in ("default", "vague", "specific"))
        scenes.add(root / "assets/scenes" / task["scene"])
    paths = sorted(scenes) + [root / "assets/robots/franka_robotiq_2f_85_flattened.usd"]
    for path in paths:
        layer = Sdf.Layer.FindOrOpen(str(path.resolve()))
        assert layer and len(layer.rootPrims), path
    return {"tasks": len(tasks), "difficulty": dict(Counter(t["difficulty_label"] for t in tasks)),
            "scene_layers": len(scenes), "robot_layer": paths[-1].name, "openusd": Usd.GetVersion()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    print(json.dumps(check(parser.parse_args().checkout), indent=2))
