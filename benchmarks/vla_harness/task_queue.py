"""Attempt-local Plus task claims; durable episode results remain in the recorder.

Claims are not results. After a crash run_resumable builds a fresh attempt queue
from committed episodes, so a claimed but unfinished task is retried.
"""
from contextlib import closing
import json
import random
import sqlite3


def task_key(task):
    return json.dumps([task["suite"], task["task_id"]], separators=(",", ":"))


def initialize_queue(path, tasks):
    keys = [task_key(task) for task in tasks]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate tasks in dynamic queue")
    random.Random(42).shuffle(keys)
    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE pending (position INTEGER PRIMARY KEY, task TEXT UNIQUE NOT NULL)")
        db.executemany("INSERT INTO pending VALUES (?, ?)", enumerate(keys))
        db.commit()


class DynamicWorkItems:
    """Sized iterable for the harness; len/peek never consume a task.

    Per-worker progress has the global task count as its upper bound. Authoritative
    progress is the aggregate of completed episodes, not those local percentages.
    """
    def __init__(self, path, items):
        self.path = str(path)
        self.items = list(items)
        if any(ep != 0 for _, _, ep in self.items):
            raise ValueError("Dynamic Plus queue requires exactly trial zero")
        self.lookup = {task_key(item[1]): item for item in self.items}
        if len(self.lookup) != len(self.items):
            raise ValueError("Duplicate dynamic work items")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def __iter__(self):
        with closing(sqlite3.connect(self.path, timeout=60)) as db:
            while True:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT position, task FROM pending ORDER BY position LIMIT 1").fetchone()
                if row is None:
                    db.commit()
                    return
                db.execute("DELETE FROM pending WHERE position=?", (row[0],))
                db.commit()
                yield self.lookup[row[1]]
