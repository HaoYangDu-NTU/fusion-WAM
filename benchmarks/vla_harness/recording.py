"""Keep the harness's shared SQLite WAL on this host's local disk."""

import os
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from vla_eval.recording import RecordingStore, db_path_for_eval


def prepare_recording_db(output_dir, eval_id):
    # /workspace is DPC on this host; /tmp is node-local XFS. SQLite resolves
    # the link before opening its WAL/SHM sidecars, so all writers share XFS.
    work_root = Path(__file__).resolve().parents[2] / "vlaharness_tmp"
    if not work_root.exists():
        work_root.symlink_to(tempfile.mkdtemp(prefix="openwam-vlaharness-", dir="/tmp"), target_is_directory=True)
    if not work_root.is_symlink():
        raise RuntimeError(f"{work_root} must link to node-local storage for SQLite WAL")
    local_dir = Path(tempfile.mkdtemp(prefix="run-", dir=work_root))
    local_db = db_path_for_eval(local_dir, eval_id)
    RecordingStore(local_db).close()
    output_db = db_path_for_eval(output_dir, eval_id)
    output_db.symlink_to(local_db)
    return local_db, output_db


def persist_recording_db(local_db, output_db):
    # Called after all owned writers exit. backup() includes committed WAL
    # pages; replacing the link publishes a standalone DB on /workspace.
    temporary = output_db.with_suffix(".sqlite.tmp")
    # Finish/check the standalone snapshot on local disk. DPC/NFS random
    # SQLite I/O is much slower than a sequential copy of the closed file.
    with tempfile.TemporaryDirectory(dir=Path(local_db).parent) as directory:
        snapshot = Path(directory) / "recording.sqlite"
        with closing(sqlite3.connect(str(local_db))) as source:
            with closing(sqlite3.connect(str(snapshot))) as destination:
                source.backup(destination)
                destination.execute("PRAGMA journal_mode=DELETE")
                if destination.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise RuntimeError(f"Recording backup failed integrity_check: {temporary}")
        shutil.copyfile(snapshot, temporary)
    os.replace(temporary, output_db)
