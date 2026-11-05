import pytest
from huggingface_hub.hf_api import RepoFile

from scripts.gr1 import download_dataset


class _FakeApi:
    def __init__(self, entries):
        self.entries = entries
        self.calls = []

    def list_repo_tree(self, repo_id, **kwargs):
        self.calls.append((repo_id, kwargs))
        return self.entries


def test_list_task_files_scopes_the_hub_query_to_one_task():
    task = "gr1_unified.example_1000"
    entry = RepoFile(path=f"{task}/data/file.parquet", size=4, oid="blob")
    api = _FakeApi([entry])

    assert download_dataset._list_task_files(api, task, "revision") == [entry]
    assert api.calls == [
        (
            download_dataset.REPO_ID,
            {
                "path_in_repo": task,
                "recursive": True,
                "revision": "revision",
                "repo_type": "dataset",
            },
        )
    ]


def test_download_file_replaces_only_lfs_pointers(tmp_path, monkeypatch):
    relative = "gr1_unified.example_1000/data/file.parquet"
    destination = tmp_path / relative
    destination.parent.mkdir(parents=True)
    destination.write_bytes(
        download_dataset.LFS_POINTER_HEADER
        + b"\noid sha256:deadbeef\nsize 4\n"
    )
    entry = RepoFile(path=relative, size=4, oid="blob")

    def fake_download(repo_id, filename, **kwargs):
        assert repo_id == download_dataset.REPO_ID
        assert filename == relative
        assert not destination.exists()
        assert kwargs["force_download"] is False
        destination.write_bytes(b"PAR1")
        return str(destination)

    monkeypatch.setattr(download_dataset, "hf_hub_download", fake_download)
    download_dataset._download_file(
        entry,
        local_dir=tmp_path,
        revision="revision",
        endpoint=download_dataset.HF_ENDPOINT,
    )

    assert destination.read_bytes() == b"PAR1"


def test_remove_lfs_pointer_preserves_real_files(tmp_path):
    path = tmp_path / "episode.parquet"
    path.write_bytes(b"PAR1-real-data")

    download_dataset._remove_lfs_pointer(path)

    assert path.read_bytes() == b"PAR1-real-data"


def test_download_file_retries_download_error_and_missing_result(tmp_path, monkeypatch):
    relative = "gr1_unified.example_1000/data/file.parquet"
    destination = tmp_path / relative
    entry = RepoFile(path=relative, size=4, oid="blob")
    calls = []
    sleeps = []

    def fake_download(*args, **kwargs):
        calls.append(None)
        if len(calls) == 1:
            raise OSError("temporary download failure")
        if len(calls) == 3:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"PAR1")
        return str(destination)

    monkeypatch.setattr(download_dataset, "hf_hub_download", fake_download)
    monkeypatch.setattr(download_dataset.time, "sleep", sleeps.append)

    download_dataset._download_file(
        entry,
        local_dir=tmp_path,
        revision="revision",
        endpoint=download_dataset.HF_ENDPOINT,
        max_retries=2,
        retry_delay=0.25,
    )

    assert len(calls) == 3
    assert sleeps == [0.25, 0.5]


def test_download_file_stops_after_retry_limit(tmp_path, monkeypatch):
    entry = RepoFile(
        path="gr1_unified.example_1000/data/file.parquet",
        size=4,
        oid="blob",
    )
    calls = []

    def fake_download(*args, **kwargs):
        calls.append(None)
        raise OSError("persistent failure")

    monkeypatch.setattr(download_dataset, "hf_hub_download", fake_download)
    monkeypatch.setattr(download_dataset.time, "sleep", lambda _: None)

    with pytest.raises(OSError, match="persistent failure"):
        download_dataset._download_file(
            entry,
            local_dir=tmp_path,
            revision="revision",
            endpoint=download_dataset.HF_ENDPOINT,
            max_retries=2,
            retry_delay=0,
        )

    assert len(calls) == 3
