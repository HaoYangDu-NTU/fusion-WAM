"""All model replicas load together; failures still clean up only owned groups."""

import json
from types import SimpleNamespace
from urllib.error import URLError
from urllib.request import proxy_bypass_environment

import pytest
import yaml

from benchmarks.vla_harness import run_eval


@pytest.mark.parametrize("failure", [None, "exit", "timeout"])
@pytest.mark.parametrize("export_steps", [False, True])
@pytest.mark.parametrize("client_queue", [False, True])
def test_parallel_model_readiness_and_cleanup(tmp_path, monkeypatch, failure, export_steps, client_queue):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.yaml").write_text("{}")
    for name in ("normalization_stats.npy", "model.safetensors"):
        (checkpoint / name).touch()
    (tmp_path / "libero/libero/assets").mkdir(parents=True)
    args = SimpleNamespace(
        benchmark="libero", ckpt_dir=checkpoint, ckpt_name="model.safetensors",
        libero_path=tmp_path, gpus="0,1", suites=",".join(run_eval.SUITES),
        task_ids="0", smoke=False, workers=2, max_batch_size=8, max_wait_time=0.01,
        render="gpu", render_gpus=None, base_port=0, output_dir=tmp_path / "run",
        eval_id="startup-test", image_format="raw", export_step_jsonl=export_steps,
        client_action_queue=client_queue,
    )
    args.base_port = 19400
    plan = run_eval.make_plan(args)
    # No actual network/GPU resources are used by these child-process doubles.
    class Socket:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def bind(self, *_): pass
    monkeypatch.setattr(run_eval.socket, "socket", Socket)
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "existing-upper.example")
    monkeypatch.setenv("no_proxy", "existing-lower.example")
    processes, ready, signals = [], set(), []
    clock = [0.0]
    monkeypatch.setattr(run_eval.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(run_eval.time, "sleep", lambda _: None)
    monkeypatch.setattr(run_eval.importlib.metadata, "version", lambda _: "3.3.2")

    class Process:
        def __init__(self, command, **kwargs):
            env = kwargs["env"]
            for key, existing in (("NO_PROXY", "existing-upper.example"), ("no_proxy", "existing-lower.example")):
                assert existing in env[key].split(",")
                for host in ("127.0.0.1", "localhost", "::1"):
                    assert proxy_bypass_environment(host, {"no": env[key]})
            self.pid = 90000 + len(processes)
            self.args = command
            self.worker = "benchmarks.vla_harness.worker" in command
            if self.worker:
                assert ready == {0, 1}, "Workers must await every replica"
            self.returncode = 0 if self.worker else None
            processes.append(self)
        def poll(self): return self.returncode
        def wait(self, **_): return self.returncode

    def kill_group(pid, sig):
        signals.append(pid)
        next(p for p in processes if p.pid == pid).returncode = -int(sig)
    monkeypatch.setattr(run_eval.subprocess, "Popen", Process)
    monkeypatch.setattr(run_eval.os, "killpg", kill_group)

    class Opener:
        def open(self, url, **_):
            assert len([p for p in processes if not p.worker]) == 2
            slot = 0 if ":19400/" in url else 1
            if failure == "timeout":
                clock[0] = 2401
                raise URLError("not ready")
            if failure == "exit":
                processes[1].returncode = 7
                raise URLError("not ready")
            ready.add(slot)
            return SimpleResponse()
    class SimpleResponse:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *_): pass
    monkeypatch.setattr(run_eval.urllib.request, "build_opener", lambda *_: Opener())
    class Monitor(run_eval.RuntimeMonitor):
        def __init__(self, output_dir, **_): super().__init__(output_dir)
        def sample(self, *_, **__): return {}
    monkeypatch.setattr(run_eval, "RuntimeMonitor", Monitor)
    monkeypatch.setattr(run_eval, "persist_recording_db", lambda *_: None)
    from vla_eval.results import merge
    exports = []
    # The installed API controls step export through recorder configuration,
    # not an export_step_jsonl argument to merge_db.
    monkeypatch.setattr(merge, "merge_db", lambda *_, **kw: exports.append(kw))
    monkeypatch.setattr(run_eval, "write_summary", lambda _: {"selected_run_complete": True})
    if failure:
        error = TimeoutError if failure == "timeout" else RuntimeError
        with pytest.raises(error):
            run_eval.launch(args, plan)
        assert not any(p.worker for p in processes)
        if failure == "exit":
            record = json.loads((args.output_dir / "process_failure.json").read_text())
            assert record["failed_processes"][0]["returncode"] == 7
    else:
        assert run_eval.launch(args, plan) == 0
        assert sum(p.worker for p in processes) == 2
    assert signals == ([90000] if failure == "exit" else [90000, 90001])
    assert all(p.poll() is not None for p in processes)
    cfg = yaml.safe_load((args.output_dir / "config.yaml").read_text())
    assert cfg["benchmarks"][0]["recording"] == {"record_step": export_steps, "record_video": False}
    assert cfg["benchmarks"][0]["episodes_per_task"] == 50
    assert json.loads((args.output_dir / "run.json").read_text())["export_step_jsonl"] is export_steps
    manifest = json.loads((args.output_dir / "run.json").read_text())
    assert manifest["execution"]["client_action_queue"] is client_queue
    assert "client.py" in manifest["execution"]["source_sha256"]
    assert exports == [{}]
