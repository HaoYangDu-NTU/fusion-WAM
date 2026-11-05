"""Compare Plus behavior/aggregation directly to the OpenWAM reference client."""

import base64
import hashlib
import importlib
import io
import json
import random
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from vla_eval.recording import NullEpisodeRecorder

from benchmarks.vla_harness import libero_plus, libero_plus_env
from benchmarks.vla_harness.libero_plus_protocol import SUITES, check_text_cache, load_config, scheduler, task_catalog
from benchmarks.vla_harness.run_eval import make_plan
from benchmarks.vla_harness.summarize import summarize, write_summary


def official_single_eval():
    interface = importlib.import_module("benchmarks.libero-plus.openwam2libero_interface")
    previous = sys.modules.get("openwam2libero_interface")
    sys.modules["openwam2libero_interface"] = interface
    try:
        return importlib.import_module("benchmarks.libero-plus.single_eval")
    finally:
        if previous is None:
            sys.modules.pop("openwam2libero_interface", None)
        else:
            sys.modules["openwam2libero_interface"] = previous


@pytest.fixture
def raw_obs():
    rng = np.random.default_rng(17)
    return {
        "agentview_image": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.02, -0.02]),
    }


@pytest.mark.parametrize("separate", [None, False, True])
def test_plus_official_images_state_language_and_action(raw_obs, separate):
    from benchmarks.vla_harness.model_server import preprocess_observation
    from openwam.deploy.obs_preprocess import ObsPreprocessor

    interface = importlib.import_module("benchmarks.libero-plus.openwam2libero_interface")
    native_action = np.array([0.2, -0.3, 0.1, 1, 0, 0, 1, 0, 0, 0.37], dtype=np.float32)
    captured = {}

    def predict(payload):
        captured.update(payload)
        return {"action": native_action.tolist()}

    pong = {"type": "pong", "representation": "eef"}
    if separate is not None:
        pong["seperate_encode_view"] = separate
    policy = interface.OpenWAMLiberoPolicy(_client=SimpleNamespace(ping=lambda: pong, predict=predict))
    prompt = "Please move the bowl to the plate, carefully."
    expected_action = policy.act(raw_obs, prompt)
    bench = libero_plus.OpenWAMLIBEROPlusBenchmark(seperate_encode_view=bool(separate))
    bench._language = prompt
    obs = bench.make_obs(raw_obs, {"language": "must never use this training prompt"})
    assert obs["prompt"] == captured["prompt"] == prompt
    for slot in ["head_camera", "left_wrist_camera"]:
        expected = np.asarray(Image.open(io.BytesIO(base64.b64decode(captured["images"][slot]))))
        np.testing.assert_array_equal(obs["images"][slot], expected)
    np.testing.assert_array_equal(obs["state"], captured["state"])
    pre = ObsPreprocessor(
        multiview=True, camera_layout=["head", "left", None], img_height=384, img_width=320,
        requires_proprio=True, seperate_encode_view=bool(separate),
    )
    actual = preprocess_observation(pre, obs)
    expected = pre.preprocess(captured)
    np.testing.assert_array_equal(np.asarray(actual["image"]), np.asarray(expected["image"]))
    assert actual["image"].size == ((448, 224) if separate else (320, 384))
    if separate:
        for slot, key in [("head_camera", "agentview_image"), ("left_wrist_camera", "robot0_eye_in_hand_image")]:
            np.testing.assert_array_equal(obs["images"][slot], raw_obs[key][::-1, ::-1])
    from benchmarks.libero.openwam2libero_interface import native_eef10_to_libero7d

    np.testing.assert_array_equal(native_eef10_to_libero7d(native_action), expected_action)


@pytest.mark.parametrize("suite", SUITES)
@pytest.mark.parametrize("success", [False, True])
@pytest.mark.parametrize("correct", [False, True])
def test_official_reset_rng_settle_actions_and_termination(monkeypatch, raw_obs, suite, success, correct):
    reference = official_single_eval()
    trace = []
    action = np.array([0.2, -0.3, 0.1, 0, 0, 0, -0.37], dtype=np.float32)
    max_steps = 700 if suite == "libero_10" else 600
    policy_steps = 3 if success else max_steps
    cfg = {**load_config(), "suite": suite, "task_id": 12, "result_dir": None, "correct_instruction_suffix": correct}

    def seed(value):
        trace.append(("global_seed", value))
        random.seed(value)
        np.random.seed(value)

    def make_env(task, config):
        trace.append(("make_env", random.random(), float(np.random.rand())))
        steps = 0

        def step(value):
            nonlocal steps
            steps += 1
            trace.append(("step", tuple(value), random.random(), float(np.random.rand())))
            done = success and steps == 30 + policy_steps
            return raw_obs, float(done), done, {}

        return SimpleNamespace(
            seed=lambda *_: pytest.fail("Plus must not call env.seed"),
            reset=lambda: trace.append(("reset",)) or raw_obs,
            set_init_state=lambda s: trace.append(("init", s)) or raw_obs,
            step=step,
            close=lambda: trace.append(("close",)),
        )

    task = SimpleNamespace(name="put_bowl_on_plate_view_0_0_100_0_0_initstate_0",
                           language="put bowl on plate view 0 0 100 0 0 initstate 0")
    expected_prompt = "put bowl on plate" if correct else task.language

    def act(obs, prompt):
        assert prompt == expected_prompt
        return action.copy()
    source = SimpleNamespace(get_task=lambda index: task, get_task_init_states=lambda index: [123, 456])
    benchmark = SimpleNamespace(get_benchmark_dict=lambda: {suite: lambda: source})
    package = ModuleType("libero.libero")
    package.benchmark = benchmark
    monkeypatch.setitem(sys.modules, "libero", ModuleType("libero"))
    monkeypatch.setitem(sys.modules, "libero.libero", package)
    monkeypatch.setattr(reference, "_write_libero_config", lambda: None)
    monkeypatch.setattr(reference, "_control_seed", seed)
    monkeypatch.setattr(reference, "_make_env_with_randomization_retries", make_env)
    monkeypatch.setattr(
        reference,
        "OpenWAMLiberoPolicy",
        lambda **_: SimpleNamespace(reset=lambda: None, act=act, close=lambda: None),
    )
    assert reference.run_eval(cfg) == 0
    expected = list(trace)
    trace.clear()

    def modules():
        reference._control_seed(cfg["seed"])
        return reference, cfg, benchmark

    monkeypatch.setattr(libero_plus_env, "official_modules", modules)
    commands = iter([("step", action.tolist())] * policy_steps + [("close", None)])
    responses = []
    connection = SimpleNamespace(recv=lambda: next(commands), send=responses.append, close=lambda: None)
    libero_plus_env.serve(connection, suite, 12, correct)
    assert trace == expected
    assert len([t for t in trace if t[0] == "global_seed"]) == 1
    assert trace[0] == ("global_seed", 10000)
    assert responses[0]["language"] == expected_prompt
    assert responses[-1]["result"][2] == success

    bench = libero_plus.OpenWAMLIBEROPlusBenchmark(suites=[suite])
    bench._recorder = NullEpisodeRecorder()
    bench._env = make_env(task, cfg)
    bench._steps, bench._max_steps = max_steps - 1, max_steps
    result = bench.step({"actions": action})
    assert bench.check_done(result)
    assert bench.get_step_result(result) == {"success": False}
    with pytest.raises(ValueError, match="trial 0"):
        bench.reset({"episode_idx": 1})
    bench.cleanup()


@pytest.fixture
def catalog(tmp_path):
    root = tmp_path / "checkout"
    base = root / "libero/libero/benchmark"
    base.mkdir(parents=True)
    mapping, metadata = {}, {}
    for n, suite in enumerate(SUITES):
        mapping[suite] = [f"{suite}_task_{i}" for i in range(11 + n)]
        metadata[suite] = [
            {"id": i + 1, "name": name, "category": scheduler.PLUS_CATEGORY_ORDER[i % 7], "difficulty_level": i % 3 + 1}
            for i, name in enumerate(mapping[suite])
        ]
    (base / "libero_suite_task_map.py").write_text("libero_task_map = " + repr(mapping))
    (base / "task_classification.json").write_text(json.dumps(metadata))
    return root, task_catalog(root)


@pytest.mark.parametrize("separate", [None, False, True])
def test_plus_plan_covers_unequal_suite_sizes_once(catalog, tmp_path, separate):
    root, tasks = catalog
    if separate is not None:
        (tmp_path / "config.yaml").write_text(
            f"model:\n  video_backbone:\n    encoder:\n      seperate_encode_view: {str(separate).lower()}\n"
        )
    args = SimpleNamespace(
        benchmark="libero-plus",
        libero_path=root,
        task_ids=None,
        gpus="0,1,2,3",
        suites=",".join(SUITES),
        smoke=False,
        workers=32,
        max_batch_size=8,
        max_wait_time=0.01,
        render="gpu",
        render_gpus=None,
        base_port=8960,
        ckpt_dir=tmp_path,
        ckpt_name="model.safetensors",
        output_dir=tmp_path,
        eval_id="test",
    )
    plan = make_plan(args)
    assert plan["seperate_encode_view"] is bool(separate)
    assert plan["num_trials"] == 1 and plan["tasks"] == tasks
    assigned = [(t["suite"], t["task_id"]) for w in plan["workers"] for t in w["tasks"]]
    assert len(assigned) == len(set(assigned)) == 50
    assert len(plan["servers"]) == 4 and len(plan["workers"]) == 32
    args.task_ids = "12"
    with pytest.raises(ValueError, match="invalid task ids"):
        make_plan(args)


def write_records(root, catalog, edit=None, smoke=False):
    root.mkdir(exist_ok=True)
    source = root / "task_catalog.json"
    source.write_text(json.dumps(catalog))
    plan = {"benchmark": "libero-plus", "tasks": catalog, "num_trials": 1, "smoke": smoke}
    (root / "run.json").write_text(
        json.dumps({"plan": plan, "task_catalog_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    )
    episodes = [{**task, "episode_id": 0, "metrics": {"success": i % 3 != 0}} for i, task in enumerate(catalog)]
    if edit:
        edit(episodes)
    (root / "OpenWAMLIBEROPlus_aggregate.json").write_text(json.dumps({"tasks": [{"episodes": episodes}]}))
    return episodes


def test_full_summary_matches_official_weighted_task_suite_category_results(catalog, tmp_path, capsys):
    _, tasks = catalog
    output = tmp_path / "harness"
    records = write_records(output, tasks)
    result = write_summary(output)
    official_root = tmp_path / "official"
    trial = scheduler.TrialRun(0, 1)
    jobs, metadata = [], {}
    for row in records:
        job = scheduler.TaskJob(row["suite"], row["task_id"])
        jobs.append(job)
        metadata[job] = scheduler.TaskMetadata(row["task_name"], row["category"], row["difficulty_level"])
        directory = scheduler._run_output_dir(official_root, job, trial) / "attempt_01"
        directory.mkdir(parents=True)
        payload = {
            "suite": job.suite,
            "task_id": job.task_id,
            "trial_start": 0,
            "trial_stop": 1,
            "num_trials": 1,
            "successes": int(row["metrics"]["success"]),
            "trials": [{"trial": 0}],
        }
        (directory / "results.json").write_text(json.dumps(payload))
    expected, complete = scheduler._summarize(official_root, jobs, trial, metadata, write_files=False)
    assert complete and result["complete"] and result["selected_run_complete"]
    for key in ["category_results", "overall"]:
        assert result[key] == expected[key]
    for actual, reference in zip(result["task_results"], expected["task_results"]):
        assert actual == {key: value for key, value in reference.items() if key != "result_paths"}
    assert result["total_average_success_rate"] == result["overall"]["successes"] / 50


@pytest.mark.parametrize("bad", ["missing", "duplicate", "error", "unknown", "extra_trial", "smoke"])
def test_invalid_plus_results_never_get_a_full_score(catalog, tmp_path, bad):
    _, tasks = catalog

    def edit(rows):
        if bad == "missing":
            rows.pop()
        elif bad == "duplicate":
            rows.append(rows[0])
        elif bad == "error":
            rows[0]["failure_reason"] = "environment_error"
        elif bad == "unknown":
            rows.append({**rows[0], "suite": "unknown"})
        elif bad == "extra_trial":
            rows.append({**rows[0], "episode_id": 1})

    write_records(tmp_path, tasks, edit=edit, smoke=bad == "smoke")
    result = summarize(tmp_path)
    assert not result["complete"] and result["total_average_success_rate"] is None
    assert all(row["success_rate"] is None for row in result["category_results"])


def test_plus_offline_text_preflight_uses_exact_prompts_and_local_deployment_cache(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "model:\n  video_backbone:\n    text_embedding_dir: /missing/training/cache\n"
    )
    cache = tmp_path / "text_embedding"
    cache.mkdir()
    (cache / "manifest.json").write_text(json.dumps({"prompts": {"original instruction": "original.safetensors"}}))
    (cache / "original.safetensors").touch()
    task = {"suite": "libero_spatial", "task_id": 1}
    languages = [{**task, "language": "exact Plus instruction"}]
    with pytest.raises(ValueError, match="lacks 1/1 exact"):
        check_text_cache(tmp_path, [task], languages)
    (cache / "manifest.json").write_text(json.dumps({"prompts": {"exact Plus instruction": "plus.safetensors"}}))
    (cache / "plus.safetensors").touch()
    assert check_text_cache(tmp_path, [task], languages)["missing"] == 0


def test_category_selection_spans_suites_and_matches_workers(catalog, tmp_path, monkeypatch):
    root, tasks = catalog
    selected_names = ["Sensor Noise", "Camera Viewpoints"]
    expected = [t for t in tasks if t["category"] in selected_names]
    args = SimpleNamespace(
        benchmark="libero-plus", libero_path=root, task_ids=None, gpus="0,1,2,3",
        suites=",".join(SUITES), categories=",".join(selected_names), smoke=False,
        workers=8, max_batch_size=8, max_wait_time=0.01, render="gpu", render_gpus=None,
        base_port=8960, ckpt_dir=tmp_path, ckpt_name="model.safetensors",
        output_dir=tmp_path, eval_id="test",
    )
    plan = make_plan(args)
    assert plan["tasks"] == expected
    assert {t["suite"] for t in expected} == set(SUITES)
    assigned = [(t["suite"], t["task_id"]) for w in plan["workers"] for t in w["tasks"]]
    assert len(assigned) == len(set(assigned)) == len(expected)
    monkeypatch.setenv("LIBERO_PLUS_PATH", str(root))
    bench = libero_plus.OpenWAMLIBEROPlusBenchmark(categories=selected_names)
    assert bench.get_tasks() == expected
    args.workers = None
    automatic = make_plan(args)
    assert len(automatic["workers"]) == min(64, len(expected))
    assert {w["render_gpu"] for w in automatic["workers"]} == {"3"}
    args.smoke = True
    smoke_plan = make_plan(args)
    smoke_bench = libero_plus.OpenWAMLIBEROPlusBenchmark(
        task_ids=sorted({t["task_id"] for t in smoke_plan["tasks"]}), categories=selected_names)
    assert smoke_bench.get_tasks() == smoke_plan["tasks"]
    assert smoke_plan["tasks"] and all(t["category"] in selected_names for t in smoke_plan["tasks"])
    args.suites = SUITES[0]
    with pytest.raises(ValueError, match="--categories"):
        make_plan(args)
    args.suites = ",".join(SUITES)
    args.benchmark = "libero"
    with pytest.raises(ValueError, match="only to libero-plus"):
        make_plan(args)


@pytest.mark.parametrize("categories", [["unknown"], [], ["Sensor Noise", "Sensor Noise"]])
def test_invalid_categories_are_rejected(catalog, categories):
    with pytest.raises(ValueError, match="categories"):
        task_catalog(catalog[0], categories=categories)


def test_category_subset_summary_does_not_claim_full_benchmark(catalog, tmp_path):
    _, tasks = catalog
    records = write_records(tmp_path, tasks)
    selected = [t for t in tasks if t["category"] == "Sensor Noise"]
    manifest_path = tmp_path / "run.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["plan"]["tasks"] = selected
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "OpenWAMLIBEROPlus_aggregate.json").write_text(json.dumps({
        "tasks": [{"episodes": [r for r in records if r["category"] == "Sensor Noise"]}]
    }))
    result = write_summary(tmp_path)
    assert result["selected_run_complete"] and not result["complete"]
    assert result["total_average_success_rate"] is None
    assert [r["category"] for r in result["category_results"]] == ["Sensor Noise"]
    assert result["overall"]["trials"] == len(selected)
    assert "suite_results" not in result
    assert (tmp_path / "summary.csv").read_text().startswith("category,")
    assert not (tmp_path / "task_results.csv").exists()
    assert not (tmp_path / "category_success_rates.csv").exists()
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert "task_results" not in saved and "missing_runs" not in saved
    assert "Overall," in (tmp_path / "summary.csv").read_text()
    assert not (tmp_path / "suite_success_rates.csv").exists()


@pytest.mark.parametrize("fail_write", [False, True])
def test_plus_config_publication_is_atomic(tmp_path, monkeypatch, fail_write):
    import yaml
    reference = official_single_eval()
    repo = tmp_path / "repo"
    repo.mkdir()
    root = tmp_path / "config"
    root.mkdir()
    target = root / "config.yaml"
    previous = {"bddl_files": "previous-complete-path"}
    target.write_text(yaml.safe_dump(previous))
    monkeypatch.setenv("LIBERO_PLUS_PATH", str(repo))
    monkeypatch.setenv("LIBERO_PLUS_CONFIG_ROOT", str(root))
    dump = reference.yaml.safe_dump

    def paused_writer(data, handle, **kwargs):
        # A reader opens the published file while the new writer is active.
        assert yaml.safe_load(target.read_text()) == previous
        if fail_write:
            handle.write("partial:")
            raise OSError("interrupted write")
        return dump(data, handle, **kwargs)

    monkeypatch.setattr(reference.yaml, "safe_dump", paused_writer)
    if fail_write:
        with pytest.raises(OSError, match="interrupted write"):
            reference._write_libero_config()
        assert yaml.safe_load(target.read_text()) == previous
    else:
        reference._write_libero_config()
        assert yaml.safe_load(target.read_text())["bddl_files"] == str(repo / "libero/libero/bddl_files")
    assert list(root.glob(".config-*")) == []
