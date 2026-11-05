"""Protocol, real harness transport/session, batching and result-integrity tests."""

import asyncio
import base64
import io
import json
from collections import defaultdict
from types import SimpleNamespace

import anyio
import numpy as np
import pytest
from PIL import Image
from vla_eval.connection import Connection
from vla_eval.model_servers.serve import serve_async
from vla_eval.recording import NullEpisodeRecorder
from vla_eval.specs import check_specs

from benchmarks.libero.openwam2libero_interface import OpenWAMLiberoPolicy, native_eef10_to_libero7d
from benchmarks.vla_harness import model_server
from benchmarks.vla_harness.libero import SUITES, OpenWAMLIBEROBenchmark, single_eval
from benchmarks.vla_harness.summarize import summarize
from openwam.deploy.obs_preprocess import ObsPreprocessor


@pytest.fixture
def raw_obs():
    rng = np.random.default_rng(7)
    quat = np.array([0.1, -0.2, 0.3, 0.9])
    quat /= np.linalg.norm(quat)
    return {
        "agentview_image": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([0.2, -0.1, 0.4]),
        "robot0_eef_quat": quat,
        "robot0_gripper_qpos": np.array([0.019, -0.02]),
    }


def preprocessor(seperate_encode_view=False):
    return ObsPreprocessor(
        multiview=True, camera_layout=["head", "left", None], img_height=384, img_width=320,
        requires_proprio=True, seperate_encode_view=seperate_encode_view,
    )


def action_chunk(value):
    actions = np.tile([0.2, -0.3, 0.4, 1, 0, 0, 1, 0, 0, 0.37], (12, 1)).astype(np.float32)
    actions[:, 0] = float(value) + np.arange(12) / 100
    return actions


class ReferenceClient:
    def __init__(self, seperate_encode_view=None):
        self.seperate_encode_view = seperate_encode_view

    def ping(self):
        result = {"type": "pong", "representation": "eef"}
        if self.seperate_encode_view is not None:
            result["seperate_encode_view"] = self.seperate_encode_view
        return result

    def predict(self, payload):
        self.payload = payload
        return {"action": action_chunk(0.2)[0].tolist()}


@pytest.mark.parametrize("separate", [None, False, True])
def test_native_handshake_advertises_only_independent_views(separate):
    from openwam.deploy.server import PolicyServer

    contract = {"representation": "eef", "action_dim": 80}
    encoder = SimpleNamespace()
    if separate is not None:
        encoder.seperate_encode_view = separate
    architecture = SimpleNamespace(
        repr_contract=contract, video_backbone=SimpleNamespace(video_encoder=encoder)
    )
    server = PolicyServer(SimpleNamespace(architecture=architecture), None)
    result = server._ckpt_contract()
    assert result == {**contract, **({"seperate_encode_view": True} if separate else {})}
    assert contract == {"representation": "eef", "action_dim": 80}


@pytest.mark.parametrize("image_format", ["png", "raw"])
@pytest.mark.parametrize("separate", [None, False, True])
def test_official_camera_pixels_state_canvas_and_action(raw_obs, monkeypatch, image_format, separate):
    from vla_eval.protocol import numpy_codec
    from vla_eval.protocol.messages import Message, MessageType, pack_message, unpack_message

    monkeypatch.setattr(numpy_codec, "_image_format", image_format)
    reference_client = ReferenceClient(separate)
    reference = OpenWAMLiberoPolicy(_client=reference_client)
    expected_action = reference.act(raw_obs, "pick up the bowl")
    benchmark = OpenWAMLIBEROBenchmark(seperate_encode_view=bool(separate))
    observation = benchmark.make_obs(raw_obs, {"language": "pick up the bowl"})
    observation = unpack_message(pack_message(Message(MessageType.OBSERVATION, observation))).payload
    expected = reference_client.payload
    for key in ("head_camera", "left_wrist_camera"):
        pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(expected["images"][key]))))
        np.testing.assert_array_equal(observation["images"][key], pixels)
    np.testing.assert_array_equal(observation["state"], expected["state"])
    actual_canvas = model_server.preprocess_observation(preprocessor(bool(separate)), observation)["image"]
    expected_canvas = preprocessor(bool(separate)).preprocess(expected)["image"]
    np.testing.assert_array_equal(np.asarray(actual_canvas), np.asarray(expected_canvas))
    for slot, camera, legacy_size in [
        ("head_camera", "agentview_image", (320, 256)),
        ("left_wrist_camera", "robot0_eye_in_hand_image", (160, 128)),
    ]:
        rotated = raw_obs[camera][::-1, ::-1]
        old_pixels = Image.fromarray(rotated).resize(legacy_size, Image.Resampling.LANCZOS)
        np.testing.assert_array_equal(observation["images"][slot], rotated if separate else np.asarray(old_pixels))
    if separate:
        # Same single BILINEAR resize used by the training reader, without a
        # preceding L-shape resize that would lose wrist detail.
        train_canvas = np.concatenate([
            np.asarray(Image.fromarray(raw_obs[key][::-1, ::-1]).resize((224, 224), Image.Resampling.BILINEAR))
            for key in ("agentview_image", "robot0_eye_in_hand_image")
        ], axis=1)
        np.testing.assert_array_equal(np.asarray(actual_canvas), train_canvas)
    np.testing.assert_array_equal(native_eef10_to_libero7d(action_chunk(0.2)[0]), expected_action)


def test_missing_wrist_is_rejected(raw_obs):
    del raw_obs["robot0_eye_in_hand_image"]
    with pytest.raises(KeyError):
        OpenWAMLIBEROBenchmark().make_obs(raw_obs, {"language": "pick"})


@pytest.mark.parametrize("suite", SUITES)
def test_native_reset_seed_settle_timeout_and_continuous_gripper(monkeypatch, raw_obs, suite):
    calls = []
    env = SimpleNamespace(
        seed=lambda n: calls.append(("seed", n)),
        reset=lambda: calls.append(("reset",)) or raw_obs,
        set_init_state=lambda s: calls.append(("init", int(s))) or raw_obs,
        step=lambda a: calls.append(("step", tuple(a))) or (raw_obs, 0.0, False, {}),
        close=lambda: calls.append(("close",)),
    )
    monkeypatch.setattr(single_eval, "_make_env_with_randomization_retries", lambda *args: env)
    benchmark = OpenWAMLIBEROBenchmark(suites=[suite])
    benchmark._recorder = NullEpisodeRecorder()
    task = {"suite": suite, "task_id": 0, "task_obj": None, "init_states": [20, 30]}
    for trial in range(2):
        benchmark.reset({**task, "episode_idx": trial})
    settle = tuple([0, 0, 0, 0, 0, 0, -1])
    assert (
        calls
        == [("seed", 42), ("reset",), ("init", 20)]
        + [("step", settle)] * 30
        + [("reset",), ("init", 30)]
        + [("step", settle)] * 30
    )
    with pytest.raises(ValueError, match="consecutive"):
        benchmark.reset({**task, "episode_idx": 3})
    value = np.array([0.2, -0.3, 0.4, 0, 0, 0, -0.37], dtype=np.float32)
    benchmark._steps = (700 if suite == "libero_10" else 600) - 1
    result = benchmark.step({"actions": value})
    np.testing.assert_array_equal(calls[-1][1], value)
    assert benchmark.check_done(result)
    assert benchmark.get_step_result(result) == {"success": False}
    benchmark.cleanup()
    assert calls[-1] == ("close",)


@pytest.mark.parametrize("workers", [1, 4, 16, 40])
def test_task_sharding_coverage_order_and_exclusive_rng(workers):
    items = [(task, {"name": str(task)}, trial) for task in range(40) for trial in range(50)]
    assigned = defaultdict(list)
    ownership = defaultdict(set)
    for worker in range(workers):
        for task, _, trial in OpenWAMLIBEROBenchmark.shard_work_items(items, workers, worker):
            assigned[task].append(trial)
            ownership[task].add(worker)
    assert set(assigned) == set(range(40))
    assert all(trials == list(range(50)) for trials in assigned.values())
    assert all(len(owners) == 1 for owners in ownership.values())


@pytest.mark.parametrize("history", [0, 1, 2])
@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("client_queue", [False, True])
def test_real_harness_batching_and_session_chunk_isolation(monkeypatch, raw_obs, history, separate, client_queue):
    from benchmarks.vla_harness.client import EvaluationConnection
    class FakeBatchEngine:
        def __init__(self, _):
            self.batches = []
            self.references = []

        def generate_batch(self, observations):
            self.batches.append([obs["prompt"] for obs in observations])
            for obs in observations:
                assert obs["image"].size == ((448, 224) if separate else (320, 384))
                assert obs["state"].shape == (10,)
                current = obs["image"].getpixel((10, 10))[0]
                if history:
                    self.references.append(tuple(image.getpixel((10, 10))[0] for image in obs["clean_condition_images"]))
                else:
                    assert "clean_condition_images" not in obs
            return [action_chunk(obs["prompt"]) for obs in observations]

    native = SimpleNamespace(
        engine=SimpleNamespace(architecture=SimpleNamespace(video_backbone=SimpleNamespace(
            video_encoder=SimpleNamespace(history_num_tubelets=int(history))))),
        _ckpt_contract=lambda: {"representation": "eef"}, _obs_preprocessor=preprocessor(separate)
    )
    monkeypatch.setattr(model_server, "load_native_server", lambda *args: native)
    monkeypatch.setattr(model_server, "BatchedOpenWAMEngine", FakeBatchEngine)
    server = model_server.OpenWAMModelServer("unused", max_batch_size=4, max_wait_time=0.2,
                                           client_action_queue=client_queue)
    benchmark = OpenWAMLIBEROBenchmark(seperate_encode_view=separate)
    assert (
        check_specs(
            server.get_action_spec(),
            benchmark.get_action_spec(),
            server.get_observation_spec(),
            benchmark.get_observation_spec(),
        )
        == []
    )
    observations = [benchmark.make_obs(raw_obs, {"language": str(0.1 * i)}) for i in range(4)]

    def at_step(i, step):
        return {**observations[i], "images": {
            key: np.full_like(value, i * 50 + step) if value is not None else None
            for key, value in observations[i]["images"].items()
        }}

    async def exchange():
        async with anyio.create_task_group() as tg:
            port = await tg.start(serve_async, server, "127.0.0.1", 0)
            client_cls = EvaluationConnection if client_queue else Connection
            conns = [client_cls(f"ws://127.0.0.1:{port}", timeout=10) for _ in range(4)]
            for conn in conns:
                await conn.connect(benchmark="openwam-libero-contract")
                await conn.start_episode({})
            for step in range(10):
                result = await asyncio.gather(*(c.act(at_step(i, step)) for i, c in enumerate(conns)))
                for i, response in enumerate(result):
                    np.testing.assert_array_equal(
                        response["actions"], native_eef10_to_libero7d(action_chunk(0.1 * i)[step])
                    )
            assert sum(map(len, server.engine.batches)) == 4
            assert max(map(len, server.engine.batches)) > 1
            await asyncio.gather(*(c.act(at_step(i, 10)) for i, c in enumerate(conns)))
            assert sum(map(len, server.engine.batches)) == 8
            # Reset one session after its first action in the second chunk.
            await conns[0].end_episode({"success": False})
            await conns[0].start_episode({})
            reset = await conns[0].act(at_step(0, 100))
            other = await conns[1].act(at_step(1, 11))
            np.testing.assert_array_equal(reset["actions"], native_eef10_to_libero7d(action_chunk(0)[0]))
            np.testing.assert_array_equal(other["actions"], native_eef10_to_libero7d(action_chunk(0.1)[1]))
            assert sum(map(len, server.engine.batches)) == 9
            if history:
                offsets = list(range(-4 * (2 * history - 1), 1, 4))
                expected = [tuple(i * 50 + max(0, step + delta) for delta in offsets)
                            for step in (0, 10) for i in range(4)]
                assert sorted(server.engine.references) == sorted(expected + [(100,) * (2 * history)])
            for conn in conns:
                await conn.end_episode({"success": False})
                await conn.close()
            # EPISODE_END has no wire ACK; wait for server-side cleanup.
            with anyio.fail_after(2):
                while server._episodes:
                    await anyio.sleep(0.01)
            assert not server._chunk_buffers
            assert not server._history_images
            tg.cancel_scope.cancel()

    anyio.run(exchange)


def _write_results(root, edit=None, smoke=False):
    tasks = [{"suite": suite, "task_id": t} for suite in SUITES for t in range(10)]
    (root / "run.json").write_text(json.dumps({"plan": {"tasks": tasks, "num_trials": 50, "smoke": smoke}}))
    records = [
        {
            "task": f"{t['suite']}/{t['task_id']}",
            "episodes": [{**t, "episode_id": ep, "metrics": {"success": ep % 2 == 0}} for ep in range(50)],
        }
        for t in tasks
    ]
    if edit:
        edit(records)
    (root / "OpenWAMLIBERO_aggregate.json").write_text(json.dumps({"tasks": records}))


def test_full_suite_success_rates(tmp_path):
    _write_results(tmp_path)
    result = summarize(tmp_path)
    assert result["selected_run_complete"]
    assert result["total_average_success_rate"] == 0.5
    assert all(r["success_rate"] == 0.5 for r in result["suites"])


@pytest.mark.parametrize("bad", ["missing", "duplicate", "error", "smoke", "unknown_extra"])
def test_incomplete_or_invalid_results_are_not_official_scores(tmp_path, bad):
    def edit(tasks):
        eps = tasks[0]["episodes"]
        if bad == "missing":
            eps.pop()
        elif bad == "duplicate":
            eps[1]["episode_id"] = 0
        elif bad == "error":
            eps[1]["failure_reason"] = "environment_error"
        elif bad == "unknown_extra":
            eps.append({"suite": "unknown", "task_id": 0, "episode_id": 0, "metrics": {"success": True}})

    _write_results(tmp_path, edit, smoke=bad == "smoke")
    assert summarize(tmp_path)["total_average_success_rate"] is None


@pytest.mark.parametrize("separate", [None, False, True])
def test_launcher_plan_covers_four_suites_across_four_replicas(tmp_path, separate):
    from benchmarks.vla_harness.run_eval import make_plan

    if separate is not None:
        (tmp_path / "config.yaml").write_text(
            f"model:\n  video_backbone:\n    encoder:\n      seperate_encode_view: {str(separate).lower()}\n"
        )
    args = SimpleNamespace(
        gpus="0,1,2,3",
        suites=",".join(SUITES),
        task_ids=None,
        smoke=False,
        workers=16,
        max_batch_size=4,
        max_wait_time=0.01,
        render="gpu",
        render_gpus=None,
        base_port=8940,
        ckpt_dir=tmp_path,
        ckpt_name="model.safetensors",
        output_dir=tmp_path,
        eval_id="test",
    )
    plan = make_plan(args)
    assert plan["seperate_encode_view"] is bool(separate)
    assert len(plan["servers"]) == 4 and len(plan["workers"]) == 16
    assert [sum(w["server"] == i for w in plan["workers"]) for i in range(4)] == [4] * 4
    selected = [(t["suite"], t["task_id"]) for w in plan["workers"] for t in w["tasks"]]
    assert len(selected) == len(set(selected)) == 40
    assert set(selected) == {(s, t) for s in SUITES for t in range(10)}
    args.smoke, args.suites, args.workers = True, "libero_spatial", 16
    plan = make_plan(args)
    assert len(plan["workers"]) == len(plan["servers"]) == 2
    assert plan["num_trials"] == 1
