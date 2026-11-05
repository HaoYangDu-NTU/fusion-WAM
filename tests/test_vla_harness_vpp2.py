"""Verify the adapter's pixels, state, sampler arguments and physical actions."""
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from benchmarks.vla_harness.vpp2_policy import SOURCE, ROOT, PROMPT, VPP2Policy
from benchmarks.vla_harness.libero import make_observation


@pytest.fixture
def policy():
    sys.path.insert(0, str(SOURCE / "src"))
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from vpp2.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    cfg = OmegaConf.load(SOURCE / "configs/libero/train_action.yaml")
    p = object.__new__(VPP2Policy)
    p.processor = instantiate(cfg.data.train.processor).eval()
    p.processor.set_normalizer_from_stats(load_dataset_stats_from_json(
        str(SOURCE / "configs/libero/dataset_stats.json")))
    p.device = "cpu"
    p._text = {}
    p.model = SimpleNamespace(torch_dtype=torch.bfloat16)
    return p


def observation():
    raw = {"robot0_eef_pos": np.array([.1, .2, .3]),
           "robot0_eef_quat": np.array([0., 0., np.sin(.3), np.cos(.3)]),
           "robot0_gripper_qpos": np.array([.025, -.018]),
           "head": np.arange(256*256*3, dtype=np.uint8).reshape(256,256,3),
           "wrist": np.full((256,256,3), 123, dtype=np.uint8)}
    obs = make_observation(raw, "pick cup", {"policy_backend": "vpp2", "head_camera_key": "head",
        "left_wrist_camera_key": "wrist", "image_transform": "rotate_180"})
    return raw, obs


def test_native_pixels_and_proprio(policy):
    from vpp2.libero.eval_helpers import compose_libero_camera_views
    raw, obs = observation()
    np.testing.assert_array_equal(obs["images"]["head_camera"], raw["head"][::-1,::-1])
    np.testing.assert_allclose(obs["state"], [.1,.2,.3,0,0,.6,.025,-.018], atol=1e-7)
    image, state = policy.prepare(obs)
    rgb = compose_libero_camera_views({"image": raw["head"][::-1,::-1], "wrist_image": raw["wrist"][::-1,::-1]},
        concatenation="horizontal", image_shapes=[[3,224,224]]*2, num_cameras=2, preprocess_mode="training_exact")
    reference = torch.tensor(rgb).permute(2,0,1).unsqueeze(0).to(torch.bfloat16)*(2/255)-1
    torch.testing.assert_close(image, reference, rtol=0, atol=0)
    assert image.shape == (1,3,224,448)
    expected = policy.processor.normalizer.normalizers["state"]["default"].forward(torch.from_numpy(obs["state"])[None])
    torch.testing.assert_close(state, expected)


@pytest.mark.parametrize("batched", [False, True])
def test_sampler_seeds_gripper_once_and_prompt_cache(policy, batched):
    _, obs = observation()
    encoded, calls = [], []
    normalized = torch.linspace(-1,1,224).reshape(1,32,7)
    def encode(prompt):
        encoded.append(prompt)
        return torch.zeros(1,128,4096), torch.ones(1,128,dtype=torch.bool)
    def infer(**kwargs):
        calls.append(kwargs)
        return {"action": normalized.clone() if batched else normalized[0].clone()}
    policy.model.encode_prompt = encode
    policy.model.infer_action = infer
    actual = policy.predict(obs)["actions"]
    policy.predict(obs)
    assert encoded == [PROMPT.format(task="pick cup")]
    expected = policy.processor.normalizer.normalizers["action"]["default"].backward(normalized).numpy()[0]
    expected[:,-1] = np.sign(-(2*expected[:,-1]-1))
    np.testing.assert_array_equal(actual, expected)
    call = calls[0]
    assert call["prompt"] is None and call["context"].shape == (1,128,4096)
    assert (call["seed"],call["video_seed"],call["num_inference_steps"],call["sigma_shift"]) == (42,43,10,5)
    assert call["num_video_frames"] == 17 and call["action_horizon"] == 32


def test_plan_uses_single_sample_official_server(tmp_path):
    from benchmarks.vla_harness.run_eval import make_plan
    args = SimpleNamespace(policy_backend="vpp2", gpus="0,1", suites="libero_spatial", task_ids=None,
        smoke=False, workers=4, max_batch_size=1, max_wait_time=.01, render="gpu", render_gpus="3",
        base_port=15940, ckpt_dir=tmp_path, ckpt_name="libero/action_step030000.pt", output_dir=tmp_path, eval_id="vpp2")
    plan = make_plan(args)
    assert len(plan["tasks"]) == 10 and plan["num_trials"] == 50
    assert all("benchmarks.vla_harness.vpp2_model_server" in s["command"] for s in plan["servers"])
    args.max_batch_size = 2
    plan = make_plan(args)
    for server in plan["servers"]:
        cmd = server["command"]
        assert cmd[cmd.index("--args.max_batch_size")+1] == "2"


def test_loaded_model_is_moved_after_checkpoint(monkeypatch, policy, tmp_path):
    from hydra import utils
    from omegaconf import OmegaConf
    from benchmarks.vla_harness import vpp2_policy
    events = []
    class Model:
        text_encoder = None
        def load_checkpoint(self, *args, **kwargs): events.append("load")
        def to(self, device): events.append(("to", device)); return self
        def eval(self): return self
        def requires_grad_(self, value): return self
    cfg = OmegaConf.to_container(OmegaConf.load(SOURCE / "configs/libero/train_action.yaml"))
    monkeypatch.setattr(vpp2_policy, "validate_assets", lambda _: None)
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"format": "vpp2_wan21_action_v1", "resolved_config": cfg})
    monkeypatch.setattr(utils, "instantiate", lambda c, **k: Model() if "create_action_model" in c._target_ else policy.processor)
    from vpp2.datasets.lerobot.utils import normalizer
    monkeypatch.setattr(normalizer, "load_dataset_stats_from_json", lambda _: policy.processor.normalizer.stats)
    previous = torch.get_default_dtype()
    VPP2Policy(tmp_path, "cuda:0")
    assert events == ["load", ("to", "cuda:0")]
    assert torch.get_default_dtype() == previous


def test_prompt_cache_deduplicates_and_offloads_encoder(policy, monkeypatch):
    moves, encoded = [], []
    policy.device = "cuda:0"
    policy.model.text_encoder = SimpleNamespace(to=lambda device: moves.append(device))
    def encode(prompt):
        encoded.append(prompt)
        return torch.full((1,128,4),float(len(encoded))), torch.ones(1,128,dtype=torch.bool)
    policy.model.encode_prompt = encode
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    first = policy.contexts(["a","a"])
    second = policy.contexts(["b","a"])
    assert encoded == ["a","b"]
    assert moves == ["cuda:0","cpu","cuda:0","cpu"]
    assert first[0] is first[1] and second[1] is first[0]
