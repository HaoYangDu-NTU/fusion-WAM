"""Official VPP2 LIBERO policy boundary; simulator and model RNG stay separate."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "third_party/vpp2"
PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def validate_assets(directory):
    directory = Path(directory)
    names = ["libero/action_step030000.pt", "libero/video_step010000.pt", "libero/dataset_stats.json",
             "Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
             "Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
             "Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
             "Wan2.1-I2V-14B-480P/google/umt5-xxl/tokenizer.json"]
    for name in names:
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    if not (SOURCE / "src/vpp2/libero/runtime.py").is_file():
        raise FileNotFoundError(SOURCE)


class VPP2Policy:
    def __init__(self, directory, device="cuda:0"):
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        validate_assets(directory)
        sys.path.insert(0, str(SOURCE / "src"))
        from vpp2.checkpoint_compat import action_config
        from vpp2.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

        directory = Path(directory).resolve()
        action = directory / "libero/action_step030000.pt"
        payload = torch.load(action, map_location="cpu", mmap=True, weights_only=True)
        self.cfg = OmegaConf.create(action_config(payload))
        del payload
        cfg = self.cfg.model
        cfg.model_id = str(directory / "Wan2.1-I2V-14B-480P")
        cfg.tokenizer_model_id = cfg.model_id + "/google/umt5-xxl"
        cfg.dit_checkpoint_path = str(directory / "libero/video_step010000.pt")
        cfg.load_text_encoder = True
        cfg.tokenizer_max_len = 128
        cfg.skip_dit_load_from_pretrain = True
        cfg.action_dit_pretrained_path = None
        self.model = instantiate(cfg, model_dtype=torch.bfloat16, device=device)
        self.model.load_checkpoint(str(action), video_checkpoint_override=cfg.dit_checkpoint_path)
        self.model = self.model.to(device).eval().requires_grad_(False)
        # 128 rendering environments also use GPU memory. Keep UMT5 on CPU
        # between new prompts so the B16 video cache has ~11 GB more room.
        if self.model.text_encoder is not None:
            self.model.text_encoder.to("cpu")
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
        self.processor = instantiate(self.cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(directory / "libero/dataset_stats.json")))
        self.device = device
        self._text = {}

    def contexts(self, prompts):
        import torch
        missing = list(dict.fromkeys(p for p in prompts if p not in self._text))
        encoder = getattr(self.model, "text_encoder", None)
        if missing:
            if encoder is not None:
                encoder.to(self.device)
            try:
                for prompt in missing:
                    self._text[prompt] = tuple(t.cpu() for t in self.model.encode_prompt(prompt))
            finally:
                if encoder is not None:
                    encoder.to("cpu")
                    if torch.device(self.device).type == "cuda":
                        torch.cuda.empty_cache()
        return [self._text[p] for p in prompts]

    def prepare(self, observation):
        import torch
        from vpp2.libero.eval_helpers import compose_libero_camera_views
        images = observation["images"]
        rgb = compose_libero_camera_views(
            {"image": images["head_camera"], "wrist_image": images["left_wrist_camera"]},
            concatenation="horizontal", image_shapes=[m["shape"] for m in self.processor.shape_meta["images"]],
            num_cameras=2, preprocess_mode="training_exact")
        image = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device, self.model.torch_dtype)
        image = image * (2.0 / 255.0) - 1.0
        state = np.asarray(observation["state"], dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("VPP2 requires finite native 8D proprioception")
        key = self.processor.shape_meta["state"][0]["key"]
        state = {"state": {key: torch.from_numpy(state).unsqueeze(0)}}
        state = self.processor.normalizer.forward(self.processor.action_state_transform(state))["state"][key]
        return image, state

    def predict(self, observation):
        import torch
        with torch.inference_mode():
            image, state = self.prepare(observation)
            prompt = PROMPT.format(task=observation["prompt"])
            context, mask = self.contexts([prompt])[0]
            result = self.model.infer_action(
                prompt=None, context=context, context_mask=mask, input_image=image, proprio=state,
                action_horizon=32, num_video_frames=17, num_inference_steps=10,
                sigma_shift=5.0, seed=42, video_seed=43, rand_device="cpu", tiled=False,
                text_cfg_scale=1.0, negative_prompt="")
            key = self.processor.shape_meta["action"][0]["key"]
            normalized = result["action"].float().cpu()
            if normalized.ndim == 2:
                normalized = normalized.unsqueeze(0)
            actions = self.processor.normalizer.normalizers["action"][key].backward(
                normalized).numpy()[0]
        # Native reference: min/max denorm -> [0,1] to [-1,1] -> invert -> binarize.
        actions[..., -1] = np.sign(-(actions[..., -1] * 2 - 1))
        if actions.shape != (32, 7) or not np.isfinite(actions).all():
            raise ValueError(f"VPP2 returned invalid actions: shape={actions.shape}, finite={np.isfinite(actions).all()}")
        return {"actions": actions}

    def predict_batch(self, observations):
        import torch
        from benchmarks.vla_harness.vpp2_batch import infer_action_batch
        with torch.inference_mode():
            prepared = [self.prepare(obs) for obs in observations]
            pairs = self.contexts([PROMPT.format(task=obs["prompt"]) for obs in observations])
            contexts, masks = zip(*pairs)
            actions = infer_action_batch(self.model, torch.cat([p[0] for p in prepared]),
                torch.cat([p[1] for p in prepared]), torch.cat(contexts), torch.cat(masks))
            key = self.processor.shape_meta["action"][0]["key"]
            actions = self.processor.normalizer.normalizers["action"][key].backward(actions).numpy()
        actions[..., -1] = np.sign(-(actions[..., -1] * 2 - 1))
        if actions.shape != (len(observations), 32, 7) or not np.isfinite(actions).all():
            raise ValueError("Invalid batched VPP2 actions")
        return [{"actions": action} for action in actions]
