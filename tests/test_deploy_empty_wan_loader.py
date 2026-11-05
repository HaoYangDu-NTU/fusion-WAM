"""Checkpoint-first loading: same tensors, strict keys and real buffers."""
from types import SimpleNamespace
import torch
import pytest
from omegaconf import OmegaConf
from safetensors.torch import save_file
import openwam.model
from openwam.deploy import model_loader
from openwam.model.architectures.base import BaseWAMArchitecture

class SmallArchitecture(torch.nn.Module):
    load_checkpoint = BaseWAMArchitecture.load_checkpoint
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.register_buffer("positions", torch.arange(4.), persistent=False)
    def set_dtype_device(self, dtype, device):
        self.to(device=device, dtype=dtype)
    def attach_normalizer(self, normalizer):
        pass

@pytest.mark.parametrize("name,empty", [("wan22_style_0_5b", True), ("wan22_style_1_3b", True), ("wan22_ti2v_5b", True), ("other", False)])
@pytest.mark.parametrize("missing", [False, True])
def test_checkpoint_first_loader(monkeypatch, tmp_path, name, empty, missing):
    cfg = OmegaConf.create({"model":{"video_backbone":{"name":name,"encoder":{"name":"vjepa21"},
        "components":[{"attr":"vae","source":"state_dict"}], "text_embedding_dir":"cached"}},
        "training":{"mixed_precision":"bf16"}})
    OmegaConf.save(cfg, tmp_path/"config.yaml")
    original = SmallArchitecture()
    state = original.state_dict()
    if missing: del state["linear.bias"]
    save_file(state, str(tmp_path/"checkpoint_step_1.safetensors"))
    monkeypatch.setattr(openwam.model, "resolve_architecture_config", lambda _: SimpleNamespace(
        canonical=SimpleNamespace(framework="dual_system", variant="joint_self_attn"),
        registry_name="dual_system_joint_self_attn", params={"video_backbone": {}}))
    def build(*_):
        model = SmallArchitecture()
        assert model.linear.weight.is_meta == empty
        assert not model.positions.is_meta
        return model
    monkeypatch.setattr(openwam.model, "build_architecture", build)
    monkeypatch.setattr(model_loader, "_build_normalizer", lambda *_: None)
    if missing:
        with pytest.raises(RuntimeError, match="Strict load failed"):
            model_loader.load_from_checkpoint_dir(str(tmp_path), device="cpu")
    else:
        _, model = model_loader.load_from_checkpoint_dir(str(tmp_path), device="cpu")
        original.to(dtype=torch.bfloat16)
        for a,b in zip(model.parameters(),original.parameters()):
            assert torch.equal(a,b)
        assert torch.equal(model.positions,original.positions)
        x=torch.arange(4,dtype=torch.bfloat16)
        assert torch.equal(model.linear(x), original.linear(x))
