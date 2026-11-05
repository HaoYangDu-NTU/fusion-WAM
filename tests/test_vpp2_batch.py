"""Sampler equivalence uses the real native VPP2 loop and scheduler on CPU."""
from collections import Counter
from types import SimpleNamespace
import sys
import time
import pytest
import torch
from benchmarks.vla_harness.vpp2_policy import SOURCE
from benchmarks.vla_harness.vpp2_batch import infer_action_batch


def model():
    sys.path.insert(0, str(SOURCE / "src"))
    from vpp2.models.wan21_14b.vpp2 import VPP2
    from vpp2.models.wan21_14b.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
    m = object.__new__(VPP2)
    torch.nn.Module.__init__(m)
    m.device = torch.device("cpu")
    m.torch_dtype = torch.float32
    m.video_expert = SimpleNamespace(video_attention_mask_mode="bidirectional", fuse_vae_embedding_in_latents=False)
    m.action_expert = SimpleNamespace(action_dim=7)
    m.action_chunk_size = 1
    m.proprio_dim = 8
    m.action_visible_video_frames = 3
    m.action_clean_first_frame_start_layer = None
    m.action_clean_first_frame_timestep_cutoff = None
    m.vae = SimpleNamespace(temporal_downsample_factor=4)
    m.proprio_encoder = torch.nn.Linear(8,4)
    m.train_video_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.)
    m.infer_action_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.)
    def encode(images):
        return images.mean((1,2,3))[:,None,None,None,None].expand(-1,16,1,1,1).clone()
    m._encode_input_image_latents_tensor = lambda input_image, **k: encode(input_image)
    m._encode_video_latents = lambda x, **k: encode(x[:,:,0])
    m._encode_clip_fea_from_image = lambda x: x.mean((1,2,3))
    def cache(**k):
        return [dict(condition=k['condition_latents'].mean((1,2,3,4)),
                     context=k['context'].mean((1,2)), noise=k['latents'].mean((1,2,3,4)))], None, 5
    m._build_frozen_video_cache = cache
    def prediction(**k):
        c = k['video_kv_cache'][0]
        return .1*k['latents_action'] + (c['condition']*.2+c['context']*.03+c['noise']*.04)[:,None,None]
    m._predict_action_noise_with_cache = prediction
    m._unpack_action_tokens = lambda x: x
    return m.eval()


@pytest.mark.parametrize("batch", [1,2,5])
def test_native_loop_per_sample_noise_and_context(batch):
    torch.manual_seed(19)
    m = model()
    images = torch.randn(batch,3,16,16)
    states = torch.randn(batch,8)
    contexts = torch.randn(batch,128,4)
    masks = torch.ones(batch,128,dtype=torch.bool)
    expected = torch.stack([m.infer_action(prompt=None, input_image=images[i:i+1], action_horizon=32,
        proprio=states[i:i+1], context=contexts[i:i+1], context_mask=masks[i:i+1],
        num_inference_steps=10, sigma_shift=5., seed=42, video_seed=43, rand_device="cpu", num_video_frames=17)['action']
        for i in range(batch)])
    actual = infer_action_batch(m, images, states, contexts, masks)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    if batch > 1:
        assert not torch.equal(actual[0], actual[-1])


def test_server_oom_split_preserves_order():
    from benchmarks.vla_harness.vpp2_model_server import VPP2ModelServer
    s = object.__new__(VPP2ModelServer)
    from benchmarks.vla_harness.execution import PhaseStats
    s.performance = PhaseStats("model")
    def predict(items):
        if len(items) > 4:
            raise torch.cuda.OutOfMemoryError("test memory boundary")
        return [{"id": i} for i in items]
    s.policy = SimpleNamespace(predict_batch=predict)
    s._micro_batch_size = 32
    s._requests = 0
    s._batch_counts = Counter()
    s._generation_seconds = 0.
    s._window_started = time.monotonic()
    outputs = s._predict(list(range(13)))
    assert outputs == [{"id": i} for i in range(13)]
    assert 1 <= s._micro_batch_size <= 4
