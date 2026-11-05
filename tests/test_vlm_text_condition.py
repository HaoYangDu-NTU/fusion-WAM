import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors.torch import save_file

from openwam.text_condition.config import cache_directory, cache_from_config, text_spec, adapt_text_projection
from openwam.text_condition.encoder import select_hidden_states, VLMTextEncoder
from openwam.model.video_backbone.wan.text_embedding_cache import TextEmbeddingCache
from openwam.model.video_backbone.wan.models.dit import WanModel
from openwam.model.video_backbone.wan_backbone import Wan22Ti2v
from openwam.model.action_backbone.separate_action_dit import ActionDiT


@pytest.mark.parametrize("kind,width,layers", [("cosmos3-nano",4096,36),("qwen3p5-4b",2560,32),("cosmos3-edge",2048,28),
                                            ("rynnbrain1p1-2b",2048,24),("rynnbrain1p1-9b",4096,32)])
@pytest.mark.parametrize("method", ["avg_all_layers", "last_layer", "concat_all_layers"])
def test_cache_identity_shape_mask_and_checkpoint_projection(tmp_path, kind, width, layers, method):
    spec = text_spec(kind, method)
    d = width * layers if method == "concat_all_layers" else width
    assert spec["shape"] == [128, d]
    mask = torch.arange(128) < 7
    context = torch.randn(128, d, dtype=torch.bfloat16).masked_fill(~mask[:, None], 0)
    save_file({"context": context, "attention_mask": mask, "seq_len": mask.sum()}, str(tmp_path/'text.safetensors'))
    (tmp_path/'manifest.json').write_text(json.dumps({"version":2, **spec, "padding_side":"right", "prompts":{"hi":"text.safetensors"}}))
    cfg = {"text_encoder_type":kind, "vlm_text_embedding_method":method, "text_embedding_dir":str(tmp_path)}
    loaded, length = cache_from_config(cfg).encode(["hi"], device="cpu")
    assert torch.equal(loaded[0], context) and length.tolist() == [7]
    with pytest.raises(ValueError, match="mismatch"):
        TextEmbeddingCache(tmp_path, encoder_type=kind, method="last_layer" if method != "last_layer" else "avg_all_layers")
    # Real video/action projections have fixed output widths despite differing raw input widths.
    dit = WanModel(dim=16, in_dim=4, ffn_dim=32, out_dim=4, text_dim=d,
                   eps=1e-6, freq_dim=16, num_heads=2, num_layers=1, has_image_input=False, patch_size=(1,1,1))
    holder = SimpleNamespace(dit=dit, dit2=None, text_embedding_cache=cache_from_config(cfg))
    adapt_text_projection(holder, cfg)
    video = Wan22Ti2v(holder)
    assert video.text_dim == d
    action = ActionDiT(action_dim=7, dim=16, ffn_dim=32, num_heads=2, num_layers=1,
                       video_dim=16, bridge_layers=[0], text_dim=video.text_dim)
    with torch.no_grad():
        assert dit.text_embedding(loaded.float()).shape == (1,128,16)
        projected, attn_mask = action._prepare_context(loaded.float(), mask[None], batch_size=1, seq_len=2, device=torch.device('cpu'), dtype=torch.float32)
    assert projected.shape == (1,128,16)
    assert attn_mask.shape == (1,2,128) and not attn_mask[:,:,7:].any()
    restored = WanModel(dim=16, in_dim=4, ffn_dim=32, out_dim=4, text_dim=d,
                        eps=1e-6, freq_dim=16, num_heads=2, num_layers=1, has_image_input=False, patch_size=(1,1,1))
    restored.load_state_dict(dit.state_dict(), strict=True)


def test_aggregation_excludes_token_embeddings_and_layernorm_per_layer():
    h = (torch.full((2,128,4), 10000.), torch.randn(2,128,4)*4+7, torch.randn(2,128,4)*2-9)
    normalized = [F.layer_norm(x, (4,), eps=1e-5) for x in h[1:]]
    torch.testing.assert_close(select_hidden_states(h,"avg_all_layers"), sum(normalized)/2)
    torch.testing.assert_close(select_hidden_states(h,"concat_all_layers"), torch.cat(normalized,-1))
    assert select_hidden_states(h,"last_layer") is h[-1]


def test_invalid_mask_and_encoder_are_rejected(tmp_path):
    spec = text_spec("qwen3p5-4b")
    (tmp_path/'manifest.json').write_text(json.dumps({"version":2, **spec,"padding_side":"right","prompts":{"hi":"x.safetensors"}}))
    save_file({"context":torch.zeros(128,2560,dtype=torch.bfloat16),"attention_mask":torch.ones(128,dtype=torch.bool),"seq_len":torch.tensor(3)}, str(tmp_path/'x.safetensors'))
    with pytest.raises(ValueError, match="right-padded"):
        cache_from_config({"text_encoder_type":"qwen3p5-4b","text_embedding_dir":str(tmp_path)}).encode(["hi"],device="cpu")
    with pytest.raises(ValueError): text_spec("invalid")
    with pytest.raises(ValueError): text_spec("qwen3p5-4b","invalid")


def test_legacy_umt5_preserved_and_new_config_crops(tmp_path):
    context = torch.randn(512,4096,dtype=torch.bfloat16)
    save_file({"context":context,"seq_len":torch.tensor(32)},str(tmp_path/'x.safetensors'))
    (tmp_path/'manifest.json').write_text(json.dumps({"version":1,"shape":[512,4096],"prompts":{"hi":"x.safetensors"}}))
    assert cache_from_config({"text_embedding_dir":str(tmp_path)}).encode(["hi"],device="cpu")[0].shape == (1,512,4096)
    new = cache_from_config({"text_encoder_type":"umt5-xxl","text_embedding_dir":str(tmp_path)}).encode(["hi"],device="cpu")[0]
    assert torch.equal(new[0], context[:128])
    with pytest.raises(ValueError): cache_from_config({"text_encoder_type":"cosmos3-nano","text_embedding_dir":str(tmp_path)})


def test_paths_separate_methods_and_plus_instruction_modes():
    paths = {cache_directory(k,m,b,corrected=c) for k in ("cosmos3-nano","qwen3p5-4b","cosmos3-edge","rynnbrain1p1-2b","rynnbrain1p1-9b")
             for m in ("last_layer","avg_all_layers","concat_all_layers")
             for b,c in (("libero",False),("libero-plus",False),("libero-plus",True))}
    assert len(paths) == 45
    assert str(cache_directory()) == 'data/umt5-embedding/libero'


def test_cosmos_conversion_selects_understanding_attention_only():
    spec = importlib.util.spec_from_file_location('converter',Path(__file__).parents[1]/'scripts/convert_cosmos3_reasoner.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    assert mod.reasoner_key('layers.0.self_attn.to_q.weight','transformer/x') == 'model.language_model.layers.0.self_attn.q_proj.weight'
    assert mod.reasoner_key('layers.0.self_attn.add_q_proj.weight','transformer/x') is None
    assert mod.reasoner_key('layers.0.mlp_moe_gen.up_proj.weight','transformer/x') is None
    assert mod.reasoner_key('proj_in.weight','transformer/x') is None
    assert mod.reasoner_key('blocks.0.attn.qkv.weight','vision_encoder/x') == 'model.visual.blocks.0.attn.qkv.weight'


@pytest.mark.parametrize('backbone', ['wan22_style_0_5b', 'wan22_style_1_3b'])
@pytest.mark.parametrize('kind', ['cosmos3-nano', 'qwen3p5-4b', 'cosmos3-edge', 'rynnbrain1p1-2b', 'rynnbrain1p1-9b'])
@pytest.mark.parametrize('method', ['last_layer', 'avg_all_layers', 'concat_all_layers'])
def test_full_backbone_geometry_and_saved_config(tmp_path, monkeypatch, backbone, kind, method):
    from tests.test_wan22_style_capacity import launcher_config
    from openwam.model.video_backbone.wan22_style_backbone import Wan22Style
    from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder
    cfg = launcher_config(backbone=backbone)
    vb_cfg = cfg.model.video_backbone
    vb_cfg.text_encoder_type, vb_cfg.vlm_text_embedding_method = kind, method
    vb_cfg.text_embedding_dir = str(tmp_path)
    spec = text_spec(kind, method)
    (tmp_path/'manifest.json').write_text(json.dumps({'version':2, **spec, 'padding_side':'right',
                                                     'model_fingerprint':'test-revision', 'prompts':{}}))
    encoder = VJEPA21VideoEncoder(torch.nn.Linear(1,1), history_num_tubelets=0, embed_dim=1408, variant='vitg')
    monkeypatch.setattr(encoder, 'save_deploy_assets', lambda *a: None)
    vb = Wan22Style.from_pretrained(cfg, external_encoder=encoder, device='meta')
    assert vb.text_encoder is None
    assert vb.text_dim == spec['shape'][1]
    assert vb.dit.text_embedding[0].in_features == spec['shape'][1]
    assert vb.dit.text_embedding[-1].out_features == vb.dim
    vb.save_deploy_assets(str(tmp_path/'checkpoint'), cfg)
    saved = OmegaConf.to_container(vb_cfg, resolve=True)
    assert saved['text_encoder_type'] == kind and saved['vlm_text_embedding_method'] == method
    assert saved['text_embedding_fingerprint'] == 'test-revision'
    assert saved['components'][0]['extra_kwargs']['text_dim'] == spec['shape'][1]
    restored = Wan22Style.from_pretrained(saved, external_encoder=encoder, device='meta')
    assert {k:tuple(v.shape) for k,v in vb.state_dict().items()} == {k:tuple(v.shape) for k,v in restored.state_dict().items()}
    saved['text_embedding_fingerprint'] = 'wrong-revision'
    with pytest.raises(ValueError, match='fingerprint'):
        Wan22Style.from_pretrained(saved, external_encoder=encoder, device='meta')


@pytest.mark.parametrize('kind', ['cosmos3-nano', 'qwen3p5-4b', 'cosmos3-edge', 'rynnbrain1p1-2b', 'rynnbrain1p1-9b'])
@pytest.mark.parametrize('method', ['last_layer', 'avg_all_layers', 'concat_all_layers'])
def test_5b_projection_adapter_only_changes_input_linear(kind, method):
    d = text_spec(kind, method)['shape'][1]
    with torch.device('meta'):
        dit = WanModel(dim=3072,in_dim=48,ffn_dim=14336,out_dim=48,text_dim=4096,eps=1e-6,
                       freq_dim=256,num_heads=24,num_layers=30,has_image_input=False,patch_size=(1,2,2))
    original = dict(dit.named_parameters())
    holder = SimpleNamespace(dit=dit,dit2=None,text_embedding_cache=SimpleNamespace(shape=(128,d)))
    adapt_text_projection(holder, {'from_scratch':True})
    assert dit.text_embedding[0].weight.shape == (3072,d)
    for key, value in dit.named_parameters():
        if not key.startswith('text_embedding.0.'):
            assert value is original[key]


@pytest.mark.parametrize('kind', ['cosmos3-nano','qwen3p5-4b','cosmos3-edge','rynnbrain1p1-2b','rynnbrain1p1-9b'])
@pytest.mark.parametrize('method', ['avg_all_layers','last_layer','concat_all_layers'])
def test_plus_preparation_uses_training_encoder_method_and_fingerprint(tmp_path, monkeypatch, kind, method):
    import numpy as np
    import yaml
    from safetensors.numpy import save_file as save_numpy
    from benchmarks.vla_harness.prepare_libero_plus_svae import prepare_checkpoint
    from benchmarks.vla_harness.libero_plus_protocol import check_text_cache
    import openwam.text_condition.config as text_config
    source, cache = tmp_path/'source', tmp_path/'plus'
    source.mkdir();cache.mkdir()
    vb = {'text_encoder_type':kind,'vlm_text_embedding_method':method,'text_embedding_fingerprint':'revision',
          'text_embedding_dir':'/training/libero','encoder':{'name':'vjepa21','seperate_encode_view':True,'add_input_bottleneck':True},
          'from_scratch':True,'components':[{'attr':'dit'}]}
    (source/'config.yaml').write_text(yaml.safe_dump({'model':{'video_backbone':vb}}))
    (source/'manifest.json').write_text('{"embed_dim":4}')
    np.save(source/'normalization_stats.npy',{})
    save_numpy({'video_backbone.video_encoder.feature_norm.weight':np.ones(4,np.float32),
                'video_backbone.dit.patch_embedding.clean_proj.weight':np.zeros((8,4,1,2,2),np.float32),
                'video_backbone.dit.patch_embedding.proj1.weight':np.zeros((2,4,1,2,2),np.float32),
                'video_backbone.dit.view_embedding.weight':np.zeros((2,8),np.float32)},source/'checkpoint.safetensors')
    manifest={'version':2,**text_spec(kind,method),'padding_side':'right','model_fingerprint':'revision','prompts':{}}
    (cache/'manifest.json').write_text(json.dumps(manifest))
    def select(k,m,benchmark,**kwargs):
        assert (k,m,benchmark,kwargs['corrected']) == (kind,method,'libero-plus',True)
        return cache
    monkeypatch.setattr(text_config,'cache_directory',select)
    output=prepare_checkpoint(source,'checkpoint.safetensors',tmp_path/'deploy',corrected=True)
    saved=yaml.safe_load((output/'config.yaml').read_text())['model']['video_backbone']
    assert saved['text_embedding_dir'] == str(cache)
    assert saved['text_encoder_type'] == kind and saved['vlm_text_embedding_method'] == method
    assert check_text_cache(output,[],[])['mode'] == 'offline'
    manifest['vlm_text_embedding_method']='last_layer' if method!='last_layer' else 'avg_all_layers'
    (cache/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='vlm_text_embedding_method'):
        check_text_cache(output,[],[])
    with pytest.raises(ValueError,match='vlm_text_embedding_method'):
        prepare_checkpoint(source,'checkpoint.safetensors',tmp_path/'wrong',corrected=True)


def test_fingerprint_includes_nested_shards_and_preserves_flat_identity(tmp_path):
    import hashlib
    import os
    from openwam.text_condition.precompute import model_fingerprint
    (tmp_path/'config.json').write_text('{}')
    flat=tmp_path/'model.safetensors'
    flat.write_bytes(b'flat')
    legacy=hashlib.sha256()
    legacy.update(b'config.json');legacy.update(b'{}')
    stat=flat.stat()
    legacy.update(f'{flat.name}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
    assert model_fingerprint(tmp_path)==legacy.hexdigest()
    nested=tmp_path/'transformer/weights.safetensors'
    nested.parent.mkdir();nested.write_bytes(b'nested')
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'transformer/weights.safetensors','b':'transformer/weights.safetensors'}}))
    before=model_fingerprint(tmp_path)
    stat=nested.stat();os.utime(nested,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000))
    assert model_fingerprint(tmp_path)!=before
