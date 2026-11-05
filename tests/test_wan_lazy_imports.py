"""Lazy optional imports retain native online tokenizer/model loading."""
from types import SimpleNamespace
import torch

def test_native_wan_loader_still_restores_weights():
    from openwam.model.video_backbone.wan.shared.core.loader.model import load_model
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    bias = torch.arange(3, dtype=torch.float32)
    model = load_model(torch.nn.Linear, None, config={"in_features":4,"out_features":3},
                       state_dict={"weight":weight,"bias":bias},
                       torch_dtype=torch.float32, device="cpu")
    assert torch.equal(model.weight, weight)
    assert torch.equal(model.bias, bias)
    assert torch.equal(model(torch.ones(4)), weight.sum(1)+bias)

def test_online_tokenizer_still_resolves_autotokenizer(monkeypatch):
    from transformers import AutoTokenizer
    from openwam.model.video_backbone.wan.models.text_encoder import HuggingfaceTokenizer
    calls=[]
    class Tokenizer:
        vocab_size=7
        def __call__(self, text, **kwargs):
            calls.append((text, kwargs))
            return SimpleNamespace(input_ids=[1,2], attention_mask=[1,1])
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda name, **kwargs: Tokenizer())
    tokenizer=HuggingfaceTokenizer("local-fixture",seq_len=8,clean="whitespace")
    assert tokenizer("hello   world",return_mask=True)==([1,2],[1,1])
    assert calls==[(["hello world"],{"return_tensors":"pt","padding":"max_length","truncation":True,"max_length":8})]
