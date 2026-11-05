"""Frozen V-JEPA rotation preserves eager rounding, strided views and prefixes."""
import unittest

import torch

from openwam.model.video_backbone.encoder.vjepa21_src.fused_rope import rotate_frozen, triton
from openwam.model.video_backbone.encoder.vjepa21_src.modules import rotate_queries_or_keys


@unittest.skipUnless(torch.cuda.is_available() and triton is not None, 'CUDA and Triton required')
class TestFusedRoPE(unittest.TestCase):
    def test_strided_rotation_matches_original_bitwise(self):
        torch.manual_seed(42)
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for n_cls, n_registers in ((0, 0), (1, 0), (0, 4), (1, 4)):
                with self.subTest(dtype=dtype, cls=n_cls, registers=n_registers):
                    tokens = 60 + n_cls + n_registers
                    qkv = torch.randn(2, tokens, 3, 4, 64, device='cuda', dtype=dtype)
                    qkv = qkv.permute(2, 0, 3, 1, 4)
                    pos = torch.arange(60, device='cuda').float() * (15 / 19)
                    for x in (qkv[0, :, :, :, 20:40], qkv[1, :, :, :, 20:40]):
                        expected = rotate_queries_or_keys(x, pos, n_registers, bool(n_cls)).to(dtype)
                        actual = rotate_frozen(x, pos, n_registers, bool(n_cls))
                        self.assertTrue(torch.equal(expected, actual),
                                        f'max error {(expected.float() - actual.float()).abs().max().item()}')

    def test_gradients_and_sample_specific_positions_use_original_path(self):
        x = torch.randn(2, 4, 60, 28, device='cuda', requires_grad=True)
        pos = torch.arange(60, device='cuda').float()
        self.assertIsNone(rotate_frozen(x, pos, 0, False))
        self.assertIsNone(rotate_frozen(x.detach(), pos.expand(2, 4, -1), 0, False))


if __name__ == '__main__':
    unittest.main()
