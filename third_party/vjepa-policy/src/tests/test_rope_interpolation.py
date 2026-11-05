import unittest

import torch

from vjepa_policy.models.backbone.modules import Block, RoPEAttention


class RoPEInterpolationTest(unittest.TestCase):
    def test_rectangular_grid_positions_use_actual_width(self):
        attention = RoPEAttention(dim=96, num_heads=3)
        ids = torch.tensor([0, 27, 28, 391, 392, 783])

        depth, height, width = attention.separate_positions(
            ids,
            H_patches=14,
            W_patches=28,
        )

        torch.testing.assert_close(depth, torch.tensor([0, 0, 0, 0, 1, 1]))
        torch.testing.assert_close(height, torch.tensor([0, 0, 1, 13, 0, 13]))
        torch.testing.assert_close(width, torch.tensor([0, 27, 0, 27, 0, 27]))

    def test_rectangular_grid_endpoints_map_to_canonical_grid(self):
        attention = RoPEAttention(
            dim=96,
            num_heads=3,
            interpolate_rope=True,
            canonical_spatial_grid=(16, 16),
        )
        ids = torch.tensor([0, 27, 364, 391])
        _, height, width = attention.separate_positions(
            ids,
            H_patches=14,
            W_patches=28,
        )

        height, width = attention._scale_spatial_positions(
            height,
            width,
            H_patches=14,
            W_patches=28,
        )

        torch.testing.assert_close(height, torch.tensor([0.0, 0.0, 15.0, 15.0]))
        torch.testing.assert_close(width, torch.tensor([0.0, 15.0, 0.0, 15.0]))

    def test_default_path_does_not_scale_positions(self):
        attention = RoPEAttention(dim=96, num_heads=3)
        height = torch.tensor([0, 7, 13])
        width = torch.tensor([0, 14, 27])

        scaled_height, scaled_width = attention._scale_spatial_positions(
            height,
            width,
            H_patches=14,
            W_patches=28,
        )

        torch.testing.assert_close(scaled_height, height)
        torch.testing.assert_close(scaled_width, width)

    def test_patch_size_defaults_match_vjepa2_1(self):
        patch16 = RoPEAttention(dim=96, num_heads=3, interpolate_rope=True)
        patch14 = RoPEAttention(
            dim=96,
            num_heads=3,
            interpolate_rope=True,
            patch_size=14,
        )

        self.assertEqual(patch16.canonical_spatial_grid, (16, 16))
        self.assertEqual(patch14.canonical_spatial_grid, (18, 18))

    def test_block_passes_interpolation_configuration_to_attention(self):
        block = Block(
            dim=96,
            num_heads=3,
            use_rope=True,
            interpolate_rope=True,
            canonical_spatial_grid=(16, 20),
        )

        self.assertTrue(block.attn.interpolate_rope)
        self.assertEqual(block.attn.canonical_spatial_grid, (16, 20))

    def test_interpolation_preserves_attention_mask_and_kv_capture(self):
        attention = RoPEAttention(
            dim=96,
            num_heads=3,
            interpolate_rope=True,
            canonical_spatial_grid=(16, 16),
        ).eval()
        tokens = torch.randn(1, 8, 96)
        token_ids = torch.arange(8).unsqueeze(0)
        attention_mask = torch.ones(8, 8, dtype=torch.bool)
        captured = []

        with torch.no_grad():
            output = attention(
                tokens,
                mask=token_ids,
                attn_mask=attention_mask,
                T=1,
                H_patches=2,
                W_patches=4,
                kv_out=captured,
            )

        self.assertEqual(output.shape, tokens.shape)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][0].shape, (1, 3, 8, 32))
        self.assertEqual(captured[0][1].shape, (1, 3, 8, 32))

    def test_explicit_kv_return_matches_capture_hook(self):
        attention = RoPEAttention(dim=96, num_heads=3).eval()
        tokens = torch.randn(1, 8, 96)
        token_ids = torch.arange(8).unsqueeze(0)
        captured = []

        with torch.no_grad():
            output, key, value = attention(
                tokens,
                mask=token_ids,
                T=1,
                H_patches=2,
                W_patches=4,
                kv_out=captured,
                return_kv=True,
            )
            default_output = attention(
                tokens,
                mask=token_ids,
                T=1,
                H_patches=2,
                W_patches=4,
            )

        self.assertEqual(output.shape, tokens.shape)
        torch.testing.assert_close(output, default_output)
        torch.testing.assert_close(key, captured[0][0])
        torch.testing.assert_close(value, captured[0][1])


if __name__ == "__main__":
    unittest.main()
