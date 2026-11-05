import unittest

import numpy as np
from PIL import Image
import torch

from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder


class TestVJEPA21BatchPreprocessing(unittest.TestCase):
    def test_matches_original_for_every_pixel_value_and_preserves_order(self):
        # Every uint8 value appears in every channel; distinct frames/clips
        # catch channel, time, or batch transposition mistakes.
        base = np.arange(16 * 16 * 3, dtype=np.int64).reshape(16, 16, 3)
        clips = [[Image.fromarray(((base + 17 * b + 31 * t) % 256).astype(np.uint8))
                  for t in range(3)] for b in range(2)]
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                encoder = VJEPA21VideoEncoder(torch.nn.Linear(1, 1), history_num_tubelets=0, embed_dim=1, variant='test')
                encoder = encoder.to(device=device, dtype=torch.bfloat16)
                reference = torch.cat([encoder.preprocess_video(clip) for clip in clips])
                result = encoder.preprocess_video_batch(clips)
                self.assertEqual(result.shape, (2, 3, 3, 16, 16))
                self.assertTrue(result.is_contiguous())
                self.assertTrue(torch.equal(result, reference), (result - reference).abs().max().item())


if __name__ == '__main__':
    unittest.main()
