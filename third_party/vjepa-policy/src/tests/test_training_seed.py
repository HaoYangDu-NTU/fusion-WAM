import argparse
import os
import random
import unittest
from unittest import mock

import numpy as np
import torch

from vjepa_policy.train_common import add_common_args, seed_process


class TrainingSeedTest(unittest.TestCase):
    def test_common_training_seed_defaults_to_seven(self):
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        args = parser.parse_args(
            [
                "--dataset-dirs",
                "/tmp/dataset",
                "--text-cache-dir",
                "/tmp/text",
                "--checkpoint",
                "/tmp/vjepa2.pt",
            ]
        )
        self.assertEqual(args.seed, 7)

    @mock.patch.dict(os.environ, {"RANK": "2"})
    def test_seed_process_is_rank_specific_and_repeatable(self):
        self.assertEqual(seed_process(7), 9)
        first = (random.random(), np.random.rand(), torch.rand(1))

        self.assertEqual(seed_process(7), 9)
        second = (random.random(), np.random.rand(), torch.rand(1))

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))


if __name__ == "__main__":
    unittest.main()
