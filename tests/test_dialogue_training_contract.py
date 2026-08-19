"""Regression tests for dialogue SFT target counting."""
import unittest

import torch

from scripts.training.finetune_neuron_dialogue import effective_sft_mask


class TestEffectiveSftMask(unittest.TestCase):
    def test_excludes_unaligned_targets_from_loss_denominator(self):
        shift_targets = torch.tensor([[10, -100, 12, 13]])
        shift_mask = torch.tensor([[True, True, True, False]])
        shift_sft_mask = torch.tensor([[False, True, True, True]])

        result = effective_sft_mask(shift_targets, shift_mask, shift_sft_mask)

        self.assertEqual(result.tolist(), [[False, False, True, False]])
        self.assertEqual(int(result.sum()), 1)

    def test_keeps_only_valid_answer_positions(self):
        shift_targets = torch.tensor([[7, 8, 9]])
        shift_mask = torch.tensor([[True, True, True]])
        shift_sft_mask = torch.tensor([[False, True, True]])

        self.assertEqual(
            effective_sft_mask(shift_targets, shift_mask, shift_sft_mask).tolist(),
            [[False, True, True]],
        )


if __name__ == "__main__":
    unittest.main()
