"""Production population assembly contract.

This test intentionally checks configuration only; it does not require local
checkpoint files.  The runtime loader owns the default production population,
so research-only arrays must not silently replace it.
"""

import unittest

from neuroplex.core.model_loader import DEFAULT_NEURON_IDS


EXPECTED_DIALOGUE_IDS = [
    "zh_aug0_dialogue",
    "zh_aug1_dialogue",
    "zh_aug2_dialogue",
    "zh_aug3_dialogue",
    "zh_std0_dialogue",
]


class TestProductionPopulationContract(unittest.TestCase):
    def test_default_population_is_five_dialogue_neurons(self):
        self.assertEqual(DEFAULT_NEURON_IDS, EXPECTED_DIALOGUE_IDS)
        self.assertEqual(len(DEFAULT_NEURON_IDS), 5)


if __name__ == "__main__":
    unittest.main()
