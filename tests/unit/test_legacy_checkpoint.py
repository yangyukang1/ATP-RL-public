from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from atprl.compat.legacy_checkpoint import inspect_state_dict


class LegacyCheckpointTests(unittest.TestCase):
    def test_safely_inspects_split_state_dict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            torch.save({"layer.weight": torch.zeros(4, 3), "layer.bias": torch.zeros(4)}, path)
            report = inspect_state_dict(path)
            self.assertEqual(report["tensor_count"], 2)
            self.assertEqual(report["first_linear"]["shape"], [4, 3])


if __name__ == "__main__":
    unittest.main()
