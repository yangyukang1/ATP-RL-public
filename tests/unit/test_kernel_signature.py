from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

from atprl.cli import _kernel_signature
from atprl.config.kernel_signature import (
    KERNEL_DIMENSION_FIELDS,
    build_kernel_signature,
    checkpoint_compatibility_key,
    kernel_signature_from_config,
    pipeline_compatibility_key,
    read_kernel_signature,
    validate_checkpoint_metadata,
    validate_checkpoint_signature,
    write_kernel_signature,
)
from atprl.config.loader import load_config


ROOT = Path(__file__).resolve().parents[2]


class KernelSignatureTests(unittest.TestCase):
    def test_signature_sidecar_round_trip(self):
        signature = build_kernel_signature(
            n=5,
            k=3,
            topology="paper_n5",
            dqn_scheduler_impl="enumerative",
            dimensions={"scheduler_output_dim": 10, "power_action_dim": 4},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_kernel_signature(tmp, signature)
            self.assertEqual(path.name, "kernel_signature.json")
            self.assertEqual(read_kernel_signature(tmp), signature)
            self.assertEqual(checkpoint_compatibility_key(signature)["k"], 3)


if __name__ == "__main__":
    unittest.main()
