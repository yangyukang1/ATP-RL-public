import os
import sys
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
