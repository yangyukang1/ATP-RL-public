from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def inspect_state_dict(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            "Safe weights-only loading failed. The file may be a full PBT pickle checkpoint. "
            "Do not use unrestricted torch.load unless the local source is explicitly trusted."
        ) from exc
    if not isinstance(payload, dict):
        raise TypeError("Legacy state-dict file must contain a dictionary")

    tensors = []
    for key, value in payload.items():
        if torch.is_tensor(value):
            tensors.append({"key": str(key), "shape": list(value.shape), "dtype": str(value.dtype)})
    first_linear = next((item for item in tensors if len(item["shape"]) == 2), None)
    return {
        "path": str(path),
        "tensor_count": len(tensors),
        "first_linear": first_linear,
        "tensors": tensors,
    }


def print_inspection(path: Path) -> None:
    print(json.dumps(inspect_state_dict(path), indent=2, ensure_ascii=False))
