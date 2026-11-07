"""Atomic checkpoint publication with POSIX locking and storage reservation."""

import contextlib, hashlib, io, json, os, threading
from pathlib import Path
import torch


class StoragePressure(RuntimeError):
    pass


_held_locks = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def sync_dir(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_mkdir(path):
    path = Path(path)
    missing = []
    p = path
    while not p.exists():
        missing.append(p)
        p = p.parent
    for p in reversed(missing):
        p.mkdir(exist_ok=True)
        sync_dir(p.parent)
        sync_dir(p)


@contextlib.contextmanager
def lock(path, blocking=True):
    import fcntl

    path = Path(path)
    durable_mkdir(path.parent)
    key = (str(path.resolve()), os.getpid(), threading.get_ident())
    if key in _held_locks:
        _held_locks[key] += 1
        try:
            yield
        finally:
            _held_locks[key] -= 1
        return
    with path.open("a+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        _held_locks[key] = 1
        try:
            yield
        finally:
            del _held_locks[key]
            fcntl.flock(f, fcntl.LOCK_UN)


def reserve(path, n):
    margin = int(os.environ.get("SAC_V2_FREE_MARGIN_BYTES", str(8 * 2**30)))
    v = os.statvfs(Path(path).parent)
    if v.f_bavail * v.f_frsize < n + margin:
        raise StoragePressure(f"Need {n} bytes plus {margin} reserve at {path}")


def atomic_bytes(path, data, *, checked=True):
    path = Path(path)
    durable_mkdir(path.parent)
    volume_lock = os.environ.get("SAC_V2_VOLUME_LOCK", "/tmp/atprl_sac_v2_volume.lock")
    with lock(volume_lock):
        if checked:
            reserve(path, len(data))
        tmp = path.with_name(path.name + ".tmp")
        try:
            with tmp.open("wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            sync_dir(path.parent)
        finally:
            tmp.unlink(missing_ok=True)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def atomic_json(path, value, **kwargs):
    return atomic_bytes(
        path,
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode(),
        **kwargs,
    )


def torch_bytes(value):
    b = io.BytesIO()
    torch.save(value, b)
    return b.getvalue()


def read_json(path):
    return json.loads(Path(path).read_text())


def prefix_sha(path, n):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while n:
            b = f.read(min(n, 4 * 1024 * 1024))
            if not b:
                raise ValueError("Missing committed journal prefix")
            h.update(b)
            n -= len(b)
    return h.hexdigest()


def descriptor(root, path):
    p = Path(path)
    return {"path": str(p.relative_to(root)), "sha256": sha(p), "bytes": p.stat().st_size}


def verify(root, ref):
    p = Path(root) / ref["path"]
    if not p.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Reference escapes job root")
    if p.stat().st_size != ref["bytes"] or sha(p) != ref["sha256"]:
        raise ValueError("Artifact integrity mismatch: " + str(p))
    return p


def append_json(path, row, max_bytes=32768):
    p = Path(path)
    durable_mkdir(p.parent)
    b = (json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n").encode()
    if len(b) > max_bytes:
        raise StoragePressure("Scientific row exceeds validated record budget")
    reserve(p, len(b))
    with p.open("ab") as f:
        f.write(b)
        f.flush()


def bounded_event(path, row, limit=4 * 1024 * 1024):
    p = Path(path)
    if p.exists() and p.stat().st_size > limit:
        os.replace(p, p.with_suffix(p.suffix + ".previous"))
        sync_dir(p.parent)
    append_json(p, row)
