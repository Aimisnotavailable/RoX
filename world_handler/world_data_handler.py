import struct
import numpy as np
from pathlib import Path

# --- RLE helpers (still useful for memory, but not used for disk I/O) ---
CHUNK_HEADER_FMT = "<I"
RUN_FMT = "<I B"
RUN_SIZE = struct.calcsize(RUN_FMT)

def pack_chunk_to_bytes(chunk_voxels: np.ndarray) -> bytes:
    if chunk_voxels.size == 0:
        return b''
    parts = []
    curr = int(chunk_voxels[0])
    cnt = 1
    for v in chunk_voxels[1:]:
        v = int(v)
        if v == curr:
            cnt += 1
        else:
            parts.append(struct.pack(RUN_FMT, cnt, curr))
            curr = v
            cnt = 1
    parts.append(struct.pack(RUN_FMT, cnt, curr))
    return b''.join(parts)

def unpack_chunk_from_bytes(data: bytes) -> np.ndarray:
    if not data:
        return np.empty(0, dtype=np.uint8)
    out = []
    offset = 0
    while offset + RUN_SIZE <= len(data):
        cnt, vid = struct.unpack_from(RUN_FMT, data, offset)
        out.append((cnt, vid))
        offset += RUN_SIZE
    total = sum(cnt for cnt, _ in out)
    arr = np.empty(total, dtype=np.uint8)
    pos = 0
    for cnt, vid in out:
        arr[pos:pos+cnt] = vid
        pos += cnt
    return arr

# --- New: per‑chunk saving and loading with compression ---
def save_chunk(path: Path, chunk_voxels: np.ndarray):
    """Save a single chunk to a compressed .npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, voxels=chunk_voxels)

def load_chunk(path: Path) -> np.ndarray:
    """Load a single chunk from a compressed .npz file; returns empty array if missing."""
    try:
        with np.load(path) as data:
            return data['voxels']
    except (FileNotFoundError, KeyError):
        # No file → treat as empty chunk
        return np.empty(0, dtype=np.uint8)