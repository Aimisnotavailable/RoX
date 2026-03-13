import struct
import numpy as np
from pathlib import Path

CHUNK_HEADER_FMT = "<I"   # uint32 length of block in bytes
RUN_FMT = "<I B"          # uint32 count, uint8 id
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
    n_runs = len(data) // RUN_SIZE
    # view as array of runs
    runs = np.frombuffer(data, dtype=np.uint8).reshape(-1)  # fallback
    # safer: iterate runs but it's still fast because it's binary
    out = []
    offset = 0
    while offset + RUN_SIZE <= len(data):
        cnt, vid = struct.unpack_from(RUN_FMT, data, offset)
        out.append((cnt, vid))
        offset += RUN_SIZE
    # preallocate
    total = sum(cnt for cnt, _ in out)
    arr = np.empty(total, dtype=np.uint8)
    pos = 0
    for cnt, vid in out:
        arr[pos:pos+cnt] = vid
        pos += cnt
    return arr

def save_world(path: Path, chunks: list):
    with open(path, "wb") as f:
        # write index: number of chunks then for each chunk: offset (uint64)
        f.write(struct.pack("<I", len(chunks)))
        index_pos = f.tell()
        f.write(b'\x00' * (8 * len(chunks)))  # placeholder offsets
        offsets = []
        for i, chunk in enumerate(chunks):
            offsets.append(f.tell())
            data = pack_chunk_to_bytes(chunk)
            f.write(struct.pack(CHUNK_HEADER_FMT, len(data)))
            f.write(data)
        # go back and write offsets
        f.seek(index_pos)
        for off in offsets:
            f.write(struct.pack("<Q", off))

def load_chunk_by_index(path: Path, idx: int) -> np.ndarray:
    with open(path, "rb") as f:
        n_chunks = struct.unpack("<I", f.read(4))[0]
        f.seek(4 + idx * 8)
        off = struct.unpack("<Q", f.read(8))[0]
        f.seek(off)
        length = struct.unpack(CHUNK_HEADER_FMT, f.read(4))[0]
        data = f.read(length)
        return unpack_chunk_from_bytes(data)

# print(load_chunk_by_index(Path("world_data/chunks/world.dat"), 0))