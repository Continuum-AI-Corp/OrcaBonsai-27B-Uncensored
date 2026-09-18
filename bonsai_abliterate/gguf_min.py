"""A minimal GGUF header reader, because stock gguf-py cannot open these packs.

PrismML's ternary GGUFs use private ggml type ids -- `PQ2_0` = 142 and `PTQ1_0` = 143 --
which sit outside upstream's range (upstream's enum ends at 43). `gguf.GGUFReader` raises

    ValueError: np.uint32(143) is not a valid GGMLQuantizationType

before it reaches a single tensor. This reads the header directly instead. It is
deliberately small: names, shapes, type ids, offsets and the key/value block are all the
adapter exporter needs from the base model. Writing the adapter still uses gguf-py, since
an adapter holds only standard F32 tensors.
"""
import struct
import numpy as np

# Bytes per 128-element block. These match the pack's own runtime/codec.py, which is the
# authoritative description of both layouts.
PRISM_BLOCK_BYTES = {142: 34, 143: 28}   # PQ2_0, PTQ1_0
PRISM_NAME = {142: "PQ2_0", 143: "PTQ1_0"}
STD_ITEMSIZE = {0: 4, 1: 2}      # F32, F16 -- enough for what we read

def _u(f, fmt):
    n = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(n))

def _str(f):
    (n,) = _u(f, "<Q")
    return f.read(n).decode("utf-8")

def _val(f, t):
    if t == 8:  return _str(f)
    if t == 9:
        (et,) = _u(f, "<I"); (n,) = _u(f, "<Q")
        return [_val(f, et) for _ in range(n)]
    fmts = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<?",10:"<Q",11:"<q",12:"<d"}
    return _u(f, fmts[t])[0]

class Gguf:
    def __init__(self, path):
        self.path = path
        f = open(path, "rb")
        magic = f.read(4)
        assert magic == b"GGUF", magic
        (self.version,) = _u(f, "<I")
        (n_tensors,) = _u(f, "<Q")
        (n_kv,) = _u(f, "<Q")
        self.kv = {}
        for _ in range(n_kv):
            k = _str(f); (t,) = _u(f, "<I")
            self.kv[k] = _val(f, t)
        self.tensors = {}
        for _ in range(n_tensors):
            name = _str(f)
            (nd,) = _u(f, "<I")
            dims = [_u(f, "<Q")[0] for _ in range(nd)]
            (tt,) = _u(f, "<I")
            (off,) = _u(f, "<Q")
            self.tensors[name] = {"ne": dims, "type": tt, "offset": off,
                                  "type_name": PRISM_NAME.get(tt, str(tt))}
        align = self.kv.get("general.alignment", 32)
        pos = f.tell()
        self.data_start = pos + (-pos) % align
        f.close()

    def raw(self, name):
        t = self.tensors[name]
        numel = 1
        for d in t["ne"]: numel *= d
        tt = t["type"]
        if tt in PRISM_BLOCK_BYTES:
            nbytes = numel // 128 * PRISM_BLOCK_BYTES[tt]
        elif tt in STD_ITEMSIZE:
            nbytes = numel * STD_ITEMSIZE[tt]
        else:
            raise NotImplementedError(f"type {tt} for {name}")
        with open(self.path, "rb") as f:
            f.seek(self.data_start + t["offset"])
            return f.read(nbytes)

    def shape_out_in(self, name):
        """(rows, cols) in the [out, in] convention the pack's codec expects."""
        return tuple(int(n) for n in self.tensors[name]["ne"][::-1])
