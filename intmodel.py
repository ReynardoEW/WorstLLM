#!/usr/bin/env python3
"""Pure-int reference of the on-chain model. It mirrors the EVM word operations.

Values live in 16 signed 16-bit lanes of a 256-bit word, or in signed scalars. All scales are
powers of 2, so requantizing is a shift. Usage: intmodel.py eval CKPT [sequences]
"""
import math, subprocess, sys
import numpy as np
import torch
from tokenizers import Tokenizer

import train as tr

L, LANES = 16, 16
M256 = (1 << 256) - 1
ONES = sum(1 << (L * i) for i in range(LANES))
BIAS = ONES << (L - 1)
LMASK = (1 << L) - 1
FX = 4  # norm output: 16 units per 1.0 of mean absolute value
ABS_S = 7  # lane shift after the norm multiply
ACC_LIM = 24000  # calibrated largest lane value, below 2^15 to leave a margin
V_LIM = 100  # attention keeps sum(p * v) in a lane: 256 * 100 < 2^15
SCORE_STEP = 0.25
TEMP = 0.8
DW, FW, VW_ = tr.DIM // LANES, tr.FFN // LANES, tr.VOCAB // LANES
LAYERS_WORDS = 8  # cached words per position: 4 heads, key and value


def keccak(b):
    return int(subprocess.check_output(["cast", "keccak", "0x" + b.hex()]).strip(), 16)


def to_s(v):
    return v - (1 << 256) if v >> 255 else v


def pack(vals):
    return sum(int(v) << (L * i) for i, v in enumerate(vals)) & M256


def lane_rshift(w, s):
    """Rounded arithmetic shift of every lane."""
    if s == 0:
        return w
    r = (w + BIAS + (ONES << (s - 1))) & M256
    return (((r >> s) & (ONES * ((1 << (L - s)) - 1))) - ONES * (1 << (L - 1 - s))) & M256


def lane_relu_rshift(w, s):
    """max(0, lane) >> s, unsigned result."""
    v = (w + BIAS) & M256
    r = v & (((v >> (L - 1)) & ONES) * ((1 << (L - 1)) - 1))
    if s == 0:
        return r
    return (((r + (ONES << (s - 1))) & M256) >> s) & (ONES * ((1 << (L - s)) - 1))


def unpack(w):
    b = (w + BIAS) & M256
    return [((b >> (L * i)) & LMASK) - (1 << (L - 1)) for i in range(LANES)]


def unpack_u(w):
    return [(w >> (L * i)) & LMASK for i in range(LANES)]


def matvec(words, x):
    return [sum(wv * xv for wv, xv in zip(row, x)) & M256 for row in words]


def norm(h_words, abs_c):
    """x = h * DIM * 2^FX / sum|h|. SWAR absolute value, one MUL sums a whole word."""
    m15 = ONES * ((1 << (L - 1)) - 1)
    total = 0
    for w in h_words:
        v = (w + BIAS) & M256
        neg = ((v >> (L - 1)) & ONES) ^ ONES
        mag = ((v ^ (neg * LMASK)) & m15) + neg
        total += ((mag * ONES) & M256) >> 240
    m = abs_c // total if total else 0
    return [y for w in h_words for y in unpack(lane_rshift((w * m) & M256, ABS_S))]


def attn(q, ks, vs, a, lut):
    scores = [to_s((q * k) & M256) >> 240 for k in ks]
    mx = max(scores)
    es = [lut[(mx - s) >> a] if (mx - s) >> a < 32 else 0 for s in scores]
    mm = (1 << 20) // sum(es)
    A = 0
    for e, v in zip(es, vs):
        A = (A + ((e * mm) >> 12) * v) & M256
    return A


def lut_for(step):
    return [min(255, round(255 * math.exp(-i * step))) for i in range(32)]


def flog2(x):
    return math.floor(math.log2(x))


def packed_matrix(W, fw, reverse=False):
    N, K = W.shape
    assert N % LANES == 0
    Wi = np.clip(np.round(W * 2.0 ** fw), -127, 127).astype(np.int64)
    rows = []
    for g in range(0, N, LANES):
        block = Wi[g:g + LANES]
        if reverse:
            block = block[::-1]
        rows.append([pack(block[:, j]) for j in range(K)])
    return rows, Wi


def fw_for(W, fin, ymax):
    """The largest weight scale that fills int8 and keeps measured sums inside a lane."""
    return min(flog2(127 / float(np.abs(W).max())), flog2(ACC_LIM / (ymax * 2.0 ** fin)))


def calibrate(ck, tokens):
    """Largest float activations on real text. These choose every power-of-2 scale."""
    m = tr.Model(ck["layers"], ck["norm"], ck["window"])
    m.load_state_dict(ck["state"])
    m.eval()
    s = {}

    def rec(name, v):
        s[name] = max(s.get(name, 0), float(v.abs().max()))

    x = torch.from_numpy(tokens.astype(np.int64))
    with torch.no_grad():
        h = m.tok(x) + m.pos(torch.arange(x.shape[1]) % ck["window"])
        for i, b in enumerate(m.blocks):
            rec("h", h)
            n = b.n1(h)
            rec("x", n)
            B, t, _ = n.shape
            q, k, v = (mm(n).view(B, t, tr.HEADS, tr.HEAD_DIM).transpose(1, 2) for mm in (b.q, b.k, b.v))
            rec(f"q{i}", q), rec(f"k{i}", k), rec(f"v{i}", v)
            sc = q @ k.transpose(-1, -2)
            rec(f"s{i}", sc.masked_fill(~b.mask[:t, :t], 0))
            a = sc.masked_fill(~b.mask[:t, :t], -1e9).softmax(-1)
            o = b.o((a @ v).transpose(1, 2).reshape(B, t, -1))
            rec(f"o{i}", o)
            h = h + o
            rec("h", h)
            f = b.f1(b.n2(h))
            rec(f"f{i}", f)
            r = torch.relu(f)
            rec(f"r{i}", r)
            o2 = b.f2(r)
            rec(f"g{i}", o2)
            h = h + o2
        rec("h", h)
        n = m.out_norm(h)
        rec("x", n)
        rec("logits", n @ m.tok.weight.T)
    return s


class IntModel:
    def __init__(self, ck, tokens):
        st = {k: v.numpy() for k, v in ck["state"].items()}
        assert ck["norm"] == 3
        self.layers, self.window = ck["layers"], ck["window"]
        self.positions = st["pos.weight"].shape[0]
        s = self.stats = calibrate(ck, tokens)
        self.FH = min(4, flog2((1 << 14) / s["h"]))
        self.abs_c = tr.DIM * (1 << FX) * (1 << ABS_S)
        split = lambda t: [[pack(r[i:i + LANES]) for i in range(0, tr.DIM, LANES)] for r in np.round(t * 2.0 ** self.FH).astype(np.int64)]
        self.emb, self.pos = split(st["tok.weight"]), split(st["pos.weight"])
        self.blocks = []
        for i in range(self.layers):
            g = lambda n: st[f"blocks.{i}.{n}.weight"]
            fq, fk = flog2(V_LIM / s[f"q{i}"]), flog2(V_LIM / s[f"k{i}"])
            while s[f"s{i}"] * 2.0 ** (fq + fk) > ACC_LIM:  # the dot product must fit the top lane
                if fq >= fk:
                    fq -= 1
                else:
                    fk -= 1
            fv, fr = flog2(V_LIM / s[f"v{i}"]), flog2(V_LIM / s[f"r{i}"])
            b = dict(fq=fq, fk=fk, fv=fv, fr=fr)
            for name, fin, ymax, rev in (("q", FX, s[f"q{i}"], True), ("k", FX, s[f"k{i}"], False), ("v", FX, s[f"v{i}"], False),
                                         ("o", fv, s[f"o{i}"], False), ("f1", FX, s[f"f{i}"], False), ("f2", fr, s[f"g{i}"], False)):
                fw = fw_for(g(name), fin, ymax)
                b[name], b[name + "_raw"] = packed_matrix(g(name), fw, rev)
                b["w" + name] = fw
            b["sq"], b["sk"], b["sv"] = b["wq"] + FX - fq, b["wk"] + FX - fk, b["wv"] + FX - fv
            b["so"] = b["wo"] + fv - self.FH
            b["sr"] = b["wf1"] + FX - fr
            b["sf"] = b["wf2"] + fr - self.FH
            for k_ in ("sq", "sk", "sv", "so", "sr", "sf"):
                assert 0 <= b[k_] < L - 1, (i, k_, b[k_])
            b["a"] = max(0, flog2(SCORE_STEP * 2.0 ** (fq + fk)))
            b["lut"] = lut_for(2.0 ** (b["a"] - fq - fk))
            self.blocks.append(b)
        self.wu = fw_for(st["tok.weight"], FX, s["logits"])
        self.unembed, self.unembed_raw = packed_matrix(st["tok.weight"], self.wu)
        self.aout = max(0, flog2(SCORE_STEP * TEMP * 2.0 ** (self.wu + FX)))
        self.lut_out = lut_for(2.0 ** (self.aout - self.wu - FX) / TEMP)
        self.audit = None  # set to a dict to record the largest true lane values

    def new_state(self):
        """t, the next input token, and the ring cache, laid out like the contract's memory."""
        return dict(t=0, cur=0, ring=[0] * (self.window * 8 * self.layers))

    def blob(self, st):
        return st["t"].to_bytes(32, "big") + st["cur"].to_bytes(32, "big") + b"".join(w.to_bytes(32, "big") for w in st["ring"])

    def load_blob(self, b):
        return dict(t=int.from_bytes(b[:32], "big"), cur=int.from_bytes(b[32:64], "big"),
                    ring=[int.from_bytes(b[i:i + 32], "big") for i in range(64, len(b), 32)])

    def _check(self, name, raw, x):
        if self.audit is not None:
            y = raw @ np.array(x, dtype=np.int64)
            self.audit[name] = max(self.audit.get(name, 0), int(np.abs(y).max()))

    def step(self, st):
        """One token through the model. Updates the cache, returns the unpacked logits."""
        W, t = self.window, st["t"]
        h = [(a + b) & M256 for a, b in zip(self.emb[st["cur"]], self.pos[t % self.positions])]
        n, slot = min(t + 1, W), t % W
        ring = st["ring"]
        for l, b in enumerate(self.blocks):
            base = l * 8 * W
            x = norm(h, self.abs_c)
            for nm in "qkv":
                self._check(nm, b[nm + "_raw"], x)
            q = [lane_rshift(w, b["sq"]) for w in matvec(b["q"], x)]
            k = [lane_rshift(w, b["sk"]) for w in matvec(b["k"], x)]
            v = [lane_rshift(w, b["sv"]) for w in matvec(b["v"], x)]
            att = []
            for i in range(tr.HEADS):
                kb, vb = base + i * 2 * W, base + i * 2 * W + W
                ring[kb + slot], ring[vb + slot] = k[i], v[i]
                att += unpack(lane_rshift(attn(q[i], ring[kb:kb + n], ring[vb:vb + n], b["a"], b["lut"]), 8))
            self._check("o", b["o_raw"], att)
            h = [(hw + lane_rshift(o, b["so"])) & M256 for hw, o in zip(h, matvec(b["o"], att))]
            x = norm(h, self.abs_c)
            self._check("f1", b["f1_raw"], x)
            r = [y for w in matvec(b["f1"], x) for y in unpack_u(lane_relu_rshift(w, b["sr"]))]
            self._check("f2", b["f2_raw"], r)
            h = [(hw + lane_rshift(o, b["sf"])) & M256 for hw, o in zip(h, matvec(b["f2"], r))]
        x = norm(h, self.abs_c)
        self._check("unembed", self.unembed_raw, x)
        return [v for w in matvec(self.unembed, x) for v in unpack(w)]

    def pick(self, logits, seed, t):
        """Temperature sampling through the same LUT and keccak draw as the contract."""
        mx = max(logits)
        es = [self.lut_out[(mx - l) >> self.aout] if (mx - l) >> self.aout < 32 else 0 for l in logits]
        r = keccak(seed.to_bytes(32, "big") + t.to_bytes(32, "big")) % sum(es)
        for i, e in enumerate(es):
            if r < e:
                return i
            r -= e
        raise AssertionError("walk ended without a token")

    def generate(self, st, seed, n, prompt=(), stops=()):
        """Feeds the prompt, samples up to n tokens, and returns the tokens to print.

        A sampled token in stops ends the tx after it prints. Prompt tokens never stop it.
        """
        t0, out = st["t"], []
        total = t0 + len(prompt) + n
        while st["t"] < total:
            logits = self.step(st)
            i = st["t"] - t0
            st["cur"] = prompt[i] if i < len(prompt) else self.pick(logits, seed, st["t"])
            st["t"] += 1
            if st["cur"] == 0:
                break
            out.append(st["cur"])
            if i >= len(prompt) and st["cur"] in stops:
                break
        return out

    def int_loss(self, tokens):
        st = self.new_state()
        total, scale = 0.0, 2.0 ** (self.wu + FX)
        for t in range(len(tokens) - 1):
            st["cur"] = int(tokens[t])
            lg = np.array(self.step(st), dtype=np.float64) / scale
            st["t"] += 1
            lg -= lg.max()
            total += -(lg[int(tokens[t + 1])] - np.log(np.exp(lg).sum()))
        return total / (len(tokens) - 1)


def calib_tokens(rng_seed=0, n=256):
    va = np.memmap(f"{tr.DATA}/valid.bin", dtype=np.uint8, mode="r")
    rng = np.random.default_rng(rng_seed)
    return np.stack([va[j:j + tr.T] for j in rng.integers(0, len(va) - tr.T - 1, n)])


def main():
    ck = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
    calib = calib_tokens()
    im = IntModel(ck, calib)
    print("FH", im.FH, "wu", im.wu, "a_out", im.aout, "stats", {k: round(v, 1) for k, v in im.stats.items()})
    for i, b in enumerate(im.blocks):
        print(i, {k: b[k] for k in ("fq", "fk", "fv", "fr", "wq", "wk", "wv", "wo", "wf1", "wf2", "sq", "sk", "sv", "so", "sr", "sf", "a")})
    m = tr.Model(ck["layers"], ck["norm"], ck["window"])
    m.load_state_dict(ck["state"])
    m.eval()
    va = np.memmap(f"{tr.DATA}/valid.bin", dtype=np.uint8, mode="r")
    rng = np.random.default_rng(1)
    runs = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    im.audit = {}
    ti = tf = 0
    for j in rng.integers(0, len(va) - 65, runs):
        seq = np.array(va[j:j + 65])
        ti += im.int_loss(seq)
        with torch.no_grad():
            x = torch.from_numpy(seq[:-1].astype(np.int64))[None]
            y = torch.from_numpy(seq[1:].astype(np.int64))[None]
            tf += torch.nn.functional.cross_entropy(m(x)[0], y[0]).item()
    print("largest true lane values (limit 32767):", im.audit)
    print(f"mean int loss {ti / runs:.4f} float {tf / runs:.4f} over {runs * 64} tokens")


if __name__ == "__main__":
    main()
