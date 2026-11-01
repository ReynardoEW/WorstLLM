#!/usr/bin/env python3
"""MLX copy of train.py's RMS-norm model. Saves a checkpoint that intmodel.py can load.
Usage: train_mlx.py NAME --layers 4 --iters 20000"""
import argparse, math, time
from functools import partial
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np, torch
from tokenizers import Tokenizer
from train import VOCAB, DIM, HEADS, HEAD_DIM, FFN, T, DATA, RUNS, EOS

NORM_KIND = [2]  # 2: rms, 3: mean absolute value

def rms(x):
    if NORM_KIND[0] == 3:
        return x / (mx.abs(x).mean(-1, keepdims=True) + 1e-6)
    return x / (mx.sqrt((x * x).mean(-1, keepdims=True)) + 1e-6)

def band(t, window):
    """Each position attends to itself and the previous window-1 positions."""
    return mx.logical_and(mx.tril(mx.ones((t, t))) > 0, mx.triu(mx.ones((t, t)), k=-(window - 1)) > 0)

class Block(nn.Module):
    def __init__(self, window, ffn_norm):
        super().__init__()
        self._window, self._ffn_norm = window, ffn_norm
        self.q, self.k, self.v = (nn.Linear(DIM, HEADS * HEAD_DIM, bias=False) for _ in range(3))
        self.o = nn.Linear(HEADS * HEAD_DIM, DIM, bias=False)
        self.f1, self.f2 = nn.Linear(DIM, FFN, bias=False), nn.Linear(FFN, DIM, bias=False)

    def __call__(self, x):
        B, t, _ = x.shape
        h = rms(x)
        q, k, v = (m(h).reshape(B, t, HEADS, HEAD_DIM).transpose(0, 2, 1, 3) for m in (self.q, self.k, self.v))
        s = q @ k.transpose(0, 1, 3, 2)
        a = mx.softmax(mx.where(band(t, self._window), s, -1e9), -1)
        x = x + self.o((a @ v).transpose(0, 2, 1, 3).reshape(B, t, -1))
        return x + self.f2(nn.relu(self.f1(rms(x) if self._ffn_norm else x)))

class Model(nn.Module):
    def __init__(self, layers, window, ffn_norm):
        super().__init__()
        self.tok, self.pos = nn.Embedding(VOCAB, DIM), nn.Embedding(window, DIM)  # positions wrap at the window
        self._positions = window
        self.blocks = [Block(window, ffn_norm) for _ in range(layers)]
        for b in self.blocks:
            for m in (b.q, b.k, b.v, b.o, b.f1, b.f2):
                m.weight = mx.random.normal(m.weight.shape) * 0.04
            for m in (b.o, b.f2):
                m.weight = m.weight / math.sqrt(2 * layers)
        self.tok.weight = mx.random.normal(self.tok.weight.shape) * 0.02
        self.pos.weight = mx.random.normal(self.pos.weight.shape) * 0.02

    def __call__(self, idx, start=0):
        x = self.tok(idx) + self.pos((mx.arange(idx.shape[1]) + start) % self._positions)
        for b in self.blocks:
            x = b(x)
        return rms(x) @ self.tok.weight.T

STARTS = {}

def batch(d, bs):
    """Half the windows start at a story boundary, so position 0 + EOS means 'begin a story'."""
    i = np.random.randint(0, len(d) - T - 1, bs)
    if id(d) not in STARTS:  # one scan only: this file has 69M tokens
        st = np.flatnonzero(np.asarray(d) == 0)
        STARTS[id(d)] = st[st < len(d) - T - 1]
    st = STARTS[id(d)]
    i[: bs // 2] = st[np.random.randint(0, len(st), bs // 2)]
    return mx.array(np.stack([d[j:j + T] for j in i]).astype(np.int32)), mx.array(np.stack([d[j + 1:j + 1 + T] for j in i]).astype(np.int32))

def loss_fn(m, x, y, start=0):
    return nn.losses.cross_entropy(m(x, start).reshape(-1, VOCAB), y.reshape(-1)).mean()

def save(m, a, v):
    st = {k: torch.from_numpy(np.array(p)) for k, p in tree_flatten(m.parameters())}
    for i in range(a.layers):
        st[f"blocks.{i}.mask"] = torch.from_numpy(np.array(band(T, a.window)))
    torch.save(dict(state=st, layers=a.layers, norm=NORM_KIND[0], val=v, window=a.window, ffn_norm=a.ffn_norm), f"{RUNS}/{a.name}.pt")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("name"); p.add_argument("--layers", type=int, default=4); p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--window", type=int, default=T); p.add_argument("--ffn-norm", dest="ffn_norm", type=int, default=1)
    p.add_argument("--norm", type=int, default=2)
    p.add_argument("--every", type=int, default=2000); p.add_argument("--bs", type=int, default=64); p.add_argument("--lr", type=float, default=3e-3)
    a = p.parse_args()
    NORM_KIND[0] = a.norm
    tr, va = (np.memmap(f"{DATA}/{s}.bin", dtype=np.uint8, mode="r") for s in ("train", "valid"))
    np.random.seed(1); mx.random.seed(1)
    m = Model(a.layers, a.window, a.ffn_norm); mx.eval(m.parameters())
    print(f"{a.name}: mlx {mx.default_device()} params={sum(v.size for _, v in tree_flatten(m.parameters())):,}", flush=True)
    opt = optim.AdamW(learning_rate=a.lr, weight_decay=0.01)
    vg = nn.value_and_grad(m, loss_fn)
    state = [m.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y, start):
        l, g = vg(m, x, y, start)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(m, g)
        return l

    t0, best = time.time(), 9e9
    for it in range(a.iters + 1):
        opt.learning_rate = mx.array(a.lr * min(1, (it + 1) / 500) * (0.5 * (1 + math.cos(math.pi * it / a.iters)) * 0.98 + 0.02))
        if it % a.every == 0:
            v = sum(loss_fn(m, *batch(va, 64)).item() for _ in range(40)) / 40
            print(f"{a.name} it {it:>6} val {v:.4f} {it / (time.time() - t0 + 1e-6):6.1f} it/s {time.time() - t0:6.0f}s", flush=True)
            if v < best:
                best = v; save(m, a, v)
        x, y = batch(tr, a.bs)
        step(x, y, mx.array(np.random.randint(0, a.window))); mx.eval(state)  # no position is special
    print(f"{a.name} best val {best:.4f}", flush=True)

if __name__ == "__main__":
    main()
