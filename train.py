#!/usr/bin/env python3
"""Tokenize TinyStories and train the gas-shaped model. Usage:
  train.py tok                                    build tokenizer + token files
  train.py fit NAME --layers 4 --norm 0 --iters N train a float model
"""
import argparse, math, os, re, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("LLM_DATA", f"{ROOT}/data")
RUNS = f"{ROOT}/runs"
VOCAB, DIM, HEADS, HEAD_DIM, FFN, T = 256, 64, 4, 16, 128, 64
EOS = "<|endoftext|>"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def clean(text):
    """ASCII only: non-ASCII quotes are normalized, the rest dropped, so on-chain bytes stay simple."""
    for a, b in (("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"), ("—", "-"), ("…", "...")):
        text = text.replace(a, b)
    return re.sub(r"[^\x00\x20-\x7e\n]|\n(?=\n)", "", text.replace(EOS, "\0")).replace("\0", EOS)


def build_tokenizer():
    with open(f"{DATA}/train.txt", encoding="utf-8", errors="ignore") as f:
        # the separator is a special token, and every printable ASCII symbol must have a token
        sample = clean(f.read(60_000_000)).replace(EOS, "\n") + "".join(chr(c) for c in range(32, 127)) * 3
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    # no initial_alphabet: only the ~96 symbols the ASCII data uses, the rest of the vocab is merges
    tr = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=[EOS], min_frequency=2, show_progress=False)
    tok.train_from_iterator([sample[i:i + 100000] for i in range(0, len(sample), 100000)], tr)
    assert tok.get_vocab_size() == VOCAB, tok.get_vocab_size()
    tok.save(f"{DATA}/tok.json")
    return tok


def encode_file(tok, name, limit=None):
    with open(f"{DATA}/{name}.txt", encoding="utf-8", errors="ignore") as f:
        text = f.read(limit)
    text = clean(text)
    out, step = [], 2_000_000
    for i in range(0, len(text), step):
        out.extend(tok.encode(text[i:i + step]).ids)
        print(f"{name}: {i + step:,}/{len(text):,} chars, {len(out):,} tokens", flush=True)
    np.array(out, dtype=np.uint8).tofile(f"{DATA}/{name}.bin")


class RMSNorm(nn.Module):
    """No parameters: the next matmul absorbs any gain. On-chain: sum of squares, isqrt, one reciprocal."""

    def forward(self, x):
        return x / (x.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)


class AbsNorm(nn.Module):
    """Mean absolute value. On-chain: SWAR abs, one MUL per word, one DIV per norm."""

    def forward(self, x):
        return x / (x.abs().mean(-1, keepdim=True) + 1e-6)


def make_norm(kind):
    return {0: nn.Identity, 1: lambda: nn.LayerNorm(DIM, bias=False), 2: RMSNorm, 3: AbsNorm}[kind]()


class Block(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(DIM, HEADS * HEAD_DIM, bias=False) for _ in range(3))
        self.o = nn.Linear(HEADS * HEAD_DIM, DIM, bias=False)
        self.f1, self.f2 = nn.Linear(DIM, FFN, bias=False), nn.Linear(FFN, DIM, bias=False)
        self.n1, self.n2 = make_norm(norm), make_norm(norm)
        self.register_buffer("mask", torch.tril(torch.ones(T, T)).bool())

    def forward(self, x):
        B, t, _ = x.shape
        h = self.n1(x)
        q, k, v = (m(h).view(B, t, HEADS, HEAD_DIM).transpose(1, 2) for m in (self.q, self.k, self.v))
        a = (q @ k.transpose(-1, -2)).masked_fill(~self.mask[:t, :t], -1e9).softmax(-1)
        x = x + self.o((a @ v).transpose(1, 2).reshape(B, t, -1))
        return x + self.f2(F.relu(self.f1(self.n2(x))))


class Model(nn.Module):
    def __init__(self, layers, norm, positions=T):
        super().__init__()
        self.tok, self.pos = nn.Embedding(VOCAB, DIM), nn.Embedding(positions, DIM)
        self.positions = positions
        self.blocks = nn.ModuleList(Block(norm) for _ in range(layers))
        self.out_norm = make_norm(norm)
        self.layers = layers
        for n, p in self.named_parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02 if "tok" in n or "pos" in n else 0.04)
        # residual outputs start small: this is what keeps a norm-free stack stable
        for b in self.blocks:
            for m in (b.o, b.f2):
                m.weight.data *= 1 / math.sqrt(2 * layers)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device) % self.positions)
        for b in self.blocks:
            x = b(x)
        return self.out_norm(x) @ self.tok.weight.T


def batch(d, bs, dev):
    i = np.random.randint(0, len(d) - T - 1, bs)
    x = np.stack([d[j:j + T] for j in i]).astype(np.int64)
    y = np.stack([d[j + 1:j + 1 + T] for j in i]).astype(np.int64)
    return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)


@torch.no_grad()
def val_loss(m, d, dev, n=40):
    m.eval()
    r = sum(F.cross_entropy(m(x).view(-1, VOCAB), y.view(-1)).item() for x, y in (batch(d, 64, dev) for _ in range(n))) / n
    m.train()
    return r


@torch.no_grad()
def sample(m, tok, dev, n=70, temp=0.8):
    m.eval()
    idx = torch.tensor([[tok.token_to_id(EOS)]], device=dev)
    for _ in range(n):
        p = (m(idx[:, -T:])[:, -1] / temp).softmax(-1)
        idx = torch.cat([idx, torch.multinomial(p, 1)], 1)
    m.train()
    return tok.decode(idx[0, 1:].tolist())


def fit(a):
    os.makedirs(RUNS, exist_ok=True)
    tok = Tokenizer.from_file(f"{DATA}/tok.json")
    tr, va = (np.memmap(f"{DATA}/{s}.bin", dtype=np.uint8, mode="r") for s in ("train", "valid"))
    np.random.seed(1)
    torch.manual_seed(1)
    m = Model(a.layers, a.norm).to(DEVICE)
    params = sum(p.numel() for p in m.parameters())
    print(f"{a.name}: layers={a.layers} norm={a.norm} params={params:,} device={DEVICE}", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    t0 = time.time()
    best = 9e9
    for it in range(a.iters + 1):
        lr = a.lr * min(1, (it + 1) / 500) * (0.5 * (1 + math.cos(math.pi * it / a.iters)) * 0.98 + 0.02)
        for g in opt.param_groups:
            g["lr"] = lr
        if it % a.every == 0:
            v = val_loss(m, va, DEVICE)
            print(f"{a.name} it {it:>6} val {v:.4f} {it / (time.time() - t0 + 1e-6):6.1f} it/s {time.time() - t0:6.0f}s", flush=True)
            if v < best:
                best = v
                torch.save(dict(state=m.state_dict(), layers=a.layers, norm=a.norm, val=v), f"{RUNS}/{a.name}.pt")
        x, y = batch(tr, a.bs, DEVICE)
        loss = F.cross_entropy(m(x).view(-1, VOCAB), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    print(f"{a.name} best val {best:.4f}\n--- sample ---\n{sample(m, tok, DEVICE)}\n---", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["tok", "fit"])
    p.add_argument("name", nargs="?")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--norm", type=int, default=0)
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--every", type=int, default=2000)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    a = p.parse_args()
    if a.cmd == "tok":
        tok = build_tokenizer()
        print("vocab sample:", sorted(tok.get_vocab(), key=len)[-30:])
        encode_file(tok, "valid")
        encode_file(tok, "train", 400_000_000)
    else:
        fit(a)
