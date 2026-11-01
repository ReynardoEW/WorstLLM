#!/usr/bin/env python3
"""Rebuild every contract from this repo and compare it byte for byte with what is on mainnet.

Needs no key and sends nothing. Needs solc 0.8.38.
Usage: LLM_DATA=data/chat python verify_rebuild.py [CKPT]
"""
import hashlib, json, os, subprocess, sys, urllib.request

os.environ.pop("LLM_NODE", None)
import torch
from tokenizers import Tokenizer
import intmodel as im, train as tr
from onchain import EMB_PER, build_main

RPC = os.environ.get("RPC", "https://ethereum-rpc.publicnode.com")
ADDRESSES = json.load(open("deployments_mainnet.json"))


def rpc(method, *params):
    r = urllib.request.Request(RPC, json.dumps(dict(jsonrpc="2.0", id=1, method=method, params=list(params))).encode(), {"content-type": "application/json", "user-agent": "curl/8"})
    out = json.load(urllib.request.urlopen(r, timeout=60))
    assert "error" not in out, out
    return out["result"]


class Compare:
    """Takes the place of the deployer: each contract the build produces is checked against the chain."""

    def __init__(self):
        self.checked = 0
        self.last = None

    def put(self, code):
        key = hashlib.sha256(code).hexdigest()
        if key not in ADDRESSES:
            sys.exit(f"FAIL: a rebuilt contract of {len(code):,} bytes is not in the deployment list")
        addr = ADDRESSES[key]["addr"]
        if bytes.fromhex(rpc("eth_getCode", addr, "latest")[2:]) != code:
            sys.exit(f"FAIL: {addr} on chain differs from the rebuild")
        self.checked += 1
        self.last = addr
        return addr

    def data(self, b):
        return self.put(b"\0" + b)


def front(engine):
    std = dict(language="Solidity", sources={"WorstLLM.sol": dict(content=open("WorstLLM.sol").read())},
               settings=dict(evmVersion="osaka", optimizer=dict(enabled=True, runs=200), outputSelection={"*": {"*": ["evm.deployedBytecode"]}}))
    out = json.loads(subprocess.run(["solc", "--standard-json"], input=json.dumps(std), capture_output=True, text=True).stdout)
    dep = out["contracts"]["WorstLLM.sol"]["WorstLLM"]["evm"]["deployedBytecode"]
    code = bytearray.fromhex(dep["object"])
    for refs in dep["immutableReferences"].values():  # ENGINE is written into the code at deploy time
        for r in refs:
            code[r["start"]:r["start"] + r["length"]] = bytes(12) + bytes.fromhex(engine[2:])
    addr = next(v["addr"] for v in ADDRESSES.values() if v.get("front"))
    if bytes.fromhex(rpc("eth_getCode", addr, "latest")[2:]) != bytes(code):
        sys.exit(f"FAIL: front contract {addr} differs from the rebuild")
    return addr


ck = torch.load(sys.argv[1] if len(sys.argv) > 1 else "runs/m16.pt", map_location="cpu", weights_only=False)
model = im.IntModel(ck, im.calib_tokens())
tok = Tokenizer.from_file(f"{tr.DATA}/tok.json")
g = Compare()
tb = bytearray()
for i in range(tr.VOCAB):
    b = b"" if i == 0 else tok.decode([i]).encode()
    tb += b.ljust(30, b"\0") + bytes([1 if b"\n" in b else 0, len(b)])
words = lambda rows: b"".join(w.to_bytes(32, "big") for row in rows for w in row)
table = g.data(bytes(tb))
embs = [g.data(words(model.emb[:EMB_PER])), g.data(words(model.emb[EMB_PER:]))]
pos = g.data(words(model.pos))
engine = g.put(build_main(model, g, table, embs, pos))
addr = front(engine)
print(f"PASS: {g.checked} contracts match mainnet byte for byte, and the front contract {addr} matches with engine {engine}")
