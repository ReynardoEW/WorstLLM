#!/usr/bin/env python3
"""M0: measure real gas/MAC of int8 matmul kernels with real txs on anvil (osaka).

Kernels (y = W x, W int8 in [-127,127], x int8 as 32-byte signed words):
  V0 naive      row bytes in data contract, one MUL per MAC
  V1 report     data contract, offset-binary, and(shr(8k,w),M), 32-bit lanes
  V2 lane24     data contract, offset-binary, mask-only phases, 24-bit lanes
  V3 stored     data contract, weights stored already spread in signed lanes
  V4 pushcode   weights are PUSH32 immediates in unrolled code chunks

Every output is checked against a pure-int reference. A negative control proves
that the check can fail. Stdlib only.
"""
import json, os, random, subprocess, sys, time, urllib.request

PORT = 8599
RPC = f"http://127.0.0.1:{PORT}"
SENDER = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
TX_CAP = 16_777_216
MOD = 1 << 256
MAX_CODE = 24_576

OPS = dict(STOP=0x00, ADD=0x01, MUL=0x02, SUB=0x03, LT=0x10, ISZERO=0x15, AND=0x16, SHR=0x1c, SAR=0x1d,
           CALLDATALOAD=0x35, CALLDATACOPY=0x37, CODECOPY=0x39, RETURNDATASIZE=0x3d, RETURNDATACOPY=0x3e, EXTCODECOPY=0x3c, POP=0x50, MLOAD=0x51,
           MSTORE=0x52, JUMPI=0x57, JUMPDEST=0x5b, GAS=0x5a, RETURN=0xf3, STATICCALL=0xfa, REVERT=0xfd)


class Asm:
    def __init__(self):
        self.b, self.labels, self.fix = bytearray(), {}, []

    def op(self, *names):
        for n in names:
            self.b.append(OPS[n])
        return self

    def push(self, v):
        if v == 0:
            self.b.append(0x5f)  # PUSH0
        else:
            n = (v.bit_length() + 7) // 8
            self.pushn(v, n)
        return self

    def pushn(self, v, n):
        self.b.append(0x5f + n)
        self.b += v.to_bytes(n, "big")
        return self

    def dup(self, n):
        self.b.append(0x7f + n)
        return self

    def swap(self, n):
        self.b.append(0x8f + n)
        return self

    def label(self, name):
        self.labels[name] = len(self.b)
        return self.op("JUMPDEST")

    def ref(self, name):
        self.b.append(0x61)
        self.fix.append((len(self.b), name))
        self.b += b"\0\0"
        return self

    def build(self):
        for pos, name in self.fix:
            self.b[pos:pos + 2] = self.labels[name].to_bytes(2, "big")
        return bytes(self.b)


# ---------- chain ----------
def rpc(method, *params):
    req = urllib.request.Request(RPC, json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": list(params)}).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=300) as r:
        out = json.load(r)
    if "error" in out:
        raise RuntimeError(f"{method}: {out['error']}")
    return out["result"]


def send(data, to=None, gas=TX_CAP):
    tx = {"from": SENDER, "data": "0x" + data.hex(), "gas": hex(gas)}
    if to:
        tx["to"] = to
    h = rpc("eth_sendTransaction", tx)
    for _ in range(600):  # anvil mines async, big txs take a moment
        rc = rpc("eth_getTransactionReceipt", h)
        if rc:
            break
        time.sleep(0.05)
    assert rc and int(rc["status"], 16) == 1, f"tx failed: {rc}"
    return rc


def deploy(runtime):
    assert len(runtime) <= MAX_CODE, f"code too big: {len(runtime)}"
    init = bytes.fromhex("61") + len(runtime).to_bytes(2, "big") + bytes.fromhex("80600a5f395ff3") + runtime
    rc = send(init)
    return rc["contractAddress"], int(rc["gasUsed"], 16)


def wrapper(kernel, xaddr, K):
    """Empty-calldata entry point. Calldata x would hit the EIP-7623 floor and hide exec gas."""
    a = Asm()
    a.pushn(K * 32, 2).push(1).push(0).pushn(int(xaddr, 16), 20).op("EXTCODECOPY")
    a.push(0).push(0).pushn(K * 32, 2).push(0).pushn(int(kernel, 16), 20).op("GAS", "STATICCALL", "ISZERO").ref("fail").op("JUMPI")
    a.op("RETURNDATASIZE").push(0).push(0).op("RETURNDATACOPY", "RETURNDATASIZE").push(0).op("RETURN")
    a.label("fail").push(0).push(0).op("REVERT")
    return deploy(a.build())[0]


def measure(kaddr, x, gas=TX_CAP):
    """Returns (return data words, kernel exec gas, full tx gasUsed). Baseline: same wrapper, no-op kernel."""
    K = len(x)
    xaddr, _ = deploy(b"\0" + b"".join(word(i) for i in x))
    w = wrapper(kaddr, xaddr, K)
    base = int(send(b"", wrapper(deploy(b"\0")[0], xaddr, K))["gasUsed"], 16)
    ret = bytes.fromhex(rpc("eth_call", {"from": SENDER, "to": w, "gas": hex(10**9)}, "latest")[2:])
    used = int(send(b"", w, gas=gas)["gasUsed"], 16)
    return [int.from_bytes(ret[i:i + 32], "big") for i in range(0, len(ret), 32)], used - base, used


def word(v):
    return (v % MOD).to_bytes(32, "big")


# ---------- reference ----------
def matmul(W, x):
    return [sum(a * b for a, b in zip(row, x)) for row in W]


def signed(v, bits):
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def unlane(acc, L, n, shift=0):
    acc >>= shift
    out = []
    for _ in range(n):
        y = signed(acc, L)
        out.append(y)
        acc = (acc - y) >> L
    return out


def ones(L, n, shift=0):
    return sum(1 << (L * i + shift) for i in range(n))


def lane_bits(K):
    L = 1
    while K * 127 * 128 >= 1 << (L - 1):
        L += 1
    return L


# ---------- stream kernels: V0..V3 share one harness ----------
class Layout:
    """groups: list of (bytes, phases). Each phase produces one accumulator word."""

    def __init__(self, name, per_group, per_phase, group_words):
        self.name, self.per_group, self.per_phase, self.group_words = name, per_group, per_phase, group_words


def pack_v0(W, K):
    return [(bytes(w & 0xff for w in row), 1) for row in W]


def pack_offset(W, K, group, phase_lanes, spacing):
    """Offset-binary transposed bytes, phase-major output order."""
    out = []
    for base in range(0, len(W), group):
        rows = W[base:base + group]
        phases = (len(rows) + phase_lanes - 1) // phase_lanes
        data = bytearray()
        for j in range(K):
            wv = 0
            for g in range(group):
                ph, l = divmod(g, phase_lanes)
                val = (rows[g][j] if g < len(rows) else 0) + 128
                wv |= val << (8 * (spacing * l + ph))
            data += wv.to_bytes(32, "big")
        out.append((bytes(data), phases))
    return out


def pack_stored(W, K, L):
    lanes = 256 // L
    out = []
    for base in range(0, len(W), lanes):
        rows = W[base:base + lanes]
        out.append((b"".join(word(sum(r[j] << (L * l) for l, r in enumerate(rows))) for j in range(K)), 1))
    return out


def segments(groups):
    """Split groups over data contracts, then into same-shape runs."""
    segs, cur = [], []
    used = 0
    for g in groups:
        if used + len(g[0]) > MAX_CODE - 1:
            segs.append(cur)
            cur, used = [], 0
        cur.append(g)
        used += len(g[0])
    segs.append(cur)
    return segs


def stream_kernel(v, K, N, addr_segs):
    a = Asm()
    OUT = K * 32
    if v in ("V1", "V2"):  # sx = 128 * sum(x), corrects the offset-binary weights
        a.push(0)
        for j in range(K):
            a.push(j * 32).op("CALLDATALOAD", "ADD")
        a.push(128).op("MUL")
    else:
        a.push(0)
    a.pushn(OUT, 2)
    uid = 0

    def body(phases):
        for ph in range(phases):
            a.push(0)
            if v == "V0":
                for j in range(K):
                    a.pushn(j, 2).op("MLOAD").push(248).op("SAR").push(j * 32).op("CALLDATALOAD", "MUL", "ADD")
            elif v == "V1":
                m = ones(32, 8) * 0xff
                for j in range(K):
                    a.push(j * 32).op("MLOAD")
                    if ph:
                        a.push(8 * ph).op("SHR")
                    a.pushn(m, 32).op("AND").push(j * 32).op("CALLDATALOAD", "MUL", "ADD")
                a.pushn(ones(32, 8), 32).dup(5).op("MUL").swap(1).op("SUB")
            elif v == "V2":
                m = ones(24, 10, 8 * ph) * 0xff
                for j in range(K):
                    a.push(j * 32).op("MLOAD").pushn(m, 32).op("AND").push(j * 32).op("CALLDATALOAD", "MUL", "ADD")
                a.pushn(ones(24, 10, 8 * ph), 32).dup(5).op("MUL").swap(1).op("SUB")
            else:  # V3
                for j in range(K):
                    a.push(j * 32).op("MLOAD").push(j * 32).op("CALLDATALOAD", "MUL", "ADD")
            a.dup(3).op("MSTORE").swap(1).push(32).op("ADD").swap(1)

    for addr, seg in addr_segs:
        src = 1
        a.push(src)
        i = 0
        while i < len(seg):
            size, phases = len(seg[i][0]), seg[i][1]
            n = 1
            while i + n < len(seg) and len(seg[i + n][0]) == size and seg[i + n][1] == phases:
                n += 1
            end = src + n * size
            uid += 1
            lbl = f"L{uid}"
            if n > 1:
                a.label(lbl)
            a.push(size).dup(2).push(0).pushn(int(addr, 16), 20).op("EXTCODECOPY")
            body(phases)
            a.push(size).op("ADD")
            if n > 1:
                a.pushn(end, 2).dup(2).op("LT").ref(lbl).op("JUMPI")
            src = end
            i += n
        a.op("POP")
    total = sum(g[1] for _, s in addr_segs for g in s)
    a.pushn(total * 32, 2).pushn(OUT, 2).op("RETURN")
    return a.build()


# ---------- V4: weights are code ----------
def code_chunks(W, K, L):
    """Unrolled PUSH32 chunks. Each chunk reads x from calldata, returns its accs."""
    groups = pack_stored(W, K, L)
    words = [[int.from_bytes(g[0][i:i + 32], "big") for i in range(0, len(g[0]), 32)] for g in groups]
    chunks, cur = [], Asm()
    n = 0

    def close():
        nonlocal cur, n
        cur.pushn(n * 32, 2).push(0).op("RETURN")
        chunks.append((cur.build(), n))
        cur, n = Asm(), 0

    for wg in words:
        size_after = len(cur.b) + K * 39 + 8 + 10
        if size_after > MAX_CODE:
            close()
        cur.push(0)
        for j, wv in enumerate(wg):
            cur.pushn(wv, 32).push(j * 32).op("CALLDATALOAD", "MUL", "ADD")
        cur.push(n * 32).op("MSTORE")
        n += 1
    close()
    return chunks


def call_kernel(calls, xwords, repeats=1):
    """calls: [(addr, acc count, K)]. Copies x to memory and STATICCALLs each chunk."""
    OUT = xwords * 32
    a = Asm()
    a.pushn(xwords * 32, 2).push(0).push(0).op("CALLDATACOPY")
    a.push(repeats).push(1)  # stack: [i, ok]
    a.label("loop")
    off = OUT
    for addr, n, K in calls:
        a.pushn(n * 32, 2).pushn(off, 2).pushn(K * 32, 2).push(0).pushn(int(addr, 16), 20).op("GAS", "STATICCALL", "AND")
        off += n * 32
    a.swap(1).push(1).swap(1).op("SUB").swap(1)
    a.dup(2).ref("loop").op("JUMPI")
    a.op("ISZERO").ref("fail").op("JUMPI")
    a.pushn(off - OUT, 2).pushn(OUT, 2).op("RETURN")
    a.label("fail").push(0).push(0).op("REVERT")
    return a.build()


# ---------- run one variant ----------
def make(v, W, K, L=None):
    """Deploy. Returns (kernel addr, decode fn, deploy gas, stored bytes)."""
    dep = 0
    stored = 0
    N = len(W)
    if v == "V4":
        L = L or lane_bits(K)
        chunks = code_chunks(W, K, L)
        calls = []
        for code, n in chunks:
            addr, g = deploy(code)
            dep += g
            stored += len(code)
            calls.append((addr, n, K))
        kaddr, kg = deploy(call_kernel(calls, K))
        lanes = 256 // L
        dec = lambda ws: [y for w in ws for y in unlane(w, L, lanes)][:N]
        return kaddr, dec, dep, stored, kg
    if v == "V0":
        groups = pack_v0(W, K)
        dec = lambda ws: [signed(w, 256) for w in ws]
    elif v == "V1":
        groups = pack_offset(W, K, 32, 8, 4)
        dec = lambda ws: [y for w in ws for y in unlane(w, 32, 8)][:N]
    elif v == "V2":
        groups = pack_offset(W, K, 30, 10, 3)
        phases = [ph for g in groups for ph in range(g[1])]
        dec = lambda ws: [y for ph, w in zip(phases, ws) for y in unlane(w, 24, 10, 8 * ph)][:N]
    else:
        L = L or lane_bits(K)
        groups = pack_stored(W, K, L)
        lanes = 256 // L
        dec = lambda ws: [y for w in ws for y in unlane(w, L, lanes)][:N]
    addr_segs = []
    for seg in segments(groups):
        blob = b"\0" + b"".join(g[0] for g in seg)
        addr, g = deploy(blob)
        dep += g
        stored += len(blob)
        addr_segs.append((addr, seg))
    kaddr, kg = deploy(stream_kernel(v, K, N, addr_segs))
    return kaddr, dec, dep, stored, kg


def run_case(v, W, x, L=None):
    K = len(x)
    kaddr, dec, dep, stored, kg = make(v, W, K, L)
    ws, gas, _ = measure(kaddr, x)
    return dec(ws) == matmul(W, x), gas, dep, stored, kg


def rand_case(rng, N, K, bound=127):
    return [[rng.randint(-bound, bound) for _ in range(K)] for _ in range(N)], [rng.randint(-128, 127) for _ in range(K)]


def worst_case(N, K):
    return [[127 if i % 2 else -127] * K for i in range(N)], [-128] * K


SHAPES = [(32, 64), (64, 64), (128, 64), (64, 128), (256, 64)]
VARIANTS = ["V0", "V1", "V2", "V3", "V4"]


def bench():
    rng = random.Random(7)
    print(f"{'kernel':7}{'shape':>10}{'exec gas':>12}{'gas/MAC':>9}{'deploy/w':>10}{'kernel dep':>12}  bit-exact", flush=True)
    for v in VARIANTS:
        for N, K in SHAPES:
            if v == "V0" and (N, K) not in [(32, 64), (64, 64)]:
                continue  # naive is the baseline, no need to pay for every shape
            W, x = rand_case(rng, N, K)
            ok, gas, dep, stored, kg = run_case(v, W, x)
            Wa, xa = worst_case(N, K)
            ok2, *_ = run_case(v, Wa, xa)
            assert ok and ok2, f"{v} {N}x{K} output mismatch random={ok} worst={ok2}"
            mac = N * K
            print(f"{v:7}{N:>5}x{K:<4}{gas:>12,}{gas / mac:>9.3f}{dep / mac:>10.1f}{kg:>12,}  yes", flush=True)


def negative_control():
    """The bit-exact check must go red when a lane is too narrow."""
    W, x = worst_case(64, 64)
    assert lane_bits(64) == 21
    ok, *_ = run_case("V3", W, x, L=20)
    assert not ok, "control failed: 20-bit lanes matched, the check cannot catch overflow"
    print("negative control: 20-bit lanes on 64 inputs mismatch the reference, as they must", flush=True)


def tight_lanes():
    """Hack: narrower lanes are safe when the real weights are small."""
    rng = random.Random(11)
    W, x = rand_case(rng, 64, 64, bound=15)
    assert max(sum(map(abs, r)) for r in W) * 128 < 1 << 17
    ok, gas, dep, *_ = run_case("V4", W, x, L=18)
    assert ok
    print(f"V4 tight  L=18 |w|<=15 64x64: exec {gas:,} gas/MAC {gas / 4096:.3f} (14 lanes)", flush=True)


def decode_cost():
    """Unpack lanes into one signed word per output, needed before requant."""
    rng = random.Random(5)
    W, x = rand_case(rng, 64, 64)
    L, lanes = 21, 12
    y = matmul(W, x)
    accs = [sum(v << (L * i) for i, v in enumerate(y[b:b + lanes])) % MOD for b in range(0, 64, lanes)]
    bias = 1 << (L - 1)
    a = Asm()
    n = 0
    for w in range(len(accs)):
        a.push(w * 32).op("CALLDATALOAD").pushn(ones(L, lanes) * bias, 32).op("ADD")
        for l in range(lanes):
            a.dup(1).push(L * l).op("SHR").push((1 << L) - 1).op("AND").push(bias).swap(1).op("SUB").push(n * 32).op("MSTORE")
            n += 1
        a.op("POP")
    a.pushn(n * 32, 2).push(0).op("RETURN")
    addr, _ = deploy(a.build())
    ws, gas, _ = measure(addr, [signed(v, 256) for v in accs])
    assert [signed(w, 256) for w in ws][:64] == y, "decode mismatch"
    print(f"lane decode: {gas:,} gas for {n} outputs = {gas / n:.1f} gas/output, bit-exact", flush=True)
    return gas / n


def config_a_tx():
    """All config A linear layers, 24 tokens, one real tx at the 16,777,216 cap."""
    rng = random.Random(3)
    layer = [(64, 64), (32, 64), (32, 64), (64, 64), (128, 64), (128, 64), (64, 128)]
    shapes = layer * 4 + [(256, 64)]
    x = [rng.randint(-128, 127) for _ in range(128)]
    calls, dep, ref = [], 0, None
    count = stored = 0
    for N, K in shapes:
        W = [[rng.randint(-127, 127) for _ in range(K)] for _ in range(N)]
        ref = (W, K)
        for code, n in code_chunks(W, K, lane_bits(K)):
            addr, g = deploy(code)
            dep += g
            count += 1
            stored += len(code)
            calls.append((addr, n, K))
    macs = sum(N * K for N, K in shapes)
    tokens = 24
    kaddr, _ = deploy(call_kernel(calls, 128, tokens))
    W, K = ref
    ws, exec_gas, used = measure(kaddr, x)
    L, lanes = lane_bits(64), 256 // lane_bits(64)
    n_acc = (256 + lanes - 1) // lanes
    got = [y for w in ws[-n_acc:] for y in unlane(w, L, lanes)][:256]
    assert got == matmul(W, x[:K]), "config A unembed output mismatch"
    assert used <= TX_CAP
    print(f"config A tx: {macs:,} MAC/token x {tokens} = {macs * tokens:,} MACs, tx gasUsed {used:,} of {TX_CAP:,} "
          f"(exec {exec_gas / (macs * tokens):.3f} gas/MAC), last matrix bit-exact", flush=True)
    print(f"config A deploy: {count} code contracts, {stored:,} bytes, {dep:,} gas", flush=True)
    return dict(macs=macs * tokens, used=used, exec=exec_gas, contracts=count, bytes=stored, deploy=dep)


def cap_check():
    """Does this anvil enforce EIP-7825? False passes in CI are the open question."""
    a, _ = deploy(bytes.fromhex("00"))
    try:
        send(b"", a, gas=TX_CAP + 1)
        print("EIP-7825: tx with cap+1 gas ACCEPTED, this anvil does not enforce the cap", flush=True)
        return False
    except Exception as e:
        print(f"EIP-7825: tx with cap+1 gas rejected: {str(e)[:110]}", flush=True)
        return True


def main():
    node = subprocess.Popen(["anvil", "--hardfork", "osaka", "--gas-limit", "60000000", "--port", str(PORT), "--silent"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                rpc("eth_chainId")
                break
            except Exception:
                time.sleep(0.1)
        negative_control()
        cap_check()
        bench()
        tight_lanes()
        decode_cost()
        config_a_tx()
    finally:
        node.terminate()


if __name__ == "__main__":
    sys.exit(main())
