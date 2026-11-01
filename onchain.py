#!/usr/bin/env python3
"""The model as hand-written EVM, on local anvil (osaka), checked against intmodel.py.

Weights are PUSH32 code. Lanes are 16 bits.

Contracts: the hand-written engine (stateless, runs under STATICCALL) and WorstLLM.sol, which
commits states, stores text and emits events.
Engine calldata: seed, new tokens, prompt length, flags, prompt ids (1 byte each), state blob.
Flags: bit 0 has state, bit 1 stop at the end of the reply. Engine output: state blob, then text.

Set LLM_NODE=geth to run on geth --dev instead of anvil.
Set LLM_NODE=hoodi or LLM_NODE=mainnet for a real network. It needs PK in the environment and signs
only for that chain id. Mainnet sends nothing unless MAINNET_DEPLOY=yes. Every send is saved before it
goes out, so an interrupted deploy resumes and never deploys a contract twice.

Usage: onchain.py CKPT deploy             deploy, or show the plan on mainnet without the flag
       onchain.py CKPT cross FILE         a fixed tx sequence, results written to FILE
       onchain.py CKPT test               exact-match, continuation and tamper tests
       onchain.py CKPT max [SEED]         longest fresh and continued tx at the cap
       onchain.py CKPT sweep [COUNT]      worst gas over seeds and prompts
       onchain.py CKPT profile            gas by section
       onchain.py CKPT chat SEED N "text" ["follow-up" ...]
"""
import hashlib, json, os, random, subprocess, sys, tempfile, time, urllib.error, urllib.request
import torch
from tokenizers import Tokenizer
import intmodel as im, train as tr
from intmodel import L, LANES, ONES, BIAS, LMASK, ABS_S, DW, FW, VW_
import m0_bench
from m0_bench import Asm, MAX_CODE, PORT, SENDER, TX_CAP, deploy, rpc, send

m0_bench.OPS.update(XOR=0x18, DIV=0x04, MOD=0x06, GT=0x11, SGT=0x13, BYTE=0x1a, KECCAK=0x20, JUMP=0x56, MCOPY=0x5e, LOG2=0xa2)

M32 = (1 << 256) - 1
B15 = 1 << (L - 1)
H, X, QW = 0x0, 0x100, 0x1200
KW, VW, OW = QW + 0x80, QW + 0x100, QW + 0x180
KWIN, VWIN, EWIN, SAMP = 0x1600, 0x1A00, 0x1E00, 0x2200
VT, VT0, VSLOT, VN32, VOFF32, VIDX, VLEN, VTOTAL, VPLEN, VWORD = (0x2400 + i * 32 for i in range(10))
SEED, TXT, STATE = 0xC0, 0x2600, 0x2FC0
CACHE = STATE + 64
VCUR = STATE + 32
MAX_PER_TX = 64
STACK_X = 14  # inputs kept on the stack: DUP costs 3 gas, PUSH + CALLDATALOAD costs 6
PROFILE = False
LABELS = []
EMB_PER = (MAX_CODE - 1) // (DW * 32)


def chunks(rows, K):
    """Unrolled PUSH32 contracts for one matrix. Returns [(code, output words)]."""
    out, n = [], 0

    def start():
        c = Asm()
        for j in range(STACK_X - 1, -1, -1):
            c.push(j * 32).op("CALLDATALOAD")
        return c

    a = start()

    def close():
        nonlocal a, n
        a.pushn(n * 32, 2).push(0).op("RETURN")
        out.append((a.build(), n))
        a, n = start(), 0

    for row in rows:
        if len(a.b) + K * 39 + 18 + STACK_X * 4 > MAX_CODE:
            close()
        a.push(0)
        for j, w in enumerate(row):
            a.pushn(w, 32)
            if j < STACK_X:
                a.dup(j + 3)
            else:
                a.push(j * 32).op("CALLDATALOAD")
            a.op("MUL", "ADD")
        a.push(n * 32).op("MSTORE")
        n += 1
    close()
    return out


# name: rpc, chain id, tip in wei, highest base fee to start or continue a deploy, in wei
NETS = {"hoodi": ("https://ethereum-hoodi-rpc.publicnode.com", 560048, 200_000_000, 50_000_000_000),
        "mainnet": ("https://ethereum-rpc.publicnode.com", 1, 20_000_000, 120_000_000)}
NET = os.environ.get("LLM_NODE", "anvil")
REMOTE = NET in NETS
DRY = NET == "mainnet" and os.environ.get("MAINNET_DEPLOY") != "yes"  # mainnet sends nothing without this flag
REMOTE_RPC, REMOTE_CHAIN, TIP, MAX_BASE = NETS.get(NET, ("", 0, 0, 0))
DEPLOY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"deployments_{NET}.json")
BATCH = 12  # unconfirmed deploys at once: public nodes limit pending txs per account


def remote_rpc(method, *params):
    req = urllib.request.Request(REMOTE_RPC, json.dumps(dict(jsonrpc="2.0", id=1, method=method, params=list(params))).encode(),
                                 {"content-type": "application/json", "user-agent": "curl/8"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.load(r)
            break
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == 5:
                raise
            time.sleep(3)
    if "error" in out:
        raise RuntimeError(f"{method}: {out['error']}")
    return out["result"]


def remote_send(data, to=None, gas=TX_CAP):
    """Signed in this process for one chain id, so the tx cannot land on another network."""
    from eth_account import Account
    acct = Account.from_key(os.environ["PK"])
    assert not DRY, "dry run must not send"
    assert int(remote_rpc("eth_chainId"), 16) == REMOTE_CHAIN, f"not {NET}, refusing to send"
    base = int(remote_rpc("eth_getBlockByNumber", "latest", False)["baseFeePerGas"], 16)
    assert base <= MAX_BASE, f"base fee {base / 1e9:.3f} gwei is above the limit, wait and rerun"
    tip = TIP
    tx = dict(chainId=REMOTE_CHAIN, nonce=int(remote_rpc("eth_getTransactionCount", acct.address, "pending"), 16), gas=gas,
              maxFeePerGas=base * 2 + tip, maxPriorityFeePerGas=tip, value=0, data=data)
    if to:
        tx["to"] = Web3Checksum(to)
    h = remote_rpc("eth_sendRawTransaction", "0x" + bytes(acct.sign_transaction(tx).raw_transaction).hex())
    for _ in range(100):
        rc = remote_rpc("eth_getTransactionReceipt", h)
        if rc:
            break
        time.sleep(4)
    assert rc and int(rc["status"], 16) == 1, f"tx failed: {h}"
    return rc


def Web3Checksum(addr):
    from eth_utils import to_checksum_address
    return to_checksum_address(addr)


class Gen:
    def __init__(self):
        self.gas, self.bytes, self.n, self.addrs = 0, 0, 0, []

    def put(self, code):
        if REMOTE:
            key = hashlib.sha256(code).hexdigest()
            done = json.load(open(DEPLOY_FILE)) if os.path.exists(DEPLOY_FILE) else {}
            if key in done:
                if "gas" not in done[key]:
                    rc = None
                    for _ in range(200):
                        rc = remote_rpc("eth_getTransactionReceipt", done[key]["sent"])
                        if rc:
                            break
                        time.sleep(3)
                    assert rc, f"earlier send not mined: {done[key]['sent']}. check it before rerunning"
                    assert int(rc["status"], 16) == 1, f"earlier deploy failed: {done[key]['sent']}. delete its entry in {DEPLOY_FILE} to retry it"
                    done[key].update(gas=int(rc["gasUsed"], 16))
                    json.dump(done, open(DEPLOY_FILE, "w"), indent=0)
                self.addrs.append(done[key]["addr"])
                self.gas += done[key]["gas"]
                self.bytes += len(code)
                self.n += 1
                return done[key]["addr"]
        if REMOTE:
            addr = self.send_create(code, key)
            self.addrs.append(addr)
            self.bytes += len(code)
            self.n += 1
            return addr
        addr, g = deploy(code)
        self.addrs.append(addr)
        self.gas += g
        self.bytes += len(code)
        self.n += 1
        return addr

    def send_create(self, code, key):
        """Send without waiting. A CREATE address depends only on sender and nonce, so it is known now."""
        import rlp
        from eth_account import Account
        from eth_utils import keccak
        acct = Account.from_key(os.environ["PK"])
        if not hasattr(self, "nonce"):
            assert int(remote_rpc("eth_chainId"), 16) == REMOTE_CHAIN, f"not {NET}, refusing to send"
            self.nonce = int(remote_rpc("eth_getTransactionCount", acct.address, "pending"), 16)
            self.pending, self.planned = [], 0
        init = bytes.fromhex("61") + len(code).to_bytes(2, "big") + bytes.fromhex("80600a5f395ff3") + code
        addr = "0x" + keccak(rlp.encode([bytes.fromhex(acct.address[2:]), self.nonce]))[12:].hex()
        limit = 60_000 + 220 * len(init)  # a tight limit lets about 11 large deploys share a 60M block, not 3
        if DRY:
            self.nonce += 1
            self.planned += limit
            return addr
        if len(self.pending) >= BATCH:
            self.wait_all()
        base = int(remote_rpc("eth_getBlockByNumber", "latest", False)["baseFeePerGas"], 16)
        assert base <= MAX_BASE, f"base fee {base / 1e9:.3f} gwei is above the limit. rerun later, it resumes"
        tx = dict(chainId=REMOTE_CHAIN, nonce=self.nonce, gas=limit, maxFeePerGas=base * 2 + TIP, maxPriorityFeePerGas=TIP, value=0, data=init)
        raw = "0x" + bytes(acct.sign_transaction(tx).raw_transaction).hex()
        h = "0x" + keccak(hexstr=raw).hex()
        done = json.load(open(DEPLOY_FILE)) if os.path.exists(DEPLOY_FILE) else {}
        done[key] = dict(addr=addr, sent=h)  # saved before sending, so a crash can never deploy it twice
        json.dump(done, open(DEPLOY_FILE, "w"), indent=0)
        remote_rpc("eth_sendRawTransaction", raw)
        self.nonce += 1
        self.pending.append((h, key, addr))
        print(f"  sent #{self.n + 1}: {addr}", flush=True)
        return addr

    def wait_all(self):
        done = json.load(open(DEPLOY_FILE)) if os.path.exists(DEPLOY_FILE) else {}
        for h, key, addr in getattr(self, "pending", []):
            for _ in range(200):
                rc = remote_rpc("eth_getTransactionReceipt", h)
                if rc:
                    break
                time.sleep(3)
            assert rc, f"not mined yet: {h}. rerun later, it resumes"
            assert int(rc["status"], 16) == 1, f"deploy failed: {h}"
            assert rc["contractAddress"].lower() == addr, "predicted address was wrong"
            done[key] = dict(addr=addr, gas=int(rc["gasUsed"], 16), tx=h)
            self.gas += done[key]["gas"]
            json.dump(done, open(DEPLOY_FILE, "w"), indent=0)
        print(f"  confirmed {len(getattr(self, 'pending', []))} deploys", flush=True)
        self.pending = []

    def finish(self):
        """Wait for every deploy, so a rerun skips them."""
        if not DRY:
            self.wait_all()

    def data(self, b):
        return self.put(b"\0" + b)


def build_main(model, g, table, embs, pos):
    W, P = model.window, model.positions
    state_size = 64 + W * 8 * model.layers * 32
    a = Asm()
    uid = [0]
    res_subs, relu_subs, lrs_subs = {}, {}, set()

    def fresh(p="L"):
        uid[0] += 1
        return f"{p}{uid[0]}"

    def probe(name):
        if PROFILE:
            if name not in LABELS:
                LABELS.append(name)
            a.op("GAS").pushn(LABELS.index(name) + 1, 1).push(0).push(0).op("LOG2")

    def call(label):
        r = fresh("R")
        a.ref(r).ref(label).op("JUMP").label(r)

    def load(addr):
        a.pushn(addr, 2).op("MLOAD")

    def store(addr):
        a.pushn(addr, 2).op("MSTORE")

    def lrs(s):
        """Rounded arithmetic shift of every lane of the top word."""
        if s == 0:
            return
        a.pushn((BIAS + (ONES << (s - 1))) & M32, 32).op("ADD").push(s).op("SHR").pushn(ONES * ((1 << (L - s)) - 1), 32).op("AND")
        a.pushn(ONES * (1 << (L - 1 - s)), 32).swap(1).op("SUB")

    def lrs_call(s):
        """Same as lrs, through one shared subroutine per shift: 21 gas more, about 100 bytes less."""
        if s == 0:
            return
        lrs_subs.add(s)
        r = fresh("R")
        a.ref(r).swap(1).ref(f"lrs{s}").op("JUMP").label(r)

    def mat(rows, K, ret):
        off = ret
        for code, n in chunks(rows, K):
            a.pushn(n * 32, 2).pushn(off, 2).pushn(K * 32, 2).pushn(X, 2).pushn(int(g.put(code), 16), 20).op("GAS", "STATICCALL", "AND")
            off += n * 32

    def named(store_, s, prefix):
        store_.setdefault(s, f"{prefix}{s}")
        call(store_[s])

    def lanes_signed(dst):
        """Unpack the biased word on top of the stack into scalars at dst."""
        for j in range(LANES):
            a.dup(1)
            if j:
                a.push(L * j).op("SHR")
            a.pushn(LMASK, 2).op("AND").pushn(B15, 2).swap(1).op("SUB")
            store(dst + j * 32)
        a.op("POP")

    # ---------- entry ----------
    a.push(1)  # ok flag, stays at the bottom, every STATICCALL ANDs into it
    a.push(0).op("CALLDATALOAD").pushn(SEED, 1).op("MSTORE")
    a.pushn(64, 1).op("CALLDATALOAD"); store(VPLEN)
    a.pushn(96, 1).op("CALLDATALOAD").pushn(1, 1).op("AND", "ISZERO").ref("nostate").op("JUMPI")
    # the front contract checks that this state was committed
    a.pushn(state_size, 2); load(VPLEN); a.pushn(128, 1).op("ADD").pushn(STATE, 2).op("CALLDATACOPY")
    a.label("nostate")
    load(STATE); a.dup(1); store(VT); store(VT0)
    a.pushn(32, 1).op("CALLDATALOAD"); load(VPLEN); a.op("ADD").dup(1)  # prompt + new tokens
    a.pushn(MAX_PER_TX, 1).op("LT").ref("fail").op("JUMPI")
    load(VT0); a.op("ADD"); store(VTOTAL)

    a.label("loop")
    load(VT)
    a.dup(1).pushn(W, 1).swap(1).op("MOD").pushn(32, 1).op("MUL"); store(VSLOT)
    a.pushn(1, 1).op("ADD").dup(1).pushn(W, 1).op("LT").ref("full").op("JUMPI")  # W < t+1
    a.ref("sized").op("JUMP")
    a.label("full").op("POP").pushn(W, 1)
    a.label("sized")  # stack: [n, ok]
    a.dup(1).pushn(32, 1).op("MUL"); store(VN32)
    a.pushn(W, 1).op("SUB").dup(1); store(VIDX)
    a.pushn(32, 1).op("MUL"); store(VOFF32)

    # ---------- embed ----------
    probe("embed")
    size = DW * 32
    load(VCUR); a.pushn(EMB_PER - 1, 1).op("LT").ref("high").op("JUMPI")
    a.pushn(size, 1); load(VCUR); a.pushn(size, 1).op("MUL").pushn(1, 1).op("ADD")
    a.push(H).pushn(int(embs[0], 16), 20).op("EXTCODECOPY").ref("embdone").op("JUMP")
    a.label("high").pushn(size, 1).pushn(EMB_PER, 1); load(VCUR); a.op("SUB").pushn(size, 1).op("MUL").pushn(1, 1).op("ADD")
    a.push(H).pushn(int(embs[1], 16), 20).op("EXTCODECOPY")
    a.label("embdone").pushn(size, 1).pushn(P, 1); load(VT); a.op("MOD").pushn(size, 1).op("MUL").pushn(1, 1).op("ADD")
    a.pushn(X, 2).pushn(int(pos, 16), 20).op("EXTCODECOPY")
    for i in range(DW):
        load(H + i * 32); load(X + i * 32); a.op("ADD"); store(H + i * 32)

    # ---------- layers ----------
    for l, b in enumerate(model.blocks):
        probe("norm")
        call("norm")
        probe("matmul qkv")
        mat(b["q"], tr.DIM, QW)
        mat(b["k"], tr.DIM, KW)
        mat(b["v"], tr.DIM, VW)
        base = CACHE + l * 8 * W * 32
        lut = int.from_bytes(bytes(b["lut"]), "big")
        for h in range(tr.HEADS):
            kb = base + h * 2 * W * 32
            vb = kb + W * 32
            probe("cache+requant")
            load(KW + h * 32); lrs_call(b["sk"]); load(VSLOT); a.pushn(kb, 3).op("ADD", "MSTORE")
            load(VW + h * 32); lrs_call(b["sv"]); load(VSLOT); a.pushn(vb, 3).op("ADD", "MSTORE")
            probe("attention")
            r = fresh("R")
            a.ref(r).pushn(b["a"], 1).pushn(lut, 32)
            load(QW + h * 32); lrs_call(b["sq"])
            a.pushn(kb, 3).ref("head").op("JUMP").label(r)
            lrs_call(8)
            r = fresh("R")
            a.ref(r).swap(1).ref(f"unp{h}").op("JUMP").label(r)
        probe("matmul o")
        mat(b["o"], tr.DIM, OW)
        probe("residual add")
        named(res_subs, b["so"], "res")
        probe("norm")
        call("norm")
        probe("matmul f1")
        mat(b["f1"], tr.DIM, OW)
        probe("relu+unpack")
        named(relu_subs, b["sr"], "relu")
        probe("matmul f2")
        mat(b["f2"], tr.FFN, OW)
        probe("residual add")
        named(res_subs, b["sf"], "res")

    # ---------- next token ----------
    probe("other")
    load(VPLEN); load(VT0); load(VT); a.op("SUB", "LT")  # t - t0 < plen
    a.ref("fromprompt").op("JUMPI")
    probe("norm")
    call("norm")
    probe("matmul unembed")
    mat(model.unembed, tr.DIM, OW)
    probe("sample")
    a.ref("sample").op("JUMP")
    a.label("fromprompt")
    load(VT0); load(VT); a.op("SUB").pushn(128, 1).op("ADD", "CALLDATALOAD").push(0).op("BYTE"); store(VCUR)
    a.label("gotnext")
    probe("other")
    load(VT); a.pushn(1, 1).op("ADD"); store(VT)
    # an end-of-chat token stops the tx and prints nothing
    load(VCUR); a.op("ISZERO").ref("done").op("JUMPI")
    a.pushn(32, 1); load(VCUR); a.pushn(32, 1).op("MUL").pushn(1, 1).op("ADD").pushn(VWORD, 2).pushn(int(table, 16), 20).op("EXTCODECOPY")
    load(VWORD); load(VLEN); a.pushn(TXT, 2).op("ADD", "MSTORE")
    load(VWORD); a.pushn(31, 1).op("BYTE"); load(VLEN); a.op("ADD"); store(VLEN)
    # a sampled token that ends the reply stops the tx, if the caller asked for that (flag bit 1)
    load(VWORD); a.pushn(30, 1).op("BYTE", "ISZERO").ref("nostop").op("JUMPI")
    a.pushn(96, 1).op("CALLDATALOAD").pushn(2, 1).op("AND", "ISZERO").ref("nostop").op("JUMPI")
    load(VT0); load(VT); a.op("SUB"); load(VPLEN); a.op("LT").ref("done").op("JUMPI")  # plen < t - t0
    a.label("nostop")
    load(VTOTAL); load(VT); a.op("LT").ref("loop").op("JUMPI")  # t < total

    a.label("done")
    a.op("ISZERO").ref("fail").op("JUMPI")
    # output: the state, then the text
    load(VT); store(STATE)
    load(VLEN); a.pushn(TXT, 2).pushn(STATE + state_size, 2).op("MCOPY")
    load(VLEN); a.pushn(state_size, 2).op("ADD").pushn(STATE, 2).op("RETURN")
    a.label("fail").push(0).push(0).op("REVERT")

    # ---------- norm: x = h * DIM * 2^FX / sum|h| ----------
    a.label("norm").push(0)
    m15 = ONES * ((1 << (L - 1)) - 1)
    for i in range(DW):
        load(H + i * 32); a.pushn(BIAS, 32).op("ADD")
        a.dup(1).pushn(L - 1, 1).op("SHR").pushn(ONES, 32).op("AND").pushn(ONES, 32).op("XOR")  # [neg, v, t]
        a.dup(1).pushn(LMASK, 2).op("MUL").dup(3).op("XOR").pushn(m15, 32).op("AND", "ADD")  # [mag, v, t]
        a.pushn(ONES, 32).op("MUL").pushn(240, 1).op("SHR")
        a.swap(1).op("POP", "ADD")
    a.push(model.abs_c).op("DIV")  # [m, ret]
    for i in range(DW):
        a.dup(1); load(H + i * 32); a.op("MUL"); lrs(ABS_S); a.pushn(BIAS, 32).op("ADD")
        lanes_signed(X + i * LANES * 32)
    a.op("POP", "JUMP")

    # ---------- attention for one head: [kb, q, lut, a, ret] -> [A] ----------
    # MCOPY moves the head's ring to fixed addresses, a computed jump enters the unrolled code.
    def entry(base, length):
        load(VIDX)
        a.pushn(length, 1).op("MUL").ref(base).op("ADD", "JUMP")

    a.label("head")
    load(VN32); a.dup(2); load(VOFF32); a.pushn(KWIN, 2).op("ADD", "MCOPY")
    load(VN32); a.pushn(W * 32, 2).dup(3).op("ADD"); load(VOFF32); a.pushn(VWIN, 2).op("ADD", "MCOPY")
    a.op("POP").pushn(1 << 255, 32)  # [mx, q, lut, a, ret]

    def item1(i):
        a.op("JUMPDEST")
        load(KWIN + i * 32); a.dup(3).op("MUL").pushn(240, 1).op("SAR")
        a.dup(1); store(EWIN + i * 32)
        a.dup(2).dup(2).op("SGT", "ISZERO")
        sk = fresh("S")
        a.ref(sk).op("JUMPI").swap(1).label(sk).op("POP")

    def item2(i):
        a.op("JUMPDEST").dup(3)
        load(EWIN + i * 32); a.dup(4).op("SUB").dup(6).op("SHR", "BYTE")
        a.dup(1); store(EWIN + i * 32)
        a.op("ADD")

    def item3(i):
        a.op("JUMPDEST")
        load(EWIN + i * 32); a.dup(3).op("MUL").pushn(12, 1).op("SHR")
        load(VWIN + i * 32); a.op("MUL", "ADD")

    def item_len(fn):
        n0, f0, lb, u0 = len(a.b), len(a.fix), dict(a.labels), uid[0]
        fn(0)
        n = len(a.b) - n0
        del a.b[n0:], a.fix[f0:]
        a.labels, uid[0] = lb, u0
        return n

    L1, L2, L3 = item_len(item1), item_len(item2), item_len(item3)
    entry("p1", L1)
    a.labels["p1"] = len(a.b)
    for i in range(W):
        item1(i)
    a.op("JUMPDEST").swap(1).op("POP").push(0)  # [sum, mx, lut, a, ret]
    entry("p2", L2)
    a.labels["p2"] = len(a.b)
    for i in range(W):
        item2(i)
    a.op("JUMPDEST").pushn(1 << 20, 3).op("DIV")  # [mm, mx, lut, a, ret]
    a.swap(3).op("POP", "POP", "POP").push(0)  # [A, mm, ret]
    entry("p3", L3)
    a.labels["p3"] = len(a.b)
    for i in range(W):
        item3(i)
    a.op("JUMPDEST").swap(1).op("POP").swap(1).op("JUMP")

    for h in range(tr.HEADS):
        a.label(f"unp{h}").pushn(BIAS, 32).op("ADD")
        lanes_signed(X + h * LANES * 32)
        a.op("JUMP")

    for s in sorted(lrs_subs):
        a.label(f"lrs{s}")
        lrs(s)
        a.swap(1).op("JUMP")

    for s, name in res_subs.items():
        a.label(name)
        for i in range(DW):
            load(OW + i * 32); lrs(s); load(H + i * 32); a.op("ADD"); store(H + i * 32)
        a.op("JUMP")

    for s, name in relu_subs.items():
        a.label(name)
        for i in range(FW):
            load(OW + i * 32); a.pushn(BIAS, 32).op("ADD")
            a.dup(1).pushn(L - 1, 1).op("SHR").pushn(ONES, 32).op("AND").pushn((1 << (L - 1)) - 1, 2).op("MUL", "AND")
            if s:
                a.pushn(ONES << (s - 1), 32).op("ADD").push(s).op("SHR").pushn(ONES * ((1 << (L - s)) - 1), 32).op("AND")
            for j in range(LANES):
                a.dup(1)
                if j:
                    a.push(L * j).op("SHR")
                a.pushn(LMASK, 2).op("AND"); store(X + (i * LANES + j) * 32)
            a.op("POP")
        a.op("JUMP")

    # ---------- sampling ----------
    lut_out = int.from_bytes(bytes(model.lut_out), "big")
    a.label("sample").pushn(lut_out, 32).push(0)  # [mx, lut, ok]
    for i in range(VW_):
        r = fresh("R")
        a.ref(r).swap(1); load(OW + i * 32); a.ref("smax").op("JUMP").label(r)
    a.push(0)  # [sum, mx, lut, ok]
    for i in range(VW_):
        r = fresh("R")
        a.ref(r).dup(4).dup(4); load(OW + i * 32); a.ref("ssum").op("JUMP").label(r)
        a.dup(1); store(SAMP + i * 32); a.op("ADD")
    load(VT); a.pushn(SEED + 32, 1).op("MSTORE")
    a.pushn(64, 1).pushn(SEED, 1).op("KECCAK", "MOD")  # [r, mx, lut, ok]
    for i in range(VW_):
        load(SAMP + i * 32)
        a.dup(1).dup(3).op("LT").ref(f"fw{i}").op("JUMPI").swap(1).op("SUB")
    a.ref("fail").op("JUMP")
    for i in range(VW_):
        a.label(f"fw{i}").op("POP").pushn(i, 1).ref("scan").op("JUMP")
    a.label("scan").dup(1).pushn(LANES, 1).op("MUL").swap(1)
    a.pushn(32, 1).op("MUL").pushn(OW, 2).op("ADD", "MLOAD").pushn(BIAS, 32).op("ADD")
    a.swap(2).swap(1)  # [base, r, v, mx, lut, ok]
    for j in range(LANES):
        a.dup(5).dup(4)
        if j:
            a.push(L * j).op("SHR")
        a.pushn(LMASK, 2).op("AND").dup(6).op("SUB").pushn(model.aout, 1).op("SHR", "BYTE")  # [e, base, r, v, mx, lut, ok]
        a.dup(1).dup(4).op("LT").ref(f"fl{j}").op("JUMPI")
        a.dup(3).op("SUB").swap(2).op("POP")
    a.ref("fail").op("JUMP")
    for j in range(LANES):
        a.label(f"fl{j}").op("POP").pushn(j, 1).op("ADD")
        store(VCUR)
        a.op("POP", "POP", "POP", "POP").ref("gotnext").op("JUMP")

    a.label("smax").pushn(BIAS, 32).op("ADD")  # [w, mx, ret] -> [mx]
    for j in range(LANES):
        a.dup(1)
        if j:
            a.push(L * j).op("SHR")
        a.pushn(LMASK, 2).op("AND").dup(3).dup(2).op("GT", "ISZERO")
        sk = fresh("S")
        a.ref(sk).op("JUMPI").swap(2).label(sk).op("POP")
    a.op("POP").swap(1).op("JUMP")

    a.label("ssum").pushn(BIAS, 32).op("ADD").push(0)  # [w, mx, lut, ret] -> [r]
    for j in range(LANES):
        a.dup(4).dup(3)
        if j:
            a.push(L * j).op("SHR")
        a.pushn(LMASK, 2).op("AND").dup(5).op("SUB").pushn(model.aout, 1).op("SHR", "BYTE", "ADD")
    a.swap(1).op("POP").swap(1).op("POP").swap(1).op("POP").swap(1).op("JUMP")
    return a.build()


class Chain:
    def __init__(self, ckpt):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model = im.IntModel(ck, im.calib_tokens())
        self.tok = Tokenizer.from_file(f"{tr.DATA}/tok.json")
        self.client = os.environ.get("LLM_NODE", "anvil")
        if REMOTE:
            global send
            m0_bench.send = send = remote_send
            m0_bench.rpc = remote_rpc
            globals()["rpc"] = remote_rpc
            from eth_account import Account
            m0_bench.SENDER = Account.from_key(os.environ["PK"]).address
        cmd = ["anvil", "--hardfork", "osaka", "--gas-limit", "60000000", "--port", str(PORT), "--silent"] + (["--steps-tracing"] if len(sys.argv) > 2 and sys.argv[2] == "trace" else [])
        if self.client == "geth":
            cmd = ["geth", "--dev", "--dev.gaslimit", "60000000", "--http", "--http.addr", "127.0.0.1", "--http.port", str(PORT), "--http.api", "eth,net,web3,debug",
                   "--rpc.gascap", "1000000000", "--datadir", tempfile.mkdtemp(), "--nodiscover", "--verbosity", "1"]
        self.node = None if REMOTE else subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(300):
            try:
                rpc("eth_chainId")
                break
            except Exception:
                time.sleep(0.1)
        if self.client == "geth":
            m0_bench.SENDER = rpc("eth_accounts")[0]
        g = self.g = Gen()
        tb = bytearray()
        for i in range(tr.VOCAB):
            b = b"" if i == 0 else self.tok.decode([i]).encode()
            assert len(b) < 30
            tb += b.ljust(30, b"\0") + bytes([1 if b"\n" in b else 0, len(b)])  # byte 30 marks the end of a reply
        words = lambda rows: b"".join(w.to_bytes(32, "big") for row in rows for w in row)
        table = g.data(bytes(tb))
        embs = [g.data(words(self.model.emb[:EMB_PER])), g.data(words(self.model.emb[EMB_PER:]))]
        pos = g.data(words(self.model.pos))
        code = build_main(self.model, g, table, embs, pos)
        self.engine = g.put(code)
        self.engine_code = code
        if REMOTE:
            g.finish()
        self.stops = {i for i in range(1, tr.VOCAB) if "\n" in self.tok.decode([i])}
        self.front = self.deploy_front()
        print(f"engine {len(code):,} bytes; {'planned' if DRY else 'deployed'} {g.n} contracts, {g.bytes:,} bytes, {g.gas:,} gas; {len(self.stops)} stop tokens; front {self.front}", flush=True)

    def deploy_front(self):
        std = dict(language="Solidity", sources={"WorstLLM.sol": dict(content=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "WorstLLM.sol")).read())},
                   settings=dict(evmVersion="osaka", optimizer=dict(enabled=True, runs=200), outputSelection={"*": {"*": ["evm.bytecode.object", "abi"]}}))
        out = json.loads(subprocess.run(["solc", "--standard-json"], input=json.dumps(std), capture_output=True, text=True).stdout)
        assert not [e for e in out.get("errors", []) if e["severity"] == "error"], out["errors"]
        init = bytes.fromhex(out["contracts"]["WorstLLM.sol"]["WorstLLM"]["evm"]["bytecode"]["object"]) + bytes(12) + bytes.fromhex(self.engine[2:])
        self.sel_generate = selector("generate(uint256,uint256,bytes,bytes)")
        self.sel_preview = selector("preview(uint256,uint256,bytes,bytes)")
        self.topic_reply = "0x%064x" % im.keccak(b"Reply(bytes32,bytes32,string)")
        self.topic_state = "0x%064x" % im.keccak(b"State(bytes32,bytes)")
        if REMOTE:
            key = hashlib.sha256(init).hexdigest()
            done = json.load(open(DEPLOY_FILE)) if os.path.exists(DEPLOY_FILE) else {}
            if DRY:
                self.g.planned += 700_000
                return "0x" + "0" * 40
            if key not in done:
                rc = send(init, None, gas=1_000_000)
                done[key] = dict(addr=rc["contractAddress"], gas=int(rc["gasUsed"], 16), tx=rc["transactionHash"], front=True)
                json.dump(done, open(DEPLOY_FILE, "w"), indent=0)
            return done[key]["addr"]
        rc = send(init, None, gas=5_000_000)
        print(f"front contract deploy: {int(rc['gasUsed'], 16):,} gas", flush=True)
        return rc["contractAddress"]

    def payload(self, seed, n, prompt=(), state=None, stop=False):
        """Calldata for the engine directly."""
        flags = (1 if state else 0) | (2 if stop else 0)
        return seed.to_bytes(32, "big") + n.to_bytes(32, "big") + len(prompt).to_bytes(32, "big") + flags.to_bytes(32, "big") + bytes(prompt) + (state or b"")

    def front_data(self, sel, seed, n, prompt=(), state=None):
        """ABI encoding of generate or preview."""
        p, s = bytes(prompt), state or b""
        pad = lambda b: b + bytes(-len(b) % 32)
        w = lambda v: v.to_bytes(32, "big")
        return sel + w(seed) + w(n) + w(128) + w(160 + len(pad(p))) + w(len(p)) + pad(p) + w(len(s)) + pad(s)

    def tx(self, seed, n, prompt=(), state=None, gas=TX_CAP):
        """generate through the front. Returns (text, state blob, gasUsed, receipt), or None if it reverts."""
        try:
            rc = send(self.front_data(self.sel_generate, seed, n, prompt, state), self.front, gas=gas)
        except AssertionError:
            return None
        reply = next(l for l in rc["logs"] if l["topics"][0] == self.topic_reply)
        st = next(l for l in rc["logs"] if l["topics"][0] == self.topic_state)
        return abi_bytes(reply["data"]).decode(), abi_bytes(st["data"]), int(rc["gasUsed"], 16), rc

    def preview(self, seed, n, prompt=(), state=None):
        """The free path, returns (text, state hash, state blob)."""
        out = rpc("eth_call", {"from": m0_bench.SENDER, "to": self.front, "data": "0x" + self.front_data(self.sel_preview, seed, n, prompt, state).hex(), "gas": hex(10**9)}, "latest")
        d = bytes.fromhex(out[2:])
        off = int.from_bytes(d[64:96], "big")
        return abi_bytes(out).decode(), "0x" + out[66:130], d[off + 32:off + 32 + int.from_bytes(d[off:off + 32], "big")]

    def engine_call(self, *a, **k):
        """Engine directly. Returns (state blob, text)."""
        size = 64 + self.model.window * 8 * self.model.layers * 32
        out = bytes.fromhex(rpc("eth_call", {"from": m0_bench.SENDER, "to": self.engine, "data": "0x" + self.payload(*a, **k).hex(), "gas": hex(10**9)}, "latest")[2:])
        return out[:size], out[size:].decode()

    def view(self, sig, arg):
        return rpc("eth_call", {"to": self.front, "data": "0x" + (selector(sig) + bytes.fromhex(arg[2:])).hex()}, "latest")

    def close(self):
        if self.node:
            self.node.terminate()


def selector(sig):
    return (im.keccak(sig.encode()) >> 224).to_bytes(4, "big")


def abi_bytes(hexdata):
    """The first dynamic string or bytes in ABI-encoded data."""
    d = bytes.fromhex(hexdata[2:])
    off = int.from_bytes(d[:32], "big")
    n = int.from_bytes(d[off:off + 32], "big")
    return d[off + 32:off + 32 + n]


def encode(c, text):
    return c.tok.encode(text).ids


def turn(text):
    return f"User: {text}\nBot:"


def test(c):
    m, seed = c.model, 7
    q = encode(c, turn("do you like dogs?"))

    # 1. engine alone: text and state match the reference, across a split and in one long run
    st = m.new_state()
    ref = m.generate(st, seed, 12, q)
    blob, text = c.engine_call(seed, 12, q)
    assert text == c.tok.decode(ref) and blob == m.blob(st), (text, c.tok.decode(ref))
    st2 = m.load_blob(blob)
    ref2 = m.generate(st2, seed, 15)
    blob2, text2 = c.engine_call(seed, 15, state=blob)
    assert text2 == c.tok.decode(ref2) and blob2 == m.blob(st2)
    long_blob, long_text = c.engine_call(seed, 27, q)
    assert long_text == text + text2 and long_blob == blob2
    print("1. engine: text and state match the reference, split equals one long run", flush=True)

    # 2. generate: stops at the end of the reply, and matches the reference with stops on
    st = m.new_state()
    ref = m.generate(st, seed, 45, q, c.stops)
    r = c.tx(seed, 45, q)
    assert r[0] == c.tok.decode(ref) and r[1] == m.blob(st), (r[0], c.tok.decode(ref))
    assert ref[-1] in c.stops, "this seed never reached a stop token, so the stop is untested"
    print(f"2. generate stopped at the end of the reply after {len(ref) - len(q)} of 45 new tokens, {r[2]:,} gas\n   {r[0]!r}", flush=True)

    # 3. preview runs the same code, costs nothing and stores nothing
    head = "0x%064x" % im.keccak(r[1])
    p_text, p_hash, p_blob = c.preview(seed, 45, q)
    assert p_text == r[0] and p_hash == head and p_blob == r[1]
    other = c.preview(seed + 99, 9, q)
    assert c.view("committed(bytes32)", other[1]) == "0x" + "0" * 64, "preview stored a state"
    print("3. preview equals generate and stores nothing", flush=True)

    # 3b. a whole free chat: preview continues from preview, and matches the reference each time
    st = m.new_state()
    blob = None
    for i, msg in enumerate(("hi", "what is 2+2?", "are you stupid?")):
        pr = encode(c, turn(msg))
        ref = m.generate(st, seed + 50 + i, 30, pr, c.stops)
        txt, h, blob = c.preview(seed + 50 + i, 30, pr, blob)
        assert txt == c.tok.decode(ref) and blob == m.blob(st), (i, txt, c.tok.decode(ref))
        assert c.view("committed(bytes32)", h) == "0x" + "0" * 64, "a free message was committed"
    print(f"3b. free chat over 3 messages matches the reference, t={st['t']}, nothing committed", flush=True)

    # 3c. proof stays paid: generate rejects a state that only preview produced
    assert c.tx(seed, 3, state=blob) is None, "generate accepted a free state"
    print("3c. generate rejects a state that only preview produced", flush=True)

    # 4. contract-readable: committed and text are in storage, and Reply decodes
    assert int(c.view("committed(bytes32)", head), 16) == 1
    assert abi_bytes(c.view("text(bytes32)", head)).decode() == r[0]
    print("4. committed(state) is true, text(state) returns the reply", flush=True)

    # 5. a follow-up continues the conversation, and a branch from the same state works
    q2 = encode(c, turn("why?"))
    st2 = m.load_blob(r[1])
    ref2 = m.generate(st2, seed + 1, 40, q2, c.stops)
    r2 = c.tx(seed + 1, 40, q2, state=r[1])
    assert r2[0] == c.tok.decode(ref2) and r2[1] == m.blob(st2)
    parent = next(l for l in r2[3]["logs"] if l["topics"][0] == c.topic_reply)["topics"][1]
    assert parent == head, "Reply does not link to its parent"
    alt = c.tx(seed + 2, 5, encode(c, turn("hello")), state=r[1])
    assert alt is not None and alt[0] != r2[0]
    print(f"5. follow-up matches the reference, t={st2['t']}, links to its parent, branch works, {r2[2]:,} gas\n   {r2[0]!r}", flush=True)

    # 6. forged states revert, in generate and in preview
    for name, i in (("cache byte", len(r[1]) // 2), ("position", 31), ("token", 63), ("last byte", len(r[1]) - 1)):
        bad = bytearray(r[1])
        bad[i] ^= 1
        assert c.tx(seed, 3, state=bytes(bad)) is None, f"forged {name} was accepted"
    assert c.tx(seed, 3, state=b"\0" * len(r[1])) is None
    assert c.tx(seed, 3, state=r[1][:-32]) is None
    try:
        c.preview(seed, 3, state=r[1][:-32])
        raise SystemExit("preview accepted a short state")
    except RuntimeError:
        pass
    print("6. generate rejects forged states: cache, position, token, last byte, zero state, short state. preview rejects a short state", flush=True)

    # 7. more than 64 tokens in one tx reverts
    assert c.tx(seed, 65) is None
    print("7. more than 64 tokens in one tx reverts", flush=True)


def full_state(c, seed):
    """A conversation past the window, so the cache is full."""
    base = c.tx(seed, 16, encode(c, turn("hello")))
    for i in range(4):
        base = c.tx(seed + i + 1, 12, encode(c, ("" if base[0].endswith("\n") else "\n") + turn("ok")), state=base[1])
    return base


def full_length_seed(c, n, prompt, state):
    """A seed whose reply never stops early. Found for free with preview, so the tx is the worst case."""
    t0 = int.from_bytes(state[:32], "big") if state else 0
    for seed in range(1000, 1400):
        try:
            _, _, blob = c.preview(seed, n, prompt, state)
        except RuntimeError:
            return None
        if int.from_bytes(blob[:32], "big") - t0 == len(prompt) + n:
            return seed
    return None


def longest(c, prompt, state):
    """The most new tokens at the cap, using only runs that reach full length."""
    lo, hi, best = 1, 64 - len(prompt) + 1, None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        seed = full_length_seed(c, mid, prompt, state)
        r = c.tx(seed, mid, prompt, state) if seed is not None else None
        if r is not None:
            lo, best = mid, r
        else:
            hi = mid
        print(f"  new tokens {mid}: {'OUT OF GAS or no full-length seed' if r is None else f'{r[2]:,} gas'}", flush=True)
    return lo, best


def find_max(c, seed):
    prompt = encode(c, turn("do you like dogs?"))
    base = full_state(c, seed)
    for label, st in (("new chat", None), ("continued chat, full cache", base[1])):
        lo, r = longest(c, prompt, st)
        print(f"{label}, {len(prompt)}-token prompt: max {lo} new tokens, {len(prompt) + lo} total, {r[2]:,} gas ({r[2] / TX_CAP:.1%})\n   {r[0]!r}", flush=True)


def sweep(c, count):
    prompts = ["hi", "what is 2+2?", "are you stupid?", "tell me a story", "!!!!!!!!", "qqqq zzzz xxxx", "the the the the", "User: User: User:", "1234567890", "I feel sad today"]
    worst, fails, full = {}, 0, 0
    base = full_state(c, 1)
    total = int(os.environ.get("SWEEP_TOTAL", 56))
    rng = random.Random(1)
    for i in range(count):
        p = encode(c, turn(prompts[i % len(prompts)]))
        for label, st in (("fresh", None), ("continued", base[1])):
            seed = None
            t0 = int.from_bytes(st[:32], "big") if st else 0
            for _ in range(150):  # only runs that never stop early count, they are the worst case
                cand = rng.getrandbits(64)
                if int.from_bytes(c.preview(cand, total - len(p), p, st)[2][:32], "big") - t0 == total:
                    seed = cand
                    break
            r = c.tx(seed if seed is not None else rng.getrandbits(64), total - len(p), p, st)
            fails += r is None
            full += seed is not None
            used = r[2] if r else 0
            if used > worst.get(label, (0,))[0]:
                worst[label] = (used, i)
        if i % 20 == 0:
            print(f"  {i}/{count}: {worst}", flush=True)
    print(f"worst gas at {total} total tokens, {count} seeds and adversarial prompts: {worst}, {full} of {count * 2} runs reached full length, failed txs: {fails}", flush=True)


def profile(c):
    """Engine directly, so the probes can log. The cache is full."""
    blob, _ = c.engine_call(1, 40, encode(c, turn("hello")))
    n = 24
    rc = send(c.payload(1, n, state=blob), c.engine, gas=TX_CAP)
    logs = [(int(l["topics"][0], 16) - 1, int(l["topics"][1], 16)) for l in rc["logs"] if len(l["topics"]) == 2 and l["data"] == "0x"]
    over = 1125 + 15
    cost = {}
    for (lab, g0), (_, g1) in zip(logs, logs[1:]):
        cost[LABELS[lab]] = cost.get(LABELS[lab], 0) + g0 - g1 - over
    real = int(rc["gasUsed"], 16) - len(logs) * over
    print(f"engine, {n} tokens, full cache: ~{real:,} gas without probes")
    for k, v in sorted(cost.items(), key=lambda kv: -kv[1]):
        print(f"{k:16}{v:>12,}  {v / real:6.1%}  {v / n:>10,.0f} per token")
    print(f"{'outside tokens':16}{real - sum(cost.values()):>12,}  {(real - sum(cost.values())) / real:6.1%}")


def chat(c, seed, n, messages):
    state, last = None, "\n"
    for i, msg in enumerate(messages):
        p = encode(c, ("" if last.endswith("\n") else "\n") + turn(msg))
        last, state, gas, _ = c.tx(seed + i, n, p, state=state)
        print(f"[tx {i + 1}: {gas:,} gas, ${gas * 0.04e-9 * 2266:.2f} at 0.04 gwei]\n{last}\n", flush=True)


def cross(c, path):
    """A fixed tx sequence. Different networks and clients must produce identical results."""
    out = []

    def rec(label, r):
        out.append(dict(label=label, gas=r[2], text=r[0], state=hashlib.sha256(r[1]).hexdigest()))
        print(f"{label}: {r[2]:,} gas {r[0]!r}", flush=True)
        if REMOTE and len(r) > 3:
            print(f"   https://hoodi.etherscan.io/tx/{r[3]['transactionHash']}", flush=True)
        return r

    p = encode(c, turn("do you like dogs?"))
    a = rec("new chat", c.tx(7, 45, p))
    b = rec("follow-up", c.tx(8, 40, encode(c, turn("why?")), state=a[1]))
    d = rec("follow-up 2", c.tx(9, 30, encode(c, turn("are you stupid?")), state=b[1]))
    e = rec("follow-up 3", c.tx(10, 30, encode(c, turn("what is 2+2?")), state=d[1]))
    txt, h, _ = c.preview(11, 25, encode(c, turn("ok")), e[1])
    out.append(dict(label="preview", text=txt, state=h))
    print(f"preview: {txt!r}", flush=True)
    near = encode(c, turn("ok"))
    n_new = int(os.environ.get("NEAR_CAP_TOTAL", 56)) - len(near)
    seed = full_length_seed(c, n_new, near, e[1])
    assert seed is not None
    rec(f"full length {len(near) + n_new} tokens", c.tx(seed, n_new, near, e[1]))
    out.append(dict(label="text(state)", text=abi_bytes(c.view("text(bytes32)", "0x%064x" % im.keccak(e[1]))).decode()))
    assert out[-1]["text"] == e[0]
    bad = bytearray(e[1])
    bad[100] ^= 1
    out.append(dict(label="forged state", reverted=c.tx(10, 3, state=bytes(bad)) is None))
    print(out[-1], flush=True)
    try:
        send(c.front_data(c.sel_generate, 10, 3, (), e[1]), c.front, gas=TX_CAP + 1)
        over = "accepted"
    except Exception as ex:
        over = f"rejected: {str(ex)[:90]}"
    print(f"{c.client}: tx with cap+1 gas {over}", flush=True)
    json.dump(out, open(path, "w"), indent=1)


def deploy_only(c):
    """Deploy and print the addresses, or only the plan in a dry run. It runs no generate tx."""
    from eth_account import Account
    a = Account.from_key(os.environ["PK"]).address
    bal = int(remote_rpc("eth_getBalance", a, "latest"), 16)
    base = int(remote_rpc("eth_getBlockByNumber", "latest", False)["baseFeePerGas"], 16)
    if DRY:
        g = c.g.planned
        print(f"DRY RUN, nothing sent. network {NET}, chain {int(remote_rpc('eth_chainId'), 16)}, deployer {a}")
        print(f"txs {c.g.n + 1}, gas limit total {g:,}, real gas about 91.4M")
        print(f"balance {bal / 1e18:.6f} ETH, base fee {base / 1e9:.4f} gwei, tip {TIP / 1e9} gwei, stops above {MAX_BASE / 1e9} gwei base fee")
        cost = 91.4e6 * (base + TIP) / 1e18
        print(f"expected cost {cost:.5f} ETH at this base fee. enough: {bal > cost * 1.5}")
        print(f"engine code hash {hashlib.sha256(c.engine_code).hexdigest()}")
        return
    print(f"DEPLOYED on {NET}\nengine {c.engine}\nfront  {c.front}\ntotal gas {c.g.gas:,}\nbalance left {bal / 1e18:.6f} ETH\nall addresses in {DEPLOY_FILE}")


def main():
    global PROFILE
    ckpt, cmd = sys.argv[1], sys.argv[2]
    PROFILE = cmd == "profile"
    c = Chain(ckpt)
    try:
        if cmd == "deploy":
            assert REMOTE, "set LLM_NODE to hoodi or mainnet"
            deploy_only(c)
        elif cmd == "test":
            test(c)
        elif cmd == "cross":
            cross(c, sys.argv[3])
        elif cmd == "max":
            find_max(c, int(sys.argv[3]) if len(sys.argv) > 3 else 1)
        elif cmd == "sweep":
            sweep(c, int(sys.argv[3]) if len(sys.argv) > 3 else 100)
        elif cmd == "profile":
            profile(c)
        elif cmd == "chat":
            chat(c, int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:])
    finally:
        c.close()


if __name__ == "__main__":
    main()
