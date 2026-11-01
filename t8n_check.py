#!/usr/bin/env python3
"""Replay the cross txs in geth's evm t8n under a named fork, compare gasUsed with other clients."""
import json, os, re, subprocess, sys, tempfile

KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"  # anvil's public test key, no real funds
anvil_json, geth_json = sys.argv[1], sys.argv[2]
rep = json.load(open(anvil_json + ".replay"))
anvil = {x["label"]: x.get("gas") for x in json.load(open(anvil_json))}
geth = {x["label"]: x.get("gas") for x in json.load(open(geth_json))}
env = dict(currentCoinbase="0x" + "c0" * 20, currentGasLimit="0x3938700", currentNumber="0x10", currentTimestamp="0x1000", currentDifficulty="0x0",
           currentRandom="0x" + "11" * 32, currentBaseFee="0x7", parentBeaconBlockRoot="0x" + "22" * 32, withdrawals=[], currentExcessBlobGas="0x0",
           parentExcessBlobGas="0x0", parentBlobGasUsed="0x0", blockHashes={}, ommers=[])


SYSTEM = {"0x000F3df6D732807Ef1319fB7B8bB8522d0Beac02", "0x0000F90827F1C53a10cb7A02335B175320002935"}


def run(fork, data, nonce_slots, gas="0x1000000"):
    alloc = {a: dict(code=c, balance="0x0", nonce="0x1", storage={}) for a, c in rep["codes"].items()}
    alloc[rep["sender"]] = dict(balance="0x" + "f" * 30, nonce="0x0", code="0x", storage={})
    alloc[rep["main"]]["storage"] = {s: "0x" + "0" * 63 + "1" for s in nonce_slots}
    # block-level system contracts since Prague: empty stubs, they run outside the tx
    for sysaddr in SYSTEM:
        alloc[sysaddr.lower()] = dict(code="0x5f5ff3", balance="0x0", nonce="0x1", storage={})
    tx = dict(type="0x0", chainId="0x7a69", nonce="0x0", gas=gas, gasPrice="0x10", to=rep["main"], value="0x0", input="0x" + data, v="0x0", r="0x0", s="0x0", secretKey=KEY, protected=True)
    with tempfile.TemporaryDirectory() as d:
        for n, v in (("alloc", alloc), ("env", env), ("txs", [tx])):
            json.dump(v, open(os.path.join(d, n + ".json"), "w"))
        r = subprocess.run(["evm", "t8n", f"--input.alloc={d}/alloc.json", f"--input.env={d}/env.json", f"--input.txs={d}/txs.json", "--output.result=stdout", "--output.alloc=", "--output.body=", f"--state.fork={fork}", "--state.chainid=31337"], capture_output=True, text=True)
    missing = re.search(r"no code at (0x[0-9a-fA-F]{40})", r.stderr)
    if missing:  # a fork's block-level system contract, it runs outside the tx
        SYSTEM.add(missing.group(1))
        return run(fork, data, nonce_slots, gas)
    assert r.returncode == 0, r.stderr[-800:]
    out = json.loads(r.stdout)["result"]
    if out.get("rejected"):
        return "rejected: " + out["rejected"][0]["error"][:70]
    rc = out["receipts"][0]
    assert rc["status"] == "0x1", rc
    return int(rc["gasUsed"], 16)


print(f"{'tx':14}{'anvil':>12}{'t8n Osaka':>12}{'match':>7}{'geth --dev':>12}{'t8n Amsterdam':>15}{'offset':>8}")
ok = True
for i, x in enumerate(rep["replay"]):
    slots = rep["slots"][:i]
    o, am = run("Osaka", x["data"], slots), run("Amsterdam", x["data"], slots)
    a, g = anvil[x["label"]], geth[x["label"]]
    ok &= o == a  # mainnet rules are the pass criterion, geth --dev runs a newer fork
    print(f"{x['label']:14}{a:>12,}{o:>12,}{str(o == a):>7}{g:>12,}{am:>15,}{g - am:>8,}", flush=True)
print("cap+1 under Osaka:", run("Osaka", rep["replay"][0]["data"], [], gas=hex(16777217)))
print("OSAKA GAS MATCHES ANVIL EXACTLY" if ok else "OSAKA MISMATCH")
sys.exit(0 if ok else 1)
