#!/usr/bin/env python3
"""Submit every deployed contract to Sourcify. Usage: verify_contracts.py CHAIN_ID RPC DEPLOY_FILE FRONT_ADDRESS"""
import json, sys, time, urllib.error, urllib.request

chain, rpc_url, path, front = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4].lower()
SOLC = "0.8.38+commit.1835d247"


def post(url, body=None):
    r = urllib.request.Request(url, json.dumps(body).encode() if body is not None else None, {"content-type": "application/json", "user-agent": "curl/8"})
    try:
        with urllib.request.urlopen(r, timeout=90) as f:
            return f.status, json.load(f)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]


def code_of(a):
    return post(rpc_url, dict(jsonrpc="2.0", id=1, method="eth_getCode", params=[a, "latest"]))[1]["result"][2:]


def yul_input(name, code):
    src = f'object "{name}" {{\n code {{ datacopy(0, dataoffset("runtime"), datasize("runtime")) return(0, datasize("runtime")) }}\n object "runtime" {{ code {{ verbatim_0i_0o(hex"{code}") }} }}\n}}\n'
    return dict(language="Yul", sources={f"{name}.yul": dict(content=src)}, settings=dict(evmVersion="osaka", optimizer=dict(enabled=False, details=dict(yul=False)), outputSelection={"*": {"*": ["evm.bytecode.object", "evm.deployedBytecode.object"]}}))


def submit(addr, std, ident):
    s, o = post(f"https://sourcify.dev/server/v2/verify/{chain}/{addr}", dict(stdJsonInput=std, compilerVersion=SOLC, contractIdentifier=ident))
    if not isinstance(o, dict) or not o.get("verificationId"):
        return "already verified" if "already" in str(o).lower() else f"rejected {s}: {str(o)[:160]}"
    for _ in range(40):
        time.sleep(3)
        _, j = post(f"https://sourcify.dev/server/v2/verify/{o['verificationId']}")
        if isinstance(j, dict) and j.get("isJobCompleted"):
            return (j.get("contract") or {}).get("runtimeMatch") or str(j.get("error"))[:160]
    return "timed out"


done = json.load(open(path))
results = {}
std = dict(language="Solidity", sources={"WorstLLM.sol": dict(content=open("WorstLLM.sol").read())}, settings=dict(evmVersion="osaka", optimizer=dict(enabled=True, runs=200), outputSelection={"*": {"*": ["evm.bytecode.object", "evm.deployedBytecode.object", "abi", "metadata"]}}))
r = submit(front, std, "WorstLLM.sol:WorstLLM")
print("front WorstLLM.sol:", r, flush=True)
for i, v in enumerate(done.values()):
    a = v["addr"].lower()
    if a == front or v.get("front"):
        continue
    code = code_of(a)
    r = submit(a, yul_input(f"WorstLLMPart{i}", code), f"WorstLLMPart{i}.yul:WorstLLMPart{i}")
    results[r] = results.get(r, 0) + 1
    print(f"  {a} {len(code) // 2:>6} bytes: {r}", flush=True)
print("summary:", results)
