#!/usr/bin/env python3
"""Submit contracts to Etherscan. Usage: etherscan_verify.py CHAIN_ID RPC FRONT ENGINE [more part addresses]"""
import json, os, sys, time, urllib.parse, urllib.request

KEY = os.environ.get("ETHERSCAN_API_KEY") or os.environ.get("etherscan_api_key")
assert KEY, "no etherscan key in the environment"
chain, rpc_url, front, engine = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
parts = sys.argv[5:]
SOLC = "v0.8.38+commit.1835d247"
API = "https://api.etherscan.io/v2/api?" + urllib.parse.urlencode(dict(chainid=chain))


def es(params, post=True):
    params = dict(params, apikey=KEY)
    body = urllib.parse.urlencode(params).encode() if post else None
    url = API if post else API + "&" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(urllib.request.Request(url, body, {"user-agent": "curl/8"}), timeout=120) as f:
        return json.load(f)


def code_of(a):
    r = urllib.request.Request(rpc_url, json.dumps(dict(jsonrpc="2.0", id=1, method="eth_getCode", params=[a, "latest"])).encode(), {"content-type": "application/json", "user-agent": "curl/8"})
    return json.load(urllib.request.urlopen(r, timeout=30))["result"][2:]


def submit(label, addr, std, name, ctor=""):
    out = es(dict(module="contract", action="verifysourcecode", contractaddress=addr, sourceCode=json.dumps(std), codeformat="solidity-standard-json-input", contractname=name, compilerversion=SOLC, constructorArguements=ctor, constructorArguments=ctor))
    print(f"{label}: submit status {out.get('status')} {str(out.get('message'))} {str(out.get('result'))[:150] if out.get('status') != '1' else ''}", flush=True)
    if out.get("status") != "1":
        return str(out.get("result"))
    guid = out["result"]
    for _ in range(30):
        time.sleep(5)
        st = es(dict(module="contract", action="checkverifystatus", guid=guid), post=False)
        if "pending" not in str(st.get("result")).lower():
            print(f"{label}: {st.get('status')} {str(st.get('result'))[:220]}", flush=True)
            return str(st.get("result"))
    print(f"{label}: still pending", flush=True)
    return "pending"


def yul(name, code):
    src = f'object "{name}" {{\n code {{ datacopy(0, dataoffset("runtime"), datasize("runtime")) return(0, datasize("runtime")) }}\n object "runtime" {{ code {{ verbatim_0i_0o(hex"{code}") }} }}\n}}\n'
    return dict(language="Yul", sources={f"{name}.yul": dict(content=src)}, settings=dict(evmVersion="osaka", optimizer=dict(enabled=False, details=dict(yul=False)), outputSelection={"*": {"*": ["evm.bytecode.object", "evm.deployedBytecode.object"]}}))


std = dict(language="Solidity", sources={"WorstLLM.sol": dict(content=open("WorstLLM.sol").read())}, settings=dict(evmVersion="osaka", optimizer=dict(enabled=True, runs=200), outputSelection={"*": {"*": ["evm.bytecode", "evm.deployedBytecode", "abi", "metadata"]}}))
submit("front", front, std, "WorstLLM.sol:WorstLLM", "0" * 24 + engine[2:].lower())
submit("engine", engine, yul("WorstLLMEngine", code_of(engine)), "WorstLLMEngine.yul:WorstLLMEngine")
for i, a in enumerate(parts):
    submit(f"part {i}", a, yul(f"WorstLLMPart{i}", code_of(a)), f"WorstLLMPart{i}.yul:WorstLLMPart{i}")
