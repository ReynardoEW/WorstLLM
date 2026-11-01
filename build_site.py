#!/usr/bin/env python3
"""site_src.html + tokenizer + contract address -> one static file, site/index.html."""
import json, re, sys
import intmodel as im
from intmodel import tr
from onchain import selector

front, chain, net, rpc, explorer, from_block, note = sys.argv[1:8]
tok = json.load(open(f"{tr.DATA}/tok.json"))
vocab = dict(tok["model"]["vocab"])
for a in tok.get("added_tokens", []):
    vocab[a["content"]] = a["id"]
merges = tok["model"]["merges"]
hexs = lambda sig: selector(sig).hex()
cfg = dict(front=front, chain=chain, net=net, rpc=rpc, explorer=explorer, from_block=from_block, net_note=note, vocab=vocab, merges=merges,
           sel_generate=hexs("generate(uint256,uint256,bytes,bytes)"), sel_preview=hexs("preview(uint256,uint256,bytes,bytes)"), sel_text="0x" + hexs("text(bytes32)"),
           topic_reply="0x%064x" % im.keccak(b"Reply(bytes32,bytes32,string)"), topic_state="0x%064x" % im.keccak(b"State(bytes32,bytes)"))
src = open("site_src.html").read()
out = re.sub(r"/\*CONFIG\*/null/\*END\*/", lambda m: json.dumps(cfg, separators=(",", ":")), src)
open("site/index.html", "w").write(out)
print(f"site/index.html {len(out):,} bytes, front {front}")
