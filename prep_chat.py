#!/usr/bin/env python3
"""TinyChat -> whole conversations in the format the contract is prompted with."""
import random, re, sys

EOS = "<|endoftext|>"
raw = open(sys.argv[1], encoding="utf-8", errors="ignore").read().split("\n")[:-1]  # the last line is cut off
convs = []
for line in raw:
    turns = re.findall(r"\[INST\]\s*(.*?)\s*\[/INST\]\s*(.*?)\s*(?=\[INST\]|$)", line)
    if turns:
        convs.append("\n".join(f"User: {q}\nBot: {a}" for q, a in turns))
random.Random(0).shuffle(convs)
n = min(5000, len(convs) // 20)
out = sys.argv[2]
open(f"{out}/valid.txt", "w").write(EOS.join(convs[:n]) + EOS)
open(f"{out}/train.txt", "w").write(EOS.join(convs[n:]) + EOS)
print(f"{len(convs):,} conversations")
print(convs[0][:400])
