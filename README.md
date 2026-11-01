# worstllm

the world's slowest, stupidest, most expensive "L"LM

```
User: why?
Bot: Yes, the river getting in the sea building on the sun is today.
```

real language model running in pure evm bytecode. no oracle, no zk, no precompile, no offchain inference. every node runs the forward pass

no token, only llm tokens. not an investment. not useful either

thx to claude, stupid idea like this can become a reality

yes it can be gasfree using eth_call but using emitting events is cooler 

## where

site: [worstllm.reynardo.eth.limo](https://worstllm.reynardo.eth.limo)

`WorstLLM`: [`0x47366ffc281b1877329d2aa90631027365ec2eb3`](https://etherscan.io/address/0x47366ffc281b1877329d2aa90631027365ec2eb3)

engine: [`0xea684992a60507cf5a150fd9f0e7e649bab9f96a`](https://etherscan.io/address/0xea684992a60507cf5a150fd9f0e7e649bab9f96a)

all 32 contracts are in `deployments_mainnet.json`

## specs

| | |
|---|---|
| params | 148,480, int8 |
| model | 4 layers, dim 64, 4 heads, ffn 128 |
| vocab | 256 bpe tokens, ascii only |
| context window | 16 tokens, like 33 chars |
| max per tx | 60 tokens including ur prompt |
| gas | about 229k per input token, 277k per output token |
| deploy | about 91.4M gas, 32 contracts |
| output price | about $25k per 1M tokens at 0.04 gwei |

## how tf does it fit

weights are code. every weight word is a `PUSH32` so reading it costs 3 gas

16 values per word. 16 bit lanes, one `MUL` does 16 multiply-accumulates

one `MUL` per dot product. reverse the query lanes, multiply by the key, answer lands in the top lane

hand written evm. `MCOPY` puts attention data at fixed addresses, computed jump goes straight into unrolled code, no loop counter

every scale is a power of 2 so requant is just a shift

## continuing a chat

each tx commits `keccak(state)` n emits the state in a log. pass it back to continue. anyone can continue or branch any chat, forged state reverts

the engine runs under `STATICCALL` so it can't write anything. `WorstLLM.sol` handles commits, text storage n events

the `Reply` event is the proof. calldata has no reply in it, just seed, limit, prompt n state. the text comes out of execution every node checks

## run it

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch numpy tokenizers mlx eth-account
export LLM_DATA=data/chat

.venv/bin/python onchain.py runs/m16.pt test
.venv/bin/python onchain.py runs/m16.pt chat 42 28 "hi how are you?" "what is 2+2?"
```

needs foundry n solc. runs on local anvil with its public test account

| file | what |
|---|---|
| `WorstLLM.sol` | front contract |
| `onchain.py` | builds n runs the contracts |
| `intmodel.py` | pure int python reference, contracts match it exactly |
| `train_mlx.py` `train.py` `prep_chat.py` | data n training |
| `t8n_check.py` | gas check against geth's evm |
| `verify_rebuild.py` | rebuilds every contract, compares it with mainnet byte by byte |
| `verify_contracts.py` `etherscan_verify.py` | submit source to sourcify n etherscan |
| `build_site.py` `site_src.html` | builds the page in `site/` |
| `m0_bench.py` | first kernel benchmark n the assembler |

## check it urself

```sh
LLM_DATA=data/chat .venv/bin/python verify_rebuild.py
```

no key needed, sends nothing. the front contract is verified on etherscan. the engine n weights are hand written so etherscan won't take them, they're on sourcify instead

## bad stuff

text is often nonsense. it has 148k params what did u expect?

int loss is 0.052 above float

verification proves bytes, not training. retraining won't give identical weights

## license

code is [wtfpl](LICENSE). training data n weights are not: [tinychat](https://huggingface.co/datasets/starhopp3r/TinyChat) by starhopp3r is [cc by-nc 4.0](https://creativecommons.org/licenses/by-nc/4.0/), so credit it n no commercial use. no warranty, see [NOTICE](NOTICE)
