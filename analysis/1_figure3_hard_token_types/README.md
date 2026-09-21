# Figure 3 — which tokens are HARD

Reports the most frequent HARD token types over the generated tokens, split by
whether the document ended up correct or wrong, plus the share of each surface
category among HARD tokens next to its share among all tokens.

```bash
python -m analysis.1_figure3_hard_token_types.token_types \
    --captures "captures/gsm8k/shard_*.json" --task gsm8k
```

Huginn-0125, GSM8K, 1319 documents, ε = 1.5 — HARD is 11.2% of generated
tokens:

```
top-20 HARD token types
  '.'  '0'  ' $'  ' of'  ' ='  ' is'  ' 1'  ' the'  '2'  ' The'
  ','  '5'  ' 2'  ' answer'  ' +'  '4'  ' to'  ' So'  '1'  ' 3'

category       of HARD    of all
WORD             49.7%     48.6%
NUMERIC          26.3%     29.3%
PUNCT            12.9%      9.9%
OPERATOR         10.4%     10.8%
```
