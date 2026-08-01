# gemma4-hack-base
Gemma 4 E4B-itの推論・マルチモーダル性能をチャットで即座に検証できるハッカソン用ベースコード。

## Gemma4詳細
モデルの詳細は下記サイト参照。
[gemma4詳細サイト](https://ai.google.dev/gemma/docs/core/model_card_4?hl=ja)

```
pip install accelerate
pip install pillow
pip install --ignore-installed blinker aiperf
pip install -U "huggingface_hub[cli]"

```
## モデルの取得

```
hf auth login
# gemma4 12B it 取得
hf download google/gemma-4-12B-it --local-dir ./model_weights_12B_it

# gemma4 12B it mattbucci量子化モデル
hf download mattbucci/gemma-4-12B-AWQ --local-dir ./model_mattbucci_12B_it_AWQ
```


```
CUDA_VISIBLE_DEVICES=0 python simple_pred.py
```

```
python calc_latency.py \
  --model ./model_weights_12B_it \
  --request-count 100 \
  --concurrency 10 \
  --isl 500 \
  --osl 200
```

```
 gemma4 12b -it simple 
=== Request Latency ===
avg : 4040.65 ms
min : 1845.23 ms
max : 7096.05 ms
p50 : 4109.49 ms

Request Throughput : 0.25 requests/sec
```


```
                                          vLLM gemma4 12B it NVIDIA AIPerf | LLM Metrics                                                    
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┓
┃                                        Metric ┃       avg ┃       min ┃       max ┃       p99 ┃       p90 ┃       p50 ┃    std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━┩
│                      Time to First Token (ms) │  1,229.91 │    679.54 │  2,786.88 │  2,786.38 │  1,317.41 │  1,146.80 │ 520.21 │
│                     Time to Second Token (ms) │    129.60 │     37.65 │    591.07 │    512.74 │    197.94 │     49.10 │ 137.68 │
│               Time to First Output Token (ms) │  1,229.91 │    679.54 │  2,786.88 │  2,786.38 │  1,317.41 │  1,146.80 │ 520.21 │
│                          Request Latency (ms) │ 11,674.18 │ 10,914.84 │ 13,119.42 │ 13,118.82 │ 12,050.05 │ 11,533.12 │ 463.58 │
│                      Inter Token Latency (ms) │     52.48 │     51.31 │     54.84 │     54.80 │     54.41 │     52.14 │   0.95 │
│              Output Token Throughput Per User │     19.06 │     18.23 │     19.49 │     19.48 │     19.39 │     19.18 │   0.34 │
│                             (tokens/sec/user) │           │           │           │           │           │           │        │
│ E2E Output Token Throughput (tokens/sec/user) │     17.16 │     15.24 │     18.32 │     17.53 │     17.46 │     17.34 │   0.62 │
│               Output Sequence Length (tokens) │    200.01 │    200.00 │    201.00 │    200.01 │    200.00 │    200.00 │   0.10 │
│                Input Sequence Length (tokens) │    500.00 │    500.00 │    500.00 │    500.00 │    500.00 │    500.00 │   0.00 │
│          Output Token Throughput (tokens/sec) │    171.15 │       N/A │       N/A │       N/A │       N/A │       N/A │    N/A │
│             Request Throughput (requests/sec) │      0.86 │       N/A │       N/A │       N/A │       N/A │       N/A │    N/A │
│                      Request Count (requests) │    100.00 │       N/A │       N/A │       N/A │       N/A │       N/A │    N/A │
└───────────────────────────────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴────────┘
```


```
                            vLLM+Awq+4bit  NVIDIA AIPerf | LLM Metrics                                                 
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┓
┃                                        Metric ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p50 ┃    std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━┩
│                      Time to First Token (ms) │ 1,248.26 │   759.29 │ 1,386.51 │ 1,386.50 │ 1,329.53 │ 1,287.72 │ 159.48 │
│                     Time to Second Token (ms) │    68.38 │     8.32 │   541.79 │   533.08 │    87.55 │    18.19 │ 153.44 │
│               Time to First Output Token (ms) │ 1,248.26 │   759.29 │ 1,386.51 │ 1,386.50 │ 1,329.53 │ 1,287.72 │ 159.48 │
│                          Request Latency (ms) │ 5,501.24 │ 4,469.48 │ 5,598.00 │ 5,597.67 │ 5,547.02 │ 5,515.33 │ 111.50 │
│                      Inter Token Latency (ms) │    21.44 │    20.88 │    24.35 │    23.91 │    21.82 │    21.19 │   0.82 │
│              Output Token Throughput Per User │    46.71 │    41.06 │    47.88 │    47.88 │    47.71 │    47.19 │   1.62 │
│                             (tokens/sec/user) │          │          │          │          │          │          │        │
│ E2E Output Token Throughput (tokens/sec/user) │    36.25 │    34.01 │    36.66 │    36.66 │    36.62 │    36.26 │   0.35 │
│               Output Sequence Length (tokens) │   199.44 │   152.00 │   200.00 │   200.00 │   200.00 │   200.00 │   4.80 │
│                Input Sequence Length (tokens) │   500.00 │   500.00 │   500.00 │   500.00 │   500.00 │   500.00 │   0.00 │
│          Output Token Throughput (tokens/sec) │   361.21 │      N/A │      N/A │      N/A │      N/A │      N/A │    N/A │
│             Request Throughput (requests/sec) │     1.81 │      N/A │      N/A │      N/A │      N/A │      N/A │    N/A │
│                      Request Count (requests) │   100.00 │      N/A │      N/A │      N/A │      N/A │      N/A │    N/A │
└───────────────────────────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴────────┘
```