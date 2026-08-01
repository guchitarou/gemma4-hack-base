"""
transformers単体でRequest Latencyを計測するスクリプト(vLLM不使用)
--concurrency に対応(バッチ処理で同時実行を再現)
"""

import time
import statistics
import argparse

from transformers import AutoProcessor, AutoModelForMultimodalLM


def build_dummy_prompt(processor, isl: int) -> str:
    unit = "これはベンチマーク用のダミーテキストです。"
    text = unit
    while len(processor.tokenizer(text)["input_ids"]) < isl:
        text += unit
    return text


def run_batch_request(model, processor, prompt: str, osl: int, device, batch_size: int) -> float:
    """
    batch_size件のリクエストを1回のgenerate()でまとめて処理し、
    その全体が完了するまでの時間(秒)を返す。
    """
    messages = [
        {"role": "system", "content": "あなたは優秀なアシスタントです。"},
        {"role": "user", "content": prompt},
    ]

    # 同じプロンプトをbatch_size件分並べる
    batch_messages = [messages] * batch_size

    inputs = processor.apply_chat_template(
        batch_messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        padding=True,  # バッチ処理には padding が必須
    ).to(device)

    start = time.perf_counter()
    model.generate(**inputs, max_new_tokens=osl)
    end = time.perf_counter()

    return end - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./model_weights_12B_it")
    parser.add_argument("--request-count", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--isl", type=int, default=500)
    parser.add_argument("--osl", type=int, default=200)
    args = parser.parse_args()

    print(f"モデルロード中: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype="auto", device_map="auto"
    )
    device = model.device

    # パディングのためtokenizerにpad_tokenが必要
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print("モデルロード完了\n")

    prompt = build_dummy_prompt(processor, args.isl)

    # concurrency件ずつバッチにまとめて、request_count件になるまで繰り返す
    n_batches = (args.request_count + args.concurrency - 1) // args.concurrency

    per_request_latencies = []  # 「1リクエストあたり」の体感レイテンシ(=バッチ全体の時間)
    n_done = 0

    print(f"計測開始: concurrency={args.concurrency}, request_count={args.request_count}\n")

    for b in range(n_batches):
        current_batch_size = min(args.concurrency, args.request_count - n_done)

        batch_latency = run_batch_request(
            model, processor, prompt, args.osl, device, current_batch_size
        )

        # このバッチに含まれる全リクエストは同時に完了するので、
        # 同じレイテンシ値をcurrent_batch_size件分記録する
        per_request_latencies.extend([batch_latency] * current_batch_size)

        n_done += current_batch_size
        print(
            f"  batch {b + 1}/{n_batches} 完了 "
            f"({n_done}/{args.request_count}件, batch_latency={batch_latency:.3f}s)"
        )

    print("\n=== Request Latency ===")
    print(f"avg : {statistics.mean(per_request_latencies) * 1000:.2f} ms")
    print(f"min : {min(per_request_latencies) * 1000:.2f} ms")
    print(f"max : {max(per_request_latencies) * 1000:.2f} ms")
    print(f"p50 : {statistics.median(per_request_latencies) * 1000:.2f} ms")


if __name__ == "__main__":
    main()