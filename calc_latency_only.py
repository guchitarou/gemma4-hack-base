"""
transformers単体でRequest Latencyのみを計測するスクリプト(vLLM不使用)
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


def run_single_request(model, processor, prompt: str, osl: int, device) -> float:
    """1リクエストを実行し、Request Latency(秒)を返す"""
    messages = [
        {"role": "system", "content": "あなたは優秀なアシスタントです。"},
        {"role": "user", "content": prompt},
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(device)

    start = time.perf_counter()
    model.generate(**inputs, max_new_tokens=osl)
    end = time.perf_counter()

    return end - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./model_weights_12B_it")
    parser.add_argument("--request-count", type=int, default=100)
    parser.add_argument("--isl", type=int, default=500)
    parser.add_argument("--osl", type=int, default=200)
    args = parser.parse_args()

    print(f"モデルロード中: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype="auto", device_map="auto"
    )
    device = model.device
    print("モデルロード完了\n")

    prompt = build_dummy_prompt(processor, args.isl)

    latencies = []
    for i in range(args.request_count):
        latency = run_single_request(model, processor, prompt, args.osl, device)
        latencies.append(latency)
        print(f"  {i + 1}/{args.request_count} 完了 (latency={latency:.3f}s)", end="\r")

    print("\n\n=== Request Latency ===")
    print(f"avg : {statistics.mean(latencies) * 1000:.2f} ms")
    print(f"min : {min(latencies) * 1000:.2f} ms")
    print(f"max : {max(latencies) * 1000:.2f} ms")
    print(f"p50 : {statistics.median(latencies) * 1000:.2f} ms")
    total_time = sum(latencies)
    throughput = args.request_count / total_time
    print(f"\nRequest Throughput : {throughput:.2f} requests/sec")


if __name__ == "__main__":
    main()