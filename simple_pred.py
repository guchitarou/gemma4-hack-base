from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "./model_weights_12B_it"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto"
)

# Prompt
messages = [
    {"role": "system", "content": "あなたは優秀なアシスタントです。"},
    {"role": "user", "content": "なぞなぞです。サラリーマンがみんな持ってる「いし」はなんですか？ヒント意思ではないです。"},
]

# Process input
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    enable_thinking=False
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

# Generate output
outputs = model.generate(**inputs, max_new_tokens=1024)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

print("=== Response ===")
print(response)
