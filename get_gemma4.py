from transformers import AutoProcessor, AutoModelForCausalLM
model_id = "google/gemma-4-E4B-it"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    cache_dir="./model_folder"
).eval()
processor = AutoProcessor.from_pretrained(model_id, cache_dir="./model_folder")