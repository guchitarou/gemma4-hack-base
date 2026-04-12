"""
Gradio を使った VLM（Vision Language Model）チャットアプリ
========================================================
- テキストと画像を入力してモデルと会話できるシンプルな例
- gradio==6.11.0 対応
"""

import os

os.environ["GRADIO_TEMP_DIR"] = "./gradio_tmp"  # カレントディレクトリ内に保存
import gradio as gr
from PIL import Image
import time
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
import torch
# モデル重みPath
model_id = "./model_weights"

# どのような回答をAIに期待するか、ここで指定。
system_text = "<|think|>あなたは親切なAIアシスタントです。"
#system_text = "あなたは親切なAIアシスタントです。"

dtype = torch.bfloat16

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=dtype,
).eval()
processor = AutoProcessor.from_pretrained(model_id)


messages = [{"role": "system", "content": [{"type": "text", "text": system_text}]}]


def initialize_model():
    """ボタンを押したときに呼ばれる関数"""
    global messages
    messages = [{"role": "system", "content": [{"type": "text", "text": system_text}]}]
    print(messages)
    yield "✅ メッセージ履歴初期化できました"


# メッセージの初期化
initialize_model()
# ============================================================
# 【2】推論関数（モデルへの問い合わせ）
# ============================================================


def generate_response(message: str, image: Image.Image | None, history: list) -> str:
    """
    テキストと（任意で）画像を受け取り、モデルの返答を返す関数。

    Args:
        message : ユーザーが入力したテキスト
        image   : アップロードされた画像（なければ None）
        history : これまでの会話履歴 [{"role": "user"/"assistant", "content": ...}, ...]

    Returns:
        str: モデルの返答テキスト
    """

    # --- 実際のモデル呼び出しはここに書く ---
    # （今はダミーの返答を返しています）

    content_list = []
    if image is not None:
        content_list.append({"type": "image", "image": image})
    content_list.append({"type": "text", "text": message})

    global messages
    print(messages)
    messages.append({"role": "user", "content": content_list})

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]
    inputs = inputs.to(model.device, dtype=model.dtype)
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=2000, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)

    has_image = image is not None
    dummy_reply = decoded

    messages.append(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": dummy_reply},
            ],
        }
    )
    return dummy_reply


# ============================================================
# 【3】チャット処理（Gradio の ChatInterface 向け）
# ============================================================


def chat(
    message: dict,  # {"text": str, "files": [PIL.Image, ...]}
    history: list,  # [{"role": ..., "content": ...}, ...]
):
    """
    Gradio の multimodal ChatInterface から呼ばれるメイン関数。
    """
    text = message.get("text", "")
    files = message.get("files", [])

    # 画像は最初の1枚だけ使用（複数枚対応したい場合はここを変更）
    image = files[0] if files else None

    response = generate_response(text, image, history)
    partial = ""
    for char in response:
        partial += char
        yield partial
        time.sleep(0.01)  # 1文字ごとに 0.01秒 待つ


# ============================================================
# 【4】Gradio UI の定義
# ============================================================
with gr.Blocks() as demo:

    gr.Markdown("# Gemma4 e4b it チャットアプリ")
    gr.Markdown("テキストと画像を送って、モデルと会話できます。")

    # ChatInterface: チャット画面を簡単に作れる Gradio のコンポーネント
    chat_interface = gr.ChatInterface(
        fn=chat,  # チャット処理関数
        multimodal=True,  # 画像の入力を有効にする
        textbox=gr.MultimodalTextbox(  # テキスト＋ファイルの入力欄
            placeholder="メッセージを入力（画像も添付できます）",
            file_types=["image"],  # 画像ファイルのみ受け付ける
            file_count="single",  # 1枚だけ
        ),
        cache_examples=False,
    )

    with gr.Row():
        init_button = gr.Button("🚀 会話の履歴初期化", variant="primary")
        status_box = gr.Textbox(
            label="ステータス",
            value="未初期化",
            interactive=False,  # ユーザーが編集できないようにする
        )

    # ボタンを押したら initialize_model() を呼び出す
    init_button.click(
        fn=initialize_model,
        inputs=None,
        outputs=status_box,
    )


# ============================================================
# 【5】アプリの起動
# ============================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # 外部からアクセスする場合（ローカルのみなら "127.0.0.1"）
        server_port=7860,  # ポート番号
        share=False,  # True にすると公開 URL が発行される（Gradio の共有機能）
    )
