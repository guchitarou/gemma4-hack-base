from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",  # vLLMはAPIキー認証不要だが、ライブラリ仕様上何か入れる必要あり
)

def main():
    response = client.chat.completions.create(
        model="./model_weights_12B_it",  # サーバー起動時に指定したモデルパス/名前と合わせる
        messages=[
            {"role": "user", "content": "自己紹介してください"}
        ],
        temperature=0.7,
        max_tokens=256,
    )
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()