# Runpod & 使用環境設定
[READMEに移動](README.md)

## 動作確認条件
* GPU環境
    * Runpod
* 使用言語
    * Python
* 主な使用ライブラリ
    * pytoch 2.4.0
    * transformers 4.53.0
    * gradio 6.11.0
* 使用ツール
    * Vscode

## フォルダ構成
```
Gemma3nChat/
├── desc_img ------------> 説明画像
├── gradio_tmp ----------> gradioのキャッシュフォルダ
├── model_weights -------> モデルの保存フォルダ
├── bee.jpg -------------> 入力画像例
├── main.py -------------> モデルの実行テストコード
├── README.md -----------> 使い方資料
├── requirements.txt ----> 使用ライブラリ情報
├── RunpodENV.md --------> 環境構築資料
└── vlm_chat_app.py -----> CHATコード　
```

# コード・ライブラリ設定
## コードの格納

Runpod上のターミナルで下記のコマンドを実行する。
```
cd /workspace
```
移動したフォルダに Gemma3nChat 内のコードとフォルダを格納します。

格納後のフォルダ構成は下記のようになります。
```
/workspace/
├── desc_img ------------> 説明画像
├── gradio_tmp ----------> gradioのキャ
　・
　・
　・
└── vlm_chat_app.py -----> CHATコード　
```
## ライブラリのインストール
Runpod のターミナルで以下のコマンドを実行し、必要なライブラリをインストールします。
```
pip install --upgrade pip
pip install -U "huggingface_hub[cli]"
pip install -r requirements.txt
```

### HuggingFaceにログイン
手動でダウンロードして Runpod 上に配置する手間を省くため、コマンドで格納する。

HuggingFaceにターミナル上でログインする。
```
hf auth login
```
上記のコマンドを実行するとトークンを求められるのでHuggingFaceで設定した自分のトークンを設定します。
```
Enter your token (input will not be visible): 
```
今回は使用しないため、下記の質問は `n` を選択します。
```
Add token as git credential? [y/N]: n
```
これでログイン作業完了。

### HuggingFaceからモデルのダウンロード
Runpodのターミナル上で以下のコマンドでダウンロードを開始する。
```
cd /workspace
hf download google/gemma-4-E4B-it --local-dir ./model_weights
```

ダウンロードが完了すると、フォルダ内に以下のファイルが格納されます。
```
/workspace/model_weights/
├── .gitattributes
├── README.md
　・
　・
　・
└── tokenizer_config.json
```

以上で環境構築は完了です。

## 追記
> モデル取得で許可不要に！

2026/06時点では、モデル取得に許可がいと記憶していたが、別モデルと同様にモデルIDを入力するだけでモデルインストールが可能になった。そのため下記の手順でモデル取得も可能。

### インストールライブラリ
```
pip install --upgrade pip
pip install accelerate
pip install pillow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```


[READMEに移動](README.md)
