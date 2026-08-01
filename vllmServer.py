import subprocess
import os
import signal

env = os.environ.copy()  # 既存の環境変数を引き継ぐ
env["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 0番だけを使う

process = subprocess.Popen([
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", "./model_weights_12B_it",
    "--port", "8000",
], env=env, preexec_fn=os.setsid)

def cleanup():
    print("\n終了処理中...")
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=15)
    except ProcessLookupError:
        pass  # すでに終了している
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)  # 強制終了
    print("終了しました")

try:
    # ここでベンチマークなど他の処理
    input("Enterキーで終了...")  # 動作確認用
except KeyboardInterrupt:
    pass
finally:
    # プロセスグループごと終了させる（子・孫プロセスも含めて）
    cleanup()