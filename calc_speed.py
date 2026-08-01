import subprocess

cmd = [
    "aiperf", "profile",
    "--model", "./model_mattbucci_12B_it_AWQ",
    "--url", "http://localhost:8000",
    "--endpoint-type", "chat",
    "--streaming",
    "--concurrency", "10",
    "--request-count", "100",
    "--isl", "500",
    "--osl", "200",
]

result = subprocess.run(cmd, capture_output=True, text=True)

print(result.stdout)
if result.returncode != 0:
    print("エラー発生:")
    print(result.stderr)