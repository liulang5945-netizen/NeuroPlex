"""阻塞等待 seed_corpus 训练达到目标 tick 或超时。"""
import json
import sys
import time

progress = r"e:\Seed\reports\seed_corpus_progress.jsonl"
target = int(sys.argv[1]) if len(sys.argv) > 1 else 800000
deadline = time.time() + (int(sys.argv[2]) if len(sys.argv) > 2 else 1700)

while time.time() < deadline:
    try:
        lines = open(progress, encoding="utf-8").read().strip().splitlines()
        entries = [json.loads(line) for line in lines if line.strip()]
    except OSError:
        entries = []
    if entries:
        last = entries[-1]
        print(
            f"ticks={last['ticks']} holdout={last['holdout_surprise']:.4f} "
            f"acc={last['online_accuracy']:.4f}",
            flush=True,
        )
        if last["ticks"] >= target:
            print("TARGET REACHED")
            break
    time.sleep(60)
else:
    print("TIMEOUT waiting for training")
