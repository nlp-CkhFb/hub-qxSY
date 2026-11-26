"""
一键跑完实验矩阵（串行，单卡；4GB 显存下不要并行）。

矩阵 = 2 个骨干 × 2 个解码器，再对两个 CRF checkpoint 做解码期硬约束：
  1. BERT    + Linear
  2. BERT    + CRF
  3. RoBERTa + Linear
  4. RoBERTa + CRF
  5. BERT+CRF 的硬约束解码（复用 2 的 checkpoint）
  6. RoBERTa+CRF 的硬约束解码（复用 4 的 checkpoint）

用法：
  python experiments/run_all.py                # 跳过已完成的步骤
  python experiments/run_all.py --force        # 全部重跑
  python experiments/run_all.py --steps 1,2    # 只跑指定步骤
  python experiments/run_all.py --smoke        # 每条 200 句、1 epoch，验证管线
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

STEPS = {
    1: {
        "name": "bert_linear_256",
        "cmd": [PY, "src/train.py", "--backbone", "bert-base-chinese", "--decoder", "linear",
                "--run_name", "bert_linear_256"],
        "metric": "outputs/metrics/disc/bert_linear_256.json",
        "ckpt": "outputs/checkpoints/bert_linear_256.pt",
    },
    2: {
        "name": "bert_crf_256",
        "cmd": [PY, "src/train.py", "--backbone", "bert-base-chinese", "--decoder", "crf",
                "--run_name", "bert_crf_256"],
        "metric": "outputs/metrics/disc/bert_crf_256.json",
        "ckpt": "outputs/checkpoints/bert_crf_256.pt",
    },
    3: {
        "name": "roberta_linear_256",
        "cmd": [PY, "src/train.py", "--backbone", "roberta-base-chinese", "--decoder", "linear",
                "--run_name", "roberta_linear_256"],
        "metric": "outputs/metrics/disc/roberta_linear_256.json",
        "ckpt": "outputs/checkpoints/roberta_linear_256.pt",
    },
    4: {
        "name": "roberta_crf_256",
        "cmd": [PY, "src/train.py", "--backbone", "roberta-base-chinese", "--decoder", "crf",
                "--run_name", "roberta_crf_256"],
        "metric": "outputs/metrics/disc/roberta_crf_256.json",
        "ckpt": "outputs/checkpoints/roberta_crf_256.pt",
    },
    5: {
        "name": "bert_crf_256_hard",
        "cmd": [PY, "src/evaluate.py", "--ckpt", "outputs/checkpoints/bert_crf_256.pt",
                "--hard_constraint"],
        "metric": "outputs/metrics/disc/bert_crf_256_hard.json",
        "dep": "outputs/checkpoints/bert_crf_256.pt",
    },
    6: {
        "name": "roberta_crf_256_hard",
        "cmd": [PY, "src/evaluate.py", "--ckpt", "outputs/checkpoints/roberta_crf_256.pt",
                "--hard_constraint"],
        "metric": "outputs/metrics/disc/roberta_crf_256_hard.json",
        "dep": "outputs/checkpoints/roberta_crf_256.pt",
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", default="1,2,3,4,5,6")
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    only = {int(x) for x in args.steps.replace(" ", "").split(",") if x}
    log_dir = ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", KMP_DUPLICATE_LIB_OK="TRUE")

    print(f"[run_all] python={PY}")
    print(f"[run_all] root={ROOT}")
    for i in sorted(only):
        step = STEPS[i]
        metric = ROOT / step["metric"]
        if metric.exists() and not args.force:
            print(f"\n[skip] step {i} {step['name']}（已有指标；--force 可重跑）")
            continue

        cmd = list(step["cmd"])
        if args.smoke:
            if "train.py" in cmd[1]:
                # 去掉原有的 --run_name 值，避免出现两个 --run_name
                if "--run_name" in cmd:
                    i = cmd.index("--run_name")
                    del cmd[i:i + 2]
                cmd += ["--limit", "200", "--epochs", "1",
                        "--run_name", step["name"] + "_smoke"]
            elif "evaluate.py" in cmd[1]:
                if "--name" in cmd:
                    i = cmd.index("--name")
                    del cmd[i:i + 2]
                cmd += ["--limit", "100", "--name", step["name"] + "_smoke"]

        if args.device != "auto" and "train.py" in cmd[1]:
            cmd += ["--device", args.device]

        print(f"\n[step {i}] {step['name']}")
        print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
        t0 = time.perf_counter()
        log_path = log_dir / f"{step['name']}{'_smoke' if args.smoke else ''}.console.log"
        with open(log_path, "w", encoding="utf-8") as log:
            log.write(" ".join(cmd) + "\n\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log,
                                  stderr=subprocess.STDOUT, text=True)
        dt = time.perf_counter() - t0
        print(f"  exit={proc.returncode} elapsed={dt/60:.1f} min log={log_path.name}")
        if proc.returncode != 0:
            print(f"[FAIL] step {i} 失败；tail：")
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            print("\n".join(tail))
            sys.exit(proc.returncode)
        if not args.smoke and not metric.exists():
            print(f"[FAIL] 未生成指标文件：{metric}")
            sys.exit(1)

    print("\n[run_all] done")


if __name__ == "__main__":
    main()
