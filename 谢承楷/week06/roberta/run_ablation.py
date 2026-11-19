"""
RoBERTa 底座 · 9 组消融实验（复现课件「一次只改一个旋钮」的表）

口径与课件一致：
  同一份训练集、同一批无重叠验证样本（8,901 条）、固定 seed=42、3 epoch。
  唯一不同的底座：chinese-roberta-wwm-ext（课件里是 bert-base-chinese）。

每组的差异只有一处：
  exp1 CLS 池化            —— 基线
  exp2 Mean 池化           --pool mean
  exp3 Max 池化            --pool max
  exp4 CLS + Mean 拼接     --pool cls_mean
  exp5 学习率 3e-5         --lr 3e-5
  exp6 max_length 32       --max_length 32
  exp7 FGM 对抗训练        --fgm
  exp8 Label Smoothing     --label_smoothing 0.1
  exp9 Focal Loss (γ=2)    --focal_gamma 2

模型权重在 HF 模型库（默认 F:\\huggingface_cache\\models\\roberta），
本目录只放代码；结果写入 code/outputs/roberta_ablation.{json,md}。

用法：
  python roberta/run_ablation.py                # 跑全部 9 组（已跑过的自动跳过，可断点续跑）
  python roberta/run_ablation.py --only exp1,exp2
  python roberta/run_ablation.py --smoke        # 冒烟：每组 20 步 / 评 300 条
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = CODE_ROOT / "outputs"
TRAIN_SCRIPT = CODE_ROOT / "roberta" / "train_roberta.py"

CONFIGS = [
    ("exp1", "CLS 池化（基线）", []),
    ("exp2", "Mean 池化", ["--pool", "mean"]),
    ("exp3", "Max 池化", ["--pool", "max"]),
    ("exp4", "CLS + Mean 拼接", ["--pool", "cls_mean"]),
    ("exp5", "学习率 3e-5", ["--lr", "3e-5"]),
    ("exp6", "max_length 32", ["--max_length", "32"]),
    ("exp7", "+ FGM 对抗训练", ["--fgm"]),
    ("exp8", "+ Label Smoothing", ["--label_smoothing", "0.1"]),
    ("exp9", "Focal Loss (γ=2)", ["--focal_gamma", "2"]),
]

EPOCH_RE = re.compile(r"epoch (\d+): val_acc=([\d.]+) val_f1=([\d.]+)")


def run_one(exp, name, extra, epochs, batch_size, smoke):
    """跑一组实验，返回该组的指标记录；失败抛异常。"""
    cmd = [sys.executable, "-u", str(TRAIN_SCRIPT),
           "--seed", "42", "--epochs", str(epochs), "--batch_size", str(batch_size),
           "--no_save", *extra]
    if smoke:
        cmd += ["--max_steps", "20", "--max_eval", "300"]

    print(f"\n===== {exp} {name} =====\n  {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=str(CODE_ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    history = []
    for line in proc.stdout:
        m = EPOCH_RE.search(line)
        if m:
            history.append({"epoch": int(m.group(1)), "val_acc": float(m.group(2)),
                            "val_f1": float(m.group(3))})
        print(line.rstrip(), flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{exp} 训练失败（exit={proc.returncode}），完整日志见上")

    best = max(history, key=lambda r: r["val_f1"])
    return {
        "exp": exp,
        "config": name,
        "args": extra,
        "best_macro_f1": best["val_f1"],
        "best_epoch": best["epoch"],
        "best_acc": best["val_acc"],
        "last_macro_f1": history[-1]["val_f1"],
        "epochs": history,
        "elapsed_s": round(time.time() - t0, 1),
    }


def write_report(results, path_md, path_json):
    base = results.get("exp1", {}).get("best_macro_f1")
    lines = [
        "# 消融实验 · RoBERTa 底座（chinese-roberta-wwm-ext）",
        "",
        "口径：同一份训练集、同一批无重叠验证样本（8,901 条）、固定 seed=42、3 epoch。"
        "每组只改一个旋钮；相对基线单位为 Macro-F1 百分点。",
        "",
        "| # | 配置 | Macro-F1 | 相对基线 |",
        "|---|---|---|---|",
    ]
    for exp, name, _ in CONFIGS:
        r = results.get(exp)
        if not r:
            continue
        f1 = r["best_macro_f1"]
        delta = "—" if exp == "exp1" or base is None else f"{(f1 - base) * 100:+.1f}"
        lines.append(f"| {exp} | {name} | {f1:.4f} | {delta} |")
    lines += ["", f"基线（exp1）= {base}。", ""]
    path_md.write_text("\n".join(lines), encoding="utf-8")
    path_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--only", default="", help="只跑指定组，如 exp1,exp3")
    ap.add_argument("--smoke", action="store_true", help="冒烟：每组 20 步 / 评 300 条")
    args = ap.parse_args()

    path_json = OUT_DIR / "roberta_ablation.json"
    path_md = OUT_DIR / "roberta_ablation.md"
    results = json.loads(path_json.read_text(encoding="utf-8")) if path_json.exists() else {}

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    todo = [c for c in CONFIGS if (not only or c[0] in only) and c[0] not in results]
    if not todo:
        print("没有待跑的组（结果已存在，或 --only 都跑完了）")
    for exp, name, extra in todo:
        try:
            rec = run_one(exp, name, extra, args.epochs, args.batch_size, args.smoke)
        except RuntimeError as e:
            print(f"[FAIL] {e}", flush=True)
            write_report(results, path_md, path_json)
            sys.exit(1)
        results[exp] = rec
        write_report(results, path_md, path_json)   # 每组落盘一次，中断也能续
        print(f"[done] {exp} best_macro_f1={rec['best_macro_f1']} ({rec['elapsed_s']}s)", flush=True)

    write_report(results, path_md, path_json)
    print("\n" + path_md.read_text(encoding="utf-8"))
    print(f"→ {path_md}\n→ {path_json}")


if __name__ == "__main__":
    main()
