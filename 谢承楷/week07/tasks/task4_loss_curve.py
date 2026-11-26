# -*- coding: utf-8 -*-
"""
任务 4：读 outputs/logs 下的 steps.jsonl，画一条训练 loss 曲线。

作用：
  steps.jsonl 是训练器每 N 步追加一行（epoch / step / loss / lr / elapsed_s），
  直接看数字看不出收敛情况，画成曲线才能判断「学习率是否合适、有没有震荡、早停点在哪个 epoch」。

做法：
  1. 按行顺序还原**全局 step**（每 epoch 内 step 会从 0 重新计数）
  2. 原始 loss（细线）+ EMA 平滑曲线（粗线），并在 epoch 边界画竖虚线
  3. 保存 PNG，同时导出抽稀后的 CSV，方便贴进报告

输入：
  outputs/logs/<run>_steps.jsonl（默认 bert_linear_256，即任务 1 本次训练产物）

输出：
  outputs/figures/loss_curve.png
  outputs/figures/loss_curve.csv

运行方式（在 homework 目录下）：
  F:\\Miniconda3\\envs\\hf\\python.exe tasks\\task4_loss_curve.py
  ... tasks\\task4_loss_curve.py --run bert_crf_256      # 指定其它 run
"""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")                      # 无界面环境也能出图
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "outputs" / "logs"
FIG_DIR = ROOT / "outputs" / "figures"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def read_steps(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def to_global_steps(rows: list[dict]) -> tuple[list[int], list[float], list[int]]:
    """返回 (全局步号, loss, epoch 起始位置)。"""
    xs, ys, epoch_starts = [], [], []
    offset, last_epoch = 0, rows[0].get("epoch", 1)
    max_step_in_epoch = max(r.get("step", 0) for r in rows if r.get("epoch", 1) == last_epoch) + 1
    for r in rows:
        ep = r.get("epoch", 1)
        if ep != last_epoch:
            offset += max_step_in_epoch
            max_step_in_epoch = max(x.get("step", 0) for x in rows if x.get("epoch") == ep) + 1
            last_epoch = ep
            epoch_starts.append(offset)
        xs.append(offset + r.get("step", 0))
        ys.append(float(r.get("loss", 0.0)))
    return xs, ys, epoch_starts


def ema(values: list[float], alpha: float = 0.5) -> list[float]:
    out, prev = [], None
    for v in values:
        prev = v if prev is None else alpha * v + (1 - alpha) * prev
        out.append(prev)
    return out


def main():
    ap = argparse.ArgumentParser(description="画 steps.jsonl 的 loss 曲线")
    ap.add_argument("--run", default="bert_linear_256", help="run 名（对应 <run>_steps.jsonl）")
    args = ap.parse_args()

    path = LOG_DIR / f"{args.run}_steps.jsonl"
    if not path.exists():
        cands = sorted(LOG_DIR.glob("*_steps.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            raise FileNotFoundError(f"{LOG_DIR} 下没有 *_steps.jsonl，请先跑一次训练")
        path = cands[0]
        print(f"[warn] 找不到 {args.run}，改用最新的 {path.name}")

    rows = read_steps(path)
    if not rows:
        raise ValueError(f"{path} 是空文件（训练可能刚开始）")
    xs, ys, epoch_starts = to_global_steps(rows)
    smooth = ema(ys, alpha=0.5)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    ax.plot(xs, ys, color="#9ecae1", linewidth=1.0, label="loss（每 N 步）")
    ax.plot(xs, smooth, color="#08519c", linewidth=2.0, label="EMA 平滑")
    for e, pos in enumerate(epoch_starts, start=2):
        ax.axvline(pos, color="#bdbdbd", linestyle="--", linewidth=1)
        ax.text(pos, max(ys) * 0.98, f" epoch {e}", fontsize=9, color="#636363", va="top")
    ax.set_xlabel("训练步数（global step）")
    ax.set_ylabel("loss")
    ax.set_title(f"训练 loss 曲线：{path.stem.replace('_steps', '')}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    png = FIG_DIR / "loss_curve.png"
    fig.savefig(png)
    plt.close(fig)

    # 抽稀导出 CSV（最多 300 行），方便贴报告
    step_n = max(1, len(xs) // 300)
    with open(FIG_DIR / "loss_curve.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["global_step", "loss", "loss_ema"])
        for i in range(0, len(xs), step_n):
            w.writerow([xs[i], f"{ys[i]:.6f}", f"{smooth[i]:.6f}"])

    print(f"读取 {path.name}：{len(rows)} 个点，{len(epoch_starts) + 1} 个 epoch")
    print(f"loss：{ys[0]:.4f} → {ys[-1]:.4f}（最低 {min(ys):.4f}，最高 {max(ys):.4f}）")
    print(f"已写入 → {png}")


if __name__ == "__main__":
    main()
