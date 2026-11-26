"""
汇总所有实验：主表、n200、per-type、per-role、成本、显著性，并出图 + results.md。

用法：
  python experiments/summarize.py
  python experiments/summarize.py --significance   # 额外跑 7 组配对检验（约 1 分钟）

输入：
  outputs/metrics/disc/{method}.json     （train.py / evaluate.py 自动生成）
  outputs/logs/{method}_steps.jsonl      （训练曲线，可选）
  report/significance.json               （significance.py 生成，可选）
输出：
  report/results.md、report/summary.csv、report/per_type.csv、report/per_role.csv、
  report/figures/*.png
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from evaluate_events import load_jsonl
from significance import load_pred_map, per_type_metrics

METRIC_DIR = ROOT / "outputs" / "metrics" / "disc"
PRED_DIR = ROOT / "outputs" / "disc"
LOG_DIR = ROOT / "outputs" / "logs"
REPORT = ROOT / "report"
FIG = REPORT / "figures"

METHODS = [
    ("bert_linear_256", "BERT + Linear", "#4C78A8"),
    ("bert_crf_256", "BERT + CRF", "#2F5D8A"),
    ("bert_crf_256_hard", "BERT + CRF + 硬约束", "#8FB8DE"),
    ("roberta_linear_256", "RoBERTa + Linear", "#E45756"),
    ("roberta_crf_256", "RoBERTa + CRF", "#B23A3A"),
    ("roberta_crf_256_hard", "RoBERTa + CRF + 硬约束", "#F2A6A0"),
]
BASE = "bert_linear_256"

# 课程仓库 outputs/metrics/disc/summary.md 里的 BERT 基线（同一 dev 930、同一评测脚本）
COURSE = {
    "bert_linear_256": {"trigger": 0.8293, "argument": 0.5630, "event": 0.1564},
    "bert_crf_256": {"trigger": 0.8320, "argument": 0.5815, "event": 0.1697},
    "bert_crf_256_hard": {"trigger": 0.8464, "argument": 0.6566, "event": 0.2177},
}



def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def prf(d):
    return {"p": d["precision"], "r": d["recall"], "f1": d["f1"],
            "tp": d.get("tp", 0), "fp": d.get("fp", 0), "fn": d.get("fn", 0)}


def collect():
    out = {}
    for key, label, color in METHODS:
        p = METRIC_DIR / f"{key}.json"
        if not p.exists():
            print(f"[skip] 缺指标：{p.name}")
            continue
        d = load_json(p)
        meta = d.get("meta") or {}
        ill = meta.get("illegal_dev") or {}
        out[key] = {
            "key": key, "label": label, "color": color,
            "trigger": prf(d["trigger"]), "argument": prf(d["argument"]),
            "event": prf(d["event_exact"]),
            "type_macro_f1": d.get("type_macro_f1"),
            "n200": d.get("n200_summary") or {},
            "per_type": d.get("per_type") or {},
            "per_role": d.get("per_role") or {},
            "meta": meta,
            "train_s": meta.get("train_s"),
            "ms_per_sentence": meta.get("ms_per_sentence"),
            "params_m": meta.get("params_m"),
            "illegal": ill.get("illegal_seq_rate") if isinstance(ill, dict) else ill,
            "illegal_per_seq": ill.get("illegal_per_seq") if isinstance(ill, dict) else None,
            "best_epoch": meta.get("best_epoch"),
            "inner_val": meta.get("inner_val_span_f1"),
            "peak_mb": meta.get("peak_gpu_mb"),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--significance", action="store_true")
    args = ap.parse_args()

    REPORT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    res = collect()
    if not res:
        print("没有可汇总的指标文件，先跑 experiments/run_all.py")
        return
    base = res.get(BASE)

    # per_type 里的 f1 是「事件级」的；这里另算「论元级」分类型 F1
    gold = load_jsonl(str(ROOT / "data/processed/dev_extract.jsonl"))
    types = sorted({t for r in res.values() for t in r["per_type"]})
    for key, r in res.items():
        pred_path = PRED_DIR / f"pred_{key}.jsonl"
        r["arg_per_type"] = (per_type_metrics(gold, load_pred_map(pred_path), types)
                             if pred_path.exists() else {})

    # ── 显著性（可选） ──────────────────────────────────────────
    sig = None
    sig_path = REPORT / "significance.json"
    if args.significance:
        from significance import write_significance
        gold = load_jsonl(str(ROOT / "data/processed/dev_extract.jsonl"))
        pairs = []
        # 每个骨干：Linear→CRF、CRF→硬约束；跨骨干：同方法
        for a, b in [("bert_linear_256", "bert_crf_256"),
                     ("bert_crf_256", "bert_crf_256_hard"),
                     ("roberta_linear_256", "roberta_crf_256"),
                     ("roberta_crf_256", "roberta_crf_256_hard"),
                     ("bert_crf_256", "roberta_crf_256"),
                     ("bert_crf_256_hard", "roberta_crf_256_hard"),
                     ("bert_linear_256", "roberta_linear_256")]:
            if (PRED_DIR / f"pred_{a}.jsonl").exists() and (PRED_DIR / f"pred_{b}.jsonl").exists():
                pairs.append((a, PRED_DIR / f"pred_{a}.jsonl", b, PRED_DIR / f"pred_{b}.jsonl"))
        sig = write_significance(pairs, sig_path, gold)
    elif sig_path.exists():
        sig = load_json(sig_path)

    # ── 主表 ───────────────────────────────────────────────────
    def cell(v, nd=4):
        return "—" if v is None else f"{v:.{nd}f}"

    rows = []
    for key, label, _ in METHODS:
        r = res.get(key)
        if not r:
            continue
        d_arg = (r["argument"]["f1"] - base["argument"]["f1"]) * 100 if base else None
        d_ev = (r["event"]["f1"] - base["event"]["f1"]) * 100 if base else None
        rows.append({
            "method": key, "display": label,
            "trigger_f1": r["trigger"]["f1"], "argument_f1": r["argument"]["f1"],
            "event_f1": r["event"]["f1"],
            "d_arg_pp": None if d_arg is None else round(d_arg, 2),
            "d_event_pp": None if d_ev is None else round(d_ev, 2),
            "n200_arg": r["n200"].get("argument_f1"),
            "n200_event": r["n200"].get("event_f1"),
            "train_s": r["train_s"], "ms_per_sentence": r["ms_per_sentence"],
            "illegal_seq_rate": r["illegal"],
            "params_m": r["params_m"], "best_epoch": r["best_epoch"],
        })

    with open(REPORT / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ── per-type / per-role ──────────────────────────────────────
    types = sorted(types, key=lambda t: -res.get(BASE, res[list(res)[0]])["arg_per_type"].get(t, {}).get("support", 0))
    per_type_rows = []
    for t in types:
        row = {"type": t}
        support = 0
        for key, r in res.items():
            pt = r["arg_per_type"].get(t)
            if pt:
                support = max(support, pt.get("support", 0))
            row[f"{key}_arg_f1"] = pt["argument_f1"] if pt else None
        row["support"] = support
        # 每类 arg F1 按方法排序，方便看谁最好
        per_type_rows.append(row)
    with open(REPORT / "per_type.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(per_type_rows[0]))
        w.writeheader()
        w.writerows(per_type_rows)

    roles = sorted({r for x in res.values() for r in x["per_role"]},
                   key=lambda r: -(sum(x["per_role"].get(r, {}).get("tp", 0)
                                       + x["per_role"].get(r, {}).get("fn", 0)
                                       for x in res.values())))
    role_rows = []
    for role in roles[:30]:
        row = {"role": role, "support": max((x["per_role"].get(role, {}).get("tp", 0)
                                             + x["per_role"].get(role, {}).get("fn", 0))
                                            for x in res.values())}
        for key, x in res.items():
            row[f"{key}_f1"] = x["per_role"].get(role, {}).get("f1")
        role_rows.append(row)
    with open(REPORT / "per_role.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(role_rows[0]))
        w.writeheader()
        w.writerows(role_rows)

    # ── 图 ─────────────────────────────────────────────────────
    make_figures(res, sig)

    # ── results.md ─────────────────────────────────────────────
    write_md(res, rows, per_type_rows, role_rows, sig)
    print(f"[summarize] → {REPORT / 'results.md'}")


def _fonts():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def make_figures(res, sig):
    plt = _fonts()
    import numpy as np

    keys = [k for k, _, _ in METHODS if k in res]
    labels = [res[k]["label"] for k in keys]
    colors = [res[k]["color"] for k in keys]
    x = np.arange(len(keys))
    w = 0.26

    # 1) 三指标分组柱
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, (metric, name) in enumerate([("trigger", "触发词 F1"), ("argument", "论元 F1"), ("event", "事件完全匹配 F1")]):
        vals = [res[k][metric]["f1"] for k in keys]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("dev 930：不同骨干 × 解码器（含解码期硬约束）")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(FIG / "main_metrics.png", dpi=180)
    plt.close(fig)

    # 2) 相对 BERT+Linear 的差值
    if BASE in res:
        fig, ax = plt.subplots(figsize=(8.5, 3.8))
        d_arg = [(res[k]["argument"]["f1"] - res[BASE]["argument"]["f1"]) * 100 for k in keys]
        d_ev = [(res[k]["event"]["f1"] - res[BASE]["event"]["f1"]) * 100 for k in keys]
        ax.barh(np.arange(len(keys)) + .18, d_arg, .34, label="论元 F1 差值", color="#4C78A8")
        ax.barh(np.arange(len(keys)) - .18, d_ev, .34, label="事件 F1 差值", color="#F58518")
        ax.set_yticks(np.arange(len(keys)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="black", lw=.8)
        ax.set_xlabel("相对 BERT + Linear 的百分点")
        ax.set_title("提升幅度（pp）")
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=.25)
        fig.tight_layout()
        fig.savefig(FIG / "delta_vs_bert_linear.png", dpi=180)
        plt.close(fig)

    # 3) per-type 论元 F1 热力图
    types = sorted({t for k in keys for t in res[k]["arg_per_type"]},
                   key=lambda t: -(res[keys[0]]["arg_per_type"].get(t, {}).get("support", 0)))
    M = np.array([[res[k]["arg_per_type"].get(t, {}).get("argument_f1", np.nan) for k in keys] for t in types])
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    im = ax.imshow(M, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([res[k]["label"] for k in keys], rotation=25, ha="right", fontsize=7)
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels(types, fontsize=7)
    for i in range(len(types)):
        for j in range(len(keys)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=.025, label="论元 F1")
    ax.set_title("分事件类型论元 F1")
    fig.tight_layout()
    fig.savefig(FIG / "per_type_argument_f1.png", dpi=180)
    plt.close(fig)

    # 4) 成本 vs 效果
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k in keys:
        r = res[k]
        if r["train_s"]:
            ax.scatter(r["train_s"] / 60, r["argument"]["f1"], s=60, color=r["color"], label=r["label"])
            ax.annotate(r["label"], (r["train_s"] / 60, r["argument"]["f1"]),
                        fontsize=6, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("训练时间（分钟）")
    ax.set_ylabel("dev 论元 F1")
    ax.set_title("效果—成本（硬约束复用 checkpoint，train_s=0 未画）")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(FIG / "cost_vs_quality.png", dpi=180)
    plt.close(fig)

    # 5) 训练曲线（内层验证 spanF1 / loss）
    steps = {}
    for k in keys:
        p = LOG_DIR / f"{k}_steps.jsonl"
        if p.exists():
            steps[k] = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if steps:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for k, hist in steps.items():
            ep = [h["epoch"] + h["step"] / max(1, max(x["step"] for x in hist)) for h in hist]
            axes[0].plot(ep, [h["loss"] for h in hist], lw=1, label=res[k]["label"])
            axes[1].plot([h["elapsed_s"] / 60 for h in hist], [h["loss"] for h in hist], lw=1)
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("training loss"); axes[0].set_title("loss vs epoch")
        axes[1].set_xlabel("minutes"); axes[1].set_ylabel("training loss"); axes[1].set_title("loss vs wall-clock")
        axes[0].legend(fontsize=6); axes[0].grid(alpha=.25); axes[1].grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(FIG / "training_curves.png", dpi=180)
        plt.close(fig)


def write_md(res, rows, per_type_rows, role_rows, sig):
    L = []
    L.append("# 实验结果：BERT vs RoBERTa、Linear vs CRF vs 硬约束\n")
    L.append("数据：DuEE-Fin 派生（13 事件类型 / 92 槽位 / 61 角色），train 5,747 句，"
             "dev 930 句；joint 149 标签 BIO；评测口径与课程基线完全一致"
             "（`src/evaluate_events.py`，trigger 文本完全匹配、论元按 (role,text) Counter）。\n")
    L.append("> 本文件由 `experiments/summarize.py` 自动生成，数字来自 `outputs/metrics/disc/*.json`；"
             "叙述性结论见根目录 `README.md`。\n")

    L.append("## 1. 主表（dev 930）\n")
    L.append("| 方法 | 触发词 F1 | 论元 F1 | 事件 F1 | Δ论元 vs BERT+Linear | Δ事件 | n200 论元 F1 | 训练(s) | 推理(ms/句) | 非法率 | 参数量 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        d_arg = "—" if r["d_arg_pp"] is None else f"{r['d_arg_pp']:+.2f}pp"
        d_ev = "—" if r["d_event_pp"] is None else f"{r['d_event_pp']:+.2f}pp"
        n200 = "—" if r["n200_arg"] is None else f"{r['n200_arg']:.4f}"
        ts = "—" if r["train_s"] is None else f"{r['train_s']:.0f}"
        ms = "—" if r["ms_per_sentence"] is None else f"{r['ms_per_sentence']:.2f}"
        ill = "—" if r["illegal_seq_rate"] is None else f"{r['illegal_seq_rate']*100:.1f}%"
        pm = "—" if r["params_m"] is None else f"{r['params_m']:.1f}M"
        L.append(f"| {r['display']} | {r['trigger_f1']:.4f} | {r['argument_f1']:.4f} | "
                 f"{r['event_f1']:.4f} | {d_arg} | {d_ev} | {n200} | {ts} | {ms} | {ill} | {pm} |")
    L.append("")

    # 与课程基线逐项对照（只对 BERT 三个方法有对应条目）
    L.append("### 1.1 与课程文件内 BERT 基线对照\n")
    L.append("课程基线来自 `baseline/`（原仓库 `outputs/metrics/disc/summary.md`），"
             "dev 930、评测脚本与本实验相同；差异主要来自本实验修正了 LayerNorm 的 "
             "`gamma/beta` 加载（见 README），以及 batch/seed 不同。\n")
    L.append("| 方法 | 指标 | 课程基线 | 本实验 | Δ(pp) |")
    L.append("|---|---|---:|---:|---:|")
    for r in rows:
        c = COURSE.get(r["method"])
        if not c:
            continue
        for mkey, metric, key in [("trigger", "触发词", "trigger_f1"),
                                  ("argument", "论元", "argument_f1"),
                                  ("event", "事件", "event_f1")]:
            L.append(f"| {r['display']} | {metric} | {c[mkey]:.4f} | "
                     f"{r[key]:.4f} | {(r[key] - c[mkey]) * 100:+.2f} |")
    L.append("")

    if sig:
        def fp(p):
            return "<0.0001" if p < 1e-4 else f"{p:.4f}"

        L.append("## 2. 配对显著性检验\n")
        L.append("配对 bootstrap：同一批句子的重采样下标同时作用于两方法，5000 次，"
                 "报告 F1 差值的 95% 区间与双侧 p；McNemar 用句子级「事件完全正确」做 2×2 精确检验。"
                 "Δ = 后者 − 前者，正数表示后者更好。\n")
        L.append("| 对比 | 指标 | Δ(pp) | 95% CI | bootstrap p | McNemar p |")
        L.append("|---|---|---:|---|---:|---:|")
        for c in sig:
            mc = c.get("mcnemar_event_exact", {}).get("p_mcnemar")
            for metric in ("trigger", "argument", "event"):
                b = c["bootstrap"][metric]
                L.append(f"| {c['a']} ↔ {c['b']} | {metric} | {b['delta_pp']:+.2f} | "
                         f"[{b['ci95_pp'][0]:+.2f}, {b['ci95_pp'][1]:+.2f}] | {fp(b['p_bootstrap'])} | {fp(mc)} |")
        L.append("")

    L.append("## 3. 分事件类型论元 F1\n")
    keys = [k for k, _, _ in METHODS if k in res]
    L.append("| 类型 | 支持数 | " + " | ".join(res[k]["label"] for k in keys) + " |")
    L.append("|---|---:|" + "---:|" * len(keys))
    for row in per_type_rows:
        vals = []
        for k in keys:
            v = row.get(f"{k}_arg_f1")
            vals.append("—" if v is None else f"{v:.3f}")
        L.append(f"| {row['type']} | {row['support']} | " + " | ".join(vals) + " |")
    L.append("")

    L.append("## 4. 高频角色论元 F1（Top 20）\n")
    L.append("| 角色 | 支持数 | " + " | ".join(res[k]["label"] for k in keys) + " |")
    L.append("|---|---:|" + "---:|" * len(keys))
    for row in role_rows[:20]:
        vals = []
        for k in keys:
            v = row.get(f"{k}_f1")
            vals.append("—" if v is None else f"{v:.3f}")
        L.append(f"| {row['role']} | {row['support']} | " + " | ".join(vals) + " |")
    L.append("")

    L.append("## 5. 图\n")
    for name, cap in [("main_metrics.png", "主指标对比"),
                      ("delta_vs_bert_linear.png", "相对 BERT+Linear 的提升"),
                      ("per_type_argument_f1.png", "分类型论元 F1 热力图"),
                      ("cost_vs_quality.png", "效果—训练成本"),
                      ("training_curves.png", "训练曲线")]:
        if (FIG / name).exists():
            L.append(f"![{cap}](figures/{name})\n")

    (REPORT / "results.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
