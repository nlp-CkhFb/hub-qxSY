# -*- coding: utf-8 -*-
"""
任务 1：四路方法（规则 / BERT / 两段式 / LLM）跑完后，汇总成一张与 PPT 第 35 页同格式的对比表。

作用：
  只从 outputs/metrics 下读取各方法**真实跑出来的**指标 JSON，绝不手抄数字、绝不填估算值；
  缺失的方法显式标注「未跑」，避免把 mock 或旧结果混进作业结论。

输入（按 homework 目录结构）：
  outputs/metrics/disc/rule.json                 规则基线
  outputs/metrics/disc/bert_linear_256.json      BERT+Linear 联合模型
  outputs/metrics/disc/pipeline.json             两段式流水线
  outputs/metrics/llm/<tag>.json                 本地大模型 zero-shot
  outputs/llm/report_<tag>.json                  LLM 的延迟 / token 统计（可选）

输出：
  outputs/metrics/summary_4methods.md             Markdown 对比表（可直接贴进报告）
  outputs/metrics/summary_4methods.csv            CSV 版

运行方式（在 homework 目录下）：
  F:\\Miniconda3\\envs\\hf\\python.exe tasks\\task1_compare.py
  # 指定 LLM 的 tag（默认自动挑 metrics/llm 下最新的一个）
  ... tasks\\task1_compare.py --llm_tag bonsai_zero_n20
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]          # homework/
METRIC_DISC = ROOT / "outputs" / "metrics" / "disc"
METRIC_LLM = ROOT / "outputs" / "metrics" / "llm"
REPORT_LLM = ROOT / "outputs" / "llm"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def f1_of(metrics: dict | None, key: str) -> str:
    """从统一评测结果里取三级 F1；缺失时返回 '未跑'，不猜。"""
    if not metrics or key not in metrics:
        return "未跑"
    return f"{metrics[key]['f1']:.4f}"


def pick_llm_tag(explicit: str) -> str:
    """没指定就用 metrics/llm 下最新的一份（按修改时间）。"""
    if explicit:
        return explicit
    files = sorted(METRIC_LLM.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0].stem if files else ""


def build_rows(llm_tag: str) -> list[dict]:
    rows = []

    # ── L0 规则 ──
    rule = load_json(METRIC_DISC / "rule.json")
    rows.append({
        "方法（层级）": "规则：词典 + 正则（L0）",
        "trigger F1": f1_of(rule, "trigger"),
        "argument F1": f1_of(rule, "argument"),
        "event F1": f1_of(rule, "event_exact"),
        "训练/成本": "0（纯 CPU）" if rule else "未跑",
        "延迟": (f"{rule['meta']['ms_per_sentence']:.2f} ms"
                 if rule and rule.get("meta") else "未跑"),
    })

    # ── L2 BERT 联合 ──
    bert = load_json(METRIC_DISC / "bert_linear_256.json")
    bert_meta = (bert or {}).get("meta") or {}
    bert_cost = "未跑"
    if bert:
        bert_cost = (f"{bert_meta.get('train_s', '-')} s"
                     f"（{bert_meta.get('bert_layers', 12)} 层 / "
                     f"{bert_meta.get('epochs', '-')} epoch）")
    rows.append({
        "方法（层级）": "BERT + Linear（L2，联合 149 标签）",
        "trigger F1": f1_of(bert, "trigger"),
        "argument F1": f1_of(bert, "argument"),
        "event F1": f1_of(bert, "event_exact"),
        "训练/成本": bert_cost,
        "延迟": (f"{bert_meta['ms_per_sentence']:.2f} ms"
                 if bert_meta.get("ms_per_sentence") is not None else "未跑"),
    })

    # ── L2 两段式 ──
    pipe = load_json(METRIC_DISC / "pipeline.json")
    pipe_meta = (pipe or {}).get("meta") or {}
    pipe_cost = "未跑"
    if pipe:
        pipe_cost = (f"{pipe_meta.get('train_s', '-')} s"
                     f"（16 个模型 / 论元 {pipe_meta.get('arg_epochs', '-')} epoch）")
    rows.append({
        "方法（层级）": "两段式 检测→类型→论元（L2）",
        "trigger F1": f1_of(pipe, "trigger"),
        "argument F1": f1_of(pipe, "argument"),
        "event F1": f1_of(pipe, "event_exact"),
        "训练/成本": pipe_cost,
        "延迟": (f"{pipe_meta['ms_per_sentence']:.2f} ms"
                 if pipe_meta.get("ms_per_sentence") is not None else "未跑"),
    })

    # ── L3 本地 LLM zero-shot ──
    llm = load_json(METRIC_LLM / f"{llm_tag}.json") if llm_tag else None
    report = load_json(REPORT_LLM / f"report_{llm_tag}.json") if llm_tag else None
    if llm:
        n = report.get("n", "-") if report else "-"
        model = report.get("model", "本地 LLM") if report else "本地 LLM"
        cost = f"{n} 条 / 本地推理 {report.get('total_s', '-')} s" if report else f"{n} 条"
        latency = (f"{report['latency_mean_s'] * 1000:.0f} ms"
                   if report and report.get("latency_mean_s") is not None else "-")
        label = f"本地 LLM zero-shot（L3，{Path(str(model)).name[:28]}，n={n}）"
    else:
        label, cost, latency = "本地 LLM zero-shot（L3）", "未跑", "未跑"
    rows.append({
        "方法（层级）": label,
        "trigger F1": f1_of(llm, "trigger"),
        "argument F1": f1_of(llm, "argument"),
        "event F1": f1_of(llm, "event_exact"),
        "训练/成本": cost,
        "延迟": latency,
    })
    return rows


def to_markdown(rows: list[dict]) -> str:
    cols = ["方法（层级）", "trigger F1", "argument F1", "event F1", "训练/成本", "延迟"]
    lines = ["# 四路方法对比（同一评测口径：common/evaluate_events.py，dev_extract 930 句）", "",
             "| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    lines.append("")
    lines.append("> 说明：表中数字全部来自本次实跑（homework/outputs/metrics），"
                 "未跑的方法显示「未跑」；LLM 为本地量化模型 zero-shot，样本数见方法名。")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="汇总四路方法指标，生成对比表")
    ap.add_argument("--llm_tag", default="", help="LLM 评测结果的 tag（默认取最新）")
    args = ap.parse_args()

    llm_tag = pick_llm_tag(args.llm_tag)
    rows = build_rows(llm_tag)
    md = to_markdown(rows)

    out_dir = ROOT / "outputs" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "summary_4methods.md"
    csv_path = out_dir / "summary_4methods.csv"
    md_path.write_text(md, encoding="utf-8")

    import csv
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(md)
    print(f"已写入 → {md_path}")
    print(f"已写入 → {csv_path}")


if __name__ == "__main__":
    main()
