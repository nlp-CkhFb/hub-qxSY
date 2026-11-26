# -*- coding: utf-8 -*-
"""
任务 2：从模型预测里挑 10 条 badcase，按「边界 / 角色串味 / 触发词歧义」三类归档。

作用：
  把「指标掉了多少」翻译成「具体错在哪」，这是论文/答辩里最容易被追问的部分。
  脚本用可复现的规则做初筛，并保留原文+金标+预测三栏，方便人工复核。

三类错误的判定规则（全部基于 span 与角色的可解释特征，不靠关键词瞎猜）：
  边界      预测 span 与金标 span 有交叠但不完全一致（多字/少字/包含关系）
  角色串味  论元文本对得上，但被贴到了另一个角色（如 收购方 ↔ 被收购方）
  触发词歧义 触发词没抽到、或触发词/事件类型判成另一个（如 质押 ↔ 解除质押）

输入：
  data/processed/dev_extract.jsonl                 金标
  outputs/disc/pred_<source>.jsonl                 预测（bert / pipeline）
  outputs/llm/pred_<tag>.jsonl                     预测（llm，配套 gold_n*.jsonl）

输出：
  outputs/badcase/badcase_10.md
  outputs/badcase/badcase_10.json

运行方式（在 homework 目录下）：
  F:\\Miniconda3\\envs\\hf\\python.exe tasks\\task2_badcase.py
  ... tasks\\task2_badcase.py --source pipeline
  ... tasks\\task2_badcase.py --source llm --llm_tag bonsai_zero_n20
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "processed" / "dev_extract.jsonl"

CAT_BOUNDARY = "边界"
CAT_ROLE = "角色串味"
CAT_TRIGGER = "触发词歧义"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def spans_overlap(a: dict, b: dict) -> bool:
    """两段 span 是否交叠（缺 offset 时退回文本包含判断）。"""
    if all(k in a for k in ("start", "end")) and all(k in b for k in ("start", "end")):
        return a["start"] < b["end"] and b["start"] < a["end"]
    ta, tb = (a.get("text") or ""), (b.get("text") or "")
    return bool(ta and tb and (ta in tb or tb in ta))


def classify(gold_ev: dict, pred_ev: dict | None) -> tuple[str, str] | None:
    """返回 (类别, 诊断说明)；无法归类时返回 None。"""
    g_trig = (gold_ev.get("trigger") or {}).get("text", "")
    g_args = gold_ev.get("args") or []

    if not pred_ev or not (pred_ev.get("trigger") or {}).get("text"):
        return CAT_TRIGGER, f"整句漏抽：金标触发词「{g_trig}」没有被识别"

    p_trig = (pred_ev.get("trigger") or {}).get("text", "")
    if pred_ev.get("type") != gold_ev.get("type"):
        return CAT_TRIGGER, (f"类型判错：金标「{gold_ev.get('type')}」→ 预测「{pred_ev.get('type')}」"
                             f"（触发词 {g_trig}→{p_trig}）")
    if p_trig != g_trig and not spans_overlap(gold_ev["trigger"], pred_ev["trigger"]):
        return CAT_TRIGGER, f"触发词替换：金标「{g_trig}」→ 预测「{p_trig}」"

    # 触发词层面的边界错误（「IPO」vs「IPO发行」这类多字/少字）
    boundary_note = ""
    if p_trig != g_trig and spans_overlap(gold_ev["trigger"], pred_ev["trigger"]):
        boundary_note = f"触发词边界：金标「{g_trig}」→ 预测「{p_trig}」"

    g_set = {(a["role"], a["text"]) for a in g_args}
    p_set = {(a["role"], a["text"]) for a in (pred_ev.get("args") or [])}
    g_rest = [a for a in g_args if (a["role"], a["text"]) not in p_set]
    p_rest = [a for a in (pred_ev.get("args") or []) if (a["role"], a["text"]) not in g_set]

    # 1) 角色串味：文本完全一致，角色不同
    for p in p_rest:
        for g in g_rest:
            if p["text"] == g["text"] and p["role"] != g["role"]:
                return CAT_ROLE, f"角色串味：金标把「{g['text']}」标成 <{g['role']}>，预测标成 <{p['role']}>"

    # 2) 边界：同一段文字被多抽/少抽了几个字
    for p in p_rest:
        for g in g_rest:
            if p["role"] == g["role"] and spans_overlap(p, g):
                return CAT_BOUNDARY, f"论元边界：<{g['role']}> 金标「{g['text']}」→ 预测「{p['text']}」"

    if boundary_note:
        return CAT_BOUNDARY, boundary_note
    return None


def load_data(source: str, llm_tag: str) -> tuple[Path, Path]:
    if source == "bert":
        return GOLD, ROOT / "outputs" / "disc" / "pred_bert_linear_256.jsonl"
    if source == "pipeline":
        return GOLD, ROOT / "outputs" / "disc" / "pred_pipeline.jsonl"
    if source == "llm":
        pred = ROOT / "outputs" / "llm" / f"pred_{llm_tag}.jsonl"
        report = ROOT / "outputs" / "llm" / f"gold_n"
        # LLM 的评测集是 dev 的采样子集，金标在 outputs/llm/gold_n*.jsonl
        cands = sorted((ROOT / "outputs" / "llm").glob("gold_n*.jsonl"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            raise FileNotFoundError("找不到 LLM 用的 gold_n*.jsonl，请先跑 llm_extract.py")
        return cands[0], pred
    raise ValueError(source)


def main():
    ap = argparse.ArgumentParser(description="挑 10 条 badcase 并归类")
    ap.add_argument("--source", default="bert", choices=["bert", "pipeline", "llm"])
    ap.add_argument("--llm_tag", default="bonsai_zero_n20")
    ap.add_argument("--per_cat", type=int, default=4, help="每类最多选几条")
    args = ap.parse_args()

    gold_path, pred_path = load_data(args.source, args.llm_tag)
    if not pred_path.exists():
        raise FileNotFoundError(f"找不到预测文件：{pred_path}")
    gold_rows = load_jsonl(gold_path)
    pred_map = {(r.get("id"), r.get("sent_id")): r for r in load_jsonl(pred_path)}
    print(f"金标 {gold_path.name}（{len(gold_rows)} 条）| 预测 {pred_path.name}")

    buckets: dict[str, list[dict]] = defaultdict(list)
    for g in gold_rows:
        pred = pred_map.get((g.get("id"), g.get("sent_id")))
        pred_ev = (pred.get("events") or [None])[0] if pred else None
        gold_ev = g.get("event")
        if not gold_ev:
            continue
        hit = classify(gold_ev, pred_ev)
        if not hit:
            continue
        cat, note = hit
        if len(buckets[cat]) >= args.per_cat:
            continue
        buckets[cat].append({
            "category": cat,
            "note": note,
            "text": g.get("text", ""),
            "gold": gold_ev,
            "pred": pred_ev,
        })

    picked = []
    for cat in (CAT_BOUNDARY, CAT_ROLE, CAT_TRIGGER):
        picked.extend(buckets.get(cat, []))
    picked = picked[:10]
    print(f"选中 {len(picked)} 条：" + "、".join(
        f"{c}{len(buckets.get(c, []))}" for c in (CAT_BOUNDARY, CAT_ROLE, CAT_TRIGGER)))

    lines = [f"# Badcase 分析（10 条，来源：{args.source}）", "",
             "| # | 类别 | 原文（截断） | 金标 | 预测 | 诊断 |",
             "|---|---|---|---|---|---|"]
    for i, b in enumerate(picked, 1):
        g = b["gold"]
        p = b["pred"] or {}
        g_args = "；".join(f"{a['role']}={a['text']}" for a in g.get("args") or [])
        p_args = "；".join(f"{a['role']}={a['text']}" for a in p.get("args") or []) or "（空）"
        g_str = f"{g.get('type')}｜{g['trigger']['text']}｜{g_args}"
        p_str = (f"{p.get('type', '—')}｜{p['trigger']['text']}｜{p_args}"
                 if p else "（未抽到事件）")
        text = b["text"].replace("|", "／")
        lines.append(f"| {i} | {b['category']} | {text[:70]}… | {g_str} | {p_str} | {b['note']} |")

    out_dir = ROOT / "outputs" / "badcase"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "badcase_10.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "badcase_10.json").write_text(
        json.dumps(picked, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n已写入 → {out_dir / 'badcase_10.md'}")


if __name__ == "__main__":
    main()
