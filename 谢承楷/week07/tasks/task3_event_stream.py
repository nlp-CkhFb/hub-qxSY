# -*- coding: utf-8 -*-
"""
任务 3：把 extract_event() 接进事件流 demo，跑通「一条公告 → 事件列表」。

作用：
  模型/规则只解决「一句话抽一个事件」，真实业务要的是**篇章级事件流**：
  一篇公告几百字 → 切句 → 每句调 extract_event() → 校验/正则兜底 → 按出现顺序拼成事件列表。
  本脚本就是这条链路的最小可运行版本，输出可直接给下游消费的 JSON。

关键接口：
  extract_event(text)          单句入口：返回该句的事件列表（0 或 1 条）
  doc_to_events(doc)           篇章入口：切句 + 逐句抽取 + 句子内坐标换全文坐标 + 组成事件流

说明：
  这里刻意**不做跨句同类型合并** —— 公告里「6 家子公司分别回购」是 6 个独立事件，
  按类型合并会丢召回（原因见 service/extract_service.py 的同名说明）。

输入：
  data/processed/dev_docs.jsonl（篇章级 dev 集，自带金标 events 便于对照）
  outputs/checkpoints/bert_linear_256.pt（任务 1 训练出的联合模型）

输出：
  outputs/demo/event_stream_demo.json

运行方式（在 homework 目录下）：
  F:\\Miniconda3\\envs\\hf\\python.exe tasks\\task3_event_stream.py
  # 指定公告 id / 指定 checkpoint
  ... tasks\\task3_event_stream.py --doc_id <公告id> --joint_ckpt outputs\\checkpoints\\bert_linear_256.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]          # homework/
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "service"))

from event_data import load_docs                     # noqa: E402
from extract_service import ExtractService, split_sentences  # noqa: E402

BERT_PATH_DEFAULT = r"F:\huggingface_cache\models\roberta"


def extract_event(text: str, svc: ExtractService) -> list[dict]:
    """
    单句事件抽取入口（作业要求的 extract_event）。

    返回该句抽到的事件（本项目一句话最多 1 个事件），字段：
      {"type", "trigger": {...}, "args": [...]}
    坐标是**句子内**字符偏移，篇章级使用前要加句子起点 offset。
    """
    return svc.extract_sentence(text)["events"]


def doc_to_events(doc: dict, svc: ExtractService) -> dict:
    """一条公告 → 事件列表（事件流）。"""
    text = doc.get("text") or ""
    stream, sentences = [], []
    for idx, (offset, seg) in enumerate(split_sentences(text)):
        events = extract_event(seg, svc)
        # 句子内坐标 → 全文坐标：下游要高亮原文时必须用全文坐标
        for ev in events:
            trig = ev.setdefault("trigger", {})
            for key in ("start", "end"):
                if trig.get(key) is not None:
                    trig[key] += offset
            for a in ev.get("args") or []:
                for key in ("start", "end"):
                    if a.get(key) is not None:
                        a[key] += offset
            stream.append({
                "type": ev["type"],
                "trigger": trig,
                "args": ev.get("args") or [],
                "sent_idx": idx,          # 记录来自第几句，便于回溯
            })
        sentences.append({
            "sent_idx": idx,
            "text": seg.strip(),
            "n_events": len(events),
            "types": [e["type"] for e in events],
        })
    return {
        "id": doc.get("id", ""),
        "title": doc.get("title", ""),
        "n_sentences": len(sentences),
        "events": stream,
        "sentences": sentences,
    }


def brief(ev: dict) -> str:
    args = "；".join(f"{a['role']}={a['text']}" for a in ev.get("args") or [])
    return f"{ev['type']}｜触发词={ev['trigger']['text']}｜{args or '（无论元）'}"


def main():
    ap = argparse.ArgumentParser(description="事件流 demo：一条公告 → 事件列表")
    ap.add_argument("--doc_id", default="", help="指定公告 id；留空自动挑金标事件最多的一篇")
    ap.add_argument("--joint_ckpt", default=str(ROOT / "outputs" / "checkpoints" / "bert_linear_256.pt"))
    ap.add_argument("--bert_path", default=BERT_PATH_DEFAULT)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    docs = load_docs("dev", ROOT / "data" / "processed")
    if args.doc_id:
        docs = [d for d in docs if d.get("id") == args.doc_id]
    else:
        # 挑「金标事件最多」的一篇，演示效果最直观（空公告看不出东西）
        docs = [max(docs[:60], key=lambda d: len(d.get("events") or []))]
    if not docs:
        print("没有找到公告")
        return
    doc = docs[0]

    svc = ExtractService(backend="joint", joint_ckpt=args.joint_ckpt,
                         bert_path=args.bert_path, device=args.device)
    svc.load()

    result = doc_to_events(doc, svc)

    print("=" * 78)
    print(f"公告 ID：{result['id']}")
    print(f"标题：{result['title'][:60]}")
    print(f"原文：{doc['text'][:160]}{'...' if len(doc['text']) > 160 else ''}")
    print(f"切句数：{result['n_sentences']}")
    print("-" * 78)
    print(f"【事件列表】共 {len(result['events'])} 个事件")
    for i, ev in enumerate(result["events"], 1):
        print(f"  {i}. [句{ev['sent_idx']}] {brief(ev)}")
    if not result["events"]:
        print("  （没有抽到事件）")
    print("-" * 78)
    gold = doc.get("events") or []
    print(f"【与金标对照】金标 {len(gold)} 个：{[e.get('type') for e in gold]}")
    print(f"              抽出 {len(result['events'])} 个：{[e['type'] for e in result['events']]}")
    print(f"              类型命中：{len(set(e.get('type') for e in gold) & set(e['type'] for e in result['events']))}")
    print("=" * 78)
    print(json.dumps(result["events"], ensure_ascii=False, indent=2)[:1500])

    out = ROOT / "outputs" / "demo" / "event_stream_demo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 → {out}")


if __name__ == "__main__":
    main()
