# -*- coding: utf-8 -*-
from __future__ import annotations
import os
# 【学生修改】：强制使用国内镜像下载，避免网络超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]   
sys.path.insert(0, str(ROOT / "common"))     

from crf import CRFClassifier
from eval_utils import CKPT_DIR, LOG_DIR, Timer, evaluate_predictions
from log_utils import TrainLogger, peak_gpu_mb
from event_data import (BioDataset, DATA_DIR, build_joint_schema, collate_bio,
                        decode_bio_spans, illegal_bio_stats, load_enums, load_extract_split,
                        load_schema, make_pred_row, row_event, spans_to_event)
from models import batch_to_device, build_model, predict_char_tags

IGNORE_INDEX = -100

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def span_f1_from_char_tags(rows: list[dict], char_tags: list[list[str]]) -> float:
    tp = fp = fn = 0
    for row, tags in zip(rows, char_tags):
        ge = row_event(row) or {}
        gold = {(a["role"], a["text"]) for a in ge.get("args") or []}
        gold.add(("__trigger__", (ge.get("trigger") or {}).get("text", "")))
        ev = spans_to_event(decode_bio_spans(tags), row["text"])
        pred = {(a["role"], a["text"]) for a in (ev or {}).get("args", [])}
        pred.add(("__trigger__", (ev or {}).get("trigger", {}).get("text", "")))
        pred.discard(("__trigger__", ""))
        gold.discard(("__trigger__", ""))
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0

def evaluate_loss_and_f1(model, loader, id2label: dict, device) -> tuple[float, float]:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            _, loss = model(batch["input_ids"], batch["attention_mask"],
                            batch["labels"], batch.get("token_type_ids"))
            total += float(loss.item())
            n += 1
    avg_loss = total / max(n, 1)
    tags, _ = predict_char_tags(model, loader, id2label, device)
    f1 = span_f1_from_char_tags(loader.dataset.rows, tags)
    return avg_loss, f1

def build_optimizer(model, args):
    enc_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        (enc_params if name.startswith("encoder.") else head_params).append(p)
    groups = [{"params": enc_params, "lr": args.lr}]
    if head_params:
        groups.append({"params": head_params, "lr": args.lr * args.head_lr_mult})
    return AdamW(groups, weight_decay=args.weight_decay)

def run_epoch(model, loader, optimizer, scheduler, device, args):
    model.train()
    total, n = 0.0, 0
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        batch = batch_to_device(batch, device)
        _, loss = model(batch["input_ids"], batch["attention_mask"],
                        batch["labels"], batch.get("token_type_ids"))
        (loss / args.grad_accum).backward()
        total += float(loss.item())
        n += 1
        if (step + 1) % args.grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        if (step + 1) % 20 == 0:
            print(f"  step {step + 1}/{len(loader)} loss={total / n:.4f}", flush=True)
    
    if len(loader) % args.grad_accum != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return total / max(n, 1)

def parse_args():
    # 把所有适合你电脑的参数全写成默认值
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", type=str, default="crf")
    parser.add_argument("--bert_path", type=str, default="bert-base-chinese")
    parser.add_argument("--bert_layers", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)   # 保护3GB显存
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head_lr_mult", type=float, default=5.0)
    parser.add_argument("--grad_accum", type=int, default=2)   # 等效 batch=8
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hard_constraint", action="store_true", default=False) # 补齐属性
    parser.add_argument("--run_name", type=str, default="bert_all_2k")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")
    
    data_dir = DATA_DIR
    schema = load_schema(data_dir / "event_schema.json")
    enums = load_enums(data_dir / "event_schema.json")
    label_scheme = build_joint_schema(schema)
    labels = label_scheme["labels"]
    id2label = label_scheme["id2label"]
    print(f"标签体系：joint，{len(labels)} 个标签")

    print(f"正在加载 Tokenizer: {args.bert_path}")
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_path)
 
    random.seed(42)  
    train_all = load_extract_split("train", data_dir)
    dev_all = load_extract_split("dev", data_dir)
    train_rows = random.sample(train_all, min(2000, len(train_all)))
    dev_rows = random.sample(dev_all, min(500, len(dev_all)))

    rng = random.Random(42)
    idxs = list(range(len(train_rows)))
    rng.shuffle(idxs)
    n_inner = max(1, int(len(train_rows) * 0.1))
    inner_idx = set(idxs[:n_inner])
    fit_rows = [r for i, r in enumerate(train_rows) if i not in inner_idx]
    val_rows = [r for i, r in enumerate(train_rows) if i in inner_idx]
    print(f"训练 {len(fit_rows)} 句 / 内层验证 {len(val_rows)} 句 / 评测 dev {len(dev_rows)} 句")

    fit_loader = DataLoader(BioDataset(fit_rows, tokenizer, label_scheme, args.max_len), batch_size=args.batch_size, shuffle=True, collate_fn=collate_bio)
    val_loader = DataLoader(BioDataset(val_rows, tokenizer, label_scheme, args.max_len), batch_size=args.batch_size, shuffle=False, collate_fn=collate_bio)
    dev_loader = DataLoader(BioDataset(dev_rows, tokenizer, label_scheme, args.max_len), batch_size=args.batch_size, shuffle=False, collate_fn=collate_bio)

    print(f"正在加载模型权重...")
    model = build_model(args, label_scheme).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数：{n_params / 1e6:.1f}M")

    optimizer = build_optimizer(model, args)
    total_steps = max(1, len(fit_loader) * args.epochs // args.grad_accum)
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{args.run_name}.pt"
    logger = TrainLogger(method=args.run_name, args=args, root=ROOT, params=n_params, notes="")

    best_f1, best_epoch = -1.0, 0
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = run_epoch(model, fit_loader, optimizer, scheduler, device, args)
        val_loss, val_f1 = evaluate_loss_and_f1(model, val_loader, id2label, device)
        _, dev_f1 = evaluate_loss_and_f1(model, dev_loader, id2label, device)
        elapsed = time.perf_counter() - t0
        print(f"epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | inner_F1={val_f1:.4f} | dev_F1={dev_f1:.4f} | {elapsed:.0f}s", flush=True)
        
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            torch.save({"state_dict": model.state_dict(), "label_scheme": label_scheme, "id2label": id2label, "args": vars(args)}, ckpt_path)
            print(f"  ★ 最优 → 保存 {ckpt_path.name}", flush=True)

    train_s = time.perf_counter() - train_start
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    
    with Timer() as timer:
        dev_tags, _ = predict_char_tags(model, dev_loader, id2label, device)
    dev_illegal = illegal_bio_stats(dev_tags)

    preds = []
    for row, tags in zip(dev_rows, dev_tags):
        ev = spans_to_event(decode_bio_spans(tags), row["text"], enums=enums)
        preds.append(make_pred_row(row, [ev] if ev else []))

    lat = timer.summary(len(dev_rows))
    res = evaluate_predictions(args.run_name, preds, meta={"train_s": round(train_s,1)}, gold_path=data_dir / "dev_extract.jsonl")

    print("\n【非法 BIO 统计】")
    print(f"  含非法序列 {dev_illegal['illegal_seqs']} 条（{dev_illegal['illegal_seq_rate']:.1%}）")
    print(f"\n训练耗时 {train_s:.0f}s | 延迟 {lat['ms_per_sentence']} ms/句")

if __name__ == "__main__":
    main()