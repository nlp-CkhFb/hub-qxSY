"""
统一训练脚本：一个骨干 × 一个解码器 → 一个事件抽取模型。

用法（在本目录 week07 下）：
  python src/train.py --backbone bert-base-chinese    --decoder linear
  python src/train.py --backbone bert-base-chinese    --decoder crf
  python src/train.py --backbone roberta-base-chinese --decoder linear
  python src/train.py --backbone roberta-base-chinese --decoder crf

设计要点：
  - 只改「骨干」这一个变量：两个模型同为 12 层 / 768 / 102M，数据、标签（149）、
    epoch、batch、学习率、评测口径全部一致。
  - dev 930 句不参与训练/选点：从 train 切 10% 做内层验证，按字符级 span F1 选 epoch。
  - 4GB 显存：batch 8 + 梯度累积 4（有效 batch 32，与课程基线一致）。
    实测 batch 16 会顶到 3.7GB、触发 WDDM 共享显存换页，吞吐反而从 123ms/样本掉到 196ms/样本。
    GTX 1650 Ti（Turing，无 tensor core）实测 fp16 比 fp32 慢 3.7×，所以默认关 AMP。
  - CRF 的 loss 在 fp32 里算（前向算法的 logsumexp 在 fp16 下容易溢出）。
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from backbones import load_tokenizer, resolve_backbone
from event_data import (
    BioDataset,
    build_joint_schema,
    collate_bio,
    decode_bio_spans,
    illegal_bio_stats,
    load_extract_split,
    load_schema,
    make_pred_row,
    row_event,
    span_to_char_tags,
    spans_to_event,
)
from eval_utils import CKPT_DIR, LOG_DIR, Timer, evaluate_predictions
from models import build_model, build_optimizer, predict_char_tags


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="bert-base-chinese",
                   help="bert-base-chinese | roberta-base-chinese | 本地模型目录")
    p.add_argument("--decoder", default="linear", choices=["linear", "crf"])
    p.add_argument("--layers", type=int, default=0, help="0=全部 12 层；>0 只保留前 N 层")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--head_lr_mult", type=float, default=5.0)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--inner_val", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", default="")
    p.add_argument("--data_dir", default="")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--amp", dest="amp", action="store_true", default=False,
                   help="fp16 混合精度。GTX 1650 Ti 无 tensor core，实测 fp16 反而慢 3.7×，默认关闭")
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help=">0 时只取前 N 条做冒烟")
    p.add_argument("--hard_constraint", action="store_true",
                   help="训练时也加硬约束（默认不加；CRF 硬约束用 evaluate.py 解码期做）")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但 torch.cuda.is_available()=False")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_inner(rows: list[dict], ratio: float, seed: int):
    """从 train 切出内层验证集：先固定排序再 shuffle，保证同一 seed 结果稳定。"""
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_val = max(1, int(round(len(rows) * ratio))) if ratio > 0 else 0
    return rows[n_val:], rows[:n_val]


def char_gold_tags(rows: list[dict], label_scheme: dict) -> list[list[str]]:
    labs = label_scheme["labels"]
    out = []
    for r in rows:
        ev = row_event(r)
        if ev is None:
            out.append(["O"] * len(r["text"]))
        else:
            ids = span_to_char_tags(r["text"], ev, label_scheme)
            out.append([labs[i] for i in ids])
    return out


def span_set(tags: list[str]):
    return {(sp["tag"], sp["start"], sp["end"]) for sp in decode_bio_spans(tags)}


def span_f1(gold_tags: list[list[str]], pred_tags: list[list[str]]):
    tp = fp = fn = 0
    for gold, pred in zip(gold_tags, pred_tags):
        g, p = span_set(gold), span_set(pred)
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


@torch.no_grad()
def evaluate_val(model, loader, gold_tags, id2label, device):
    char_tags, _ = predict_char_tags(model, loader, id2label, device)
    # 预测的字符级长度可能因句尾空格比原文短，补 O 后再比 span
    padded = []
    for tags, gold in zip(char_tags, gold_tags):
        padded.append(tags + ["O"] * max(0, len(gold) - len(tags)))
    return span_f1(gold_tags, padded)


def train(args):
    set_seed(args.seed)
    device = pick_device(args.device)
    amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # 输入长度动态变化，自动选最快 kernel
    bb = resolve_backbone(args.backbone)
    data_dir = Path(args.data_dir) if args.data_dir else ROOT / "data" / "processed"

    schema = load_schema()
    label_scheme = build_joint_schema(schema)
    id2label = label_scheme["id2label"]

    run_name = args.run_name or f"{bb['key'].split('-')[0]}_{args.decoder}_{args.max_len}"
    print(f"[run] {run_name} | {bb['display']} | decoder={args.decoder} | device={device} | amp={amp}")
    print(f"[data] {data_dir}")

    rows = load_extract_split("train", data_dir=data_dir)
    if args.limit:
        rows = rows[: args.limit]
    train_rows, val_rows = split_inner(rows, args.inner_val, args.seed)
    dev_rows = load_extract_split("dev", data_dir=data_dir)
    if args.limit:
        dev_rows = dev_rows[: max(50, args.limit // 10)]
    print(f"[data] train={len(train_rows)} inner_val={len(val_rows)} dev={len(dev_rows)}")

    tokenizer = load_tokenizer(bb["path"])
    print(f"[tok] {type(tokenizer).__name__} vocab={tokenizer.vocab_size}")
    train_ds = BioDataset(train_rows, tokenizer, label_scheme, max_length=args.max_len)
    val_ds = BioDataset(val_rows, tokenizer, label_scheme, max_length=args.max_len)
    dev_ds = BioDataset(dev_rows, tokenizer, label_scheme, max_length=args.max_len, with_labels=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_bio,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_bio,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_bio,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    val_gold = char_gold_tags(val_rows, label_scheme)
    model = build_model(
        bb["path"], label_scheme, decoder=args.decoder, num_layers=args.layers,
        dropout=args.dropout, hard_constraint=args.hard_constraint,
    ).to(device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] params={params_m:.1f}M | layers={model.encoder.num_layers_used} "
          f"| token_type_ids={model.encoder.accepts_token_type_ids}")

    optimizer = build_optimizer(model, lr=args.lr, head_lr_mult=args.head_lr_mult,
                                weight_decay=args.weight_decay)
    accum = max(1, args.grad_accum)
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    steps_path = LOG_DIR / f"{run_name}_steps.jsonl"
    steps_f = open(steps_path, "w", encoding="utf-8")
    history = []
    best_f1, best_epoch, best_path = -1.0, 0, None
    t_train0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running, seen, micro = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            token_type_ids = batch["token_type_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                emit = model.emissions(input_ids, attention_mask, token_type_ids)
            emit = emit.float()
            if model.use_crf:
                tags = labels.clone()
                tags[tags == -100] = 0
                loss = model.crf.neg_log_likelihood(emit, tags, attention_mask)
            else:
                loss = F.cross_entropy(emit.view(-1, model.num_tags), labels.view(-1),
                                       ignore_index=-100)
            loss = loss / accum
            scaler.scale(loss).backward()

            running += float(loss.item()) * accum * input_ids.size(0)
            seen += input_ids.size(0)
            micro += 1

            if micro % accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                if step % args.log_every == 0 or step == len(train_loader):
                    rec = {
                        "epoch": epoch, "step": step, "loss": round(running / max(seen, 1), 5),
                        "lr": round(scheduler.get_last_lr()[0], 8),
                        "grad_norm": round(grad_norm, 4),
                        "elapsed_s": round(time.perf_counter() - t0, 1),
                        "samples": seen,
                    }
                    steps_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    steps_f.flush()
                    print(f"  e{epoch} step {step}/{len(train_loader)} loss={rec['loss']:.4f} "
                          f"lr={rec['lr']:.2e} gnorm={grad_norm:.2f} {rec['elapsed_s']}s", flush=True)

        val = evaluate_val(model, val_loader, val_gold, id2label, device)
        epoch_s = time.perf_counter() - t0
        rec = {"epoch": epoch, "train_loss": round(running / max(seen, 1), 5),
               "inner_val": {k: round(v, 5) for k, v in val.items()}, "epoch_s": round(epoch_s, 1)}
        history.append(rec)
        print(f"[epoch {epoch}] loss={rec['train_loss']:.4f} inner_val spanF1={val['f1']:.4f} "
              f"P={val['precision']:.4f} R={val['recall']:.4f} ({epoch_s/60:.1f} min)", flush=True)

        if val["f1"] > best_f1:
            best_f1, best_epoch = val["f1"], epoch
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            best_path = CKPT_DIR / f"{run_name}.pt"
            tmp = best_path.with_suffix(".tmp")
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
                "backbone": bb,
                "label_scheme": label_scheme,
                "best_epoch": best_epoch,
                "inner_val_span_f1": best_f1,
            }, tmp)
            tmp.replace(best_path)
            print(f"  [ckpt] saved {best_path.name} (inner_val F1={best_f1:.4f})", flush=True)

    steps_f.close()
    train_s = time.perf_counter() - t_train0
    train_peak_mb = (torch.cuda.max_memory_allocated() / 1024 ** 2) if device.type == "cuda" else 0.0

    # ── 用内层验证选出的最优 epoch 在 dev 上评测（dev 只在这里用一次）──
    if best_path is not None:
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    model.eval()

    with Timer() as t:
        char_tags, info = predict_char_tags(model, dev_loader, id2label, device)
    infer = t.summary(len(dev_rows))

    preds = []
    for row, tags, score in zip(dev_rows, char_tags, info["scores"]):
        spans = decode_bio_spans(tags)
        ev = spans_to_event(spans, row["text"])
        preds.append(make_pred_row(row, [ev] if ev else [], {"confidence": round(float(score), 4)}))

    tok_ill = illegal_bio_stats(info["token_tags"])
    char_ill = illegal_bio_stats(char_tags)
    meta = {
        "display_name": f"{bb['display']} + {args.decoder.upper()}",
        "backbone": bb["key"],
        "backbone_display": bb["display"],
        "backbone_path": bb["path"],
        "decoder": args.decoder,
        "num_layers": model.encoder.num_layers_used,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": accum,
        "effective_batch": args.batch_size * accum,
        "lr": args.lr,
        "head_lr_mult": args.head_lr_mult,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "amp": amp,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "seed": args.seed,
        "params_m": round(params_m, 1),
        "train_s": round(train_s, 1),
        "ms_per_sentence": infer["ms_per_sentence"],
        "sentences_per_s": infer["sentences_per_s"],
        "peak_gpu_mb": round(train_peak_mb, 1),
        "infer_peak_gpu_mb": infer["peak_gpu_mb"],
        "best_epoch": best_epoch,
        "inner_val_span_f1": round(best_f1, 5),
        "n_truncated": dev_ds.n_truncated,
        "n_truncated_rate": round(dev_ds.n_truncated / max(len(dev_rows), 1), 4),
        "illegal_seq_rate": round(char_ill["illegal_seq_rate"], 4),
        "illegal_dev": char_ill,
        "illegal_token": tok_ill,
        "illegal_char": char_ill,
        "hard_constraint_train": bool(args.hard_constraint),
    }
    with open(LOG_DIR / f"{run_name}_train.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "backbone": bb, "history": history, "meta": meta},
                  f, ensure_ascii=False, indent=2)

    print(f"[dev] evaluating {len(dev_rows)} sentences ...")
    res = evaluate_predictions(run_name, preds, meta=meta, verbose=True)
    print(f"[dev] trigger F1={res['trigger']['f1']:.4f} arg F1={res['argument']['f1']:.4f} "
          f"event F1={res['event_exact']['f1']:.4f} | token illegal={tok_ill['illegal_seq_rate']:.4f} "
          f"| {infer['ms_per_sentence']:.1f} ms/句")
    return res


if __name__ == "__main__":
    train(parse_args())
