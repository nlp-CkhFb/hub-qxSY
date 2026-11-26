"""
从 checkpoint 复评（不重训），并支持「解码期硬约束」。

为什么要单独一个脚本：
  1. 硬约束只是在维特比解码时把非法转移的分数置为 -1e4，不改变任何权重，
     用同一个 checkpoint 重解一次就能得到结果 —— 不需要、也不应该重训 3 个 epoch。
  2. 课程基线里 bert_crf_256_hard 的 train_s=0，就是这么来的；本脚本复现同一口径。

用法：
  python src/evaluate.py --ckpt outputs/checkpoints/bert_crf_256.pt
  python src/evaluate.py --ckpt outputs/checkpoints/bert_crf_256.pt --hard_constraint
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch.utils.data import DataLoader

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from backbones import load_tokenizer
from event_data import (
    BioDataset,
    build_bio_constraint_mask,
    build_start_constraint_mask,
    collate_bio,
    decode_bio_spans,
    illegal_bio_stats,
    load_extract_split,
    make_pred_row,
    spans_to_event,
)
from eval_utils import Timer, evaluate_predictions
from models import build_model, predict_char_tags


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--name", default="", help="默认 <ckpt名>_hard / <ckpt名>_repro")
    p.add_argument("--split", default="dev", choices=["dev", "train"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=0, help="0=沿用 checkpoint 的 max_len")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--hard_constraint", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck_args = ckpt.get("args", {})
    bb = ckpt["backbone"]
    scheme = ckpt["label_scheme"]
    id2label = scheme["id2label"]
    max_len = args.max_len or ck_args.get("max_len", 256)
    decoder = ck_args.get("decoder", "crf")

    name = args.name or (ckpt_path.stem + ("_hard" if args.hard_constraint else "_repro"))
    print(f"[eval] {name} | {ckpt_path.name} | decoder={decoder} | hard={args.hard_constraint} "
          f"| device={device}")

    model = build_model(
        bb["path"], scheme, decoder=decoder,
        num_layers=ck_args.get("layers", 0) or 0,
        dropout=ck_args.get("dropout", 0.1),
        hard_constraint=args.hard_constraint,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    if args.hard_constraint:
        if not model.use_crf:
            raise ValueError("hard_constraint 只对 CRF 解码器有意义")
        # 关键坑：constraint/start_constraint 是 persistent buffer，load_state_dict 会用
        # 训练时的「全允许」覆盖回来。必须 load 之后重新装，并断言确实禁掉了东西。
        with torch.no_grad():
            model.crf.constraint.copy_(build_bio_constraint_mask(scheme))
            model.crf.start_constraint.copy_(build_start_constraint_mask(scheme))
        n_forbidden = int((~model.crf.constraint).sum())
        assert n_forbidden > 0, "硬约束没有生效：没有禁止任何转移对"
        print(f"[hard] forbidden transitions={n_forbidden}, "
              f"forbidden starts={int((~model.crf.start_constraint).sum())}")

    tokenizer = load_tokenizer(bb["path"])
    rows = load_extract_split(args.split, data_dir=ROOT / "data" / "processed")
    if args.limit:
        rows = rows[: args.limit]
    ds = BioDataset(rows, tokenizer, scheme, max_length=max_len, with_labels=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_bio,
                        num_workers=0, pin_memory=(device.type == "cuda"))

    model.eval()
    with Timer() as t:
        char_tags, info = predict_char_tags(model, loader, id2label, device)
    timing = t.summary(len(rows))

    preds = []
    for row, tags, score in zip(rows, char_tags, info["scores"]):
        ev = spans_to_event(decode_bio_spans(tags), row["text"])
        preds.append(make_pred_row(row, [ev] if ev else [], {"confidence": round(float(score), 4)}))

    tok_ill = illegal_bio_stats(info["token_tags"])
    char_ill = illegal_bio_stats(char_tags)
    meta = {
        "display_name": f"{bb['display']} + {decoder.upper()}"
                         + (" + 硬约束解码" if args.hard_constraint else ""),
        "backbone": bb["key"],
        "backbone_path": bb["path"],
        "decoder": decoder,
        "hard_constraint": bool(args.hard_constraint),
        "train_s": 0,
        "best_epoch": ckpt.get("best_epoch", 0),
        "inner_val_span_f1": ckpt.get("inner_val_span_f1", 0.0),
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "max_len": max_len,
        "batch_size": args.batch_size,
        "device": str(device),
        "ms_per_sentence": timing["ms_per_sentence"],
        "peak_gpu_mb": timing["peak_gpu_mb"],
        "n_truncated": ds.n_truncated,
        "illegal_seq_rate": round(char_ill["illegal_seq_rate"], 4),
        "illegal_dev": char_ill,
        "illegal_token": tok_ill,
        "illegal_char": char_ill,
        "note": "解码期硬约束：同一 checkpoint，仅 mask 非法转移/非法开头后重跑维特比",
    }

    # 复评脚本默认把 train_s 设 0（这是历史上给硬约束解码用的口径）。
    # 普通复评要保留训练成本，否则汇总表会把「训练 25 分钟」写成 0。
    train_log = ROOT / "outputs" / "logs" / f"{ckpt_path.stem}_train.json"
    if train_log.exists():
        try:
            train_meta = json.loads(train_log.read_text(encoding="utf-8")).get("meta", {})
        except Exception:
            train_meta = {}
        for k in ("train_s", "params_m", "gpu", "epochs", "lr", "head_lr_mult",
                  "n_truncated", "n_truncated_rate", "illegal_train", "peak_gpu_mb"):
            if k in train_meta and k not in ("train_s", "peak_gpu_mb"):
                meta[k] = train_meta[k]
        if args.hard_constraint:
            meta["base_train_s"] = train_meta.get("train_s", 0)
            meta["base_train_peak_gpu_mb"] = train_meta.get("peak_gpu_mb", 0)
        else:
            meta["train_s"] = train_meta.get("train_s", meta["train_s"])
            meta["peak_gpu_mb"] = train_meta.get("peak_gpu_mb", meta["peak_gpu_mb"])
    res = evaluate_predictions(name, preds, meta=meta, verbose=True)
    print(f"[dev] trigger={res['trigger']['f1']:.4f} arg={res['argument']['f1']:.4f} "
          f"event={res['event_exact']['f1']:.4f} | token illegal="
          f"{tok_ill['illegal_seq_rate']:.4f} char illegal={char_ill['illegal_seq_rate']:.4f}")


if __name__ == "__main__":
    main()
