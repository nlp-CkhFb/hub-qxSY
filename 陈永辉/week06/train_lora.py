"""Qwen3-0.6B + LoRA 指令微调（15 类新闻标题分类）。"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common import LABELS, evaluate, get_device, load_eval, load_splits

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
SYSTEM = ("你是一个新闻标题分类助手。把标题分到以下类别之一，只输出类别名："
          + "、".join(LABELS))


def chat_prompt(tok, text):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class SFTDataset(Dataset):
    def __init__(self, data, tok, max_length=128):
        self.items = []
        for d in data:
            prompt = chat_prompt(tok, d["sentence"])
            p = tok(prompt, add_special_tokens=False)["input_ids"]
            a = tok(LABELS[d["label"]], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            ids = (p + a)[:max_length]
            labels = ([-100] * len(p) + a)[:max_length]
            self.items.append((ids, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ids, labels = self.items[i]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels)}


def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), L), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = b["input_ids"]
        labels[i, :n] = b["labels"]
        mask[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


@torch.no_grad()
def predict(model, tok, samples, device, max_new=8):
    model.eval()
    preds, golds = [], []
    for d in samples:
        prompt = chat_prompt(tok, d["sentence"])
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        label = next((i for i, name in enumerate(LABELS) if name == text), -1)
        if label < 0:
            label = next((i for i, name in enumerate(LABELS) if name in text), -1)
        preds.append(label)
        golds.append(d["label"])
    return golds, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=str(ROOT / "pretrain_models" / "Qwen3-0.6B"))
    ap.add_argument("--num_train", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", default=str(OUT / "lora_adapter"))
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise SystemExit(
            f"找不到 Qwen3-0.6B：{model_dir}\n"
            f"请把模型目录放到 week6/pretrain_models/Qwen3-0.6B，"
            f"或用 --model_dir 指定。")

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    device = get_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"device={device} dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train, _ = load_splits()
    train = train[: args.num_train]
    evals, meta = load_eval(args.n_eval, args.seed)
    print(f"train={len(train)} eval={len(evals)} clean={meta['n_clean']} dup={meta['n_dup']}")

    try:
        base = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=dtype)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype)
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r,
                     lora_alpha=args.lora_alpha, lora_dropout=0.05,
                     target_modules=["q_proj", "v_proj"])
    model = get_peft_model(base, cfg).to(device)
    model.print_trainable_parameters()

    from functools import partial
    coll = partial(collate, pad_id=tok.pad_token_id)
    dl = DataLoader(SFTDataset(train, tok, args.max_length), batch_size=args.batch_size,
                    shuffle=True, collate_fn=coll, drop_last=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    steps = min(len(dl), args.max_steps) if args.max_steps else len(dl)
    best_f1, best_metrics, log = 0.0, None, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(dl):
            out = model(input_ids=b["input_ids"].to(device),
                        attention_mask=b["attention_mask"].to(device),
                        labels=b["labels"].to(device))
            (out.loss / args.grad_accum).backward()
            if (i + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            running += out.loss.item()
            n += 1
            if i % 50 == 0:
                print(f"  e{epoch} step {i}/{steps} loss={out.loss.item():.4f}", flush=True)
            if args.max_steps and n >= args.max_steps:
                break

        golds, preds = predict(model, tok, evals, device)
        m = evaluate(golds, preds)
        log.append({"epoch": epoch, "loss": round(running / max(1, n), 4),
                    "acc": m["acc"], "macro_f1": m["macro_f1"],
                    "sec": round(time.time() - t0, 1)})
        print(f"epoch {epoch}: acc={m['acc']} macro_f1={m['macro_f1']} ({log[-1]['sec']}s)")

        if m["macro_f1"] > best_f1:
            best_f1, best_metrics = m["macro_f1"], m
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.save_dir)
            tok.save_pretrained(args.save_dir)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lora_metrics.json").write_text(
        json.dumps({"meta": meta, "best": best_metrics, "log": log}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"best macro_f1={best_f1} -> {args.save_dir}")
    if best_metrics:
        print("per-class recall:", json.dumps(best_metrics["per_class_recall"], ensure_ascii=False))


if __name__ == "__main__":
    main()