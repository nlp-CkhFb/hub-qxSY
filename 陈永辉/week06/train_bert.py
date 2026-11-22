"""BERT 微调（15 类新闻标题分类）。"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common import LABELS, evaluate, get_device, load_eval, load_splits

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


class DS(Dataset):
    def __init__(self, data, tok, max_length=64):
        enc = tok([d["sentence"] for d in data], truncation=True, max_length=max_length,
                  padding="max_length", return_tensors="pt")
        self.ids = enc["input_ids"]
        self.mask = enc["attention_mask"]
        self.y = torch.tensor([d["label"] for d in data])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "attention_mask": self.mask[i], "labels": self.y[i]}


class BertPool(nn.Module):
    def __init__(self, model_dir, num_labels=len(LABELS), pool="max", dropout=0.1):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(str(model_dir))
        self.pool = pool
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hs = out.last_hidden_state
        if self.pool == "cls":
            vec = hs[:, 0]
        else:
            m = attention_mask.unsqueeze(-1).float()
            if self.pool == "mean":
                vec = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
            else:
                vec = hs.masked_fill(m == 0, -1e9).max(1).values
        return self.fc(self.drop(vec))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, golds = [], []
    for b in loader:
        logits = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        preds.extend(logits.argmax(-1).cpu().tolist())
        golds.extend(b["labels"].tolist())
    return golds, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=str(ROOT / "pretrain_models" / "bert-base-chinese"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr_mult", type=float, default=5.0)
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--pool", default="max", choices=["cls", "mean", "max"])
    ap.add_argument("--num_train", type=int, default=0)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--no_class_weight", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default=str(OUT / "bert.pt"))
    args = ap.parse_args()

    from transformers import AutoTokenizer

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    print(f"device={device}")

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    train, _ = load_splits()
    evals, meta = load_eval(args.n_eval, args.seed)
    if args.num_train:
        train = train[: args.num_train]
    print(f"train={len(train)} eval={len(evals)} clean={meta['n_clean']} dup={meta['n_dup']}")

    train_dl = DataLoader(DS(train, tok, args.max_length), batch_size=args.batch_size, shuffle=True)
    eval_dl = DataLoader(DS(evals, tok, args.max_length), batch_size=args.batch_size)

    model = BertPool(args.model_dir, pool=args.pool).to(device)
    head = list(model.fc.parameters()) + list(model.drop.parameters())
    body = [p for n, p in model.named_parameters() if not n.startswith("fc") and not n.startswith("drop")]
    opt = torch.optim.AdamW([
        {"params": body, "lr": args.lr},
        {"params": head, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=0.01)

    if args.no_class_weight:
        weight = None
    else:
        from sklearn.utils.class_weight import compute_class_weight
        y = np.array([d["label"] for d in train])
        present = np.unique(y)
        cw = compute_class_weight("balanced", classes=present, y=y)
        weight = torch.ones(len(LABELS), dtype=torch.float)
        weight[present] = torch.tensor(cw, dtype=torch.float)
        weight = weight.to(device)
    crit = nn.CrossEntropyLoss(weight=weight)

    steps = min(len(train_dl), args.max_steps) if args.max_steps else len(train_dl)
    total = steps * args.epochs
    warm = int(total * 0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))

    best_f1, best_metrics, log = 0.0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        for i, b in enumerate(train_dl):
            opt.zero_grad(set_to_none=True)
            loss = crit(model(b["input_ids"].to(device), b["attention_mask"].to(device)),
                        b["labels"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            n += 1
            if i % 100 == 0:
                print(f"  e{epoch} step {i}/{steps} loss={loss.item():.4f}", flush=True)
            if args.max_steps and n >= args.max_steps:
                break

        golds, preds = predict(model, eval_dl, device)
        m = evaluate(golds, preds)
        log.append({"epoch": epoch, "loss": round(running / max(1, n), 4),
                    "acc": m["acc"], "macro_f1": m["macro_f1"],
                    "sec": round(time.time() - t0, 1)})
        print(f"epoch {epoch}: acc={m['acc']} macro_f1={m['macro_f1']} ({log[-1]['sec']}s)")

        if m["macro_f1"] > best_f1:
            best_f1, best_metrics = m["macro_f1"], m
            OUT.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "pool": args.pool,
                        "metrics": m, "args": vars(args)}, args.save)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bert_metrics.json").write_text(
        json.dumps({"meta": meta, "best": best_metrics, "log": log}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"best macro_f1={best_f1} -> {args.save}")
    if best_metrics:
        print("per-class recall:", json.dumps(best_metrics["per_class_recall"], ensure_ascii=False))


if __name__ == "__main__":
    main()