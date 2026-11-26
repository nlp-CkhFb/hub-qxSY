"""BERT 联合 BIO 事件抽取。"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common import (DATA, decode_spans, evaluate, get_device, load_jsonl, load_schema,
                    make_labels, row_event, span_to_tags, spans_to_event)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


class BioDS(Dataset):
    def __init__(self, rows, tok, label2id, max_len=256):
        self.rows, self.tok, self.label2id, self.max_len = rows, tok, label2id, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        text = row["text"]
        enc = self.tok(list(text), is_split_into_words=True, max_length=self.max_len,
                       truncation=True, padding=False)
        wids = enc.word_ids()
        ev = row_event(row)
        char_tags = (span_to_tags(text, ev, self.label2id) if ev
                     else [self.label2id["O"]] * len(text))
        labels, prev = [], None
        for w in wids:
            if w is None:
                labels.append(-100)
            elif w != prev:
                labels.append(char_tags[w] if w < len(char_tags) else -100)
                prev = w
            else:
                labels.append(-100)
        return {"input_ids": torch.tensor(enc["input_ids"]),
                "attention_mask": torch.tensor(enc["attention_mask"]),
                "labels": torch.tensor(labels),
                "word_ids": torch.tensor([-1 if w is None else w for w in wids]),
                "idx": i}


def collate(batch):
    L = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": torch.zeros(len(batch), L, dtype=torch.long),
           "attention_mask": torch.zeros(len(batch), L, dtype=torch.long),
           "labels": torch.full((len(batch), L), -100, dtype=torch.long),
           "word_ids": torch.full((len(batch), L), -1, dtype=torch.long),
           "idx": torch.tensor([b["idx"] for b in batch])}
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        for k in ("input_ids", "attention_mask", "labels", "word_ids"):
            out[k][i, :n] = b[k]
    return out


class Tagger(nn.Module):
    def __init__(self, model_dir, num_labels):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(str(model_dir))
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, ids, mask):
        out = self.bert(input_ids=ids, attention_mask=mask, return_dict=True)
        return self.fc(self.drop(out.last_hidden_state))


@torch.no_grad()
def predict(model, loader, id2label, rows, device):
    model.eval()
    preds = []
    for b in loader:
        logits = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        ids = logits.argmax(-1).cpu().tolist()
        wids = b["word_ids"].tolist()
        for bi, ri in enumerate(b["idx"].tolist()):
            row = rows[ri]
            text = row["text"]
            tags, prev = ["O"] * len(text), None
            for j, w in enumerate(wids[bi]):
                if w < 0 or w == prev:
                    continue
                if w < len(tags):
                    tags[w] = id2label[ids[bi][j]]
                prev = w
            ev = spans_to_event(decode_spans(tags), text)
            preds.append({"id": row.get("id", ri), "events": [ev] if ev else []})
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=str(ROOT / "pretrain_models" / "bert-base-chinese"))
    ap.add_argument("--train", default=str(DATA / "train_extract.jsonl"))
    ap.add_argument("--dev", default=str(DATA / "dev_extract.jsonl"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr_mult", type=float, default=5.0)
    ap.add_argument("--num_train", type=int, default=0)
    ap.add_argument("--n_eval", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default=str(OUT / "bert.pt"))
    args = ap.parse_args()

    from transformers import AutoTokenizer

    torch.manual_seed(args.seed)
    device = get_device()
    schema = load_schema()
    labels, label2id, id2label = make_labels(schema)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    train = load_jsonl(args.train)
    dev = load_jsonl(args.dev)
    if args.num_train:
        train = train[: args.num_train]
    if args.n_eval:
        dev = dev[: args.n_eval]
    print(f"device={device} labels={len(labels)} train={len(train)} dev={len(dev)}")

    train_dl = DataLoader(BioDS(train, tok, label2id, args.max_len), batch_size=args.batch_size,
                          shuffle=True, collate_fn=collate, drop_last=True)
    dev_dl = DataLoader(BioDS(dev, tok, label2id, args.max_len), batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate)

    model = Tagger(args.model_dir, len(labels)).to(device)
    head = list(model.fc.parameters()) + list(model.drop.parameters())
    body = [p for n, p in model.named_parameters() if not n.startswith("fc") and not n.startswith("drop")]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr},
                             {"params": head, "lr": args.lr * args.head_lr_mult}],
                            weight_decay=0.01)
    crit = nn.CrossEntropyLoss(ignore_index=-100)
    steps = min(len(train_dl), args.max_steps) if args.max_steps else len(train_dl)
    total = steps * args.epochs
    warm = int(total * 0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))

    best, log = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        for i, b in enumerate(train_dl):
            opt.zero_grad(set_to_none=True)
            logits = model(b["input_ids"].to(device), b["attention_mask"].to(device))
            loss = crit(logits.reshape(-1, logits.size(-1)), b["labels"].to(device).view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            n += 1
            if i % 50 == 0:
                print(f"  e{epoch} step {i}/{steps} loss={loss.item():.4f}", flush=True)
            if args.max_steps and n >= args.max_steps:
                break

        preds = predict(model, dev_dl, id2label, dev, device)
        m = evaluate(dev, preds)
        score = m["event"]["f1"]
        log.append({"epoch": epoch, "loss": round(running / max(1, n), 4),
                    "trigger_f1": m["trigger"]["f1"], "argument_f1": m["argument"]["f1"],
                    "event_f1": score, "sec": round(time.time() - t0, 1)})
        print(f"epoch {epoch}: trigger={m['trigger']['f1']} arg={m['argument']['f1']} "
              f"event={score} ({log[-1]['sec']}s)")

        if score > best:
            best = score
            OUT.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "metrics": m, "args": vars(args)}, args.save)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bert_metrics.json").write_text(
        json.dumps({"best_event_f1": best, "log": log}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best event_f1={best} -> {args.save}")


if __name__ == "__main__":
    main()
