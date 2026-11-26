"""
微调实验：同一个分类头、同一套超参，只换预训练骨干。

控制变量：
  - 同样的 7k 训练集 / 1k 开发集 / 2k 测试集；
  - 同样的 max_len=40、batch=32、epochs=2、lr=2e-5、warmup=10%、class weights；
  - 同样的随机种子和评测函数。
这样 BERT 与 RoBERTa 的差异才能归因到“预训练权重”，而不是训练技巧。
"""

import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

import config
import models
import metrics


class TextDataset(Dataset):
    def __init__(self, records, tokenizer, max_len: int):
        self.enc = tokenizer(
            [r["sentence"] for r in records],
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor([r["label"] for r in records], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}, self.labels[i]


def _collate(batch):
    encs, labels = zip(*batch)
    out = {k: torch.stack([e[k] for e in encs]) for k in encs[0]}
    return out, torch.stack(labels)


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    preds, probs = [], []
    t0 = time.perf_counter()
    for enc, _ in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc)
        prob = torch.softmax(logits, dim=-1)
        preds.append(prob.argmax(dim=-1).cpu())
        probs.append(prob.cpu())
    elapsed = time.perf_counter() - t0
    return (
        torch.cat(preds).numpy().astype(np.int64),
        torch.cat(probs).numpy(),
        elapsed,
    )


def run_finetune(
    kind: str,
    train,
    dev,
    test,
    epochs: int = None,
    batch_size: int = None,
    lr: float = None,
    max_len: int = None,
    max_steps: int = None,
    seed: int = None,
    save_model: bool = False,
):
    epochs = epochs or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE
    lr = lr or config.LR
    max_len = max_len or config.MAX_LEN
    seed = config.SEED if seed is None else seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config.NUM_THREADS)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t_start = time.perf_counter()
    tokenizer = models.load_tokenizer(str(config.BACKBONES[kind]["dir"]))
    train_ds = TextDataset(train, tokenizer, max_len)
    dev_ds = TextDataset(dev, tokenizer, max_len)
    test_ds = TextDataset(test, tokenizer, max_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=64, shuffle=False, collate_fn=_collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False, collate_fn=_collate
    )

    model = models.build_classifier(kind, config.NUM_LABELS).to(device)

    # 类别不平衡：与课程代码保持一致，使用 sqrt-inverse 风格前先做标准 inverse
    counts = np.bincount([r["label"] for r in train], minlength=config.NUM_LABELS)
    weights = len(train) / (config.NUM_LABELS * np.maximum(counts, 1))
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY
    )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * config.WARMUP_RATIO)),
        num_training_steps=max(1, total_steps),
    )

    best_f1 = -1.0
    best_state = None
    global_step = 0
    epoch_run = 0

    for epoch in range(epochs):
        model.train()
        epoch_run = epoch + 1
        running_loss = 0.0
        for step, (enc, labels) in enumerate(train_loader):
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(**enc)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            global_step += 1
            running_loss += loss.item()

            if global_step % 50 == 0:
                print(
                    f"  [{kind}] epoch {epoch + 1} step {global_step}/{total_steps} "
                    f"loss={running_loss / 50:.4f}",
                    flush=True,
                )
                running_loss = 0.0

            if max_steps is not None and global_step >= max_steps:
                break

        dev_pred, _, _ = _predict(model, dev_loader, device)
        dev_true = np.array([r["label"] for r in dev], dtype=np.int64)
        dev_f1 = metrics.evaluate(dev_true, dev_pred, labels=range(config.NUM_LABELS))[
            "macro_f1"
        ]
        print(
            f"  [{kind}] epoch {epoch + 1} dev macro-F1={dev_f1:.4f}",
            flush=True,
        )
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if max_steps is not None and global_step >= max_steps:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_seconds = time.perf_counter() - t_start

    test_pred, test_prob, test_forward = _predict(model, test_loader, device)
    infer_seconds = test_forward

    if save_model:
        config.ensure_dirs()
        torch.save(model.state_dict(), config.OUT_DIR / f"{kind}_ft_model.pt")

    return {
        "pred": test_pred,
        "prob": test_prob,
        "train_seconds": train_seconds,
        "infer_seconds": infer_seconds,
        "extra": {
            "epochs_run": epoch_run,
            "steps": global_step,
            "batch_size": batch_size,
            "lr": lr,
            "max_len": max_len,
            "best_dev_macro_f1": round(float(best_f1), 6),
            "params_m": round(models.count_params(model) / 1e6, 3),
            "checkpoint_mb": config.BACKBONES[kind]["size_mb"],
            "device": device,
        },
    }
