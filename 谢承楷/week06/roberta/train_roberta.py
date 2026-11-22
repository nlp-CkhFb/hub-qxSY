"""
RoBERTa 底座 · 消融实验训练脚本（独立于 bert/，不改动原有 BERT 代码）

教学重点：
  1. 架构只改两处——**池化方式**和**损失函数**，RoBERTa 本体 12 层不动。
  2. 四个池化口径：cls / mean / max / cls_mean（CLS 与 Mean 拼接，768×2）。
  3. 池化前必须 mask 掉 [PAD]，否则 Mean/Max 会把补位统计进去。
  4. 四个训练技巧本文件都实现了，但这个任务上基本不涨：
       类别加权（默认开）  长尾类 recall 从「全错」回到「能中一部分」
       Label Smoothing    软标签稀释有限信号
       Focal Loss         难样本主要来自标签口径重叠，不是「学不会」
       FGM 对抗训练       几乎不动，训练时间还 ×1.5

口径（与课件表格一致）：
  同一份训练集、同一批无重叠验证样本（8,901 条）、固定 seed=42、3 epoch。
  模型权重只从本地 HF 模型库读取，不联网、不写入项目目录。

用法：
  python roberta/train_roberta.py                       # cls 池化，3 epoch
  python roberta/train_roberta.py --pool max --epochs 5
  python roberta/train_roberta.py --pool cls_mean        # CLS + Mean 拼接
  python roberta/train_roberta.py --lr 3e-5              # 学习率对照
  python roberta/train_roberta.py --max_length 32        # 截断长度对照
  python roberta/train_roberta.py --label_smoothing 0.1  # 软标签
  python roberta/train_roberta.py --focal_gamma 2        # Focal Loss
  python roberta/train_roberta.py --fgm                  # FGM 对抗训练
  python roberta/train_roberta.py --max_steps 20 --max_eval 300 --no_save   # 冒烟
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_clean_val, load_splits
from common.labels import LABELS

CODE_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = CODE_ROOT / "outputs"

# 模型权重放在 HF 模型库，代码目录里不放权重
MODEL_DIR = Path(os.environ.get("ROBERTA_MODEL_DIR", r"F:\huggingface_cache\models\roberta"))


class TnewsDataset(Dataset):
    """把 {"sentence","label"} 变成 RoBERTa 的输入张量"""

    def __init__(self, data, tok, max_length=64):
        self.data, self.tok, self.max_length = data, tok, max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        enc = self.tok(d["sentence"], truncation=True, max_length=self.max_length,
                       padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "labels": torch.tensor(d["label"])}


class RobertaClassifier(nn.Module):
    """
    RoBERTa + 池化 + 线性头。

    pool 四选一：
      cls      —— 取 [CLS] 向量（预训练就为它设计，但不一定最适合短文本）
      mean     —— 对所有有效 token 取均值（短文本常更稳）
      max      —— 逐维取最大（抓最显著特征，噪声也大）
      cls_mean —— CLS 与 Mean 拼接（768×2，表达更全但参数翻倍）
    """

    def __init__(self, model_dir, num_labels=15, pool="cls", dropout=0.1):
        super().__init__()
        from transformers import AutoModel
        self.roberta = AutoModel.from_pretrained(str(model_dir))
        self.pool = pool
        hidden = self.roberta.config.hidden_size          # roberta-base = 768
        pooled = hidden * 2 if pool == "cls_mean" else hidden
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(pooled, num_labels)

    def forward(self, input_ids, attention_mask):
        # return_dict=True：transformers 5.x 里不传会拿到 tuple，容易下标取错
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hs = out.last_hidden_state                     # [B, L, 768]
        if self.pool == "cls":
            vec = hs[:, 0]
        else:
            m = attention_mask.unsqueeze(-1).float()   # [B, L, 1]
            if self.pool == "mean":
                vec = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
            elif self.pool == "cls_mean":
                mean = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
                vec = torch.cat([hs[:, 0], mean], dim=-1)
            else:                                      # max：PAD 位置压到极小值再取 max
                vec = hs.masked_fill(m == 0, -1e9).max(1).values
        return self.classifier(self.dropout(vec))


class FGM:
    """
    FGM（Fast Gradient Method）对抗训练 —— 三步：
      ① 正常前向 + 反传，拿到参数的梯度 g；
      ② 在 **word_embeddings** 上沿梯度方向加一点扰动：r = eps * g / ||g||；
      ③ 用被扰动的权重再前向 + 反传一次，两次梯度一起更新，最后把权重还原。
    """

    def __init__(self, model, emb_name="word_embeddings", eps=1.0):
        self.model, self.emb_name, self.eps = model, emb_name, eps
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(self.eps * param.grad / norm)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def focal_loss(logits, targets, weight=None, gamma=2.0):
    """
    Focal Loss：把「已经学会的样本」的权重压下去，让梯度集中在难样本上。
        L = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    logp = F.log_softmax(logits, dim=-1)
    logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
    p_t = logp_t.exp()
    alpha = weight[targets] if weight is not None else 1.0
    return (-alpha * ((1.0 - p_t) ** gamma) * logp_t).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    ys, ps = [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        ps.extend(logits.argmax(-1).cpu().tolist())
        ys.extend(batch["labels"].tolist())
    return {
        "acc": round(float(accuracy_score(ys, ps)), 4),
        "macro_f1": round(float(f1_score(ys, ps, average="macro")), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="cls", choices=["cls", "mean", "max", "cls_mean"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5, help="RoBERTa 层学习率，越小越保守")
    ap.add_argument("--head_lr_mult", type=float, default=5.0, help="分类头 lr = 底座 lr × 该倍数")
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--no_class_weight", action="store_true")
    # ↓ 四个训练技巧
    ap.add_argument("--label_smoothing", type=float, default=0.0, help="软标签 ε，常用 0.05~0.1")
    ap.add_argument("--focal_gamma", type=float, default=0.0, help="Focal Loss 的 γ，>0 才启用（与类别加权共用权重）")
    ap.add_argument("--fgm", action="store_true", help="打开 FGM 对抗训练（训练时间约 ×1.5）")
    ap.add_argument("--fgm_eps", type=float, default=1.0, help="FGM 扰动强度 ε")
    # ↓ 冒烟/调试用
    ap.add_argument("--max_steps", type=int, default=0, help="每个 epoch 最多跑多少步，0=全跑（冒烟用）")
    ap.add_argument("--max_eval", type=int, default=0, help="验证集最多评多少条，0=全评（冒烟用）")
    ap.add_argument("--no_save", action="store_true", help="不落盘 checkpoint（消融用，省 390 MB）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not MODEL_DIR.is_dir() or not (
            (MODEL_DIR / "pytorch_model.bin").exists() or any(MODEL_DIR.glob("*.safetensors"))):
        raise FileNotFoundError(f"未找到本地 RoBERTa 权重：{MODEL_DIR}（需要 config.json + 权重 + vocab.txt）")

    from transformers import AutoTokenizer

    # 固定随机种子：消融实验只允许一个旋钮变
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    train, _ = load_splits()
    val, meta = load_clean_val(seed=args.seed)          # 去重后的干净验证集，8,901 条
    if args.max_eval:
        val = val[: args.max_eval]

    train_dl = DataLoader(TnewsDataset(train, tok, args.max_length), batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(TnewsDataset(val, tok, args.max_length), batch_size=args.batch_size)

    model = RobertaClassifier(MODEL_DIR, num_labels=len(LABELS), pool=args.pool).to(device)

    # 分层学习率：头是随机初始化的，需要学得更快；底座要小步，否则冲掉预训练知识
    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    base_params = [p for n, p in model.named_parameters() if not n.startswith("classifier") and not n.startswith("dropout")]
    opt = torch.optim.AdamW([
        {"params": base_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=0.01)

    # 类别加权：23× 不均衡，不加权时尾部类 recall 会趋近 0
    if args.no_class_weight:
        weight = None
    else:
        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=[d["label"] for d in train])
        weight = torch.tensor(cw, dtype=torch.float).to(device)

    # 损失函数：Focal 与 Label Smoothing 不叠加，Focal 优先
    if args.focal_gamma > 0:
        if args.label_smoothing > 0:
            print("[warn] 同时给了 --focal_gamma 和 --label_smoothing，按 Focal 跑", flush=True)
        crit = lambda logits, y: focal_loss(logits, y, weight, args.focal_gamma)   # noqa: E731
        crit_desc = f"Focal(gamma={args.focal_gamma})"
    else:
        crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
        crit_desc = "CE" + (f"+LS({args.label_smoothing})" if args.label_smoothing > 0 else "")

    steps_per_epoch = min(len(train_dl), args.max_steps) if args.max_steps else len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * 0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: step / max(1, warmup) if step < warmup else max(0.0, (total_steps - step) / max(1, total_steps - warmup)))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))   # AMP：显存约降 40%，训练更快
    fgm = FGM(model, eps=args.fgm_eps) if args.fgm else None

    print(f"[cfg] backbone=roberta({MODEL_DIR.name}) pool={args.pool} loss={crit_desc} "
          f"class_weight={'off' if weight is None else 'on'} "
          f"fgm={'on(eps=%s)' % args.fgm_eps if fgm else 'off'} epochs={args.epochs} "
          f"steps/epoch={steps_per_epoch} eval_n={len(val)} save={'off' if args.no_save else 'on'}", flush=True)

    best, log = 0.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, n_step = time.time(), 0.0, 0
        for i, batch in enumerate(train_dl):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                loss = crit(model(ids, mask), y)
            scaler.scale(loss).backward()

            if fgm is not None:
                fgm.attack()
                with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    loss_adv = crit(model(ids, mask), y)
                scaler.scale(loss_adv).backward()
                fgm.restore()

            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            n_step += 1
            if i % 200 == 0:
                print(f"  epoch {epoch} step {i}/{steps_per_epoch} loss={loss.item():.4f}", flush=True)
            if args.max_steps and n_step >= args.max_steps:
                break

        res = evaluate(model, val_dl, device)
        log.append({"epoch": epoch, "train_loss": round(running / max(1, n_step), 4),
                    "val_acc": res["acc"], "val_macro_f1": res["macro_f1"],
                    "elapsed_s": round(time.time() - t0, 1)})
        print(f"epoch {epoch}: val_acc={res['acc']} val_f1={res['macro_f1']}", flush=True)

        if res["macro_f1"] > best and not args.no_save:
            best = res["macro_f1"]
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "pool": args.pool,
                        "max_length": args.max_length, "model_dir": str(MODEL_DIR),
                        "val_acc": res["acc"], "val_macro_f1": res["macro_f1"], "args": vars(args)},
                       OUT_DIR / "roberta_ckpt.pt")

    if not args.no_save:
        (OUT_DIR / "roberta_train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"最佳 val Macro-F1 = {best} → {OUT_DIR / 'roberta_ckpt.pt'}")
    else:
        print("--no_save：本次不落盘（消融模式），ckpt 与日志都没写")


if __name__ == "__main__":
    main()
