"""
在预训练字符级语言模型（best_model.pt）上做 SFT 监督微调，然后进入命令行对话。
训练数据为 sft_train.jsonl，模板：用户：{instruction}\n助手：{output}
用法:
    python sft.py                          # 从 --base 微调 + 对话
    python sft.py --base best_model_v2.pt  # 用新的预训练底座微调
    python sft.py --resume sft_model.pt    # 在已有 SFT 权重上继续训练 N 轮
    python sft.py --skip_train             # 直接加载 sft_model.pt 对话
"""

import os
import json
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from language_model import LM

PROMPT_TEMPLATE = "用户：{instruction}\n助手："


# ─────────────────────────── 数据 ───────────────────────────

class SFTDataset(Dataset):
    """
    一条样本 = 用户提问 + 助手回答。
    只对“助手：”之后的内容计算 loss，提问部分作为条件输入不参与监督。
    """

    def __init__(self, path, char2idx, max_len):
        self.max_len = max_len
        self.samples = []
        dropped = total = 0

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                prompt = PROMPT_TEMPLATE.format(instruction=item["instruction"])
                answer = item["output"] + "\n"      # 用换行作为回答结束标记

                ids, mask = [], []
                for text, flag in ((prompt, 0), (answer, 1)):
                    for ch in text:
                        total += 1
                        if ch not in char2idx:      # 字表外的字符直接丢弃（与预训练一致）
                            dropped += 1
                            continue
                        ids.append(char2idx[ch])
                        mask.append(flag)

                ids, mask = ids[:max_len], mask[:max_len]
                if len(ids) < 2 or sum(mask) == 0:
                    continue
                self.samples.append((torch.tensor(ids), torch.tensor(mask)))

        self.drop_ratio = dropped / max(total, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, mask = self.samples[idx]
        x = ids[:-1]                                # 输入：错一位
        y = ids[1:]                                 # 目标：错一位
        m = mask[1:]                                # 目标位置的监督标记
        y = torch.where(m.bool(), y, torch.full_like(y, -100))
        return x, y


def collate_fn(batch):
    max_len = max(len(x) for x, _ in batch)
    xs, ys = [], []
    for x, y in batch:
        pad = max_len - len(x)
        xs.append(torch.cat([x, torch.zeros(pad, dtype=torch.long)]))
        ys.append(torch.cat([y, torch.full((pad,), -100, dtype=torch.long)]))
    return torch.stack(xs), torch.stack(ys)


# ─────────────────────────── 模型 ───────────────────────────

def build_model(ckpt, device, dropout=None):
    """按 checkpoint 里记录的配置重建结构，strict=True 严格加载权重。"""
    a = ckpt["args"]
    model = LM(
        vocab_size=len(ckpt["char2idx"]),
        embed_dim=a["embed_dim"],
        hidden_dim=a["hidden_dim"],
        num_layers=a["num_layers"],
        model_type=a.get("model", "transformer"),
        dropout=a["dropout"] if dropout is None else dropout,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model.to(device)


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            n = (y != -100).sum().item()
            if n == 0:
                continue
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item() * n
            total_tokens += n
    return total_loss / max(total_tokens, 1)


def finetune(args, base_ckpt, device, resume=False):
    char2idx, idx2char = base_ckpt["char2idx"], base_ckpt["idx2char"]
    model = build_model(base_ckpt, device, dropout=args.dropout)

    train_ds = SFTDataset(args.train_data, char2idx, args.seq_len)
    val_ds = SFTDataset(args.val_data, char2idx, args.seq_len)
    print(f"训练样本: {len(train_ds)}  验证样本: {len(val_ds)}  "
          f"字表外字符丢弃比例: {train_ds.drop_ratio:.2%}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_ckpt = None
    if resume:
        # 以当前权重的验证 loss 为起点，避免续训第一轮用更差的权重覆盖原 checkpoint
        best_val = evaluate(model, val_loader, criterion, device)
        best_ckpt = base_ckpt
        print(f"续训起点验证 Loss: {best_val:.4f}")
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}")
    print("-" * 32)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = (y != -100).sum().item()
            total_loss += loss.item() * n
            total_tokens += n

        tr_loss = total_loss / max(total_tokens, 1)
        va_loss = evaluate(model, val_loader, criterion, device)

        marker = "  *" if va_loss < best_val else ""
        if va_loss < best_val:
            best_val = va_loss
            best_ckpt = {
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": {**base_ckpt["args"], **vars(args)},
            }
            torch.save(best_ckpt, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {va_loss:>10.4f}"
              f"   ppl={math.exp(va_loss):.2f}{marker}")

    if best_ckpt is base_ckpt:
        print(f"\nSFT 完成。没有优于起点的轮次，未更新 {args.save}")
    else:
        print(f"\nSFT 完成。最佳验证 Loss: {best_val:.4f}  已保存至 {args.save}")
    return best_ckpt


# ─────────────────────────── 生成 / 对话 ───────────────────────────

@torch.no_grad()
def generate(model, char2idx, idx2char, prompt, args, device):
    """自回归逐字生成，遇到换行（回答结束标记）停止。"""
    ids = [char2idx[ch] for ch in prompt if ch in char2idx]
    if not ids:
        return ""

    max_ctx = args.seq_len
    out_chars = []

    model.eval()
    for _ in range(args.max_new_tokens):
        x = torch.tensor([ids[-max_ctx:]], dtype=torch.long, device=device)
        logits = model(x)[0, -1].float() / max(args.temperature, 1e-5)

        if args.top_k and args.top_k < logits.numel():
            kth = torch.topk(logits, args.top_k).values[-1]
            logits[logits < kth] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()

        if idx2char[next_id] == "\n" and out_chars:
            break
        ids.append(next_id)
        out_chars.append(idx2char[next_id])

    return "".join(out_chars)


def chat(model, char2idx, idx2char, args, device):
    print("\n进入对话模式（输入 exit 退出；字表外的字符会被忽略）")
    print("-" * 56)
    while True:
        try:
            question = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question in ("exit", "quit", "q"):
            break
        if not question:
            continue

        prompt = PROMPT_TEMPLATE.format(instruction=question)
        answer = generate(model, char2idx, idx2char, prompt, args, device)
        print(f"助手：{answer}\n")


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",       default="best_model.pt")
    parser.add_argument("--train_data", default="sft_train.jsonl")
    parser.add_argument("--val_data",   default="sft_val.jsonl")
    parser.add_argument("--save",       default="sft_model.pt")
    parser.add_argument("--resume",     default="", help="从已有 SFT checkpoint 继续训练，如 sft_model.pt")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--seq_len",    type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=8e-4)
    parser.add_argument("--dropout",    type=float, default=0.1,
                        help="SFT 阶段 dropout：数据量小时比预训练的 0.3 更容易拟合")
    parser.add_argument("--temperature",     type=float, default=0.7)
    parser.add_argument("--top_k",           type=int,   default=20)
    parser.add_argument("--max_new_tokens",  type=int,   default=100)
    parser.add_argument("--skip_train", action="store_true", help="跳过训练，直接加载 --save 对话")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print(f"device: {device}")

    if args.skip_train:
        if not os.path.exists(args.save):
            raise FileNotFoundError(f"未找到 {args.save}，请先运行 python sft.py 完成微调。")
        ckpt = torch.load(args.save, map_location="cpu", weights_only=False)
        print(f"加载 SFT 权重: {args.save}")
    elif args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"未找到 {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        print(f"从 SFT checkpoint 继续训练: {args.resume}")
        ckpt = finetune(args, resume_ckpt, device, resume=True)
    else:
        base_ckpt = torch.load(args.base, map_location="cpu", weights_only=False)
        print(f"加载预训练权重: {args.base}（词表 {len(base_ckpt['char2idx'])}）")
        ckpt = finetune(args, base_ckpt, device)

    model = build_model(ckpt, device)
    chat(model, ckpt["char2idx"], ckpt["idx2char"], args, device)


if __name__ == "__main__":
    main()
