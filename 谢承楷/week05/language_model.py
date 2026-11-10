"""
字符级语言模型训练脚本。
结构：Embedding + 正弦位置编码 + 因果掩码 TransformerEncoder，逐字预测下一个字，打印 Loss 与 PPL。

用法:
    python language_model.py
        默认配置从零训练，结束覆盖 best_model.pt
    python language_model.py --save best_model_v2.pt
        换个文件名保存，保留旧权重
    python language_model.py --num_layers 4 --embed_dim 256 --hidden_dim 512 --lr 5e-4 --save best_model_v2.pt
        更大的模型（参数量约 5.2M，比默认配置更能记住语料）
    python language_model.py --resume best_model_v2.pt --epochs 20 --lr 5e-4 --save best_model_v2.pt
        从已有 checkpoint 继续训练，模型结构与字表按 checkpoint 恢复，--lr 必须与之前一致

参数说明（括号内为默认值）:
    --epochs      训练轮数（20）
    --stride      切块间隔（64，不重叠；改成 1 会把样本数放大约 64 倍，一轮要跑很久）
    --seq_len     每条样本的字符数（64），需与续训时保持一致
    --batch_size  批大小（128）
    --embed_dim   隐藏维度（128），必须能被 4 整除，因为头数固定为 4
    --hidden_dim  FFN 隐层的一半（256），实际 dim_feedforward = 2 × hidden_dim
    --num_layers  Transformer 层数（8）
    --dropout     dropout（0.3）
    --lr          学习率（1e-3），续训时不会自动沿用 checkpoint 里的值，要显式传入
    --max_vocab   字表上限（4000，0 = 不限）
    --val_ratio   验证集比例（0.05）
    --seed        随机种子（42），同时决定训练/验证集的划分
    --corpus      语料文件匹配模式（*.txt）
    --save        checkpoint 输出路径（best_model.pt）
    --resume      续训起点 checkpoint（空 = 从零训练）

checkpoint 保存内容：model_state / char2idx / idx2char / args；
只在验证 PPL 创新低时保存，因此里面始终是最优权重，不含优化器状态。
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    """
    构建字表：按出现频率取前 max_vocab 个字（None = 全部字符）。
    语料里的生僻字超出字表后会被 CharDataset 过滤掉 —— 字表越小模型越小，
    但覆盖不足会丢信息，这是词表大小的取舍。
    """
    from collections import Counter
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """
    字符级语言模型的训练样本：输入 x = 连续 seq_len 个字符，目标 y = 往后错一位。

    stride 控制切块间隔：
        stride=1  —— 每个字符位置都当一个样本（样本大量重叠，一期语料能切出上百万个样本）
        stride=seq_len —— 不重叠切块（标准做法，样本数 = 语料长度 / seq_len）
    语料一大，stride=1 会让一轮训练变得极慢，这不是模型慢，而是样本数被放大了 seq_len 倍。
    """

    def __init__(self, text, char2idx, seq_len, stride=1):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────

class LM(nn.Module):
    """
    字符级 Transformer 语言模型：
    Embedding + 正弦位置编码 → 因果掩码 TransformerEncoder → 全词表分类。
    model_type 参数仅用于记录配置，保留是为了与 pinyin_ime.py 的加载逻辑兼容。
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # 正弦位置编码：无需训练参数，persistent=False 不写入 checkpoint
        self.register_buffer("pos_enc", self._sinusoidal_pe(4096, embed_dim), persistent=False)
        # 注意 embed_dim 需能被 nhead 整除
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)

    @staticmethod
    def _sinusoidal_pe(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)   # (1, max_len, d_model)

    def forward(self, x):
        T = x.size(1)
        e = self.drop(self.embed(x) + self.pos_enc[:, :T])          # (B, T, D)
        # 因果掩码：每个位置只能看到自己和之前的位置（语言模型不能偷看未来）
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        out = self.encoder(e, mask=mask)                            # (B, T, D)
        logits = self.fc(self.drop(out))                            # (B, T, V)
        return logits


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="transformer", choices=["transformer"])
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=64, help="切块间隔：1=密集重叠，seq_len=不重叠")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--embed_dim",  type=int,   default=256)
    parser.add_argument("--hidden_dim", type=int,   default=512)
    parser.add_argument("--num_layers", type=int,   default=4)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_model.pt")
    parser.add_argument("--resume",     default="", help="从已有 checkpoint 继续训练，如 best_model_v2.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: {args.model.upper()}")

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    # 续训时字表和模型结构都取自 checkpoint，保证与已有权重严格对应
    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        for key in ("model", "embed_dim", "hidden_dim", "num_layers"):
            setattr(args, key, resume_ckpt["args"][key])
        char2idx, idx2char = resume_ckpt["char2idx"], resume_ckpt["idx2char"]
        print(f"从 {args.resume} 继续训练（结构按 checkpoint 恢复）")
    else:
        char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len, args.stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 模型
    model = LM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        model_type=args.model,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state"], strict=True)
        print(f"已加载权重: {args.resume}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")
    if resume_ckpt is not None:
        # 用当前权重的验证 PPL 作为起点，避免续训第一轮用更差的权重覆盖原 checkpoint
        _, best_val_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"续训起点验证 PPL: {best_val_ppl:.2f}")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()
