"""
字符级语言模型训练脚本 —— Transformer 版（带因果 mask，GPT 式 decoder-only）。
由 language_model.py 改造而来：把 LSTM 换成带因果 mask(causal mask) 的 Transformer，
数据流程与训练流程保持一致，便于对比。

与 language_model_transformer_nomask.py（无 mask 版）的唯一区别在 forward 里的 mask：
    有 mask —— 位置 i 只能看到 <= i 的 token，训练目标与"逐字生成"的推理场景一致，是标准语言模型。
    无 mask —— 位置 i 还能"偷看"未来的 token，训练 loss 会虚假地低，无法用于从左到右生成。

用法:
    python language_model_transformer_masked.py --epochs 20
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

class PositionalEncoding(nn.Module):
    """
    正弦位置编码：RNN 靠逐步递归天然带有顺序信息，而 Transformer 的自注意力
    对输入顺序不敏感（不看位置就是无序词袋），必须显式把"位置"加进输入。
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))        # (1, max_len, d_model)，随模型搬设备
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                  # x: (B, T, D)
        return self.drop(x + self.pe[:, : x.size(1)])


class LM(nn.Module):
    """embed -> (+位置编码) -> TransformerEncoder(带因果 mask) -> Linear 投回词表。
    hidden_dim 在这里充当 FFN 宽度(dim_feedforward)。"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, nhead, dropout):
        super().__init__()
        assert embed_dim % nhead == 0, "embed_dim 必须能被 nhead 整除"
        self.d_model = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = PositionalEncoding(embed_dim, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=hidden_dim,
            dropout=dropout, batch_first=True,
            norm_first=True,   # pre-LN 结构，小模型训练更稳
        )
        # norm_first 时需要在编码器末尾补一个 LayerNorm；关掉 nested-tensor 快速路径以免报警告
        self.encoder = nn.TransformerEncoder(
            layer, num_layers, norm=nn.LayerNorm(embed_dim), enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):                                  # x: (B, T)
        T = x.size(1)
        # 因果 mask：上三角（不含对角线）置 -inf，softmax 后这些位置的注意力权重为 0，
        # 即位置 i 只允许注意 <= i 的位置 —— 看不到"未来"，与逐字生成时只见前文一致。
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        e = self.pos(self.embed(x) * math.sqrt(self.d_model))
        out = self.encoder(e, mask=mask)                   # (B, T, D)
        logits = self.fc(self.drop(out))                   # (B, T, V)
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


def probe_future_use(model, loader, criterion, device, vocab_size):
    """
    "未来依赖"诊断：把每条序列的后半段位置替换成随机 token，只统计前半段位置的预测 loss。
        有因果 mask：前半段位置在数学上根本看不到后半段 → 两次 loss 应严格相等。
        无 mask：前半段的预测可以"借力"后半段的内容 → 未来一变，预测就变，两次 loss 会出现差异。
    返回 (干净输入的 loss, 后半段被破坏后的 loss)。
    """
    model.eval()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    T = x.size(1)
    half = T // 2
    with torch.no_grad():
        logits = model(x)
        loss_clean = criterion(logits[:, :half].reshape(-1, vocab_size),
                               y[:, :half].reshape(-1))
        x_corrupt = x.clone()
        x_corrupt[:, half:] = torch.randint(0, vocab_size, (x.size(0), T - half), device=device)
        logits = model(x_corrupt)
        loss_corrupt = criterion(logits[:, :half].reshape(-1, vocab_size),
                                 y[:, :half].reshape(-1))
    return loss_clean.item(), loss_corrupt.item()


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=1, help="切块间隔：1=密集重叠，seq_len=不重叠")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--embed_dim",  type=int,   default=128)
    parser.add_argument("--hidden_dim", type=int,   default=256, help="Transformer 中用作 FFN 宽度")
    parser.add_argument("--nhead",      type=int,   default=4,   help="注意力头数，需整除 embed_dim")
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=3e-4, help="Transformer 对学习率比 LSTM 敏感，取小一些")
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_transformer_masked.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: Transformer(causal-mask)")

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

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
        nhead=args.nhead,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")

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

    # —— "未来依赖"诊断：验证模型 ——
    with torch.no_grad():
        counts = torch.bincount(val_ds.data, minlength=vocab_size).double()
    p = counts[counts > 0] / counts.sum()
    uni_entropy = -(p * p.log()).sum().item()
    loss_clean, loss_corrupt = probe_future_use(model, val_loader, criterion, device, vocab_size)
    print(f"\n[诊断] ln(V) = {math.log(vocab_size):.4f}（均匀瞎猜）；字符一元熵 = {uni_entropy:.4f}（完全无上下文时的理论最优）")
    print(f"[诊断] 把序列后半段换成随机 token 后，前半段位置的预测 loss: {loss_clean:.4f} -> {loss_corrupt:.4f}（变化 {loss_corrupt - loss_clean:+.4f}）")
    print("[诊断] 两次 loss 完全相同 ⇒ 预测与未来内容严格无关（因果 mask 把未来挡死），训练 PPL 真实可信。")


if __name__ == "__main__":
    main()
