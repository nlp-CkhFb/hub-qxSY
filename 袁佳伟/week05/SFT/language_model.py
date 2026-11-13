"""
字符级语言模型训练脚本，支持 RNN / LSTM 切换，含 PPL 计算。
用法:
    python language_model.py --model lstm --epochs 20
    python language_model.py --model rnn  --epochs 20
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据 ───────────────────────────

SPECIAL_TOKENS = {"<pad>": 0, "<eos>": 1}


def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    """
    构建字符表：内置 <pad> 和 <eos>，保证生成时能明确停止，避免无限循环输出符号。
    """
    from collections import Counter
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    chars = [c for c in chars if c not in SPECIAL_TOKENS]

    char2idx = dict(SPECIAL_TOKENS)
    for c in chars:
        if c not in char2idx:
            char2idx[c] = len(char2idx)
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """
    字符级语言模型的训练样本：输入 x = 连续 seq_len 个字符，目标 y = 往后错一位。

    stride 控制切块间隔：
        stride=1  —— 每个字符位置都当一个样本（样本大量重叠，一期语料能切出上百万个样本）
        stride=seq_len —— 不重叠切块（标准做法，样本数 = 语料长度 / seq_len）
    语料一大，stride=1 会让一轮训练 变得极慢，这不是模型慢，而是样本数被放大了 seq_len 倍。
    """

    def __init__(self, text, char2idx, seq_len, stride=1):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        ids = [char2idx[c] for c in text if c in char2idx]
        eos_id = char2idx.get("<eos>")
        if eos_id is not None:
            ids = ids + [eos_id]
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
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout):
        super().__init__()
        # self.embed = nn.Embedding(vocab_size, embed_dim)
        self.tok = nn.Embedding(vocab_size, embed_dim)      # 字符 id → d 维向量，训练中是会学的

        # 这里使用 Transformer 替代 RNN/LSTM
        # norm_first=True 表示先 LayerNorm 再 MultiheadAttention，效果更好(Pre-LN)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=hidden_dim, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        # TransformerEncoder 由多个 encoder_layer 堆叠而成，num_layers 控制堆叠层数, norm=nn.LayerNorm(embed_dim) 用于对输出进行归一化
        # 把同一个 layer 堆 N 次 + 最后再来一个 LayerNorm
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(embed_dim), enable_nested_tensor=False)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.tok.weight   # 输入/输出 embedding 共享
        nn.init.normal_(self.tok.weight, std=0.02)
        self.drop = nn.Dropout(dropout)
        # self.fc = nn.Linear(embed_dim, vocab_size)
        

    def forward(self, x):
         # idx: (batch, T) 的字符 id；输出: (batch, T, vocab_size) 的未归一化 logits
        x = self.tok(x)                     # (B, T, d) 查表
        x = self.drop(x)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)   # 下三角 0、右上 -inf
        x = self.transformer_encoder(x, mask=mask)
        # 输出层直接给 logits，不在模型里做 softmax：
        # F.cross_entropy 内部自带 log_softmax，归一化两次会让数值变差、梯度变小（常见坑）。
        # 只有 generate 采样时才显式 softmax，因为那里确实要概率分布。
        return self.head(x)


        # out = self.transformer_encoder(e)
        # logits = self.fc(self.drop(out))   # (B, T, V)
        # return logits


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    #model.train(train)
    total_loss = 0.0
    total_tokens = 0
    if train:
        model.train()
    else:
        model.eval()

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
    parser.add_argument("--model",      default="transformer", choices=["rnn", "lstm", "transformer"])
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=64, help="切块间隔：1=密集重叠，seq_len=不重叠")
    parser.add_argument("--batch_size", type=int,   default=128)
    # 默认这套超参（d=192 / 4 层 / 4 头）参数量约 2.66M，正好和本周 LSTM 基线的 2.46M 对齐在 ±10% 内 ——
    # 作业要求「先对齐参数量再比 PPL」，否则比出来的不是结构差异而是模型大小的差异
    parser.add_argument("--embed_dim",  type=int,   default=192)
    parser.add_argument("--hidden_dim", type=int,   default=4*192)  # feedforward_dim = 4 * d
    parser.add_argument("--num_layers", type=int,   default=4)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_model.pt")
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
    print(f"模型: TransformerEncoderLayer × {args.num_layers}（d={args.embed_dim} "
          f"FFN={4*args.embed_dim} Pre-LN）  "f"参数量 {total_params/1e6:.2f}M")

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

# 用生成的模型进行文本生成的示例
def generate_text(model, start_text, char2idx, idx2char, length=100, temperature=1.0, top_k=20):
    model.eval()
    eos_id = char2idx.get("<eos>")
    input_ids = torch.tensor([char2idx[c] for c in start_text if c in char2idx], dtype=torch.long).unsqueeze(0)
    input_ids = input_ids.to(next(model.parameters()).device)

    generated = start_text

    with torch.no_grad():
        for _ in range(length):
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :] / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                top_v, top_i = torch.topk(next_token_logits, k=min(top_k, next_token_logits.size(-1)))
                probs = torch.softmax(top_v, dim=-1)
                sample_idx = torch.multinomial(probs, num_samples=1).item()
                next_token_id = int(top_i[sample_idx].item())
            else:
                probabilities = torch.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probabilities, num_samples=1).item()

            if eos_id is not None and next_token_id == eos_id:
                break

            next_char = idx2char.get(next_token_id, "")
            if not next_char:
                break

            generated += next_char
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=input_ids.device)], dim=1)

    return generated


if __name__ == "__main__":
    main()
    # # 加载模型并生成文本示例
    # checkpoint = torch.load("best_model.pt", map_location=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    # model_args = checkpoint["args"]
    # char2idx = checkpoint["char2idx"]          # 4000
    # idx2char = checkpoint["idx2char"]
    # model = LM(
    #     vocab_size=len(char2idx),
    #     embed_dim=model_args["embed_dim"],      # 128
    #     hidden_dim=model_args["hidden_dim"],    # 512
    #     num_layers=model_args["num_layers"],    # 2
    #     model_type=model_args["model"],         # "transformer"
    #     dropout=model_args["dropout"],          # 0.3
    # )
    # print(model_args)
    # print(f"加载模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    # model.load_state_dict(checkpoint["model_state"])
    # model.eval()
    # generated_text = generate_text(model, "几年前经过各方的共同努力，中东和谈", char2idx, idx2char, length=50, temperature=1.0)
    # print(generated_text)

    # 在已有模型上进行sft训练
   


