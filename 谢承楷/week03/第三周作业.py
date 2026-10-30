# -*- coding: utf-8 -*-
"""
train_langid_cls.py —— 多语种「语种识别」分类器（综合实战脚本 · 课堂 12 语种版）

任务：一句话 → 它是什么语言（12 分类）
模型：Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear → Softmax
损失：CrossEntropyLoss      优化器：Adam
数据：data/langid/v12（课堂 12 语种）
运行：CPU 即可，单线程实测约 35 秒。

    python train_langid_cls.py                       # 课堂 12 语种
    python train_langid_cls.py --save langid_rnn.pt  # 存档给 eval_langid.py 评测

【本课想让学生看到的四件事】
  1. 同一份代码、同一张字符表，换数据就能处理 12 种语言文字
  2. 字符级不需要分词器：中文按字、英文按字母、印地语按字母+元音标记，一视同仁
  3. 组件是一个个加进去的：去掉 LayerNorm / Dropout / RNN 都能现场对比
  4. 错得最有意思的是 es ↔ pt（同一套拉丁字母），不是"模型没学会"

【这份脚本是「12 类版」—— 语种列表和字符表都是写死的】
  · LABELS 直接列出了这 12 个语种，模型的输出维度 = len(LABELS)
  · 字符表不从训练集现算，而是读 data/langid/vocab/vocab_v12.txt（课堂用的那张表）
  要换数据集/加语种时，这两处都得跟着改 —— 下一页的作业就是这个。

【字符表是怎么来的】见 build_char_vocab()：
  把训练集每句话按字拆开 → 统计每个字的出现次数 → 按次数从高到低编号。
  已经生成好的表在 data/langid/vocab/vocab_v12.txt，每行「字符 \\t id \\t 出现次数」。
"""

from typing import Any


import os
import csv
import time
import random
import argparse
import unicodedata
from collections import Counter, OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────────────────
SEED = 42             # 随机种子，固定后每次跑出来的结果一致（方便课堂复现对比）
MAX_LEN = 40          # 中文中位长度约 10 字符、德语约 39，统一截到 40
EMBED_DIM = 64        # 字符 Embedding 向量维度（每个字符变成 64 维向量）
HIDDEN_DIM = 64       # RNN 隐状态维度 → 也是 MaxPool 后特征向量的维度
DROPOUT = 0.3         # Dropout 层"丢弃率"，训练时随机把 30% 的神经元置 0 防过拟合
BATCH_SIZE = 64       # DataLoader 每次喂多少条样本给模型（训练时 shuffle）
LR = 1e-3             # Adam 优化器学习率，控制每步参数更新的幅度
EPOCHS = 8            # 训练轮数：把整个训练集完整过 8 遍
NUM_THREADS = 1       # 小模型单线程比 16 线程快 2.7 倍（实测），别改成 16

HERE = os.path.dirname(os.path.abspath(__file__))  # 本脚本所在的绝对目录
DATA_DIR = os.path.join(HERE, "data", "langid", "v12")          # 训练/验证/测试 CSV 所在目录
VOCAB_PATH = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")  # 字符表 txt 文件路径

# ─── 运行模式开关（改这里就行，不用敲命令行参数）───────────────────────
# True  → 运行只生成 vocab_v12.txt，生成完就退出（构建字符表模式）
# False → 正常训练
# 命令行加 --build-vocab 也能临时触发，效果一样
RUN_BUILD_VOCAB = False

# 课堂用的 12 个语种：模型的输出就是这 12 个格子
LABELS = ["ar", "en", "es", "hi", "ja", "ko", "pt", "ru", "th", "tr", "vi", "zh"]



# ─── 1. 文本归一化（必须和建数据时一致）────────────────────────────────
def normalize(text):
    """
    NFKC 归一化 + 转小写 + 只保留「字母(L)」与「组合符号(M)」。

    为什么保留 M：印地语、泰语的元音是组合标记（कि 里的 ि），
    只留字母会把印地语砍成裸辅音，模型就没法区分。
    """
    t = unicodedata.normalize("NFKC", text).lower()   # NFKC 把"等价"的 Unicode 字符合并（如全角→半角），再转小写
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))   # 只保留字母(L)和组合符号(M)


# ─── 2. 字符表构建 ─────────────────────────────────────────────────────
def build_char_vocab(sentences, min_freq=1):
    """
    字符表 = 训练集里所有出现过的字符，每个字符发一个整数编号。

    第 0 号 <PAD>：补齐用（Embedding 里 padding_idx=0，向量恒为 0、不参与训练）
    第 1 号 <UNK>：没见过的字符（字符级下几乎用不到）
    其余按出现次数从高到低编号，次数相同按字符编码排序（保证可复现）

    多语种的关键点：这张表不需要按语种分开建。汉字、拉丁字母、西里尔字母、
    阿拉伯字母、天城文、泰文全都排在同一张表里 —— 这就是"字符级天然多语种"。

    课堂主流程没有直接调用它（字符表是从 vocab_v12.txt 读的）；
    一旦换了训练集或加了语种，就要用它重新生成一张表，
    否则新语种的字符会全部落到 1 号 <UNK>，模型等于睁眼瞎。
    """
    freq = Counter[Any]()   # 计数器：{字符: 出现次数}
    for s in sentences:   # s = 一句原始训练文本
        # normalize(text=一句原始字符串) → 归一化后的字符序列（用于统计字符频次）
        freq.update(normalize(s))
    # 按 (出现次数降序, 字符编码升序) 排序；min_freq 以下的字符扔掉
    chars = sorted(((c, n) for c, n in freq.items() if n >= min_freq),
                   key=lambda kv: (-kv[1], kv[0]))
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])   # 字符表：先放两个特殊标记
    for ch, _ in chars:   # ch = 字符本身，_ = 出现次数（这里不用）
        vocab[ch] = len(vocab)   # 按当前已有长度编号，保证从 2 开始递增
    return vocab


def load_vocab_file(path):
    """
    读已经生成好的字符表（每行「字符 \\t id \\t 出现次数」，以 # 开头的是注释）。

    课堂用的就是 data/langid/vocab/vocab_v12.txt，它是从 v12 训练集统计出来的。
    → 换了训练集、加了语种，这张表就得重新生成（用上面的 build_char_vocab）。
    """
    vocab = OrderedDict()   # {字符: id}，保持插入顺序
    with open(path, encoding="utf-8") as f:   # path = 字符表 txt 文件的绝对路径
        for line in f:
            if line.startswith("#") or not line.strip():   # 跳过注释行和空行
                continue
            ch, idx, _freq = line.rstrip("\n").split("\t")   # 每行格式：字符 \t id \t 频次
            vocab[ch] = int(idx)   # 把 id 存成 int（文件里读出来是 str）
    return vocab

# this string starts with the specified prefix
def encode(text, vocab, maxlen=MAX_LEN):
    """
    将一段文本编码为固定长度的字符 ID 序列（长度 = maxlen）。

    步骤：
      1. normalize 归一化后逐字查表，不在字符表里的字符映射到 1 号 <UNK>
      2. 超过 maxlen 的尾部直接截断 → 信息丢弃（见下方说明）
      3. 不足 maxlen 的尾部用 0 号 <PAD> 补齐

    ⚠️ 截断的取舍（为什么尾部可以扔）：
      这个任务是「语种识别」——模型要抓的是文字外观特征（汉字、拉丁字母、
      西里尔字母……），前 40 个字符已经足够暴露语种特征，尾部对判断语种几乎
      没有贡献。如果要利用全文信息，需要换成 BiRNN/Transformer + 变长输入，
      但那会让课堂小模型变重，这里用简单截断换速度。

    返回：长度恒为 maxlen 的 Python list，元素为 int。
    """
    # 归一化 → 逐字查 ID（未命中则取 <UNK>=1）→ 截断到 maxlen（尾部丢弃）
    # normalize(text=原始文本) → 归一化字符序列；vocab.get(ch, 1) 查字符 ID，没见过的给 1 号 <UNK>
    ids = [vocab.get(ch, 1) for ch in normalize(text)][:maxlen]
    # 尾部补 <PAD>=0，使所有样本长度一致（RNN 需要 batch 内等长）
    ids += [0] * (maxlen - len(ids))
    return ids


# ─── 3. 读数据 ─────────────────────────────────────────────────────────
def load_split(data_dir, name):
    """
    读取 data_dir 下的 name.csv 文件（name ∈ {train, dev, test}）。

    CSV 要求：至少有 text 和 label 两列。
    返回：[(text, label), ...] 的 list，每个元素是 (字符串文本, 语种标签字符串)。
    """
    path = os.path.join(data_dir, f"{name}.csv")   # 拼出完整 CSV 路径，如 data/langid/v12/train.csv
    rows = []   # 收集所有 (text, label) 元组
    with open(path, encoding="utf-8") as f:
        # csv.DictReader 会自动把首行当表头，每行变成一个 dict
        for r in csv.DictReader(f):   # r = {"text": "某句话", "label": "en"} 这样的 dict
            rows.append((r["text"], r["label"]))   # 取出 text 和 label 两列
    return rows


class TextDataset(Dataset):
    """
    PyTorch Dataset 实现：把 (text, label) 原始数据预处理成模型可以直接吃的张量。

    X[i] → 一条编码后的字符 ID 序列（shape = [MAX_LEN]，dtype=long）
    y[i] → 一个整数类别标签（0 ~ num_classes-1）
    """

    def __init__(self, rows, vocab, label2id):
        """
        rows:      [(text, label_str), ...] —— 原始字符串数据
        vocab:     字符表 dict，{字符: id}
        label2id:  标签到整数的映射 dict，如 {"ar": 0, "en": 1, ...}
        """
        # 一次性把所有文本编码完 —— 小数据集放内存没问题，避免 __getitem__ 里重复计算
        # encode(text=字符串, vocab=字符表, maxlen=MAX_LEN) → 固定长度的 ID list
        self.X = [encode(t, vocab) for t, _ in rows]   # t=每条文本字符串, vocab=字符表
        # 把字符串标签转成整数类别 ID
        self.y = [label2id[l] for _, l in rows]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        # DataLoader 取样本时会调用这里；返回 (字符ID张量, 类别整数)
        return torch.tensor(self.X[i], dtype=torch.long), self.y[i]


# ─── 4. 模型 ───────────────────────────────────────────────────────────
class LangIDClassifier(nn.Module):
    """
    Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear

    MaxPool 前面那两行 mask 是为了让补齐位不参与池化：
    补齐位喂进去的是全 0 向量，RNN 照样会往前走，池化时就会把
    "读过一串 0" 的状态也算进来。masked_fill 成 -inf 后取最大值就干净了。
    """

    def __init__(self, vocab_size, num_classes,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        """
        vocab_size:  字符表大小（=len(vocab)），决定 Embedding 矩阵行数
        num_classes: 输出类别数（=len(LABELS)=8），决定最后 Linear 层输出维度
        embed_dim:   Embedding 输出维度（默认 64）
        hidden_dim:  RNN 隐状态维度 / Linear 输入维度（默认 64）
        dropout:     Dropout 概率（默认 0.3）
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, L) 字符 ID 序列（B=batch size, L=MAX_LEN=40）
        mask = (x != 0).unsqueeze(-1)              # (B, L, 1)  补齐位(=0) 为 False，其余为 True
        h, _ = self.rnn(self.embedding(x))         # (B, L, H)  RNN 输出每个位置的隐状态；_ 是最后时刻隐状态，不用
        h = h.masked_fill(~mask, -1e9)             # 把补齐位置的隐状态填成 -∞，这样 max 不会选到它们
        pooled = h.max(dim=1)[0]                   # (B, H)  在 L 维度上取 max → 每句一个 HIDDEN_DIM 特征向量
        pooled = self.dropout(self.norm(pooled))   # LayerNorm 把特征规整 → Dropout 随机丢一些防过拟合
        return self.fc(pooled)                     # (B, C)  全连接层把 H 维投到 C=12 类；输出未过 softmax 的 logits


# ─── 5. 评测 ───────────────────────────────────────────────────────────
def predict(model, loader):
    """
    用训练好的模型对 DataLoader 里的所有样本做推理。

    返回：(preds, golds) —— 两个 list，分别是模型预测的类别 ID 和真实类别 ID。
    model.eval() 会关闭 Dropout 等训练专属层；torch.no_grad() 关闭梯度计算省显存/时间。
    """
    model.eval()   # 切换到推理模式 —— 关闭 Dropout、关闭 BatchNorm 的 batch 统计
    preds, golds = [], []   # preds=模型预测的类别 ID 列表；golds=真实类别 ID 列表
    with torch.no_grad():   # 关闭梯度追踪（推理不需要反向传播），省显存也更快
        for X, y in loader:   # X=(B, MAX_LEN) 字符 ID 张量；y=(B,) 真实类别整数
            # model(X) 输出 (B, C) logits；argmax(dim=1) 取每行最大值的下标 → 类别 ID
            preds += model(X).argmax(dim=1).tolist()   # .tolist() 把张量转成 Python list
            golds += y.tolist()
    return preds, golds


def macro_f1(preds, golds, num_classes):
    """每一类各自算 F1，再对类别取平均（少数类不会被大类淹掉）。"""
    f1s = []   # 收集每个类别的 F1 分数，最后对它求平均
    for c in range(num_classes):   # c = 当前要计算的类别 ID
        tp = sum(p == c and g == c for p, g in zip(preds, golds))   # 真阳性：预测=c 且 真实=c
        fp = sum(p == c and g != c for p, g in zip(preds, golds))   # 假阳性：预测=c 但 真实≠c
        fn = sum(p != c and g == c for p, g in zip(preds, golds))   # 假阴性：预测≠c 但 真实=c
        if tp + fp + fn == 0:   # 该类在测试集中完全没出现也没被预测到 → 跳过
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0   # 精确率：预测为 c 的里面有多少真的是 c
        rec = tp / (tp + fn) if tp + fn else 0.0     # 召回率：所有真 c 里被找到了多少
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)   # F1 = 2PR/(P+R)
    return sum(f1s) / len(f1s)   # macro-F1：对所有类别的 F1 简单平均


def confusion(preds, golds, labels):
    """
    构建混淆矩阵。mat[真实类][预测类] = 该组合的样本数。

    对角线 mat[i][i] 是预测正确的；mat[g][p] (g≠p) 是把真实 g 错判成 p 的数量。
    """
    n = len(labels)   # 类别总数（=12）
    mat = [[0] * n for _ in range(n)]   # 初始化 n×n 零矩阵；mat[真实类][预测类] = 计数
    for p, g in zip(preds, golds):   # p=预测类别ID，g=真实类别ID
        mat[g][p] += 1  # 行=真实标签，列=预测标签
    return mat


def print_confusion(mat, labels, top=3):
    """只打印错的格子，按错误数排序 —— 课堂上看 zh/ja 那一格就够了。"""
    # pairs 收集所有"错误组合"（真实→预测），g≠p 排除对角线正确的
    pairs = [(mat[g][p], labels[g], labels[p])   # (错误数, 真实语种代码, 预测语种代码)
             for g in range(len(labels)) for p in range(len(labels)) if g != p]
    pairs.sort(reverse=True)   # 按错误数从大到小排
    print("  最常混淆的组合（真实 → 预测）：")
    for cnt, g, p in pairs[:top]:   # 只看前 top 个（默认 3 个）
        if cnt == 0:
            break
        print(f"    {g} → {p}: {cnt} 条")


# ─── 6. 构建 v12 字符表 ────────────────────────────────────────────────
def build_v12_vocab():
    """
    从 v12/dev.csv 构建字符表，保存为 vocab_v12.txt（格式：每行「字符 \t id \t 出现次数」）。

    用法：python train_langid_cls.py --build-vocab

    ⚠️ 注意：通常 vocab 应从 train.csv 构建（训练集字符覆盖更全），
       这里按需求用 dev.csv。若后续发现新语种字符被 <UNK> 吞掉，
       改成从 train.csv 重建即可。
    """
    dev_path = os.path.join(HERE, "data", "langid", "v12", "dev.csv")
    out_dir = os.path.join(HERE, "data", "langid", "vocab")
    out_path = os.path.join(out_dir, "vocab_v12.txt")

    # 确保输出目录存在
    os.makedirs(out_dir, exist_ok=True)

    # 1. 读 v12/dev.csv，收集所有 text
    sentences = []
    with open(dev_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sentences.append(r["text"])
    print(f"读取 {dev_path}：{len(sentences)} 条")

    # 2. 用已有的 build_char_vocab 构建字符表
    vocab = build_char_vocab(sentences)
    print(f"  字符表大小：{len(vocab)}（含 <PAD>、<UNK>）")

    # 3. 重新统计每个字符的出现次数（build_char_vocab 内部 Counter 没返回）
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))

    # 4. 写入 vocab_v12.txt，格式：字符 \t id \t 出现次数
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        f.write("<PAD>\t0\t0\n")
        f.write("<UNK>\t1\t0\n")
        # 遍历 vocab 中 id >= 2 的字符，按 id 升序写入
        for ch, idx in vocab.items():
            if idx >= 2:
                f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")
    print(f"  ✓ 已保存：{out_path}")


# ─── 7. 训练 ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()   # 命令行参数解析器
    ap.add_argument("--data-dir", default=DATA_DIR)   # 数据目录（默认 data/langid/v12）
    ap.add_argument("--vocab", default=VOCAB_PATH,
                    help="字符表文件（默认用 vocab_v12.txt）")
    ap.add_argument("--epochs", type=int, default=EPOCHS)   # 训练轮数
    ap.add_argument("--per-class", type=int, default=0,
                    help="只用每类前 N 条训练，用来演示「数据少了会过拟合」")
    ap.add_argument("--threads", type=int, default=NUM_THREADS)   # PyTorch 线程数
    ap.add_argument("--save", default="", help="保存模型到该路径，供 eval_langid.py 使用")
    ap.add_argument("--build-vocab", action="store_true",
                    help="只从 v12/dev.csv 构建字符表 vocab_v12.txt，不训练")
    args = ap.parse_args()   # 把命令行参数解析成 args 对象，args.data_dir / args.vocab 等都在这

    # ── 构建字符表模式：文件开关 RUN_BUILD_VOCAB=True 或命令行加 --build-vocab 都能触发 ──
    if args.build_vocab or RUN_BUILD_VOCAB:
        build_v12_vocab()
        return

    torch.set_num_threads(args.threads)   # 限制 PyTorch CPU 线程数
    random.seed(SEED)                     # Python 随机数种子（影响 random.sample / shuffle）
    torch.manual_seed(SEED)               # PyTorch 随机种子（影响模型初始化、dropout 等）

    # load_split(data_dir=CSV所在目录, name=文件名前缀) → [(text, label_str), ...]
    train_rows = load_split(args.data_dir, "train")   # data_dir, "train" → 读 train.csv
    dev_rows   = load_split(args.data_dir, "dev")     # data_dir, "dev"   → 读 dev.csv
    test_rows  = load_split(args.data_dir, "test")    # data_dir, "test"  → 读 test.csv

    # 数据里的标签必须和 LABELS 对得上，否则模型的输出格子根本不够用
    seen = sorted({l for _, l in train_rows})   # 训练集里实际出现过的所有语种标签（去重+排序）
    extra = [l for l in seen if l not in LABELS]   # 在数据里出现、但不在预定义 LABELS 里的"多余"标签
    if extra:
        raise SystemExit(
            f"数据里出现了 LABELS 之外的标签：{' '.join(extra)}\n"
            f"  当前 LABELS（{len(LABELS)} 类）：{' '.join(LABELS)}\n"
            f"  模型输出维度 = len(LABELS)，多出来的语种没有对应的输出格子。"
        )
    label2id = {l: i for i, l in enumerate(LABELS)}   # 标签→整数映射，如 {"ar": 0, "en": 1, ...}
    num_classes = len(LABELS)   # 类别总数（=8），决定模型最后 Linear 层的输出维度

    if args.per_class:
        # 把训练集按语种分组 → 每组随机抽 min(args.per_class, 该类总数) 条
        # 用固定种子 Random(SEED) 保证每次跑出来抽的样本一样，便于课堂对比
        by_label = {}   # {语种标签: [该语种的所有文本]}，分组字典
        for t, l in train_rows:   # t=文本字符串, l=语种标签
            by_label.setdefault(l, []).append(t)   # setdefault: l 不存在就先放个空 list
        # 抽完后重新展平成 list
        train_rows = [(t, l) for l, ts in by_label.items()
                      for t in random.Random(SEED).sample(ts, min(args.per_class, len(ts)))]
        random.Random(SEED).shuffle(train_rows)   # 打乱，避免按语种聚在一起

    # load_vocab_file(path=字符表 txt 文件路径) → {字符: id} 的 OrderedDict
    vocab = load_vocab_file(args.vocab)   # path=args.vocab 默认指向 vocab_v12.txt
    print(f"数据目录：{args.data_dir}")
    print(f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条")
    print(f"  类别 {num_classes} 个：{' '.join(LABELS)}")
    print(f"  字符表 {len(vocab)} 个字符（来自 {os.path.basename(args.vocab)}）")
    print(f"  随机猜的准确率基准：{1/num_classes:.1%}")

    # TextDataset(rows=原始数据, vocab=字符表, label2id=标签→整数映射)
    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id),   # rows=训练集, vocab, label2id
                              batch_size=BATCH_SIZE, shuffle=True)
    dev_loader   = DataLoader(TextDataset(dev_rows, vocab, label2id),     # rows=验证集
                              batch_size=128)
    test_loader  = DataLoader(TextDataset(test_rows, vocab, label2id),   # rows=测试集
                              batch_size=128)

    # LangIDClassifier(vocab_size=字符总数, num_classes=输出类别数)
    model = LangIDClassifier(len(vocab), num_classes)   # vocab_size, num_classes=8
    print(f"  模型参数量：{sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.CrossEntropyLoss()   # 损失函数：内部做 log_softmax + NLL，传入 logits 和类别整数
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # Adam 优化器：自适应学习率，lr=1e-3

    t0 = time.time()   # 训练开始时间戳，后面算总耗时用
    for epoch in range(1, args.epochs + 1):   # epoch 从 1 编号到 EPOCHS
        model.train()    # 切换到训练模式 —— 开启 Dropout、激活 BatchNorm 等
        total = 0.0      # 累计这个 epoch 里所有 batch 的 loss，最后算平均 loss
        for X, y in train_loader:   # X=(B, MAX_LEN) 字符 ID 张量；y=(B,) 真实类别整数
            # 前向传播：喂入字符 ID 序列 → 输出 (B, C) logits
            logits = model(X)   # logits: (B, 8) 未过 softmax 的打分，每行对应一个样本对 8 个类别的分数
            # CrossEntropyLoss 内部会做 log_softmax，然后算 NLL；y 必须是类别整数
            loss = criterion(logits, y)   # loss: 当前 batch 的交叉熵损失（标量）
            optimizer.zero_grad()   # 清空上一步累积的梯度（PyTorch 默认累加梯度，不清就会叠加）
            loss.backward()         # 反向传播：计算所有可训练参数对 loss 的梯度，存在 .grad 属性里
            optimizer.step()        # 用梯度更新参数（Adam 的一步自适应更新）
            total += loss.item()    # .item() 把 0-d 张量转成 Python float，累加这个 batch 的 loss
        # 每个 epoch 结束后，在验证集上跑一遍，看当前模型的准确率
        # predict(model=当前模型, loader=DataLoader) → (preds列表, golds列表)
        preds, golds = predict(model, dev_loader)   # 在验证集上跑一遍
        acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)   # 验证集准确率 = 预测对的 / 总数
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={total/len(train_loader):.4f}  "
              f"dev_acc={acc:.4f}")
    elapsed = time.time() - t0   # 整个训练过程耗时（秒）= 当前时间 - 开始时间戳
    print(f"\n训练耗时：{elapsed:.0f} 秒（{args.epochs} 个 epoch，{args.threads} 线程）")

    # 测试集
    preds, golds = predict(model, test_loader)    # model, loader=测试集 DataLoader
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    # macro_f1(preds=预测ID列表, golds=真实ID列表, num_classes=类别总数) → 单个 F1 分数
    f1 = macro_f1(preds, golds, num_classes)
    print(f"\n测试集：accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")
    # confusion(preds, golds, labels=语种标签列表) → 混淆矩阵
    # print_confusion(mat, labels, top=3) → 只打印 top 个最高的混淆对
    print_confusion(confusion(preds, golds, LABELS), LABELS)

    # 逐类准确率
    print("\n  各语种准确率：")
    for c, lab in enumerate(LABELS):   # c=类别ID(0~7), lab=语种代码如 "ar" "en" ...
        idx = [i for i, g in enumerate(golds) if g == c]   # 测试集中真实标签为 c 的样本下标列表
        hit = sum(preds[i] == c for i in idx)   # 这些样本里被正确预测为 c 的数量
        print(f"    {lab}: {hit}/{len(idx)} = {hit/max(1,len(idx)):.3f}")

    # 推理示例
    print("\n--- 推理示例 ---")
    model.eval()  # 推理模式
    demos = [t for t, _ in test_rows[:4]]  # 取测试集前 4 条的原始文本做 demo
    with torch.no_grad():
        for text in demos:
            # encode(text=原始文本, vocab=字符表) → [id1, id2, ..., 0, 0] 长 MAX_LEN 的 list
            # 外面套一层 [ ... ] 是为了加 batch 维度，变成 shape (1, MAX_LEN) 才能喂模型
            logits = model(torch.tensor([encode(text, vocab)]))   # encode(text, vocab)
            # softmax 把 logits 转成概率分布；[0] 取 batch 中第一条
            prob = torch.softmax(logits, dim=1)[0]   # softmax 把 logits 转成 12 类概率分布（加起来=1）；[0] 取 batch 第一条
            pid = int(prob.argmax())  # 概率最大的类别 ID（0~11）
            print(f"  [{LABELS[pid]} {prob[pid]:.2f}]  {text[:50]}")

    if args.save:
        # 保存一个 dict，包含：
        #   state_dict —— 模型权重（eval_langid.py 用它来加载）
        #   vocab / labels / maxlen / 超参数 —— 推理时需要这些来复现预处理和模型结构
        torch.save({"state_dict": model.state_dict(), "vocab": dict(vocab),
                    "labels": LABELS, "maxlen": MAX_LEN,
                    "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                    "dropout": DROPOUT}, args.save)
        print(f"\n模型已保存：{args.save}")


if __name__ == "__main__":
    main()
