# -*- coding: utf-8 -*-
"""
predict_langid.py —— 用训练好的 langid_v12.pt 做语种识别

用法：
    python predict_langid.py こんにちは、元気ですか     # 传入句子（可多个，逗号分隔）
    python predict_langid.py -f 补充句子.txt           # 从文本文件读取（每行一句）
要点：
    1. 存档里有 vocab 和 labels —— 推理的所有配件都在文件里，不依赖数据目录
    2. 重建模型必须用存档里存的 vocab/dims，和训练时保持"同一家"的规格
    3. 推理只用 model.eval() + torch.no_grad()，不更新任何参数
"""

import sys
import os
import argparse

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "langid_v12.pt")

# Windows 控制台 UTF-8 输出（印地语/泰语字符）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 与训练脚本同款的归一化 —— 前后处理必须一致，这是语言识别生效的一半
import unicodedata


def normalize(text):
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


def encode(text, vocab, maxlen):
    ids = [vocab.get(ch, 1) for ch in normalize(text)][:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


# 与训练脚本同款的模型结构（维度从存档读，不写死）
class LangIDClassifier(torch.nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = torch.nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        h, _ = self.rnn(self.embedding(x))
        h = h.masked_fill(~mask, -1e9)
        pooled = h.max(dim=1)[0]
        pooled = self.dropout(self.norm(pooled))
        return self.fc(pooled)


def load_model():
    bundle = torch.load(HERE + "/langid_v12.pt", map_location="cpu")
    vocab = bundle["vocab"]
    labels = bundle["labels"]
    maxlen = bundle["maxlen"]
    model = LangIDClassifier(len(vocab), len(labels),
                             bundle["embed_dim"], bundle["hidden_dim"],
                             bundle["dropout"])
    model.load_state_dict(bundle["state_dict"])   # 把训练好的权重灌进模型
    model.eval()                                  # 推理模式：关闭 Dropout 等训练行为
    return model, vocab, labels, maxlen


def infer(model, vocab, labels, maxlen, text):
    ids = torch.tensor([encode(text, vocab, maxlen)])
    with torch.no_grad():                         # 不建计算图，纯前向
        logits = model(ids)
    prob = torch.softmax(logits, dim=1)[0]        # logits → 12 类概率
    pid = int(prob.argmax())
    return labels[pid], float(prob[pid]), prob.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="*", help="要识别的句子（可多个）")
    ap.add_argument("-f", "--file", default="", help="每行一句的文本文件")
    args = ap.parse_args()

    if not args.texts and not args.file:
        print(__doc__)
        return

    model, vocab, labels, maxlen = load_model()
    print(f"已加载 {MODEL_PATH}：{len(vocab)} 字典字符、{len(labels)} 语种 "
          f"({', '.join(labels)})\n")

    texts = args.texts or []
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            texts += [line.strip() for line in fh if line.strip()]

    for t in texts:
        lang, conf, prob = infer(model, vocab, labels, maxlen, t)
        top = sorted(zip(labels, prob), key=lambda kv: -kv[1])[:2]
        second = f"（次选 {top[1][0]} {top[1][1]:.2f}）" if top[1][1] > 0.15 else ""
        print(f"  [{lang} {conf:.2f}]{second}  {t[:60]}")


if __name__ == "__main__":
    main()
