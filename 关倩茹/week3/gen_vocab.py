# -*- coding: utf-8 -*-
import os
import csv
import unicodedata
from collections import Counter, OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(HERE, "data", "langid", "v12", "train.csv")
OUT_VOCAB = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")


def normalize(text):
    """和老师代码里的normalize保持完全一致"""
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


def build_char_vocab(sentences, min_freq=1):
    """完全照搬老师train_langid_cls.py里面的build_char_vocab函数"""
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))
    chars = sorted(((c, n) for c, n in freq.items() if n >= min_freq),
                   key=lambda kv: (-kv[1], kv[0]))
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
    for ch, _ in chars:
        vocab[ch] = len(vocab)
    return vocab


def save_vocab(vocab, out_path):
    # 自动创建vocab文件夹，不存在就新建
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# char\tid\tcount\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t0\n")


def load_train_text(csv_path):
    texts = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
    return texts


if __name__ == "__main__":
    print("读取训练集：", TRAIN_CSV)
    all_texts = load_train_text(TRAIN_CSV)
    print(f"句子数量：{len(all_texts)}")

    # ✅ 这里就是调用 build_char_vocab
    vocab = build_char_vocab(all_texts)
    print(f"构建字符表完成，字符总数：{len(vocab)}")

    save_vocab(vocab, OUT_VOCAB)
    print(f"\n✅ 成功生成 vocab_v12.txt：{OUT_VOCAB}")
