# -*- coding: utf-8 -*-
"""
build_vocab.py —— 独立提取 12 语种字符表并保存为 vocab_v12.txt
"""
import os
import csv
import unicodedata
from collections import Counter, OrderedDict

# 配置路径
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "langid", "v12")
VOCAB_PATH = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")

def normalize(text):
    """与训练脚本完全一致的文本归一化规则"""
    t = unicodedata.normalize("NFKC", text).lower()
    # 只保留字母(L)和组合符号(M)
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))

def build_and_save_vocab():
    train_csv = os.path.join(DATA_DIR, "train.csv")
    
    if not os.path.exists(train_csv):
        print(f" 找不到训练数据文件：{train_csv}")
        print("请确认你已经把 12 语种的 train.csv 放到了上述路径下。")
        return

    print(f"正在读取训练集: {train_csv}")
    freq = Counter()
    
    # 1. 遍历 CSV 提取所有字符
    with open(train_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # 确保 CSV 里有 text 列
        if "text" not in reader.fieldnames:
            print(" CSV 文件缺少 'text' 列！")
            return
            
        for row in reader:
            text = row["text"]
            normalized_text = normalize(text)
            freq.update(normalized_text)
    
    # 2. 排序：按出现次数从高到低，次数相同按字符编码排序
    chars = sorted(((c, n) for c, n in freq.items()), key=lambda kv: (-kv[1], kv[0]))
    
    # 3. 构建词表字典
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
    for ch, _ in chars:
        vocab[ch] = len(vocab)
        
    # 4. 创建保存目录
    os.makedirs(os.path.dirname(VOCAB_PATH), exist_ok=True)
    
    # 5. 写入 txt 文件
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        f.write("# char\tid\tfreq\n")
        for ch, idx in vocab.items():
            # 获取该字符的出现频率（<PAD> 和 <UNK> 设为 0）
            count = freq.get(ch, 0)
            # 处理可能存在的特殊字符（如制表符），防止打乱文件格式
            if ch == '\t':
                ch = '\\t'
            f.write(f"{ch}\t{idx}\t{count}\n")
            
    print(f"成功！共提取到 {len(vocab)} 个字符。")
    print(f"词表已保存至: {VOCAB_PATH}")

if __name__ == "__main__":
    build_and_save_vocab()