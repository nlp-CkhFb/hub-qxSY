import json
import time
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score

# ====== ：定义与数据集匹配的标签字典 ======
# 必须与 test.json 里的 label (0-14) 严格对应。
# 规则法的核心就是看文本里有没有这些关键词。
LABELS = ["故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事", "旅游", "国际", "证券", "农业", "电竞"]

def load_my_data(file_path, max_n=None):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if max_n:
        data = data[:max_n]
    return data

def rule_classify(text, fallback_label):
    """
    L0 规则分类核心逻辑：
    只要标题里出现了某个类别的名字（比如出现了“体育”），就直接判定为该类别。
    """
    for idx, label_name in enumerate(LABELS):
        if label_name in text:
            return idx
    # 如果一个类别词都没匹配上，就退化成“猜训练集里最多的那一类” (Majority Baseline)
    return fallback_label

print("加载数据...")
train_data = load_my_data("train.json")
test_data = load_my_data("val.json", max_n=500)

# ====== ：计算兜底类别 (Majority Baseline) ======
# 规则法如果没命中，不能瞎猜，要用训练集里出现频率最高的类作为“兜底”。
train_labels = [d["label"] for d in train_data]
fallback_label = Counter(train_labels).most_common(1)[0][0]

print(f"规则未命中时的兜底类别 ID: {fallback_label} (即 {LABELS[fallback_label]})")

print("开始运行规则/关键词分类...")
start_time = time.time()

y_true = []
y_pred = []
hit_count = 0 # 统计命中率

for item in test_data:
    text = item["sentence"]
    true_label = item["label"]
    
    # 判断是否包含任何类别的关键词
    if any(name in text for name in LABELS):
        hit_count += 1
        
    pred_label = rule_classify(text, fallback_label)
    
    y_true.append(true_label)
    y_pred.append(pred_label)

# 计算评估指标
macro_f1 = f1_score(y_true, y_pred, average='macro')
acc = accuracy_score(y_true, y_pred)
hit_rate = hit_count / len(test_data)

print("-" * 30)
print("L0 (规则/关键词) 结果：")
print(f"关键词命中率: {hit_rate:.2%}")
print(f"Macro-F1: {macro_f1:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"耗时: {time.time() - start_time:.4f} 秒")
print("-" * 30)