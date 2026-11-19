import json
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# ====== ：适配 CPU 算力 ======
# 强制限制线程数为 4，防止 CPU 被榨干导致死机。
torch.set_num_threads(4)

def load_my_data(file_path, limit):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 强行截断，适配 CPU
    data = data[:limit]
    return Dataset.from_dict({
        "text": [d["sentence"] for d in data],
        "label": [d["label"] for d in data]
    })

print("加载本地数据（截断版）...")
# 训练集只取 1000 条，测试集 500 条
train_dataset = load_my_data("train.json", limit=1000)
test_dataset = load_my_data("val.json", limit=500)

# ====== ：换用轻量级模型 ======
# 为了在 CPU 上尽快出结果，我从 bert-base 换成了参数量更小的 rbt3 (3层 Transformer)。
model_name = "hfl/rbt3" 
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    # 依据 PPT 结论，短文本截断到 64 完全够用，大幅节约算力。
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

print("正在处理文本...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds)
    }

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=15)

# ====== 极简训练配置 ======
# epochs 设为 2，防止过拟合；不保存模型节省硬盘。
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,              
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    eval_strategy="epoch",           
    save_strategy="no",              
    report_to="none"                 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print("开始极速微调...")
trainer.train()

results = trainer.evaluate()
print("\n" + "-" * 30)
print("L2 (BERT 极速版) 结果：")
print(f"Macro-F1: {results['eval_macro_f1']:.4f}")
print(f"Accuracy: {results['eval_accuracy']:.4f}")
print("-" * 30)