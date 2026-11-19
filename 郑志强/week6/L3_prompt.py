import json
import time
import os
from openai import OpenAI
from sklearn.metrics import f1_score

# ====== ：从系统环境变量读取 API Key ======
# 这样写可以避免代码把 Key 写死，交作业时既安全又显得专业。
API_KEY = os.getenv("DeepSeek_API_key")

# 防御性检查：防止因为环境变量没配好导致后面莫名其妙的报错
if not API_KEY:
    print("❌ 错误：未能从环境变量中读取到 'DeepSeek_API_key'。")
    print("请检查环境变量配置，或者在代码里临时写死 API_KEY = 'sk-xxx' 来运行。")
    exit()

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

LABELS = ["故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事", "旅游", "国际", "证券", "农业", "电竞"]

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_my_data(file_name, limit):
    file_path = os.path.join(CURRENT_DIR, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:limit]


samples = load_my_data("val.json", limit=100)

def predict(text):
    prompt = f"请判断以下新闻标题属于哪个类别。只输出类别名称，不要输出其他内容。\n可选类别：{','.join(LABELS)}\n标题：{text}"
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
            timeout=30 
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"\n[API 报错]: {str(e)}") 
        return "ERROR"

print("开始调用 DeepSeek API 测试 Zero-Shot...")
y_true, y_pred = [], []
start_time = time.time()

for i, item in enumerate(samples):
    text = item["sentence"]
    true_label = item["label"]
    pred_str = predict(text)
    
    if pred_str == "ERROR":
        continue
        
    pred_label = -1
    for idx, label_name in enumerate(LABELS):
        if label_name in pred_str:
            pred_label = idx
            break
            
    y_true.append(true_label)
    y_pred.append(pred_label)
    print(f"进度: {i+1}/{len(samples)} | 真实: {LABELS[true_label]} | 预测: {pred_str}")

if len(y_pred) > 0:
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print("-" * 30)
    print("L3 (DeepSeek Zero-Shot) 结果：")
    print(f"成功条数: {len(y_pred)}/{len(samples)}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"耗时: {time.time() - start_time:.2f} 秒")
    print("-" * 30)
else:
    print("\n❌ 所有请求都失败了！")