# -*- coding: utf-8 -*-
import os
import json
import random
import time
from pathlib import Path
from openai import OpenAI

# 引入老师写的公共组件
import sys
sys.path.insert(0, "common") 
from event_data import load_schema, load_enums, spans_to_event, make_pred_row, DATA_DIR
from event_prompt import build_messages, parse_model_output
from eval_utils import evaluate_predictions

# ====== 【请填入你的真实 Key】 ======
API_KEY = os.getenv("DeepSeek_API_key")
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

def main():
    data_dir = DATA_DIR
    schema = load_schema(data_dir / "event_schema.json")
    
    # 为了省钱省时间，只抽 50 条测试
    with open(data_dir / "dev_extract.jsonl", "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    random.seed(42)
    dev_rows = random.sample(rows, min(500, len(rows)))
    
    preds = []
    print("开始调用 DeepSeek API 进行零样本抽取...")
    for i, row in enumerate(dev_rows):
        text = row["text"]
        messages = build_messages(text, schema, few_shot=False)
        
        try:
            # 调用 API，温度设为 0 保证结果确定性
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0,
                max_tokens=100
            )
            raw_out = response.choices[0].message.content
            
            # 借用老师的解析函数，把文本转成结构化数据，并用 text.find() 补齐 offset
            events, info = parse_model_output(raw_out, text, schema)
            
            # 把事件组装成评测标准格式
            ev = events[0] if events else None
            preds.append(make_pred_row(row, [ev] if ev else []))
            
            if (i+1) % 10 == 0:
                print(f"进度: {i+1}/50")
                time.sleep(1) # 防限流
                
        except Exception as e:
            print(f"API 报错: {e}")
            preds.append(make_pred_row(row, []))
            
    print("\n正在计算指标...")
    meta = {"display_name": "DeepSeek API (Zero-shot)"}
    res = evaluate_predictions("api_zeroshot", preds, meta=meta, gold_path=data_dir / "dev_extract.jsonl")
    print(f"Trigger F1: {res['trigger']['f1']:.4f}")
    print(f"Argument F1: {res['argument']['f1']:.4f}")

if __name__ == "__main__":
    main()