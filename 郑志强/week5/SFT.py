import os
import json
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 限制 CPU 线程，防止卡死
torch.set_num_threads(4)

# =====================================================================
# 模块一：数据读取与预处理 (Data Preparation)
# =====================================================================
def load_jsonl(file_path):
    """读取 JSONL 文件"""
    data = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}，请确认路径是否正确。")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                data.append({"prompt": item["instruction"], "response": item["output"]})
    return data

def build_vocab_from_data(train_data):
    """基于训练集构建词表"""
    chars = set()
    for item in train_data:
        chars.update(item["prompt"])
        chars.update(item["response"])
    
    chars = sorted(list(chars))
    # 初始化特殊字符
    vocab = {"<PAD>": 0, "<UNK>": 1, "<EOS>": 2}
    for c in chars:
        vocab[c] = len(vocab)
    
    idx2char = {i: c for c, i in vocab.items()}
    return vocab, idx2char

class SFTDataset(Dataset):
    """SFT 数据集，核心：Prompt 部分的 Loss 屏蔽 (-100)"""
    def __init__(self, data_list, vocab, max_len=128):
        self.data_list = data_list
        self.vocab = vocab
        self.max_len = max_len
        self.pad_id = vocab["<PAD>"]
        self.eos_id = vocab["<EOS>"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        prompt = item["prompt"]
        response = item["response"]

        # 1. 转换为 ID
        prompt_text = prompt + "\n"   #加个换行符
        prompt_ids = [self.vocab.get(c, self.vocab["<UNK>"]) for c in prompt_text]
        response_ids = [self.vocab.get(c, self.vocab["<UNK>"]) for c in response] + [self.eos_id]

        # 2. 拼接
        input_ids = prompt_ids + response_ids

        # 3. 构造 Labels：Prompt 部分不计 Loss
        labels = [-100] * len(prompt_ids) + response_ids

        # 4. 截断与 Padding
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            labels = labels[:self.max_len]
        
        pad_len = self.max_len - len(input_ids)
        input_ids = input_ids + [self.pad_id] * pad_len
        labels = labels + [-100] * pad_len

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


# =====================================================================
# 模块二：模型定义 (Model)
# =====================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SFTTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        # 1. 因果掩码：只能看前面的词
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # 2. Padding掩码：把等于0的位置（<PAD>）全设为 True 屏蔽掉
        # x == 0 的地方是 PAD
        padding_mask = (x == 0) 

        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        # 3. 把两个 Mask 都传给 Transformer
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.fc_out(x)


# =====================================================================
# 模块三：训练与评估 (Training & Evaluation)
# =====================================================================
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="SFT Training", ncols=100)
    
    for input_ids, labels in pbar:
        input_ids, labels = input_ids.to(device), labels.to(device)
        
        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)


# =====================================================================
# 模块四：自回归生成 (Text Generation - 新增)
# =====================================================================
def generate(model, prompt_text, vocab, idx2char, device, max_new_tokens=50):
    model.eval()
    
    # 把输入的 prompt 转成 id
    input_ids = [vocab.get(c, vocab["<UNK>"]) for c in prompt_text]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    generated_ids = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_tensor)
            # 取最后一个位置的输出
            next_token_logits = logits[0, -1, :].clone()
            
            # 1. 重复惩罚（Repetition Penalty）：防止“复读机”
            # 如果之前生成过这个词，就降低它的分数
            for token_id in set(generated_ids):
                if next_token_logits[token_id] > 0:
                    next_token_logits[token_id] /= 1.2 # 惩罚系数，越大惩罚越狠
                else:
                    next_token_logits[token_id] *= 1.2
            
            # 2. 温度缩放（Temperature）
            temperature = 0.3
            next_token_logits = next_token_logits / temperature
            
            # 3. Top-P (Nucleus Sampling) 核采样
            top_p = 0.9
            # 将 logits 降序排序
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            # 计算累积概率
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 找出累积概率超过 top_p 的位置，将其后面的词全部屏蔽 (设为负无穷)
            sorted_indices_to_remove = cumulative_probs > top_p
            # 将第一处超过 top_p 的位置保留（因为 cumsum 是累加的，要保留边界）
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # 把要移除的词的 logits 设为极小值
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float('Inf')
            
            # 重新计算概率并采样
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            # 如果预测到了 <EOS> (结束符)，就停止生成
            if next_token_id == vocab["<EOS>"]:
                break
                
            generated_ids.append(next_token_id)
            
            # 把新生成的字加到输入序列末尾，继续预测下一个
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]], device=device)], dim=1)
    
    # 将生成的 ids 翻译回文字
    generated_text = "".join([idx2char.get(i, "") for i in generated_ids])
    return generated_text


# =====================================================================
# 模块五：主控函数 (Main)
# =====================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备: {device}")

    # 1. 加载真实数据 (请确保这两个 jsonl 文件和代码在同一目录)
    print("正在加载数据...")
    train_data = load_jsonl("sft_train.jsonl")
    val_data = load_jsonl("sft_val.jsonl")
    print(f"训练集大小: {len(train_data)} 条 | 验证集大小: {len(val_data)} 条")
    
    # 2. 构建词表
    vocab, idx2char = build_vocab_from_data(train_data)
    print(f"词表大小: {len(vocab)}")

    # 3. 构建 Dataset 和 DataLoader
    # max_len 根据你的实际数据长度调整，如果句子长可以设大点
    max_len = 128 
    train_dataset = SFTDataset(train_data, vocab, max_len=max_len)
    val_dataset = SFTDataset(val_data, vocab, max_len=max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 4. 初始化模型
    model = SFTTransformer(vocab_size=len(vocab), d_model=128, nhead=4, num_layers=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 5. 开始训练
    epochs = 10  # 3000条数据，10-20轮大概能收敛得差不多
    print("\n开始 SFT 微调训练...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # 6. 训练结束，展示预测效果（重点看这里！）
    print("\n" + "="*50)
    print("展示预测效果 (从验证集抽取 3 条进行测试)")
    print("="*50)
    
    model.eval()
    for i in range(min(3, len(val_data))):
        sample = val_data[i]
        prompt = sample["prompt"]
        true_response = sample["response"]
        
        # 调用生成函数
        pred_response = generate(model, prompt, vocab, idx2char, device, max_new_tokens=100)
        
        print(f"\n【测试样本 {i+1}】")
        print(f"输入 (Prompt): {prompt}")
        print(f"真实回答: {true_response}")
        print(f"模型预测: {pred_response}")
    print("="*50)

if __name__ == "__main__":
    main()