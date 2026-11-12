import json
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import argparse

from language_model import LM

PAD_ID = 0
IGNORE_INDEX = -100
EOS_TOKEN = "<eos>"


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_prompt(instruction):
    # 统一的 prompt 模板；减少中文冒号的重复诱导，避免生成时持续输出 :
    return f"Instruction:\n{instruction}\nResponse:\n"


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, char2idx):
        self.samples = load_jsonl(jsonl_path)
        self.char2idx = char2idx
        self.eos_id = char2idx.get(EOS_TOKEN, char2idx.get("\n"))

        # 检查数据中是否有未登录字符，理论上不应该出现未登录字符，如果出现了，说明数据或词表有问题，这里简化处理直接打印警告
        unknown_chars = set()
        for item in self.samples:
            prompt = build_prompt(item["instruction"])
            response = item["output"]

            unknown_chars.update(
                c for c in prompt + response
                if c not in self.char2idx
            )

        if unknown_chars:
            print(
                f"[Warning] {jsonl_path}: "
                f"发现 {len(unknown_chars)} 种未登录字符"
            )
            print(f"字符示例：{list(unknown_chars)[:20]}")
        else:
            print(f"[OK] {jsonl_path}: 所有字符均在词表中")

    def _encode(self, text):
        ids = [self.char2idx[c] for c in text if c in self.char2idx]
        return ids or [PAD_ID]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt = build_prompt(item["instruction"])
        response = item["output"]

        prompt_ids = self._encode(prompt)
        response_ids = self._encode(response)
        if self.eos_id is not None:
            response_ids = response_ids + [self.eos_id]
        return prompt_ids, response_ids


def collate_fn(batch):
    """
    处理不同长度样本的关键：对每个 batch 里的样本统一 pad 到当前 batch 的最大长度。
    attention_mask 用来告诉模型哪些位置是真实 token，哪些是 padding。
    """
    prompt_list, response_list = zip(*batch)
    max_len = max(len(p) + len(r) for p, r in zip(prompt_list, response_list))

    input_ids_batch, labels_batch = [], []

    for prompt_ids, response_ids in zip(prompt_list, response_list):
        full_ids = prompt_ids + response_ids
        prompt_len = len(prompt_ids)
        total_len = len(full_ids)

        labels = [-100] * total_len
        for i in range(total_len - 1):
            if i + 1 >= prompt_len:
                labels[i] = full_ids[i + 1]

        pad_len = max_len - total_len
        full_ids = full_ids + [PAD_ID] * pad_len
        labels = labels + [IGNORE_INDEX] * pad_len

        input_ids_batch.append(torch.tensor(full_ids, dtype=torch.long))
        labels_batch.append(torch.tensor(labels, dtype=torch.long))

    return (
        torch.stack(input_ids_batch),
        torch.stack(labels_batch),
    )


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)

        # 只计算真实 token 的 loss，padding 位置忽略
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        # 这里用 labels 中的 -100 让 CrossEntropyLoss 忽略 pad 和 prompt 部分。

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * (labels != IGNORE_INDEX).sum().item()
        total_tokens += (labels != IGNORE_INDEX).sum().item()

    return total_loss / max(total_tokens, 1)


def generate_text(model, prompt, char2idx, idx2char, device, max_new_tokens=60, temperature=0.7, top_k=20):
    """从 prompt 开始做一次字符级采样生成，并在遇到 eos 时停止。"""
    model.eval()

    eos_id = char2idx.get(EOS_TOKEN, char2idx.get("\n"))
    prompt_ids = [char2idx[ch] for ch in prompt if ch in char2idx]
    if not prompt_ids:
        return prompt

    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated = prompt

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                v, i = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)), dim=-1)
                probs = torch.softmax(v, dim=-1)
                sample = torch.multinomial(probs, num_samples=1)
                next_id = int(i.gather(-1, sample).item())
            else:
                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()

            next_char = idx2char.get(next_id, "")
            if not next_char:
                break

            generated += next_char
            if eos_id is not None and next_id == eos_id:
                print(f"[Info] 遇到 EOS，提前结束生成。")
                break

            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)

    return generated


def main():
    checkpoint_path = "tf_mask_model.pt"
    train_path = os.path.join("sft_train", "sft_train.jsonl")
    val_path = os.path.join("sft_train", "sft_val.jsonl")

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model_args = checkpoint["args"]
    char2idx = checkpoint["char2idx"]
    idx2char = checkpoint["idx2char"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model = LM(
        vocab_size=len(char2idx),
        embed_dim=model_args["embed_dim"],
        hidden_dim=model_args["hidden_dim"],
        num_layers=model_args["num_layers"],
        model_type=model_args["model"],
        dropout=model_args["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.train()

    train_ds = SFTDataset(train_path, char2idx)
    val_ds = SFTDataset(val_path, char2idx)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    for epoch in range(1, 20):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        model.eval()
        val_loss = 0.0
        val_tokens = 0
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                logits = model(input_ids)
                loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                val_loss += loss.item() * (labels != IGNORE_INDEX).sum().item()
                val_tokens += (labels != IGNORE_INDEX).sum().item()
        val_loss = val_loss / max(val_tokens, 1)
        model.train()

        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        torch.save({
            "model_state": model.state_dict(),
            "char2idx": char2idx,
            "idx2char": idx2char,
            "args": model_args,
        }, "sft_model.pt")

    # 文本生成验证：用一个真实验证样本 prompt 生成回答，确认模型输出是否稳定
    if val_ds.samples:
        sample = val_ds.samples[0]
        prompt = build_prompt(sample["instruction"])
        print("prompt:", prompt)
        generated = generate_text(
            model,
            prompt,
            char2idx,
            idx2char,
            device,
            max_new_tokens=80,
            temperature=0.8,
            top_k=20,
        )
        print("\n=== 生成验证 ===")
        print(generated)

def test_generation():
    checkpoint_path = "sft_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model_args = checkpoint["args"]
    char2idx = checkpoint["char2idx"]
    idx2char = checkpoint["idx2char"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model = LM(
        vocab_size=len(char2idx),
        embed_dim=model_args["embed_dim"],
        hidden_dim=model_args["hidden_dim"],
        num_layers=model_args["num_layers"],
        model_type=model_args["model"],
        dropout=model_args["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # 测试生成
    test_prompt = "Instruction:\n给我推荐一本有趣的书。 我喜欢推理小说和历史类书籍。\nResponse:\n"
    generated_text = generate_text(
        model,
        test_prompt,
        char2idx,
        idx2char,
        device,
        max_new_tokens=100,
        temperature=0.8,
        top_k=20,
    )
    print("\n=== 测试生成 ===")
    print(generated_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "generate"], default="generate")
    args = parser.parse_args()

    if args.mode == "train":
        main()
    else:
        test_generation()
        