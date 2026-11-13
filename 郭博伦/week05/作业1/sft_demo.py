"""sft_demo.py —— 用训练好的 tf_best.pt 做简单测试：3 个原题 + 3 个跨度较大的新题。"""

import os
import torch

from transformer_lm import TransformerLM, SEP, load_sft

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "tf_best.pt")


def build_model(ck, device):
    c = ck["args"]
    model = TransformerLM(len(ck["char2idx"]), c["d_model"], c["heads"], c["layers"],
                          c["seq_len"], 0.0).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model


@torch.no_grad()
def generate(model, prompt, char2idx, idx2char, seq_len, n_chars=60, temperature=0.6):
    ids = [char2idx[ch] for ch in (prompt + SEP) if ch in char2idx]
    out = list(ids)
    for _ in range(n_chars):
        logits = model(torch.tensor([out[-seq_len:]], device=next(model.parameters()).device))[0, -1]
        if temperature <= 0.01:
            nxt = int(logits.argmax())
        else:
            nxt = int(torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1))
        out.append(nxt)
    return "".join(idx2char[i] for i in out[len(ids):])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(CKPT):
        raise FileNotFoundError(f"找不到 {CKPT}，请先运行 python transformer_lm.py 训练出权重")

    ck = torch.load(CKPT, map_location=device)
    char2idx, idx2char = ck["char2idx"], ck["idx2char"]
    seq_len = ck["args"]["seq_len"]
    model = build_model(ck, device)

    # 1. 三个原题（直接从验证集取前三条，属于训练时见过的任务类型）
    val_pairs = load_sft(os.path.join(HERE, "sft_train", "sft_val.jsonl"))
    originals = [val_pairs[i][0] for i in range(3)]

    # 2. 三个跨度较大、风格不同的简单题目
    simple = [
        "请把数字 1024 和 512 相加，并给出结果。",
        "写一句关于秋天的简短句子。",
        "列出三种常见的水果。",
    ]

    print("=" * 60)
    print("【原题测试】")
    for i, q in enumerate(originals, 1):
        print(f"\n[{i}] 问: {q}")
        print(f"    答: {generate(model, q, char2idx, idx2char, seq_len)}")

    print("\n" + "=" * 60)
    print("【跨度较大的新题测试】")
    for i, q in enumerate(simple, 1):
        print(f"\n[{i}] 问: {q}")
        print(f"    答: {generate(model, q, char2idx, idx2char, seq_len)}")


if __name__ == "__main__":
    main()
