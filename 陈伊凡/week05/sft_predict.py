# -*- coding: utf-8 -*-
"""
加载 SFT 微调后的模型（sft_model.pt）进行指令问答。

用法:
    python sft_predict.py "你的指令"     # 单条生成
    python sft_predict.py               # 交互模式
"""

import argparse
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from language_model import TransformerLM
from sft_finetune import PROMPT_PREFIX, PROMPT_SEP, UNK, EOS, encode

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="sft_model.pt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("instruction", nargs="*", help="不传则进入交互模式")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"未找到 {args.ckpt}，请先运行 python sft_finetune.py 训练。")
        sys.exit(1)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    char2idx, idx2char = ckpt["char2idx"], ckpt["idx2char"]
    a = ckpt["args"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerLM(
        vocab_size=len(char2idx), embed_dim=a["embed_dim"], num_layers=a["num_layers"],
        nhead=a["nhead"], ffn_dim=a["ffn_dim"], dropout=0.0,   # 推理关闭 dropout
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    def answer(instruction: str) -> str:
        prompt = PROMPT_PREFIX + instruction.strip() + PROMPT_SEP
        ids = encode(prompt, char2idx)
        max_len = model.pos.num_embeddings
        with torch.no_grad():
            for _ in range(100):
                if len(ids) >= max_len:
                    break
                x = torch.tensor([ids[-max_len:]], device=device)
                logits = model(x)[0, -1] / args.temperature
                if args.top_k:
                    v, _ = torch.topk(logits, args.top_k)
                    logits[logits < v[-1]] = float("-inf")
                nxt = int(torch.multinomial(torch.softmax(logits, -1), 1))
                if nxt == char2idx[EOS]:
                    break
                ids.append(nxt)
        return "".join(idx2char[i] for i in ids[len(encode(prompt, char2idx)):]
                       if i not in (char2idx[UNK], char2idx[EOS]))

    if args.instruction:
        print(answer(" ".join(args.instruction)))
        return

    print("交互模式，输入指令回车生成，q 退出：")
    while True:
        try:
            text = input("\n指令> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in {"q", "quit", "exit"}:
            break
        print("回答> " + answer(text))


if __name__ == "__main__":
    main()
