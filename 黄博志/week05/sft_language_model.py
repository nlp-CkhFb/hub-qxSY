"""
字符级 SFT（监督微调）语言模型
==============================

由 transformer_language_model.py（预训练版）改造而来，原文件保持不动。

预训练和 SFT 的本质区别只有一条：**哪些位置算 loss**。

    预训练：  [今天天气很好]                每个 token 都预测下一个 token
    SFT：     用户：今天天气怎么样？\n助手：今天多云，20~26℃。<eos>
              └────────── label = -100，不算 loss ──────────┘└── 只在这里算 ──┘

模型结构完全沿用第 4 周的 Decoder-only Transformer（Embedding + 位置编码 +
因果自注意力），改的是数据、label 掩码和生成方式：

    Dataset  ：从连续文本 → (指令, 回答) 对，prompt 段 label 置 -100
    Collate  ：变长样本右侧 padding（pad 位置 label 同样是 -100）
    Generate ：按同样的模板拼 prompt，自回归生成到 <eos> 为止

数据格式（--data）支持 .json / .jsonl，三种记录都认：
    1. Alpaca:    {"instruction": ..., "input": ..., "output": ...}
    2. 简单对:     {"prompt": ..., "response": ...}
    3. 多轮(简化): {"messages": [{"role": "user", ...}, {"role": "assistant", ...}, ...]}
                   —— 最后一个 assistant 作为回答，之前所有 user 内容拼成指令

不传 --data 时使用内置 60 条中文演示数据（关于天气、算术、常识、翻译等），开箱即跑。

用法（默认参数就是跑内置 60 条演示数据；数据量小，需要多训几百轮才能记住）:
    # 1) 从第 4 周的预训练权重继续 SFT（推荐，在 week05 目录下执行）
    python sft_language_model.py --init best_transformer.pt --epochs 200

    # 2) 不用预训练权重，直接在演示数据上从零训练
    python sft_language_model.py --epochs 200

    # 3) 用自己的 Alpaca 数据
    python sft_language_model.py --data my_sft.jsonl --init best_transformer.pt

    # 4) 只加载 SFT 权重做生成
    python sft_language_model.py --load best_sft.pt --prompt 你好 --n_gen 80

说明:
    - 从零训练时词表只包含 SFT 数据里的字符；用 --init 时会继承预训练词表，
      SFT 数据里多出来的新字会追加到词表末尾（embedding / 输出层自动扩行）。
    - 训练和推理必须使用同一套模板（用户：/ 助手：），否则效果会明显变差。
    - 内置 60 条数据很少，训练后期验证 loss 上升、训练 loss 接近 0 是正常现象
      （模型在背题）：默认 --select last 保存最后一轮，验证 loss 只用来观察。
      真实 SFT 数据（几百条以上）通常只训 1~3 轮，并按验证 loss 选权重
      （--select best）。
"""

import argparse
import json
import math
import random
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 对话模板：字符级模型直接用普通汉字做分隔，不需要额外的 special token
USER_TAG = "用户："
ASSISTANT_TAG = "助手："


# ─────────────────────────── 内置演示数据（60 条） ───────────────────────────

DEMO_PAIRS = [
    ("你好", "你好！有什么可以帮你的吗？"),
    ("你是谁", "我是一个字符级小模型，刚做完指令微调。"),
    ("你叫什么名字", "我叫小Char，是一个字符级语言模型。"),
    ("谢谢", "不客气，随时找我。"),
    ("再见", "再见，祝你今天顺利。"),
    ("今天天气怎么样", "我查不到实时天气，出门前记得看天气预报。"),
    ("明天会下雨吗", "我看不到实时天气，建议你查一下天气预报。"),
    ("现在几点了", "我没有时钟，你可以看看手机上的时间。"),
    ("1加1等于几", "1加1等于2。"),
    ("3乘以7等于多少", "3乘以7等于21。"),
    ("2的10次方是多少", "2的10次方是1024。"),
    ("100减37等于多少", "100减37等于63。"),
    ("12除以4等于多少", "12除以4等于3。"),
    ("北京是哪个国家的首都", "北京是中国的首都。"),
    ("中国的首都是哪里", "中国的首都是北京。"),
    ("太阳从哪边升起", "太阳从东边升起。"),
    ("地球绕着什么转", "地球绕着太阳转。"),
    ("一年有多少天", "平年365天，闰年366天。"),
    ("一天有多少小时", "一天有24小时。"),
    ("水的化学式是什么", "水的化学式是H2O。"),
    ("把“我爱学习”翻译成英文", "I love learning."),
    ("把“Good morning”翻译成中文", "早上好。"),
    ("把“今天很冷”翻译成英文", "It is very cold today."),
    ("用英文说“谢谢”", "Thank you."),
    ("推荐一本好书", "推荐《小王子》，短小但很耐读。"),
    ("推荐一部电影", "推荐《海上钢琴师》，很安静也很动人。"),
    ("推荐一种运动", "推荐快走，简单又容易坚持。"),
    ("怎么学好编程", "多写代码，多看别人的代码，再改自己的。"),
    ("怎么保持专注", "关掉通知，一次只做一件事。"),
    ("写一句鼓励自己的话", "慢慢来，比较快。"),
    ("什么是机器学习", "机器学习是让程序从数据中学习规律的方法。"),
    ("什么是过拟合", "训练集上很好、新数据上变差，就是过拟合。"),
    ("什么是SFT", "SFT是用指令和回答数据教模型按指令作答。"),
    ("什么是Transformer", "Transformer是靠自注意力处理序列的神经网络。"),
    ("什么是词向量", "词向量是把词映射成稠密向量的表示。"),
    ("什么是语言模型", "语言模型是预测下一个词或字的模型。"),
    ("讲个笑话", "程序员最怕什么？怕演示前五分钟代码崩了。"),
    ("你会写诗吗", "我只会凑字，比如：风来云去月当头。"),
    ("今天心情不好", "那就先休息一下，别急着解决所有事。"),
    ("我睡不着", "试试放下手机，做几次深呼吸。"),
    ("怎么减肥", "少吃多动，规律作息，别追求太快。"),
    ("感冒了怎么办", "多喝水多休息，严重的话及时看医生。"),
    ("你喜欢什么颜色", "我没有眼睛，不过数据里蓝色出现得很多。"),
    ("你会唱歌吗", "我不会唱，但可以给你写一句歌词。"),
    ("你吃饭了吗", "我不吃饭，只吃字符。"),
    ("你多大年纪", "我刚被训练出来，还没有生日。"),
    ("你是人吗", "不是，我是一个语言模型。"),
    ("帮我写个请假条", "老师您好，我因身体不适请假一天，望批准。"),
    ("帮我起个网名", "叫“字符旅人”怎么样？"),
    ("总结一下SFT", "用指令-回答对做监督微调，让模型学会按指令回答。"),
    ("用一句话介绍Python", "Python是一门简洁易读的通用编程语言。"),
    ("用一句话介绍北京", "北京是中国的首都，也是历史悠久的城市。"),
    ("什么是注意力机制", "让模型按相关性给不同位置分配权重的机制。"),
    ("为什么要做验证集", "用来检查模型有没有过拟合。"),
    ("学习率太大怎么办", "调小学习率，或者加一点预热。"),
    ("训练loss不下降怎么办", "检查数据、学习率和模型结构，先小规模调试。"),
    ("什么是batch size", "一次喂给模型的样本数量。"),
    ("什么是epoch", "把训练集完整过一遍叫一个epoch。"),
    ("祝我生日快乐", "生日快乐！愿你新的一岁顺顺利利。"),
    ("晚安", "晚安，做个好梦。"),
]

# 训练过程中每个 epoch 用来观察效果的固定问题
SAMPLE_PROMPTS = ["你好", "1加1等于几", "什么是SFT"]


# ─────────────────────────── 数据 ───────────────────────────

def make_prompt(instruction):
    """把指令包成推理时的提示词（与训练时的 prompt 完全一致）。"""
    return f"{USER_TAG}{instruction}\n{ASSISTANT_TAG}"


def load_json_any(path):
    """读 .json（数组）或 .jsonl（每行一个对象）。"""
    text = Path(path).read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_pairs(path):
    """把不同格式的 SFT 数据统一成 [(instruction, response), ...]。"""
    pairs = []
    for obj in load_json_any(path):
        if "messages" in obj:
            # 多轮简化：最后一个 assistant 是回答，之前 user 的内容拼成指令
            users, resp = [], ""
            for m in obj["messages"]:
                content = (m.get("content") or "").strip()
                if not content:
                    continue
                if m.get("role") == "assistant":
                    resp = content
                elif m.get("role") == "user":
                    users.append(content)
            inst = "\n".join(users)
            if inst and resp:
                pairs.append((inst, resp))
        else:
            inst = (obj.get("instruction") or obj.get("prompt") or "").strip()
            inp = (obj.get("input") or "").strip()
            out = (obj.get("output") or obj.get("response") or "").strip()
            if inp:
                inst = f"{inst}\n{inp}"
            if inst and out:
                pairs.append((inst, out))
    return pairs


def build_char_vocab(text):
    return {c: i for i, c in enumerate(sorted(set(text)))}


def encode_example(instruction, response, char2idx, seq_len, eos_id, mask_prompt=True):
    """
    prompt + response + <eos>，prompt 部分 label = -100。
    超长时优先保留回答：截掉指令开头，再过长的回答截断。
    """
    prompt_ids = [char2idx[c] for c in make_prompt(instruction) if c in char2idx]
    resp_ids = [char2idx[c] for c in response if c in char2idx]
    if not resp_ids:
        return None, None, False

    truncated = False
    if len(resp_ids) > seq_len - 1:
        resp_ids = resp_ids[:seq_len - 1]
        truncated = True
    keep_prompt = seq_len - 1 - len(resp_ids)
    if len(prompt_ids) > keep_prompt:
        prompt_ids = prompt_ids[-keep_prompt:] if keep_prompt > 0 else []
        truncated = True

    # 与预训练脚本一致：输入右移一位，label 是“当前位置的下一个 token”
    ids = prompt_ids + resp_ids + [eos_id]
    tokens = ids[:-1]
    labels = ([-100] * len(prompt_ids) if mask_prompt else list(prompt_ids)) + resp_ids + [eos_id]
    targets = labels[1:]
    return tokens, targets, truncated


class SFTDataset(Dataset):
    def __init__(self, pairs, char2idx, seq_len, eos_id, mask_prompt=True):
        self.seq_len = seq_len
        self.samples = []
        self.skipped = 0
        self.truncated = 0
        for inst, resp in pairs:
            ids, labels, trunc = encode_example(
                inst, resp, char2idx, seq_len, eos_id, mask_prompt)
            if ids is None:
                self.skipped += 1
                continue
            self.truncated += int(trunc)
            self.samples.append((ids, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, labels = self.samples[idx]
        return torch.tensor(ids), torch.tensor(labels)


def collate_batch(batch, pad_id):
    """右侧 padding。因果注意力下真实 token 看不到后面的 pad，因此不需要 attention mask。"""
    max_len = max(len(ids) for ids, _ in batch)
    x = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    y = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(batch):
        x[i, :len(ids)] = ids
        y[i, :len(labels)] = labels
    return x, y


# ─────────────────────────── 模型（与预训练版一致） ───────────────────────────

class TransformerLM(nn.Module):
    """Decoder-only Transformer：Embedding + 位置编码 + N 层因果自注意力 + 输出投影。"""

    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=256, dropout=0.1, max_len=64, tie_weights=True):
        super().__init__()
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"
        self.block_size = max_len
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, vocab_size, bias=not tie_weights)
        if tie_weights:
            self.head.weight = self.embed.weight

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((max_len, max_len), float("-inf")), diagonal=1),
            persistent=False,
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        _, t = x.shape
        assert t <= self.block_size, f"序列长度 {t} 超过 block_size {self.block_size}"
        pos = torch.arange(t, device=x.device).unsqueeze(0)
        h = self.drop(self.embed(x) + self.pos(pos))
        mask = self.causal_mask[:t, :t]
        for layer in self.layers:
            h = layer(h, src_mask=mask)
        return self.head(self.norm(h))


def build_model(cfg, vocab_size):
    return TransformerLM(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["ff_dim"],
        dropout=cfg["dropout"],
        max_len=cfg["seq_len"],
        tie_weights=cfg["tie_weights"],
    )


# ─────────────────────────── 生成 ───────────────────────────

@torch.no_grad()
def generate(model, char2idx, idx2char, special, instruction, max_new=60,
             temperature=0.7, top_k=20, device="cpu"):
    """按模板拼出“用户：…\\n助手：”，自回归生成，遇到 <eos> 停止。"""
    model.eval()
    prompt_ids = [char2idx[c] for c in make_prompt(instruction) if c in char2idx]
    if not prompt_ids:
        prompt_ids = [char2idx.get("你", 0)]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    eos_id, pad_id = special["eos"], special["pad"]
    out = []
    for _ in range(max_new):
        cond = idx[:, -model.block_size:]
        logits = model(cond)[:, -1, :]
        logits[:, pad_id] = -1e9                      # 不允许生成 pad
        if temperature <= 0:                          # greedy
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k and top_k < logits.size(-1):
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        if next_id.item() == eos_id:                  # 模型认为回答结束
            break
        idx = torch.cat([idx, next_id], dim=1)
        out.append(idx2char[next_id.item()])
    return "".join(out)


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, device, optimizer=None,
              scheduler=None, grad_clip=1.0):
    train = optimizer is not None
    model.train(train)
    total_loss, total_tokens, correct = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        flat_y = y.reshape(-1)
        mask = flat_y != -100
        n = int(mask.sum().item())
        total_loss += loss.item() * n
        total_tokens += n
        correct += (logits.reshape(-1, logits.size(-1)).argmax(-1)[mask] == flat_y[mask]).sum().item()

    avg_loss = total_loss / max(1, total_tokens)
    acc = correct / max(1, total_tokens)
    return avg_loss, math.exp(min(avg_loss, 50.0)), acc


def make_scheduler(optimizer, total_steps, warmup_steps, min_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_pretrained_weights(model, ckpt):
    """继承预训练权重；词表变大时 embedding / pos / head 按行拷贝，新行保留随机初始化。"""
    state = ckpt["model_state"]
    with torch.no_grad():
        n_old = state["embed.weight"].shape[0]
        model.embed.weight[:n_old].copy_(state["embed.weight"])
        if model.head.weight is not model.embed.weight:
            model.head.weight[:n_old].copy_(state["head.weight"])
            if model.head.bias is not None and "head.bias" in state:
                model.head.bias[:n_old].copy_(state["head.bias"])
        p_old = state["pos.weight"]
        n_pos = min(p_old.shape[0], model.pos.weight.shape[0])
        model.pos.weight[:n_pos].copy_(p_old[:n_pos])

    rest = {k: v for k, v in state.items()
            if not k.startswith(("embed.", "pos.", "head."))}
    missing, unexpected = model.load_state_dict(rest, strict=False)
    if unexpected:
        print(f"  警告：预训练权重中有 {len(unexpected)} 个多余的 key 被忽略")
    return missing


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="字符级 SFT：用指令-回答数据微调 Transformer 语言模型")
    parser.add_argument("--data", default="", help="SFT 数据 .json/.jsonl（Alpaca / prompt-response / messages）")
    parser.add_argument("--max_samples", type=int, default=0, help="只用前 N 条数据，0=全部")
    parser.add_argument("--init", default="", help="预训练 checkpoint（如 best_transformer.pt）")
    parser.add_argument("--load", default="", help="加载 SFT checkpoint 后只生成，不训练")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seq_len", type=int, default=0, help="0=自动（有 init 用其配置，否则 64）")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=128, help="仅从零训练时生效")
    parser.add_argument("--nhead", type=int, default=4, help="仅从零训练时生效")
    parser.add_argument("--num_layers", type=int, default=3, help="仅从零训练时生效")
    parser.add_argument("--ff_dim", type=int, default=256, help="仅从零训练时生效")
    parser.add_argument("--dropout", type=float, default=0.1, help="仅从零训练时生效")
    parser.add_argument("--tie_weights", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--lr", type=float, default=0.0,
                        help="0=自动（小数据集 1e-3；大数据集 init 2e-5、从零 5e-4）")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--mask_prompt", default=True, action=argparse.BooleanOptionalAction,
                        help="prompt 部分是否屏蔽 loss（--no-mask_prompt 可做对照实验）")
    parser.add_argument("--max_vocab", type=int, default=4000, help="仅从零训练时生效")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save", default="best_sft.pt")
    parser.add_argument("--prompt", default="你好", help="生成用的用户输入")
    parser.add_argument("--n_gen", type=int, default=60)
    parser.add_argument("--sample_chars", type=int, default=30, help="训练中采样字数，0=不采样")
    parser.add_argument("--sample_every", type=int, default=10, help="每多少轮采样一次")
    parser.add_argument("--select", choices=["last", "best"], default="last",
                        help="保存最后一轮(last)还是验证 loss 最低的一轮(best)；小数据集建议 last")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: Transformer (SFT)")

    # ── 只加载 SFT 权重做生成 ──
    if args.load:
        ckpt = torch.load(args.load, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        char2idx = ckpt["char2idx"]
        idx2char = ckpt["idx2char"]
        special = ckpt["special"]
        model = build_model(cfg, ckpt["vocab_size"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        print(f"\n用户：{args.prompt}")
        print(f"助手：{generate(model, char2idx, idx2char, special, args.prompt, args.n_gen, args.temperature, args.top_k, device)}")
        return

    # ── 数据 ──
    if args.data:
        pairs = load_pairs(args.data)
        print(f"SFT 数据: {args.data}（{len(pairs)} 条）")
    else:
        pairs = list(DEMO_PAIRS)
        print(f"未指定 --data，使用内置演示数据（{len(pairs)} 条）")
    if args.max_samples:
        pairs = pairs[:args.max_samples]
    if len(pairs) < 4:
        raise ValueError("SFT 数据太少（<4 条），请检查 --data 或 --max_samples")

    random.shuffle(pairs)
    split = int(len(pairs) * (1 - args.val_ratio))
    train_pairs, val_pairs = pairs[:split], pairs[split:]
    if not val_pairs:
        val_pairs = train_pairs[:1]
        print("警告：验证集为空，临时用 1 条训练样本充当验证集")

    # ── 词表 / 配置 ──
    base = None
    if args.init:
        base = torch.load(args.init, map_location=device, weights_only=False)
        cfg = dict(base["config"])
        cfg["seq_len"] = args.seq_len or cfg["seq_len"]
        char2idx = dict(base["char2idx"])
        print(f"预训练权重: {args.init}（{len(char2idx)} 个字）")
    else:
        cfg = {
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "ff_dim": args.ff_dim,
            "dropout": args.dropout,
            "seq_len": args.seq_len or 64,
            "tie_weights": args.tie_weights,
        }
        char2idx = {}

    sft_text = "".join(make_prompt(i) + r for i, r in pairs)
    if base is None:
        char2idx = build_char_vocab(sft_text)
        if args.max_vocab and len(char2idx) > args.max_vocab:
            top = list(char2idx)[:args.max_vocab]
            char2idx = {c: i for i, c in enumerate(top)}
    else:
        n_before = len(char2idx)
        for c in sorted(set(sft_text)):
            if c not in char2idx:
                char2idx[c] = len(char2idx)
        print(f"词表: {n_before} → {len(char2idx)}（SFT 数据新增 {len(char2idx) - n_before} 个字符）")

    seq_len = cfg["seq_len"]
    pad_id = len(char2idx)          # 特殊 token 追加在词表末尾，保证旧 id 不变
    eos_id = pad_id + 1
    vocab_size = len(char2idx) + 2
    idx2char = {i: c for c, i in char2idx.items()}
    special = {"pad": pad_id, "eos": eos_id}
    print(f"词表大小: {vocab_size}（其中 pad={pad_id}, eos={eos_id}）  序列长度: {seq_len}")

    train_ds = SFTDataset(train_pairs, char2idx, seq_len, eos_id, args.mask_prompt)
    val_ds = SFTDataset(val_pairs, char2idx, seq_len, eos_id, args.mask_prompt)
    print(f"训练样本: {len(train_ds)}  验证样本: {len(val_ds)}  "
          f"截断: train={train_ds.truncated} val={val_ds.truncated}  "
          f"跳过(回答为空): {train_ds.skipped + val_ds.skipped}")

    collate = partial(collate_batch, pad_id=pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, collate_fn=collate,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            drop_last=False, collate_fn=collate,
                            num_workers=args.num_workers)

    # ── 模型 ──
    model = build_model(cfg, vocab_size).to(device)
    if base is not None:
        load_pretrained_weights(model, base)
        print("已加载预训练权重" + ("（词表已扩展）" if len(char2idx) > len(base["char2idx"]) else ""))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    if args.lr:
        lr = args.lr
    elif len(train_ds) < 100:
        lr = 1e-3                     # 内置 60 条演示数据：步数少，需要较大 lr
    elif base is not None:
        lr = 2e-5                     # 预训练模型 + 真实 SFT 数据：小 lr
    else:
        lr = 5e-4
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95),
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(1, min(args.warmup, total_steps // 10))
    scheduler = make_scheduler(optimizer, total_steps, warmup_steps)
    print(f"学习率: {lr:g}  warmup: {warmup_steps} 步  总步数: {total_steps}")

    # ── 训练 ──
    best_val_loss = float("inf")
    best_epoch = 0

    def save_ckpt():
        torch.save({
            "model_state": model.state_dict(),
            "config": cfg,
            "vocab_size": vocab_size,
            "char2idx": char2idx,
            "idx2char": idx2char,
            "special": special,
            "args": vars(args),
        }, args.save)

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}  {'Val Acc':>8}")
    print("-" * 72)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl, tr_acc = run_epoch(model, train_loader, criterion, device,
                                            optimizer, scheduler, args.grad_clip)
        with torch.no_grad():
            va_loss, va_ppl, va_acc = run_epoch(model, val_loader, criterion, device)

        marker = "  *" if va_loss < best_val_loss else ""
        if va_loss < best_val_loss:
            best_val_loss, best_epoch = va_loss, epoch
            if args.select == "best":
                save_ckpt()

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}  {va_acc:>8.2%}{marker}")

        if args.sample_chars > 0 and (epoch % args.sample_every == 0 or epoch == args.epochs):
            seen = set()
            for q in [args.prompt] + SAMPLE_PROMPTS[:2]:
                if q in seen:
                    continue
                seen.add(q)
                a = generate(model, char2idx, idx2char, special, q,
                             args.sample_chars, args.temperature, args.top_k, device)
                print(f"        用户：{q}  →  {a}")

    if args.select == "last":
        save_ckpt()
        print(f"\n训练完成。已保存最后一轮权重（第 {args.epochs} 轮）；"
              f"训练中最低验证 loss {best_val_loss:.4f}（第 {best_epoch} 轮，仅供参考）")
        title = "===== 训练后生成（最后一轮） ====="
    else:
        ckpt = torch.load(args.save, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"\n训练完成。已加载最佳验证权重（第 {best_epoch} 轮，loss {best_val_loss:.4f}）")
        title = "===== 最佳模型生成 ====="

    prompts, seen = [], set()
    for q in [args.prompt] + SAMPLE_PROMPTS:
        if q not in seen:
            prompts.append(q)
            seen.add(q)
    print("\n" + title)
    for q in prompts:
        a = generate(model, char2idx, idx2char, special, q, args.n_gen,
                     args.temperature, args.top_k, device)
        print(f"用户：{q}\n助手：{a}\n")

    print(f"之后可单独生成：python sft_language_model.py --load {args.save} --prompt 你好 --n_gen 80")


if __name__ == "__main__":
    main()
