import sys
from pathlib import Path
root = Path(__file__).parent.parent
sys.path.append(str(root))
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from week04.LargeLanguageModel import LargeLanguageModel
import torch
import json
import os


# ================== 参数 ============================
base_file = os.path.abspath(__file__)
base_path = os.path.dirname(base_file)

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(device)
n_head = 16
n_embed = 512
seq_len = 128
n_layer = 2
LR = 3e-5
max_epochs = 3
pre_model_path = os.path.join(base_path, "model_pre.bin")
sft_model_path = os.path.join(base_path, "model_sft.bin")


SPECIAL = ["<user>", "</user>", "<assistant>", "</assistant>"]
# 最长优先
SPECIAL = sorted(SPECIAL, key=lambda x: len(x), reverse=True)

def tokenize(text: str, char2idx, unk=101):
    tokens = []
    p = 0
    n = len(text)
    while p < n:
        hit = None
        for w in SPECIAL:
            if text.startswith(w, p):
                hit = w
                break
        if hit:
            tokens.append(hit)
            p += len(hit)
        else:
            tokens.append(text[p])
            p += 1
    ids = [char2idx.get(t, unk) for t in tokens]
    return ids

class SFTCharDataset(Dataset): 
    def __init__(self, sft_list, char2idx, seq_len, stride=1):
        super().__init__()
        self.data = []
        self.seq_len = seq_len
        self.stride = max(1, stride)
        self.char2idx = char2idx
        for sft in sft_list:
            sft_text = f"<user>{sft['instruction']}</user><assistant>{sft['output']}</assistant>"
            assist_start = sft_text.find("<assistant>") - 11
            assist_end = sft_text.find("</assistant>", assist_start)
            sft_token_list = tokenize(sft_text, char2idx, 100)
            # for i, token in enumerate(sft_text):           
            #     sft_token_list.append(char2idx.get(token, 101))
            
            x = sft_token_list[: -1]
            y = sft_token_list[1:]
            for i in range(1, assist_start):
                y[i-1] = -100

            if len(x) > seq_len:
                continue
            else:
                gap = seq_len - len(x)
                x += [100] * gap
                y += [-100] * gap
            self.data.append((torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x , y = self.data[index]
        x = x.to(non_blocking=True, device=device)
        y = y.to(non_blocking=True, device=device)
        return x, y 
    

    

def load_dataset(file, char2idx):
    sft_list = []
    with open(file=file, mode="r", encoding="utf-8") as f:
        for line in f.readlines():
            sft_list.append(json.loads(line))
    return SFTCharDataset(sft_list, char2idx, seq_len)

def get_vocab(filename):
    vocab = {}
    idx2vocab = {}
    with open(os.path.join(base_path, filename), mode="r", encoding="utf-8") as file:
        for idx, line in enumerate(file):
            token = line.rstrip("\n") # 去掉换行符！重点，不然token带\n
            vocab[token] = idx
            idx2vocab[idx] = token
    print("total vocab: ", len(vocab))
    return vocab, idx2vocab

def train_sft():
    vocab, _ = get_vocab("./vocab.txt")
    model = LargeLanguageModel(
            vocab_size=len(vocab),
            n_embed=n_embed,
            n_layer=n_layer
            ).to(device=device)
    

    ckpt = torch.load(pre_model_path, map_location=device)
    model_dict = model.state_dict()

    # 只保留共同 key，并且形状一致
    new_ckpt = {}
    for k, v in ckpt.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            new_ckpt[k] = v

    model_dict.update(new_ckpt)
    model.load_state_dict(model_dict)


    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train_dataset = load_dataset(os.path.join(base_path, "sft_train.jsonl"), vocab)
    train_loader = DataLoader(train_dataset, batch_size= 32, shuffle=True, drop_last=True)
    model.train()
    for i in range(1, max_epochs + 1):
        train_loss_sum = torch.tensor(0.0, device=device)
        batch = 1
        for x, y in train_loader:
            logits = model(x)    # B, V; y: B, S, E
            logits_flat = logits.reshape(-1, logits.size(-1))
            y_flat = y.reshape(-1)
            loss = F.cross_entropy(logits_flat, y_flat, ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss
            print(f"f'{i}轮, {batch}批训练完成, 平均loss: {loss.item()}")
            batch +=1
        print(f'{i}轮训练完成, 平均loss: {train_loss_sum.item() / len(train_loader)}')

    torch.save(model.state_dict(), sft_model_path)

def predict(text):
    vocab, idx2vocab = get_vocab("./vocab.txt")
    model = LargeLanguageModel(
            vocab_size=len(vocab),
            n_embed=n_embed,
            n_layer=n_layer
            ).to(device=device)
    model.load_state_dict(torch.load(sft_model_path, map_location=device))
    temperature = 0.7
    model.eval()
    # for i in range(1, 100):
    predict_in_loop(model= model, text= text, vocab= vocab, idx2vocab= idx2vocab, i= 1)

def predict_in_loop(model, text, vocab, idx2vocab, i):
    if i >=40:
        return
    i += 1
    x = tokenize(text, vocab, 100)
    logits = model.forward(torch.tensor(x, dtype=torch.long, device= device).unsqueeze(0))
    last_logit = logits[0, -1, :]
    # next_idx = torch.argmax(last_logit)
    probs = F.softmax(last_logit / 0.7, dim=-1)
    next_idxs = torch.multinomial(probs, num_samples=1)
    for next_idx in next_idxs:
        new_text = text
        print(f"'{text}'的下一个字符是: '{idx2vocab.get(next_idx.item())}'")
        new_text = new_text + idx2vocab.get(next_idx.item())
        if new_text.endswith("</assistant>"):
            print(f'text: {new_text}')
            return;
        predict_in_loop(model= model, text= new_text, vocab= vocab, idx2vocab= idx2vocab, i= i)

if __name__ == "__main__":
    #train_sft()
    predict("<user>你喜欢北京、上海还是南京</user><assistant>")