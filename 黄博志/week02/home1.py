# ai 完善

import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
  def __init__(self, input_size, hidden_size = 5):
    super(TorchModel, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)   # hidden_size = 类别数 = 5
    self.loss = nn.functional.cross_entropy          # 分类用交叉熵（输入 logits，目标 Long）

  def forward(self, x, y = None):
    y_pred = self.fc1(x)                             # (batch, 5) logits，不做 sigmoid
    if y is not None:
      loss = self.loss(y_pred, y)                    # y 是 LongTensor，正好是 cross_entropy 要的类型
      return loss
    return y_pred

def mock_data():
  X = np.random.random(5)
  x = np.array(X)
  return X, np.argmax(x)

def build_dataset(totalNum):
  X_list = []
  y_list = []
  for _ in range(totalNum):
    X, y = mock_data()
    X_list.append(X)
    y_list.append(y)
  # 先拼成 ndarray 再转 tensor，避免 "Creating a tensor from a list of numpy.ndarrays is extremely slow"
  return torch.FloatTensor(np.array(X_list)), torch.LongTensor(y_list)

def testModel(model):
  # 测试模型
  model.eval()
  test_sample = 500
  test_x, test_y = build_dataset(test_sample)
  with torch.no_grad():
    logits = model(test_x)                           # 一次算完整批
    pred = torch.argmax(logits, dim=1)               # 在 5 个 logits 上取 argmax
    correct = int((pred == test_y).sum())
    wrong = test_sample - correct
  print(f"测试结果: 正确 {correct}, 错误 {wrong}, acc={correct / test_sample:.4f}")
  return correct / test_sample

def main():
  epoch_num = 20
  batch_size = 20
  train_sample = 5000
  input_size = 5
  learning_rate = 0.01

  model = TorchModel(input_size)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  log = []
  train_x, train_y = build_dataset(train_sample)

  for epoch in range(epoch_num):
    model.train()
    watch_loss = []
    # 每个 epoch 打乱一次，收敛更稳（原代码按固定顺序切 batch）
    perm = torch.randperm(train_sample)
    train_x, train_y = train_x[perm], train_y[perm]

    for i in range(train_sample // batch_size):
      x = train_x[i * batch_size : (i + 1) * batch_size]
      y = train_y[i * batch_size : (i + 1) * batch_size]
      loss = model(x, y)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      watch_loss.append(loss.item())
    print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    acc = testModel(model)
    log.append([acc, float(np.mean(watch_loss))])

  torch.save(model.state_dict(), "model.bin")

  with open("train_log.json", "w", encoding="utf-8") as f:
    json.dump(log, f, ensure_ascii=False, indent=2)

  print(f"最终 acc={log[-1][0]:.4f}, 最终 loss={log[-1][1]:.4f}")
  print([round(a, 3) for a, _ in log])

  # 画图：存文件，不阻塞
  plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
  plt.rcParams["axes.unicode_minus"] = False
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
  ax1.plot(range(1, len(log) + 1), [l[0] for l in log], "o-", label="acc")
  ax1.set_title("准确率 acc"); ax1.set_xlabel("epoch"); ax1.grid(alpha=0.3); ax1.legend()
  ax2.plot(range(1, len(log) + 1), [l[1] for l in log], "s-", color="tab:red", label="loss")
  ax2.set_title("平均损失 loss"); ax2.set_xlabel("epoch"); ax2.grid(alpha=0.3); ax2.legend()
  fig.tight_layout()
  fig.savefig("training_curve.png", dpi=130)
  print("曲线已保存: training_curve.png")
  plt.show(block=False)

main()