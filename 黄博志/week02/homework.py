import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
  def __init__(self, input_size, hidden_size = 5):
    super(TorchModel, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    # self.activation = torch.sigmoid
    self.activation = torch.softmax
    self.loss = nn.functional.mse_loss  # loss函数采用均方差损失

  def forward(self, x, y = None):
    x = self.fc1(x)
    y_pred = self.activation(x)
    if y is not None:
      print(f'loss:-> {self.loss(y_pred, y)}')
      return self.loss(y_pred, y)
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
  return torch.FloatTensor(X_list), torch.LongTensor(y_list)

def testModel(model):
  # 测试模型
  model.eval()
  test_sample = 100
  test_x, test_y = build_dataset(test_sample)
  correct, wrong = 0, 0
  with torch.no_grad():
    for i in range(test_sample):
      x = test_x[i]
      y_pred = model(x)
      y = test_y[i]
      if y_pred == y:
        correct += 1
      else:
        wrong += 1
  print(f"测试结果: 正确 {correct}, 错误 {wrong}")
  return correct / (correct + wrong)

def main():
  epoch_num = 20
  batch_size = 20
  train_sample = 500
  input_size = 5
  learning_rate = 0.01
  
  model = TorchModel(input_size)
  # 选择优化器
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  log = []
  # 创建训练集，正常任务是读取训练集 自己模拟数据集(train_sample条)
  train_x, train_y = build_dataset(train_sample)
  
  # 开始训练
  for epoch in range(epoch_num):
    model.train()
    watch_loss = []
    for i in range(train_sample // batch_size):
      x = train_x[i* batch_size : (i+1)*batch_size]
      y = train_y[i* batch_size : (i+1)*batch_size]
      print(f'x={x} y ={y}')
      loss = model(x, y)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      watch_loss.append(loss.item())
    print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    acc = testModel(model)
    log.append([acc, float(np.mean(watch_loss))])
    print(f"epoch: {epoch + 1}, loss: {loss.item()}")
    
  torch.save(model.state_dict(), "model.bin")
  # 画图
  print(log)
  plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
  plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
  plt.legend()
  plt.show()
  return

main()
