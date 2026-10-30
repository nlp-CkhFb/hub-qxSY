# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：对于任意五维向量，它最大数字在第几维就属于第几类

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes=5):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # ⭐ 5进5出
        self.loss = nn.functional.cross_entropy           # ⭐ 交叉熵，内部自带softmax

    def forward(self, x, y=None):
        logits = self.linear(x)   # ⭐ (batch, 5) 每一列 = "属于第几类的分数"
        if y is not None:
            return self.loss(logits, y)   # ⭐ 直接用原始分数，不要sigmoid
        else:
            return logits

def build_sample():
    x = np.random.random(5)
    return x, int(x.argmax())     # ⭐ 类别用 0~4（CrossEntropy要求从0开始）

def build_dataset(total_sample_num):
    X, Y = [], []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)               # ⭐ 不再包 [y]
    return torch.FloatTensor(X), torch.LongTensor(Y)   # ⭐ 标签用LongTensor

def evaluate(model):
    model.eval()
    x, y = build_dataset(100)
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1)   # ⭐ 每行取分数最高的类
        correct = (pred == y).sum().item()   # ⭐ 直接对 сравнав compare
    acc = correct / len(y)
    print("正确率：%f" % acc)
    return acc


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size): 
            #取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model_five.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(input_vec))
        preds = torch.argmax(logits, dim=1)  # 0~4
    for vec, p in zip(input_vec, preds):
        print("输入：%s, 预测最大维在第 %d 维" % (vec, int(p) + 1))  # 加1还原成"第几维"


if __name__ == "__main__":
    main()
    test_vec = [[0.06335137, 0.59953506, 0.93969451, 0.88549044, 0.81499611],
                [0.63825458, 0.68603344, 0.01722402, 0.79542396, 0.25463495],
                [0.00582085, 0.39419221, 0.15727197, 0.80350137, 0.64281110],
                [0.27086765, 0.90601916, 0.04010908, 0.63662753, 0.64029941],
                [0.42820235, 0.64868134, 0.80239463, 0.63950809, 0.98280802],
                [0.82318797, 0.15476396, 0.87119633, 0.04534388, 0.35057092],
                [0.36548631, 0.11073427, 0.72729356, 0.73181785, 0.93788803],
                [0.84342257, 0.02632086, 0.57144678, 0.01250233, 0.47026098],
                [0.57663115, 0.96097366, 0.20113700, 0.05857677, 0.32533467],
                [0.66463414, 0.85344086, 0.26544823, 0.25455131, 0.01143826]]
    predict("model_five.bin", test_vec)
