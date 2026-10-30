import numpy as np
import torch
import torch.nn as nn

#创建5维数据
def build_sample():
    x = np.random.random(5)
    y = find_max_index(x)
    return x, y
#批量生成数据
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(np.argmax(x))
    return torch.FloatTensor(np.array(X)), torch.LongTensor(Y) #long是64为整数，float是32位浮点数
#校验函数
def find_max_index(list):
    max_num = list.min()
    for i, num in enumerate(list):
        if num > max_num:
            max_num = num
            y = i
    return y
#准确率
def evaluate(model, samples, labels):
    model.eval()        #评估模式
    correct = 0
    for x, y in zip(samples, labels):
        pred = model(x.unsqueeze(0)).argmax(dim=1).item()
        if pred == y:  # 预测和标准答案一致就算对
            correct += 1
    return correct / len(labels)

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(5, 5)  # 线性层：输入5维输出1维
        #self.activation = torch.softmax   #多分类采用softmax
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵，此时交叉熵自带softmax

    def forward(self, x, y = None):
        y_pred = self.linear(x)
        if y is None:
            return y_pred  # 只预测时返回logits:经过线性层直接输出的原始数据
        return self.loss(y_pred, y)


def main():
    # 定义参数
    epoch_num = 100  # 训练轮数
    learning_rate = 0.2  # 学习率 设置成0.01发现loss还是太高，准确率不够
    batch_size = 20  # 每次训练样本个数
    total_sample_num = 500  # 每轮训练总共训练的样本总数
    model = TorchModel(5)
    train_x, train_y = build_dataset(total_sample_num)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epoch_num):
        model.train()       #训练模式
        watch_loss = []
        for batch_index in range(total_sample_num // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, train_x, train_y)  # 测试本轮模型结果
        print("本轮训练集准确率:%.2f" % acc)

if __name__ == "__main__":
    main()
