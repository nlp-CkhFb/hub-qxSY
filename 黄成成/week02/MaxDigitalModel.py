## 注释
### 五维向量，预测最大数在第几维
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class MaxDigitalModel(nn.Module):

    def __init__(self, input_size):
        super(MaxDigitalModel, self).__init__()
        self.linear = nn.Linear(input_size, input_size);
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        logits = self.linear(x)
        if y is not None:
            return self.loss_fn(logits, y)
        return x


## 数据
def build_sample():
    x = np.random.random(5)
    idx = np.argmax(x)
    y = np.zeros(5)
    y[idx] = 1
    return x, y

def build_dataset(sample_number):
    X = []
    Y = []
    for i in range(sample_number):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)

x, y = build_sample()
print(x)
print(y)

def main():
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 400  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率
    model = MaxDigitalModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    train_x, train_y = build_dataset(train_sample)
    print(train_x.shape)
    print(train_y.shape)
    for epoch in range(1, epoch_num):
        model.train()
        watch_loss = []
        for batch in range(train_sample // batch_size):
            x = train_x[batch * batch_size : batch * batch_size + batch_size]
            y = train_y[batch * batch_size : batch * batch_size + batch_size]
            y_idx = y.argmax(dim=-1)
            loss = model(x, y_idx)
            loss.backward()
            optim.step()
            optim.zero_grad()
            print(loss.item())
            watch_loss.append(loss.item())
            ##print(f"=========\n第${epoch}轮, 第${batch}批，loss: ${loss.item()}")
        acc = verify(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    print("训练完成")
    print(model.state_dict())
    torch.save(model.state_dict(), "model.bin")
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

def verify(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_predict = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_predict, y):  # 与真实标签进行对比
            if torch.argmax(y_p) == torch.argmax(y_t):
                correct += 1  #样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def predict(model_path, input_vec):
    input_size = 5
    model = MaxDigitalModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))
        result = torch.softmax(result, dim=-1)  # 模型预测
    for vec, res in zip(input_vec, result):
        print(f"输入：{vec}, 概率值：{res}")  # 打印结果

if __name__ == "__main__":
    # main()
    test_vec = [[0.81889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.39349776,0.59416669,0.92579291,0.41567412,0.8358894]]
    predict("model.bin", test_vec)
        