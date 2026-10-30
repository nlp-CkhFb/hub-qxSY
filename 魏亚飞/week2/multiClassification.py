# coding:utf8
"""
任务:根据水果的大小,预测价格档位
输入:[重量(千克), 直径(分米)] 共 2 个数
输出:3 个概率,对应三档价格 —— 0便宜(2元/斤) 1中等(8元/斤) 2贵(15元/斤)
模型:一层线性层 y = Wx + b,损失函数用交叉熵,梯度下降训练
规律:重量越大,价格档位越高
注：本作业由AI进行辅助
"""
import torch
import torch.nn as nn
import numpy as np


# ============ 第一步:造数据 ============
# 任务的规律写在这里:重量 < 0.5千克 → 便宜档;0.5~0.8 → 中等档;> 0.8 → 贵档
# 每个样本 = 一对(输入, 正确答案),模型靠大量这样的配对学会规律
def build_sample():
    w = np.random.uniform(200, 1200)               # 重量,单位克,随机 200~1200
    d = w / 1000 + np.random.normal(0, 0.05)       # 直径,与重量同尺度,随重量变大,带一点测量误差
    if w < 500:
        y = 0                                      # 便宜档
    elif w < 800:
        y = 1                                      # 中等档
    else:
        y = 2                                      # 贵档
    # 关键位置A:重量除以1000再返回 —— 特征归一化
    # 重量(200~1200)和直径(2~12)数值差100倍,不归一化会让梯度学习偏向大数值维度,收敛极慢
    return [w / 1000, d], y


# 攒一批样本:把单个样本函数重复调用 n 次,打包成模型能吃的张量
def build_dataset(total_sample_num):
    X, Y = [], []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))


# ============ 第二步:模型 ============
class PriceModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # 关键位置①:线性层,就是  y = Wx + b
        # 一个 2×3 的权重矩阵 W + 3 个偏置 b,共 9 个"待学数字"
        # 作用:把 2 维输入,加工成 3 个打分(每个价格档一个)
        self.linear = nn.Linear(input_size, 3)
        # 关键位置②:损失函数交叉熵
        # 输入(打分, 正确档位),输出一个数:错得越离谱,数越大
        self.loss = nn.functional.cross_entropy

    # 前向传播:数据流过机器的全过程
    def forward(self, x, y=None):
        y_pred = self.linear(x)                  # 输入过线性层,得到 3 个打分
        if y is not None:
            # 关键位置③:训练模式 —— 手里有正确答案,返回"错了多少"
            return self.loss(y_pred, y)
        else:
            # 关键位置④:考试模式 —— 没有答案,把打分变成概率分布(总和=1)
            return torch.softmax(y_pred, dim=1)


# ============ 第三步:模拟考 ============
# 每训练完一轮,用 100 道没学过的新题测正确率,观察模型是否真的在进步
def evaluate(model):
    model.eval()                                 # 切到评估模式
    x, y = build_dataset(100)
    correct = 0
    # 只需要前向结果,不需要梯度;不关掉会白白多算一遍梯度
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if torch.argmax(y_p) == int(y_t):    # 概率最大的档位 == 正确档位?
                correct += 1
    print("  模拟考:100 个水果,答对 %d,正确率 %.2f" % (correct, correct / 100))
    return correct / 100


# ============ 第四步:训练 ============
def main():
    # 配置参数
    epoch_num = 50          # 全部数据过 50 遍(一轮 epoch = 5000 个样本全过一遍)
    batch_size = 32         # 每批 32 个样本(一次更新权重用 32 道)
    train_sample = 5000     # 每轮 5000 个样本,即每轮 156 步
    input_size = 2          # 输入 2 维
    learning_rate = 0.02    # 每次修正权重的幅度(太小学不完,太大会震荡)

    model = PriceModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   # 优化器:负责执行"更新权重"这个动作
    train_x, train_y = build_dataset(train_sample)

    # 训练主循环:外层一轮 = 全部数据过一遍;内层一批 = 更新一次权重
    for epoch in range(epoch_num):
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]

            loss = model(x, y)       # 关键位置⑥:前向传播 —— 用当前权重算一遍,得出"错了多少"
            loss.backward()          # 关键位置⑦:反向传播 —— 链式法则从 loss 逐层往回推,
                                     # 算出 9 个数字里每一个"该往哪边调、调多少"(即梯度)
            optim.step()             # 关键位置⑧:真正修改 —— 每个权重 = 旧权重 − lr × 梯度
            optim.zero_grad()        # 关键位置⑨:梯度清零 —— 不清零,下一批的梯度会累加在旧梯度上,乱套
            watch_loss.append(loss.item())

        print("第%d轮 平均loss:%.4f" % (epoch + 1, np.mean(watch_loss)))
        evaluate(model)              # 每轮结束,模拟考一次,看正确率有没有爬升

    torch.save(model.state_dict(), "price_model.pt")   # 关键位置⑩:把学好的 9 个数字存盘
    print("训练完成,权重已保存")


# ============ 第五步:用训练好的模型预测 ============
def predict(model_path, input_vec):
    model = PriceModel(len(input_vec[0]))
    model.load_state_dict(torch.load(model_path))    # 关键位置⑪:把存盘的权重装回模型,"恢复学习成果"
    model.eval()                                     # 切到预测模式
    with torch.no_grad():                            # 预测不需要梯度
        prob = model(torch.FloatTensor(np.array(input_vec)))
    for vec, p in zip(input_vec, prob):
        idx = int(torch.argmax(p))                   # 概率最大的档位 = 最终判断
        print("输入 [重量%.2f千克, 直径%.1f分米] → 价格档:%d (概率 %.2f)"
              % (vec[0], vec[1], idx, float(p[idx])))


if __name__ == "__main__":
    main()
    # 三只新水果:小、中、大
    predict("price_model.pt",
            [[0.3, 0.35], [0.7, 0.72], [1.1, 1.15]])
