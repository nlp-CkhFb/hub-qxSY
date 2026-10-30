# week03 作业：12 类语种分类

- 任务：一句话判断属于哪种语言，12 类（ar en es hi ja pt ru zh ko th vi tr），数据在 `data/langid/v12`
- 模型：字符级，Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear
- 字符表：课程没给，用课堂的 `build_char_vocab()` 从 `train.csv` 统计生成，共 3276 个字符
- 训练：Adam，lr = 0.001，batch = 64，8 轮
- 结果：测试集准确率 0.963，macro-F1 0.964，曲线见 `train-curve.png`
- 运行：`python train_langid_cls.py`
- 文件：`train_langid_cls.py`、`data/langid/`、`langid_rnn_v12.pt`、`train-curve.png`
