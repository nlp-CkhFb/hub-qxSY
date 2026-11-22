[cfg] pool=cls loss=CE class_weight=off fgm=off epochs=3 steps/epoch=834 eval_n=10000 save=on
评测 8901 条（干净集）| 1.2 ms/条 | ECE=0.0779
| 方法 | 准确率 | Macro-F1 | 解析失败 |
|---|---|---|---|
| RoBERTa 微调 | 0.5714 | 0.5624 | 0 |

[cfg] pool=cls loss=CE class_weight=on fgm=off epochs=3 steps/epoch=834 eval_n=10000 save=on
评测 8901 条（干净集）| 1.2 ms/条 | ECE=0.1134
| 方法 | 准确率 | Macro-F1 | 解析失败 |
|---|---|---|---|
| RoBERTa 微调 | 0.5633 | 0.5518 | 0 |

[cfg] pool=cls loss=CE+LS(0.1) class_weight=on fgm=off epochs=3 steps/epoch=834 eval_n=10000 save=on
评测 8901 条（干净集）| 1.2 ms/条 | ECE=0.0284
| 方法 | 准确率 | Macro-F1 | 解析失败 |
|---|---|---|---|
| RoBERTa 微调 | 0.5648 | 0.552 | 0 |

[cfg] pool=cls loss=Focal(gamma=2.0) class_weight=on fgm=off epochs=3 steps/epoch=834 eval_n=10000 save=on
评测 8901 条（干净集）| 1.2 ms/条 | ECE=0.0378
| 方法 | 准确率 | Macro-F1 | 解析失败 |
|---|---|---|---|
| RoBERTa 微调 | 0.5644 | 0.553 | 0 |

[cfg] pool=cls loss=CE class_weight=on fgm=on(eps=1.0) epochs=3 steps/epoch=834 eval_n=10000 save=on
评测 8901 条（干净集）| 1.2 ms/条 | ECE=0.0969
| 方法 | 准确率 | Macro-F1 | 解析失败 |
|---|---|---|---|
| RoBERTa 微调 | 0.5609 | 0.5487 | 0 |

[cfg] pool=max loss=CE class_weight=on fgm=off epochs=5 steps/epoch=834 eval_n=10000 save=on
评测 8901 条（干净集）| 1.2 ms/条 | ECE=0.1475
| 方法 | 准确率 | Macro-F1 | 解析失败 |
|---|---|---|---|
| RoBERTa 微调 | 0.567 | 0.5578 | 0 |

[cfg] pool=mean loss=CE class_weight=on fgm=off epochs=5 steps/epoch=834 eval_n=10000 save=on
评测 8901 条（干净集）| 1.2 ms/条 | ECE=0.1472
| 方法 | 准确率 | Macro-F1 | 解析失败 |
|---|---|---|---|
| RoBERTa 微调 | 0.5704 | 0.5646 | 0 |