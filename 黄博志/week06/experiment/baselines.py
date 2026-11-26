"""
传统基线：关键词规则、TF-IDF + 线性模型。

这两个基线的作用不是追求高分，而是给“换预训练模型”一个参照：
  - 规则：几乎零成本，能看出任务本身有多依赖关键词；
  - TF-IDF：强传统基线，能看出预训练模型相对词袋特征到底提升了多少。
"""

import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import config

# 15 类的人工关键词。只根据类别语义写，不看测试集，避免“偷看答案”。
RULE_KEYWORDS = {
    "故事": ["故事", "传说", "民间", "寓言", "童话", "神话", "典故", "奇闻"],
    "文化": ["文化", "文物", "非遗", "书法", "绘画", "博物馆", "传统", "民俗", "诗词", "考古"],
    "娱乐": ["电影", "明星", "导演", "演员", "综艺", "演唱会", "剧组", "票房", "娱乐", "歌手"],
    "体育": ["比赛", "球队", "球员", "冠军", "联赛", "足球", "篮球", "奥运", "体育", "赛季"],
    "财经": ["经济", "金融", "银行", "理财", "贷款", "利率", "通胀", "财经", "GDP", "消费"],
    "房产": ["房价", "楼市", "房地产", "开发商", "购房", "二手房", "物业", "房产", "租房", "土地"],
    "汽车": ["汽车", "车型", "新车", "发动机", "车企", "轿车", "SUV", "新能源车", "驾照", "变速箱"],
    "教育": ["教育", "学校", "学生", "老师", "高考", "大学", "留学", "考试", "招生", "课程"],
    "科技": ["科技", "手机", "芯片", "人工智能", "互联网", "数码", "电脑", "5G", "华为", "苹果"],
    "军事": ["军事", "军队", "战机", "导弹", "海军", "空军", "武器", "国防", "士兵", "航母"],
    "旅游": ["旅游", "景区", "游客", "旅行", "酒店", "机票", "自驾", "攻略", "门票", "民宿"],
    "国际": ["国际", "美国", "俄罗斯", "日本", "欧洲", "外交", "联合国", "总统", "外长", "欧盟"],
    "证券": ["股票", "股市", "证券", "涨停", "跌停", "大盘", "个股", "牛市", "熊市", "A股"],
    "农业": ["农业", "农村", "农民", "种植", "养殖", "粮食", "农产品", "化肥", "猪肉", "秋收"],
    "电竞": ["电竞", "游戏", "战队", "选手", "英雄联盟", "王者荣耀", "版本", "副本", "玩家", "联赛"],
}


def _rule_predict_one(text: str) -> int:
    scores = []
    for i, lab in enumerate(config.LABELS):
        score = sum(1 for kw in RULE_KEYWORDS[lab] if kw in text)
        scores.append(score)
    best = int(np.argmax(scores))
    # 一个关键词都没命中时，归到“故事”这个最泛的类
    return best


def run_rule(train, test):
    t0 = time.perf_counter()
    pred = np.array([_rule_predict_one(r["sentence"]) for r in test], dtype=np.int64)
    infer = time.perf_counter() - t0
    return {
        "pred": pred,
        "prob": None,
        "train_seconds": 0.0,
        "infer_seconds": infer,
        "extra": {"n_keywords": sum(len(v) for v in RULE_KEYWORDS.values())},
    }


def _build_tfidf():
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=2,
        max_features=200000,
        sublinear_tf=True,
    )


def run_tfidf(train, test, kind: str = "svc"):
    """kind: 'svc' | 'nb' | 'lr'。"""
    train_texts = [r["sentence"] for r in train]
    test_texts = [r["sentence"] for r in test]
    y_train = np.array([r["label"] for r in train], dtype=np.int64)

    # 训练耗时 = 特征拟合 + 分类器拟合
    t0 = time.perf_counter()
    vectorizer = _build_tfidf()
    X_train = vectorizer.fit_transform(train_texts)
    if kind == "svc":
        clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, dual=True)
    elif kind == "nb":
        clf = MultinomialNB(alpha=0.1)
    elif kind == "lr":
        clf = LogisticRegression(
            C=4.0, max_iter=2000, class_weight="balanced", n_jobs=-1
        )
    else:
        raise ValueError(f"未知 TF-IDF 模型：{kind}")
    clf.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0

    # 推理耗时 = 测试集 transform + predict
    t1 = time.perf_counter()
    X_test = vectorizer.transform(test_texts)
    pred = clf.predict(X_test).astype(np.int64)
    prob = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
    infer_seconds = time.perf_counter() - t1

    return {
        "pred": pred,
        "prob": prob,
        "train_seconds": train_seconds,
        "infer_seconds": infer_seconds,
        "extra": {
            "feature_dim": int(X_train.shape[1]),
            "classifier": kind,
        },
    }
