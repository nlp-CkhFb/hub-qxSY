import json
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score

# ====== ：数据接入 ======
#直接读取本地的 json 文件，适配我的路径。
def load_my_data(file_path, max_n=None):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if max_n:
        data = data[:max_n]
    texts = [d["sentence"] for d in data]
    labels = [d["label"] for d in data]
    return texts, labels

print("加载本地数据...")
# 为了 CPU 跑得快，我只取前 2000 条训练，500条测试
X_train, y_train = load_my_data("train.json", max_n=2000)
X_test, y_test = load_my_data("val.json", max_n=500)

print("开始训练 TF-IDF + SVM...")
start_time = time.time()

# ======：特征提取方式 ======
# 老师PPT指出短文本字 n-gram 比词更稳。所以我将 analyzer 设为 'char'，并保留 (1,2) 抓取双字词。
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2), max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ====== 【学生修改点 3】：模型鲁棒性 ======
# 考虑到数据集可能存在类别不平衡，加入 class_weight='balanced' 提升长尾类表现。
svm = LinearSVC(class_weight='balanced', max_iter=2000)
svm.fit(X_train_vec, y_train)

# 预测与评估
preds = svm.predict(X_test_vec)
macro_f1 = f1_score(y_test, preds, average='macro')
acc = accuracy_score(y_test, preds)

print("-" * 30)
print(f"L1 (TF-IDF+SVM) 结果：")
print(f"Macro-F1: {macro_f1:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"耗时: {time.time() - start_time:.2f} 秒")
print("-" * 30)