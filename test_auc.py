import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

# 示例多分类标签
y_true_multiclass = np.array([0, 1, 2, 1, 0, 2, 2])

# 将多分类标签转换为二进制矩阵
y_true_binary = label_binarize(y_true_multiclass, classes=np.unique(y_true_multiclass))
print(y_true_binary)
# 假设有一些预测概率，这里用随机数模拟
y_probs_binary = np.random.rand(len(y_true_multiclass), np.max(y_true_multiclass) + 1)
print(y_probs_binary)

# 计算 ROC AUC
roc_auc = roc_auc_score(y_true_binary, y_probs_binary, average='macro')

print(f"ROC AUC: {roc_auc}")