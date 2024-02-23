#推理速度

import torch
import time
import torch.nn as nn
import os
import numpy as np
import g_mlp
import librosa
import data
import pit_criterion
import torch.optim as optim
import csv
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import f1_score
from randconv import randconv

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from g_mlp import gMLP,BinaryClassifier,testMixNet,MultiClassifier
from g_mlp import mixNet,maskNet,testLSTMNet,end2end_test_B
from pit_criterion import new_loss
from data import MyDataLoader, MyDataset
from conv_tasnet import TemporalConvNet
from customLoader import CustomDataset ,MultiDataset,SimpleMultiDataset
from torch.utils.data import Dataset, DataLoader

M, N, L, T = 16, 1, 20, 12
B, H, P, X, R, C, norm_type, causal = 128, 32, 3, 8, 1, 2, "gLN", False
model = end2end_test_B(N, B, H, P, X, R, C, 128)

# 创建虚拟数据集
X_lda = np.random.rand(16, 128)
y_lda = np.random.randint(0, 4, 16)

# 将数据集划分为训练集和测试集

# 使用LDA进行特征降维
lda = LDA(n_components=2)  # 假设降维到4维
X_train_lda = lda.fit_transform(X_lda, y_lda)
#X_test_lda = lda.transform(X_test)

# 使用SVM进行多分类
svm = SVC(kernel='linear')
svm.fit(X_train_lda, y_lda)

#model = MultiClassifier(0,128,10)
#model = testMixNet()
#model = testLSTMNet()
model = model.cuda()
random_tensor = torch.randn(16,120000)
random_tensor = random_tensor.cuda()
confounder = torch.randn(128).cuda()
# 记录开始时间
start_time = time.time()

# 初始化推理次数计数器
inference_count = 0

# 在一分钟内多次运行推理操作
while (time.time() - start_time) < 60:
    # 进行推理
    #random_tensor = randconv(random_tensor, 5, False, 1.0)
    model.forward(random_tensor,confounder)
    
    
    #lda
    # X_test = lda.transform(X_lda)
    # y_pred = svm.predict(X_test)
    # 记录推理次数
    inference_count += 1

# 记录结束时间c
end_time = time.time()

# 计算一分钟内的推理次数
inference_rate = inference_count / (end_time - start_time)

print(f"模型一分钟内可以进行推理的次数为：{inference_count}次")
print(f"推理速率为：{inference_rate}次/秒")