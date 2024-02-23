#性能监测 CPU/GPU/Memory
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
import psutil
import GPUtil as GPU

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


def print_gpu_info():
    gpu = GPU.getGPUs()[0]
    print("GPU Name:", gpu.name)
    print("GPU Memory Total:", gpu.memoryTotal, "MB")
    print("GPU Memory Used:", gpu.memoryUsed, "MB")
    print("GPU Utilization:", gpu.load*100, "%")

def print_cpu_info():
    cpu_percent = psutil.cpu_percent()
    process = psutil.Process()
    memory_info = process.memory_info()
    print("CPU Utilization:", cpu_percent, "%")
    print("Memory Usage - RSS (Resident Set Size):", memory_info.rss / (1024 * 1024), "MB")
    print("Memory Usage - VMS (Virtual Memory Size):", memory_info.vms / (1024 * 1024), "MB")
    
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
random_tensor = torch.randn(16,128)
random_tensor = random_tensor.cuda()
confounder = torch.randn(128).cuda()
# 记录开始时间
# start_time = time.time()

# # 初始化推理次数计数器
# inference_count = 0

# 在一分钟内多次运行推理操作
# while (time.time() - start_time) < 60:
    # 进行推理
    #random_tensor = randconv(random_tensor, 5, False, 1.0)
print_cpu_info()
num_iterations = 5000  # 进行10次推理
for i in range(num_iterations):
    #random_tensor = randconv(random_tensor, 5, False, 1.0)
    #model.forward(random_tensor)
    #model.forward(random_tensor,confounder)
    X_test = lda.transform(X_lda)
    y_pred = svm.predict(X_test)
print_cpu_info()
for i in range(num_iterations):
    #random_tensor = randconv(random_tensor, 5, False, 1.0)
    #model.forward(random_tensor)
    #model.forward(random_tensor,confounder)
    X_test = lda.transform(X_lda)
    y_pred = svm.predict(X_test)    



    # #lda
    # X_test = lda.transform(X_lda)
    # y_pred = svm.predict(X_test)
