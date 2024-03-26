#用来画t-SNE特征图
import torch
from sklearn.manifold import TSNE
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

from sklearn.metrics import cohen_kappa_score

from g_mlp import mixNet,maskNet,confounderNet,end2end_train,end2end_test_B
from pit_criterion import new_loss,confounder_loss,end2end_train_loss,end2end_test_loss
from data import MyDataLoader, MyDataset
from conv_tasnet import TemporalConvNet
from customLoader import CustomDataset, End2EndCustomDataset,MultiDataset,SimpleMultiDataset
from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":
    M, N, L, T = 16, 1, 20, 12
    B, H, P, X, R, C, norm_type, causal = 128, 32, 3, 8, 1, 2, "gLN", False
    data_list = []
    transformed_data_list = []
    label_list = []
    cv_dataset = SimpleMultiDataset('D:\\frequencyProcess\\expMulti_10T\\cv')
    cv_loader = DataLoader(cv_dataset, batch_size=1)
    model = end2end_test_B(N, B, H, P, X, R, C, 128)
    model = model.cpu()
    checkpoint = torch.load('End2End_test.pth')
    model.load_state_dict(checkpoint)
    model.eval()
    
    for data, left_label, right_label in cv_loader:
        data = torch.squeeze(data)
        data_list.append(data.cpu())
        label_list.append(left_label.cpu())
        transformed_data = model.case_study(data)
        transformed_data_list.append(transformed_data.cpu())
        
    print(len(data_list))
    data = torch.stack(data_list)
    transformed_datas = torch.stack(transformed_data_list)
    labels = torch.stack(label_list).numpy()
    
    print(data.shape)
    # 初始化T-SNE模型
    tsne = TSNE(n_components=2, random_state=0)

    # 对数据进行降维
    embedded_data = tsne.fit_transform(data.numpy())
    embedded_transformed = tsne.fit_transform(transformed_datas.detach().numpy())
    
    print(embedded_data.shape)
    # 将数据按标签分组
    grouped_data = {}
    grouped_transformed = {}
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        grouped_data[label] = embedded_data[indices]
        grouped_transformed[label] = embedded_transformed[indices]

    for label, group in grouped_data.items():
        print(group)
        plt.scatter(group[:, 0], group[:, 1], label=f'Device {label}')

    plt.title('Raw data',fontsize=20, fontweight='bold')
    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')
    plt.legend()
    plt.show()

    for label, group in grouped_transformed.items():
        plt.scatter(group[:, 0], group[:, 1], label=f'Device {label}')

    plt.title('After extration',fontsize=20, fontweight='bold')
    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')
    plt.legend()
    plt.show()
