#从target_path中提取confounder
import torch
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

from g_mlp import gMLP
from g_mlp import mixNet,maskNet,end2end_train
from pit_criterion import new_loss
from data import MyDataLoader, MyDataset
from conv_tasnet import TemporalConvNet
from customLoader import CustomDataset
from torch.utils.data import Dataset, DataLoader

target_path = 'D:\\frequencyProcess\\confounder\\port4'

def get_file_paths(aim):
        file_paths = []
        for root, dirs, files in os.walk(aim):
            for file in files:
                if file.endswith(".csv"):
                    file_paths.append(os.path.join(root, file))
        return file_paths
        
if __name__ == "__main__":
    M, N, L, T = 16, 1, 20, 12
    B, H, P, X, R, C, norm_type, causal = 128, 32, 3, 8, 1, 2, "gLN", False
    model = end2end_train(N, B, H, P, X, R, C, 128)
    model = model.cuda()
    checkpoint = torch.load('End2End_train.pth')
    model.load_state_dict(checkpoint)
    model.eval()
    all_files = get_file_paths(target_path)
    all_tensors = []
    all_confouders = []
    for file in all_files:
        data = np.loadtxt(file, delimiter=",", dtype=np.float32)
        data = torch.from_numpy(data).float()
        data = data.unsqueeze(dim = 0)
        all_tensors.append(data)
        
    for tensor in all_tensors:
        tensor = tensor.cuda()
        output = model.cal_confounder(tensor)
        #print('output.shape',output.shape)
        all_confouders.append(output)
        
    stacked_tensor = torch.stack(all_confouders)
    print('stack_tensor.shape',stacked_tensor.shape)
    confounder = tensor.mean(dim = 0)
    print('confounder.shape',confounder.shape)
    
    with open('port4.csv', 'w+',newline='') as file1:
        writer = csv.writer(file1)
        writer.writerow(confounder.cpu().numpy())