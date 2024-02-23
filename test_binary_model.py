#!/usr/bin/env python
# Created on 2023/03
# Author: HUA
# 用来跑测试模型

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
from randconv import randconv

from g_mlp import gMLP,BinaryClassifier
from g_mlp import mixNet,maskNet
from pit_criterion import new_loss
from data import MyDataLoader, MyDataset
from conv_tasnet import TemporalConvNet
from customLoader import CustomDataset
from torch.utils.data import Dataset, DataLoader




group_size = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# print(tensor_read)
# print(tensor_read.shape)

if __name__ == "__main__":


    # with open(address+'/newWave'+str(index)+'-'+str(i)+'.csv', 'r') as file:
    # reader = csv.reader(file)
    # row = next(reader)
    # tensor_read = torch.from_numpy(np.array(row, dtype=np.float32))
    # N = 1
    # L = 20
    # B = 128
    # H = 32
    # P = 3
    # X = 8
    # R = 2
    # norm_type = 'gLN'
    # causal = 0
    # mask_nonlinear = 'relu'
    # C = 2
    M, N, L, T = 16, 1, 20, 12
    B, H, P, X, R, C, norm_type, causal = 128, 32, 3, 8, 1, 2, "gLN", False
    net_type = ''
    # 实例化模型

    #model = mixNet(N, B, H, P, X, R, C, 128)
    if net_type == 'mask':
        model = maskNet(N, B, H, P, X, R, C, 128)
    else:
        model = BinaryClassifier(128)
        
    model = model.cuda()

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    #train_dataset = CustomDataset('D:\\frequencyProcess\\testNewBiOneDay\\tr')
    #train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    #cv_dataset = CustomDataset('D:\\frequencyProcess\\D10\\cv')
    #cv_loader = DataLoader(cv_dataset, batch_size=16, shuffle=True, num_workers=2)
    #cv_dataset = CustomDataset('D:\\frequencyProcess\\test\\cv\\mix')
    #cv_loader = DataLoader(cv_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_dataset = CustomDataset('D:\\frequencyProcess\\testModel\\tt')
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=2)
    
    # # 训练模型
    # num_epochs = 200
    # for epoch in range(num_epochs):
        # # 训练模式
        # #start_time = time.time()
        # #count = 0
        # model.train()
        # train_loss = 0
        # train_correct0 = 0
        # train_correct1 = 0
        # total_samples = 0
        # left_micro_f1_scores = []
        # left_macro_f1_scores = []
        # # print('train_loader.shape',train_loader.shape)
        # for data, left_label, right_label in tqdm(train_loader):
            # data = data.cuda()
            # left_label = left_label.cuda().float()
            # outputs0 = model(data)
            # loss = criterion(outputs0, left_label.unsqueeze(1))
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # train_loss += loss.item() * data.size(0)
            # predicted0 = (outputs0 > 0.5).float()
            # train_correct0 += (predicted0 == left_label.unsqueeze(1)).sum().item()
            # # print(predicted0)
            # # print(left_label)
            # left_micro_f1 = f1_score(predicted0.cpu(), left_label.unsqueeze(1).cpu(), average='micro')
            # left_macro_f1 = f1_score(predicted0.cpu(), left_label.unsqueeze(1).cpu(), average='macro')
            # left_micro_f1_scores.append(left_micro_f1)
            # left_macro_f1_scores.append(left_macro_f1)
            
        # left_average_micro_f1 = np.mean(left_micro_f1_scores)
        # left_average_macro_f1 = np.mean(left_macro_f1_scores)
        # print("Left Average Micro-F1:", left_average_micro_f1)
        # print("Left Average Macro-F1:", left_average_macro_f1)
        # train_loss /= len(train_loader.dataset)
        # train_accuracy0 = train_correct0 / len(train_loader.dataset)
        # print('============')
        # print('train_loss', train_loss)
        # print('train_accuracy0', train_accuracy0)
        
        
        # model.eval()
        # cv_loss = 0
        # cv_correct0 = 0
        # cv_correct1 = 0
        # total_samples = 0
        # left_micro_f1_scores = []
        # left_macro_f1_scores = []
        # with torch.no_grad():
            # for data, left_label, right_label in tqdm(cv_loader):
                # data = data.cuda()
                # left_label = left_label.cuda().float()
                # outputs0 = model(data)
                # loss = criterion(outputs0, left_label.unsqueeze(1))
                # cv_loss += loss.item() * data.size(0)
                # predicted0 = (outputs0 > 0.5).float()
                # cv_correct0 += (predicted0 == left_label.unsqueeze(1)).sum().item()
                # left_micro_f1 = f1_score(predicted0.cpu(), left_label.unsqueeze(1).cpu(), average='micro')
                # left_macro_f1 = f1_score(predicted0.cpu(), left_label.unsqueeze(1).cpu(), average='macro')
                # left_micro_f1_scores.append(left_micro_f1)
                # left_macro_f1_scores.append(left_macro_f1)
                
            # cv_loss /= len(cv_loader.dataset)
            # cv_accuracy0 = cv_correct0 / len(cv_loader.dataset)
            # left_average_micro_f1 = np.mean(left_micro_f1_scores)
            # left_average_macro_f1 = np.mean(left_macro_f1_scores)
            # print("Left Average Micro-F1:", left_average_micro_f1)
            # print("Left Average Macro-F1:", left_average_macro_f1) 
            # print('cv_loss', cv_loss)
            # print('cv_accuracy0', cv_accuracy0)
            # print('==========')
        
    checkpoint = torch.load('Classifier_mlp.pth')
    model.load_state_dict(checkpoint)
    model.eval()
    tt_loss = 0
    tt_correct0 = 0
    tt_correct1 = 0
    left_micro_f1_scores = []
    left_macro_f1_scores = []
    with torch.no_grad():
        for data, left_label, right_label in tqdm(test_loader):
            data = data.cuda()
            image = torch.randn(1, 3, 128)
            output = randconv(image, 10, False, 1.0)
            left_label = left_label.cuda().float()
            outputs0 = model(data)
            loss = criterion(outputs0, left_label.unsqueeze(1))
            tt_loss += loss.item() * data.size(0)
            predicted0 = (outputs0 > 0.5).float()
            tt_correct0 += (predicted0 == left_label.unsqueeze(1)).sum().item()
            left_micro_f1 = f1_score(predicted0.cpu(), left_label.unsqueeze(1).cpu(), average='micro')
            left_macro_f1 = f1_score(predicted0.cpu(), left_label.unsqueeze(1).cpu(), average='macro')
            left_micro_f1_scores.append(left_micro_f1)
            left_macro_f1_scores.append(left_macro_f1)        
        tt_loss /= len(test_loader.dataset)
        tt_accuracy0 = tt_correct0 / len(test_loader.dataset)
        tt_accuracy1 = tt_correct1 / len(test_loader.dataset)
        left_average_micro_f1 = np.mean(left_micro_f1_scores)
        left_average_macro_f1 = np.mean(left_macro_f1_scores)
        print("Left Average Micro-F1:", left_average_micro_f1)
        print("Left Average Macro-F1:", left_average_macro_f1)
        print('tt_loss', tt_loss)
        print('tt_accuracy0', tt_accuracy0)
        #torch.save(model.state_dict(), "Classifier.pth")
        print('==========')
       
    # # 测试模式
    # model.eval()
    # test_loss = 0
    # test_correct = 0
    # with torch.no_grad():
    #     for data, labels in device_loader:
    #         # 将数据和标签转换为张量
    #         data = data.float()
    #         labels = labels.float()
    #
    #         # 向前传递
    #         outputs = model(data)
    #         loss = criterion(outputs, labels.unsqueeze(1))
    #
    #         test_loss += loss.item() * data.size(0)
    #         predicted = (outputs > 0.5).float()
    #         # print(predicted)
    #         test_correct += (predicted == labels.unsqueeze(1)).sum().item()
    #         test_loss /= len(test_loader.dataset)
    #         test_accuracy = test_correct / len(device_loader.dataset)
    # print('test_accuracy', test_accuracy)
    # torch.save(model.state_dict(), "speClassifier.pth")
    # print('below is device accuracy')