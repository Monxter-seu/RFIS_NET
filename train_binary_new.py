#!/usr/bin/env python
# Created on 2023/03
# Author: HUA
# 用来训练直接分类的效果

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

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef

from g_mlp import gMLP,BinaryClassifier,testMixNet,MultiClassifier
from g_mlp import mixNet,maskNet,testLSTMNet
from pit_criterion import new_loss
from data import MyDataLoader, MyDataset
from conv_tasnet import TemporalConvNet
from customLoader import CustomDataset ,MultiDataset,SimpleMultiDataset
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
        model = MultiClassifier(0,128,10)
        # model = testMixNet()
        # model = testLSTMNet()
        
    model = model.cuda()
    best_accuracy = 0.0
    best_model = None
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    train_dataset = SimpleMultiDataset('D:\\frequencyProcess\\expMulti_10T\\tr')
    #train_dataset = CustomDataset('D:\\frequencyProcess\\testNewBiOneDay\\tr')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    #cv_dataset = CustomDataset('D:\\frequencyProcess\\D10\\cv')
    #cv_loader = DataLoader(cv_dataset, batch_size=16, shuffle=True, num_workers=2)
    cv_dataset = SimpleMultiDataset('D:\\frequencyProcess\\expMulti_10T\\cv')
    cv_loader = DataLoader(cv_dataset, batch_size=32, shuffle=True)
    test_dataset = SimpleMultiDataset('D:\\frequencyProcess\\expMulti_10T\\tt')
    #test_dataset = CustomDataset('D:\\frequencyProcess\\testNewBiOneDay\\tt')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    # 训练模型
    num_epochs = 300
    for epoch in range(num_epochs):
        # 训练模式
        #start_time = time.time()
        #count = 0
        model.train()
        train_loss = 0
        train_correct0 = 0
        train_correct1 = 0
        total_samples = 0
        left_micro_f1_scores = []
        left_macro_f1_scores = []
        # print('train_loader.shape',train_loader.shape)
        for data, left_label, right_label in tqdm(train_loader):
            data = data.cuda()
            
            #rbstest
            chunks = torch.chunk(data, 4, dim=1)
            data = torch.cat([chunks[i] for i in torch.randperm(4)], dim=1)
            
            #randconv
            # data = randconv(data, 5, False, 1.0)
            
            left_label = left_label.long().cuda()
            
            outputs0 = model(data)
            loss = criterion(outputs0, left_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            _, predicted0 = torch.max(outputs0, 1)
            train_correct0 += (predicted0 == left_label.unsqueeze(1)).sum().item()
            # print(predicted0)
            # print(left_label)
            left_micro_f1 = f1_score(predicted0.cpu(), left_label.cpu(), average='micro')
            left_macro_f1 = f1_score(predicted0.cpu(), left_label.cpu(), average='macro')
            left_micro_f1_scores.append(left_micro_f1)
            left_macro_f1_scores.append(left_macro_f1)
            
        left_average_micro_f1 = np.mean(left_micro_f1_scores)
        left_average_macro_f1 = np.mean(left_macro_f1_scores)
        print("Left Average Micro-F1:", left_average_micro_f1)
        print("Left Average Macro-F1:", left_average_macro_f1)
        train_loss /= len(train_loader.dataset)
        train_accuracy0 = train_correct0 / len(train_loader.dataset)
        print('============')
        print('train_loss', train_loss)
        print('train_accuracy0', train_accuracy0)
        
        cv_loss = 0
        cv_correct0 = 0
        cv_correct1 = 0
        left_micro_f1_scores = []
        left_macro_f1_scores = []
        kappa_scores = []
        with torch.no_grad():
            for data, left_label, right_label in tqdm(cv_loader):
                data = data.cuda()
                left_label = left_label.long().cuda()                  
                outputs0 = model(data)
                loss = criterion(outputs0, left_label)
                cv_loss += loss.item() * data.size(0)
                _, predicted0 = torch.max(outputs0, 1)
                cv_correct0 += (predicted0 == left_label.unsqueeze(1)).sum().item()
                # print(predicted0)
                # print(left_label)
                left_micro_f1 = f1_score(predicted0.cpu(), left_label.cpu(), average='micro')
                left_macro_f1 = f1_score(predicted0.cpu(), left_label.cpu(), average='macro')
                kappa_score = cohen_kappa_score(predicted0.cpu(), left_label.cpu())
                #print(kappa_score)
                left_micro_f1_scores.append(left_micro_f1)
                left_macro_f1_scores.append(left_macro_f1)
                kappa_scores.append(kappa_score)
                
            #print(y_test_bin_total.shape)
            #print(y_probs_total.shape)
            cv_loss /= len(test_loader.dataset)
            cv_accuracy0 = cv_correct0 / len(test_loader.dataset)
            cv_accuracy1 = cv_correct1 / len(test_loader.dataset)
            left_average_micro_f1 = np.mean(left_micro_f1_scores)
            left_average_macro_f1 = np.mean(left_macro_f1_scores)
            average_kappa = np.mean(kappa_scores)
            accuracy_score = left_average_micro_f1 + left_average_macro_f1 + average_kappa
            print('score==',accuracy_score)
            if accuracy_score>best_accuracy:
                best_accuracy = accuracy_score
                best_model = model.state_dict()
            print("Left Average Micro-F1:", left_average_micro_f1)
            print("Left Average Macro-F1:", left_average_macro_f1)
            print("Average kappa", average_kappa)
            print('cv_loss', cv_loss)
            print('cv_accuracy0', cv_accuracy0)
            print('==========')
        
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
        
    model.load_state_dict(best_model)
    model.eval()
    tt_loss = 0
    tt_correct0 = 0
    tt_correct1 = 0
    left_micro_f1_scores = []
    left_macro_f1_scores = []
    kappa_scores = []
    with torch.no_grad():
        for data, left_label, right_label in tqdm(test_loader):
            data = data.cuda()
            left_label = left_label.long().cuda()                  
            outputs0 = model(data)
            loss = criterion(outputs0, left_label)
            tt_loss += loss.item() * data.size(0)
            _, predicted0 = torch.max(outputs0, 1)
            tt_correct0 += (predicted0 == left_label.unsqueeze(1)).sum().item()
            # print(predicted0)
            # print(left_label)
            left_micro_f1 = f1_score(predicted0.cpu(), left_label.cpu(), average='micro')
            left_macro_f1 = f1_score(predicted0.cpu(), left_label.cpu(), average='macro')
            kappa_score = cohen_kappa_score(predicted0.cpu(), left_label.cpu())
            #print(kappa_score)
            left_micro_f1_scores.append(left_micro_f1)
            left_macro_f1_scores.append(left_macro_f1)
            kappa_scores.append(kappa_score)
            
        #print(y_test_bin_total.shape)
        #print(y_probs_total.shape)
        tt_loss /= len(test_loader.dataset)
        tt_accuracy0 = tt_correct0 / len(test_loader.dataset)
        tt_accuracy1 = tt_correct1 / len(test_loader.dataset)
        left_average_micro_f1 = np.mean(left_micro_f1_scores)
        left_average_macro_f1 = np.mean(left_macro_f1_scores)
        average_kappa = np.mean(kappa_scores)
        
        print("Left Average Micro-F1:", left_average_micro_f1)
        print("Left Average Macro-F1:", left_average_macro_f1)
        print("Average kappa", average_kappa)
        print('tt_loss', tt_loss)
        print('tt_accuracy0', tt_accuracy0)
        torch.save(model.state_dict(), "Classifier_Mixstyle.pth")
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