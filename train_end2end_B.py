#!/usr/bin/env python
# Created on 2023/03
# Author: HUA
# trian网络

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

from sklearn.metrics import cohen_kappa_score

from g_mlp import mixNet,maskNet,confounderNet,end2end_train,end2end_test_B
from pit_criterion import new_loss,confounder_loss,end2end_train_loss,end2end_test_loss
from data import MyDataLoader, MyDataset
from customLoader import CustomDataset, End2EndCustomDataset,MultiDataset
from torch.utils.data import Dataset, DataLoader


def get_file_paths(aim):
        file_paths = []
        for root, dirs, files in os.walk(aim):
            for file in files:
                if file.endswith(".csv"):
                    file_paths.append(os.path.join(root, file))
        return file_paths

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
        model = end2end_test_B(N, B, H, P, X, R, C, 128)
        
    model = model.cuda()
    # checkpoint = torch.load('representer.pth')
    # model.net.mlp1.fc1.weight.data = checkpoint['net.fc1.weight']
    # model.net.mlp1.fc2.weight.data = checkpoint['net.fc2.weight']
    
    # for param in  model.net.mlp1.fc1.parameters():
        # param.requires_grad = False
    # for param in  model.net.mlp1.fc2.parameters():
        # param.requires_grad = False
    #加载confounder
    target_path = 'D:\\frequencyProcess\\confounder_dic'
    all_files = get_file_paths(target_path)
    all_confouders = []
    for file in all_files:
        data = np.loadtxt(file, delimiter=",", dtype=np.float32)
        data = torch.from_numpy(data).float()
        all_confouders.append(data)
    
    # 定义损失函数和优化器
    #criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    train_dataset = MultiDataset('D:\\frequencyProcess\\expMulti_10T\\tr')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    cv_dataset = MultiDataset('D:\\frequencyProcess\\expMulti_10T\\cv')
    cv_loader = DataLoader(cv_dataset, batch_size=32, shuffle=True)
    #cv_dataset = CustomDataset('D:\\frequencyProcess\\test\\cv\\mix')
    #cv_loader = DataLoader(cv_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_dataset = MultiDataset('D:\\frequencyProcess\\expMulti_10T\\tt')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    best_accuracy = 0.0
    best_model = None
    
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
        train_correct2 = 0
        total_samples = 0
        left_micro_f1_scores = []
        left_macro_f1_scores = []
        right_micro_f1_scores = []
        right_macro_f1_scores = []
        confounder_micro_f1_scores = []
        confounder_macro_f1_scores = []
        loss_first_list =[]
        loss_second_list = []
        loss_third_list =[]
        for data, left_label, right_label in tqdm(train_loader):
            #print(left_label)
            for confounder in all_confouders:
                data = data.cuda()
                confounder = confounder.cuda()
                #confounder = torch.zeros(128).cuda()
                left_label = left_label.cuda().float()
                right_label = right_label.cuda().float()
                classifier_output0,classifier_output1,confounder_output,output = model(data,confounder)
                loss = end2end_test_loss(output, classifier_output0, left_label, classifier_output1
                ,right_label, confounder_output)
                # if index%10 == 0:
                    # print('loss:====',loss_first,loss_second,loss_third)
                    # loss_first_list.append(loss_first)
                    # loss_second_list.append(loss_second)
                    # loss_third_list.append(loss_third)   
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)
                classifier_output0 = classifier_output0.view(classifier_output0.size(0)
                                                     * classifier_output0.size(1),classifier_output0.size(2))
                classifier_output1 = classifier_output1.view(classifier_output1.size(0)
                                                     * classifier_output1.size(1),classifier_output1.size(2))
                confounder_output = confounder_output.view(confounder_output.size(0)
                                                     * confounder_output.size(1),confounder_output.size(2))
                _, predicted0 = torch.max(classifier_output0, 1)
                _, predicted1 = torch.max(classifier_output1, 1)
                _, predicted2 = torch.max(confounder_output, 1)
                train_correct0 += (predicted0 == left_label.view(-1)).sum().item()
                train_correct2 += (predicted2 == left_label.view(-1)).sum().item()
                predicted0 = predicted0.view(-1,1)
                left_micro_f1 = f1_score(predicted0.cpu(), left_label.view(-1,1).cpu(), average='micro')
                left_macro_f1 = f1_score(predicted0.cpu(), left_label.view(-1,1).cpu(), average='macro')
                
                train_correct1 += (predicted1 == right_label.view(-1)).int().sum().item()
                predicted1 = predicted1.view(-1,1)
                right_macro_f1 = f1_score(predicted1.cpu(), right_label.view(-1,1).cpu(), average='macro')
                right_micro_f1 = f1_score(predicted1.cpu(), right_label.view(-1,1).cpu(), average='micro')
                
                
                predicted2 = predicted2.view(-1,1)
                confounder_micro_f1 = f1_score(predicted2.cpu(), left_label.view(-1,1).cpu(), average='micro')
                confounder_macro_f1 = f1_score(predicted2.cpu(), left_label.view(-1,1).cpu(), average='macro')

                left_micro_f1_scores.append(left_micro_f1)
                left_macro_f1_scores.append(left_macro_f1)
                right_micro_f1_scores.append(right_micro_f1)
                right_macro_f1_scores.append(right_macro_f1)
                confounder_micro_f1_scores.append(confounder_micro_f1)
                confounder_macro_f1_scores.append(confounder_macro_f1)
        
        # print('loss_first',loss_first_list)
        # print('loss_second',loss_second_list)
        # print('loss_third',loss_third_list)
        left_average_micro_f1 = np.mean(left_micro_f1_scores)
        left_average_macro_f1 = np.mean(left_macro_f1_scores)
        right_average_micro_f1 = np.mean(right_micro_f1_scores)
        right_average_macro_f1 = np.mean(right_macro_f1_scores)
        confounder_average_micro_f1 = np.mean(confounder_micro_f1_scores)
        confounder_average_macro_f1 = np.mean(confounder_macro_f1_scores)
        print("Left Average Micro-F1:", left_average_micro_f1)
        print("Left Average Macro-F1:", left_average_macro_f1)
        print("Right Average Micro-F1:", right_average_micro_f1)
        print("Right Average Macro-F1:", right_average_macro_f1)
        print("Confounder Average Micro-F1:", confounder_average_micro_f1)
        print("Confounder Average Macro-F1:", confounder_average_macro_f1)    
        
        train_loss /= (5*len(train_loader.dataset))
        train_accuracy0 = train_correct0 / (5*len(train_loader.dataset))
        train_accuracy1 = train_correct1 / (5*len(train_loader.dataset))
        train_accuracy2 = train_correct2 / (5*len(train_loader.dataset))
        print('============')
        print('train_loss', train_loss)
        print('train_accuracy0', train_accuracy0)
        print('train_accuracy1', train_accuracy1)  
        print('confounder_accuracy1', train_accuracy2,flush=True)  


        model.eval()
        cv_loss = 0
        cv_correct0 = 0
        cv_correct1 = 0
        cv_correct2 = 0
        left_micro_f1_scores = []
        left_macro_f1_scores = []
        right_micro_f1_scores = []
        right_macro_f1_scores = []
        kappa_scores = []
        con_kappa_scores = []
        # confounder_micro_f1_scores = []
        # confounder_macro_f1_scores = []
        with torch.no_grad():
            for data, left_label, right_label in tqdm(cv_loader):
                for confounder in all_confouders:
                    data = data.cuda()
                    confounder = confounder.cuda()
                    left_label = left_label.cuda().float()
                    right_label = right_label.cuda().float()
                    classifier_output0,classifier_output1,confounder_output,output = model(data,confounder)
                    loss = end2end_test_loss(output, classifier_output0, left_label, classifier_output1
                    ,right_label, confounder_output)
                    cv_loss += loss.item() * data.size(0)
                    classifier_output0 = classifier_output0.view(classifier_output0.size(0)
                                                         * classifier_output0.size(1),classifier_output0.size(2))
                    classifier_output1 = classifier_output1.view(classifier_output1.size(0)
                                                         * classifier_output1.size(1),classifier_output1.size(2))
                    confounder_output = confounder_output.view(confounder_output.size(0)
                                                         * confounder_output.size(1),confounder_output.size(2))
                    _, predicted0 = torch.max(classifier_output0, 1)
                    _, predicted1 = torch.max(classifier_output1, 1)
                    _, predicted2 = torch.max(confounder_output, 1)
                    cv_correct0 += (predicted0 == left_label.view(-1)).sum().item()
                    cv_correct2 += (predicted2 == left_label.view(-1)).sum().item()
                    predicted0 = predicted0.view(-1,1)
                    left_micro_f1 = f1_score(predicted0.cpu(), left_label.view(-1,1).cpu(), average='micro')
                    left_macro_f1 = f1_score(predicted0.cpu(), left_label.view(-1,1).cpu(), average='macro')
                    kappa_score = cohen_kappa_score(predicted0.cpu(), left_label.view(-1,1).cpu())

                    cv_correct1 += (predicted1 == right_label.view(-1)).int().sum().item()
                    predicted1 = predicted1.view(-1,1)
                    right_macro_f1 = f1_score(predicted1.cpu(), right_label.view(-1,1).cpu(), average='macro')
                    right_micro_f1 = f1_score(predicted1.cpu(), right_label.view(-1,1).cpu(), average='micro')
                    
                    
                    predicted2 = predicted2.view(-1,1)
                    confounder_micro_f1 = f1_score(predicted2.cpu(), left_label.view(-1,1).cpu(), average='micro')
                    confounder_macro_f1 = f1_score(predicted2.cpu(), left_label.view(-1,1).cpu(), average='macro')
                    con_kappa_score = cohen_kappa_score(predicted2.cpu(), left_label.view(-1,1).cpu())
                    left_micro_f1_scores.append(left_micro_f1)
                    left_macro_f1_scores.append(left_macro_f1)
                    right_micro_f1_scores.append(right_micro_f1)
                    right_macro_f1_scores.append(right_macro_f1)
                    confounder_micro_f1_scores.append(confounder_micro_f1)
                    confounder_macro_f1_scores.append(confounder_macro_f1)
                    kappa_scores.append(kappa_score)
                    con_kappa_scores.append(con_kappa_score)
                    
            cv_loss /= (2*len(cv_loader.dataset))
            cv_accuracy0 = cv_correct0 / (2*len(cv_loader.dataset))
            cv_accuracy1 = cv_correct1 / (2*len(cv_loader.dataset))
            cv_accuracy2 = cv_correct2 / (2*len(cv_loader.dataset))
            
            left_average_micro_f1 = np.mean(left_micro_f1_scores)
            left_average_macro_f1 = np.mean(left_macro_f1_scores)
            right_average_micro_f1 = np.mean(right_micro_f1_scores)
            right_average_macro_f1 = np.mean(right_macro_f1_scores)
            confounder_average_micro_f1 = np.mean(confounder_micro_f1_scores)
            confounder_average_macro_f1 = np.mean(confounder_macro_f1_scores)
            average_kappa_score = np.mean(kappa_scores)
            average_con_kappa_score = np.mean(con_kappa_scores)
            accuracy_score = confounder_average_micro_f1 + confounder_average_macro_f1 + average_con_kappa_score
            if accuracy_score>best_accuracy:
                best_accuracy = accuracy_score
                best_model = model.state_dict()
                
            print("Left Average Micro-F1:", left_average_micro_f1)
            print("Left Average Macro-F1:", left_average_macro_f1)
            print("kappa score",average_kappa_score)
            print("Right Average Micro-F1:", right_average_micro_f1)
            print("Right Average Macro-F1:", right_average_macro_f1)
            print("Confounder Average Micro-F1:", confounder_average_micro_f1)
            print("Confounder Average Macro-F1:", confounder_average_macro_f1)  
            print("average con kappa score", average_con_kappa_score)
            print('cv_loss', cv_loss)
            print('cv_accuracy0', cv_accuracy0)
            print('cv_accuracy1', cv_accuracy1)
            print('confounder_accuracy1', cv_accuracy2,flush=True)  
        

    model.load_state_dict(best_model)
    model.eval()
    tt_loss = 0
    tt_correct0 = 0
    tt_correct1 = 0
    tt_correct2 = 0
    left_micro_f1_scores = []
    left_macro_f1_scores = []
    right_micro_f1_scores = []
    right_macro_f1_scores = []
    kappa_scores = []
    con_kappa_scores = []
    # confounder_micro_f1_scores = []
    # confounder_macro_f1_scores = []
    with torch.no_grad():
        for data, left_label, right_label in tqdm(test_loader):
            for confounder in all_confouders:
                data = data.cuda()
                confounder = confounder.cuda()
                #confounder = torch.zeros(128).cuda()
                left_label = left_label.cuda().float()
                right_label = right_label.cuda().float()
                classifier_output0,classifier_output1,confounder_output,output = model(data,confounder)
                loss = end2end_test_loss(output, classifier_output0, left_label, classifier_output1
                ,right_label, confounder_output)
                tt_loss += loss.item() * data.size(0)
                classifier_output0 = classifier_output0.view(classifier_output0.size(0)
                                                     * classifier_output0.size(1),classifier_output0.size(2))
                classifier_output1 = classifier_output1.view(classifier_output1.size(0)
                                                     * classifier_output1.size(1),classifier_output1.size(2))
                confounder_output = confounder_output.view(confounder_output.size(0)
                                                     * confounder_output.size(1),confounder_output.size(2))
                _, predicted0 = torch.max(classifier_output0, 1)
                _, predicted1 = torch.max(classifier_output1, 1)
                _, predicted2 = torch.max(confounder_output, 1)
                tt_correct0 += (predicted0 == left_label.view(-1)).sum().item()
                tt_correct2 += (predicted2 == left_label.view(-1)).sum().item()
                predicted0 = predicted0.view(-1,1)
                left_micro_f1 = f1_score(predicted0.cpu(), left_label.view(-1,1).cpu(), average='micro')
                left_macro_f1 = f1_score(predicted0.cpu(), left_label.view(-1,1).cpu(), average='macro')
                kappa_score = cohen_kappa_score(predicted0.cpu(), left_label.view(-1,1).cpu())

                tt_correct1 += (predicted1 == right_label.view(-1)).int().sum().item()
                predicted1 = predicted1.view(-1,1)
                right_macro_f1 = f1_score(predicted1.cpu(), right_label.view(-1,1).cpu(), average='macro')
                right_micro_f1 = f1_score(predicted1.cpu(), right_label.view(-1,1).cpu(), average='micro')
                
                
                predicted2 = predicted2.view(-1,1)
                confounder_micro_f1 = f1_score(predicted2.cpu(), left_label.view(-1,1).cpu(), average='micro')
                confounder_macro_f1 = f1_score(predicted2.cpu(), left_label.view(-1,1).cpu(), average='macro')
                con_kappa_score = cohen_kappa_score(predicted2.cpu(), left_label.view(-1,1).cpu())
                left_micro_f1_scores.append(left_micro_f1)
                left_macro_f1_scores.append(left_macro_f1)
                right_micro_f1_scores.append(right_micro_f1)
                right_macro_f1_scores.append(right_macro_f1)
                confounder_micro_f1_scores.append(confounder_micro_f1)
                confounder_macro_f1_scores.append(confounder_macro_f1)
                kappa_scores.append(kappa_score)
                con_kappa_scores.append(con_kappa_score)
                
        tt_loss /= (2*len(test_loader.dataset))
        tt_accuracy0 = tt_correct0 / (2*len(test_loader.dataset))
        tt_accuracy1 = tt_correct1 / (2*len(test_loader.dataset))
        tt_accuracy2 = tt_correct2 / (2*len(test_loader.dataset))
        
        left_average_micro_f1 = np.mean(left_micro_f1_scores)
        left_average_macro_f1 = np.mean(left_macro_f1_scores)
        right_average_micro_f1 = np.mean(right_micro_f1_scores)
        right_average_macro_f1 = np.mean(right_macro_f1_scores)
        confounder_average_micro_f1 = np.mean(confounder_micro_f1_scores)
        confounder_average_macro_f1 = np.mean(confounder_macro_f1_scores)
        average_kappa_score = np.mean(kappa_scores)
        average_con_kappa_score = np.mean(con_kappa_scores)
        print("Left Average Micro-F1:", left_average_micro_f1)
        print("Left Average Macro-F1:", left_average_macro_f1)
        print("kappa score",average_kappa_score)
        print("Right Average Micro-F1:", right_average_micro_f1)
        print("Right Average Macro-F1:", right_average_macro_f1)
        print("Confounder Average Micro-F1:", confounder_average_micro_f1)
        print("Confounder Average Macro-F1:", confounder_average_macro_f1)  
        print("average con kappa score", average_con_kappa_score)
        print('tt_loss', tt_loss)
        print('tt_accuracy0', tt_accuracy0)
        print('tt_accuracy1', tt_accuracy1)
        print('confounder_accuracy1', tt_accuracy2,flush=True)  
        torch.save(model.state_dict(), "End2End_test.pth")
        print('==========PLANB_B')
       
    
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