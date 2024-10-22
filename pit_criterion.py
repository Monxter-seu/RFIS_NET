# Created on 2018/12,edited on 202\/10
# Author: Kaituo XU,Minxu Hua
# 损失函数

from itertools import permutations

import torch
import torch.nn.functional as F
import numpy as np

EPS = 1e-8

def end2end_test_loss(output, classifier_output0, left_label,
                       classifier_output1, right_label, confounder_classifier_output):
    #criterion1 = torch.nn.BCELoss()
    criterion3 = torch.nn.CrossEntropyLoss()
    left_label = left_label.view(-1)
    left_label = left_label.long()
    right_label = right_label.view(-1)
    right_label = right_label.long()
    #print(classifier_output1)
    #print(left_label.shape)
    #print(output.shape)
    #print('classifier_output==',classifier_output0.shape)
    # print('left_label',left_label)
    classifier_output0 = classifier_output0.view(classifier_output0.size(0)* classifier_output0.size(1),classifier_output0.size(2))
    loss_first_class = criterion3(classifier_output0,left_label)
    variance = torch.var(output,dim=1)
    loss_second_class =  torch.div(torch.sum(variance), right_label.size(0), rounding_mode='trunc')
    classifier_output1 = classifier_output1.view(classifier_output1.size(0)
                                                 * classifier_output1.size(1),classifier_output1.size(2))
    loss_third_class = criterion3(classifier_output1, right_label)
    confounder_classifier_output = confounder_classifier_output.view(confounder_classifier_output.size(0)
                                                           * confounder_classifier_output.size(1),confounder_classifier_output.size(2))
    loss_forth_class = criterion3(confounder_classifier_output,left_label)
    total_loss = loss_first_class + loss_second_class + 0*loss_third_class + loss_forth_class
                    
    return total_loss
    
def end2end_train_loss(output, classifier_output0, left_label,
                       classifier_output1, right_label):
    criterion1 = torch.nn.BCELoss()
    criterion3 = torch.nn.CrossEntropyLoss()
    right_label = right_label.view(-1)
    right_label = right_label.long()
    #print(classifier_output1)
    #print(right_label)
    #print(output.shape)
    # print('classifier_output==',classifier_output)
    # print('left_label',left_label)
    classifier_output0 = torch.squeeze(classifier_output0,dim=-1)
    loss_first_class = criterion1(classifier_output0,left_label)
    variance = torch.var(output,dim=1)
    loss_second_class =  torch.div(torch.sum(variance), right_label.size(0), rounding_mode='trunc')
    classifier_output1 = classifier_output1.view(classifier_output1.size(0)
                                                 * classifier_output1.size(1),classifier_output1.size(2))
    loss_third_class = criterion3(classifier_output1, right_label)
    
    total_loss = 0*loss_first_class + loss_second_class + loss_third_class
    return total_loss
    
    
def variance_loss(output,classifier_output,left_label):
    criterion1 = torch.nn.BCELoss()
    #print(output.shape)
    # print('classifier_output==',classifier_output)
    # print('left_label',left_label)
    classifier_output = torch.squeeze(classifier_output,dim=-1)
    loss_first_class = criterion1(classifier_output,left_label)
    variance = torch.var(output,dim=1)
    loss_second_class = torch.sum(variance)
    
    total_loss = loss_first_class + loss_second_class
    return total_loss
    
def new_loss(output1,output2,left_label,right_label,right_ratio):
    #source_first_class = source_label[:, 0].unsqueeze(1)
    #source_second_class = source_label[:, 1:7]
    #print('source_second_class=====', source_second_class)
    #estimate_first_class = estimate_label[:, 0].unsqueeze(1).float()
    #estimate_second_class = estimate_label[:, 1].long()
    #print('estimate_second_class', estimate_second_class)
    
    right_label = right_label.long()
    criterion1 = torch.nn.BCELoss()
    criterion2 = torch.nn.CrossEntropyLoss()

    loss_first_class = criterion1(output1, left_label.unsqueeze(1))
    #print('loss_first_class',loss_first_class)
    # print('output2.shape',output2.shape)
    # print('right_label.shape',right_label.shape)
    loss_second_class = criterion2(output2, right_label)
    #print('loss_second_class', loss_second_class)

    total_loss = loss_first_class + loss_second_class*right_ratio

    return total_loss
    
def confounder_loss(output1,output2,output3,left_label,right_label,right_ratio=1,output3_ratio=1):
    #source_first_class = source_label[:, 0].unsqueeze(1)
    #source_second_class = source_label[:, 1:7]
    #print('source_second_class=====', source_second_class)
    #estimate_first_class = estimate_label[:, 0].unsqueeze(1).float()
    #estimate_second_class = estimate_label[:, 1].long()
    #print('estimate_second_class', estimate_second_class)
    
    right_label = right_label.long()
    criterion1 = torch.nn.BCELoss()
    criterion2 = torch.nn.CrossEntropyLoss()

    loss_first_class = criterion1(output1, left_label.unsqueeze(1))
    #print('loss_first_class',loss_first_class)
    loss_second_class = criterion2(output2, right_label)
    #print('loss_second_class', loss_second_class)
    loss_confounder_class = criterion1(output3, left_label.unsqueeze(1))
    
    total_loss = loss_first_class + 0*loss_second_class*right_ratio+loss_confounder_class*output3_ratio

    return total_loss

def cal_loss(source, estimate_source, source_lengths):
    """
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B]
    """
    max_snr, perms, max_snr_idx = cal_si_snr_with_pit(source,
                                                      estimate_source,
                                                      source_lengths)
    loss = 0 - torch.mean(max_snr)
    reorder_estimate_source = reorder_source(estimate_source, perms, max_snr_idx)
    return loss, max_snr, estimate_source, reorder_estimate_source


def cal_si_snr_with_pit(source, estimate_source, source_lengths):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    assert source.size() == estimate_source.size()
    B, C, T = source.size()
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
    s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)  # [B, C, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]

    # Get max_snr of each utterance
    # permutations, [C!, C]
    perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
    # one-hot, [C!, C, C]
    index = torch.unsqueeze(perms, 2)
    perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
    # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
    snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])
    max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
    # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
    max_snr /= C
    return max_snr, perms, max_snr_idx


def reorder_source(source, perms, max_snr_idx):
    """
    Args:
        source: [B, C, T]
        perms: [C!, C], permutations
        max_snr_idx: [B], each item is between [0, C!)
    Returns:
        reorder_source: [B, C, T]
    """
    B, C, *_ = source.size()
    # [B, C], permutation whose SI-SNR is max of each utterance
    # for each utterance, reorder estimate source according this permutation
    max_snr_perm = torch.index_select(perms, dim=0, index=max_snr_idx)
    # print('max_snr_perm', max_snr_perm)
    # maybe use torch.gather()/index_select()/scatter() to impl this?
    reorder_source = torch.zeros_like(source)
    for b in range(B):
        for c in range(C):
            reorder_source[b, c] = source[b, max_snr_perm[b][c]]
    return reorder_source


def get_mask(source, source_lengths):
    """
    Args:
        source: [B, C, T]
        source_lengths: [B]
    Returns:
        mask: [B, 1, T]
    """
    B, _, T = source.size()
    mask = source.new_ones((B, 1, T))
    for i in range(B):
        mask[i, :, source_lengths[i]:] = 0
    return mask


if __name__ == "__main__":
    # torch.manual_seed(123)
    # B, C, T = 2, 3, 12
    # # fake data
    # source = torch.randint(4, (B, C, T))
    # estimate_source = torch.randint(4, (B, C, T))
    # source[1, :, -3:] = 0
    # estimate_source[1, :, -3:] = 0
    # source_lengths = torch.LongTensor([T, T-3])
    # print('source', source)
    # print('estimate_source', estimate_source)
    # print('source_lengths', source_lengths)
    #
    # loss, max_snr, estimate_source, reorder_estimate_source = cal_loss(source, estimate_source, source_lengths)
    # print('loss', loss)
    # print('max_snr', max_snr)
    # print('reorder_estimate_source', reorder_estimate_source)

    a = torch.ones(4, 2, 6)
    b = torch.ones(4, 2)
    losss = new_loss(a, b)
    print('new_loss===', losss)