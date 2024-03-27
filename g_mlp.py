# Created on 2023/10
# Author: Minxu Hua
# the value of C must be 2 in this case

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random

EPS = 1e-8
NET = 'transformer'

#
class dayShiftNet(nn.Module):
    def __init__(self):
        super(dayShiftNet, self).__init__()
        self.net = mlpNet()
        self.classifier0 = BinaryClassifier(128)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
    
    def forward(self, mixture):
        output = self.net(mixture)
        classifier_output = self.classifier0(output)
        return output,classifier_output

class testMixNet(nn.Module):
    def __init__(self):
        super(testMixNet, self).__init__()
        self.mixstyle = MixStyle(p=0.5, alpha=0.1)
        self.classifier0 = MultiClassifier(0, 128,10)
    
    def forward(self, mixture):
        #print('mixture_shape',mixture.shape)
        output = self.mixstyle(mixture)
        classifier_output = self.classifier0(output)
        return classifier_output


class testLSTMNet(nn.Module):
    def __init__(self):
        super(testLSTMNet, self).__init__()
        self.net = nn.LSTM(128, 128, 2, batch_first=True)
        self.classifier0 = MultiClassifier(0, 128,10)
    
    def forward(self, mixture):
        #print('mixture_shape',mixture.shape)
        output,( hn,cn) = self.net(mixture)
        classifier_output = self.classifier0(output)
        return classifier_output
        
class end2end_test_B(nn.Module):
    def __init__(self, N, B, H, P, X, R, C, K, norm_type="gLN", causal=False,
                 mask_nonlinear='relu'):
        super(end2end_test_B, self).__init__()    
        self.norm_type = norm_type
        self.causal = causal
        self.mask_nonlinear = mask_nonlinear
        self.representation = mlpNet()
        #self.representation = TransformerLayer(d_model=128,nhead=8)
        #self.representation = LSTMModel()
        #self.representation = CNNLayer()
        self.contextExtraction = mlpNet()
        self.confounderNet = mlpNet()
        self.fingerprintExtraction = mlpNet()
        #self.fingerprintExtraction = TransformerLayer(d_model=128,nhead=8)
        #self.fingerprintExtraction = LSTMModel()
        #self.fingerprintExtraction = CNNLayer()
        #self.net = TemporalConvNet(N, B, H, P, X, R, C, norm_type, causal, mask_nonlinear)
        #self.net = mlpNet()
        self.classifier0 = MultiClassifier(0, 128,10)
        self.classifier1 = MultiClassifier(0, 128,6)
        self.confounderClassifier = MultiClassifier(0, 128,10)
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture, confounder):
        feature = self.representation(mixture)
        fingerFeature = self.fingerprintExtraction(feature)
        contextFeature =self.contextExtraction(feature)
        confounderFeature = fingerFeature + confounder
        classifier_output_confounder = self.confounderClassifier(confounderFeature)
        classifier_output0 = self.classifier0(fingerFeature)
        classifier_output1 = self.classifier1(contextFeature)
        return classifier_output0, classifier_output1,classifier_output_confounder, feature
        
    def case_study(self, mixture):
        mixture_1 = self.representation(mixture)
        return mixture_1
        
    def cal_confounder(self, mixture):
        mixture_1 = self.representation(mixture)
        #print(mixture_1.shape)
        #[bs,2,1,128]
        confounder = self.contextExtraction(mixture_1)
        return confounder
        
#end2end训练模型(只输出confounder，不加入coufounder)
class end2end_train(nn.Module):
    def __init__(self, N, B, H, P, X, R, C, K, norm_type="gLN", causal=False,
                 mask_nonlinear='relu'):
        super(end2end_train, self).__init__()    
        self.norm_type = norm_type
        self.causal = causal
        self.mask_nonlinear = mask_nonlinear
        self.representation = mlpNet()
        self.contextExtraction = mlpNet()
        self.fingerprintExtraction = mlpNet()
        #self.net = TemporalConvNet(N, B, H, P, X, R, C, norm_type, causal, mask_nonlinear)
        #self.net = mlpNet()
        self.classifier0 = BinaryClassifier(128)
        self.classifier1 = MultiClassifier(0, 128,6)
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture):
        feature = self.representation(mixture)
        fingerFeature = self.fingerprintExtraction(feature)
        contextFeature =self.contextExtraction(feature)
        classifier_output0 = self.classifier0(fingerFeature)
        classifier_output1 = self.classifier1(contextFeature)
        return classifier_output0, classifier_output1, feature
        
    def cal_confounder(self, mixture):
        mixture_1 = self.representation(mixture)
        #print(mixture_1.shape)
        #[bs,2,1,128]
        confounder = self.contextExtraction(mixture_1)
        return confounder
        
#新网络用来计算confounder以及进行推理
class confounderNet(nn.Module):
    def __init__(self, N, B, H, P, X, R, C, K, norm_type="gLN", causal=False,
                 mask_nonlinear='relu'):
        super(confounderNet, self).__init__()
        self.N, self.B, self.H, self.P, self.X, self.R, self.C, self.K = N, B, H, P, X, R, C, K
        self.norm_type = norm_type
        self.causal = causal
        self.mask_nonlinear = mask_nonlinear
        self.net = RepresentLayer('mlp','mlp')
        #self.net = TemporalConvNet(N, B, H, P, X, R, C, norm_type, causal, mask_nonlinear)
        #self.net = mlpNet()
        self.classifier0 = BinaryClassifier(128)
        self.classifier1 = MultiClassifier(0, 128,6)
        self.classifier2 = BinaryClassifier(128)
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)


    def forward(self, mixture, confounder):
        #mixture = mixture.unsqueeze(1)
        mixture_1 = self.net(mixture)
        #print(mixture_1.shape)
        #[bs,2,1,128]
        mixture_2 = mixture_1.squeeze(dim=2)
        channel_0 = mixture_2[:, 0, :]
        channel_1 = mixture_2[:, 1, :]
        classifier_output0 = self.classifier0(channel_0)
        #classifier_output0 = self.classifier0(mixture_1)
        #self.confounder = channel_1
        channel_2 = channel_0 + torch.unsqueeze(confounder, 0).expand(channel_0.size(0), -1)
        classifier_output1 = self.classifier1(channel_1)
        classifier_output2 = self.classifier2(channel_2)
        #classifier_output0 = self.classifier0(mixture)
        #classifier_output1 = self.classifier1(mixture)
        #combined_classifier_output = torch.cat((classifier_output0, classifier_output1), dim=1)
        #return combined_classifier_output
        return classifier_output0, classifier_output1, classifier_output2
        
        
class maskNet(nn.Module):
    def __init__(self, N, B, H, P, X, R, C, K, norm_type="gLN", causal=False,
                 mask_nonlinear='relu'):
        super(maskNet, self).__init__()
        self.N, self.B, self.H, self.P, self.X, self.R, self.C, self.K = N, B, H, P, X, R, C, K
        self.norm_type = norm_type
        self.causal = causal
        self.mask_nonlinear = mask_nonlinear
        self.net = mlpNet()
        self.sigmoid = nn.Sigmoid()
        #self.net = TemporalConvNet(N, B, H, P, X, R, C, norm_type, causal, mask_nonlinear)
        #self.net = mlpNet()
        self.classifier0 = BinaryClassifier(128)
        self.classifier1 = MultiClassifier(0, 128,6)
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)


    def forward(self, mixture):
        #mixture = mixture.unsqueeze(1)
        mixture_1 = self.net(mixture)
        #print(mixture_1.shape)
        #[bs,2,1,128]
        channel_0_mask = self.sigmoid(mixture_1)
        channel_1_mask = torch.ones(channel_0_mask.size(0),channel_0_mask.size(1)).cuda()-channel_0_mask
        channel_0 = mixture*channel_0_mask
        channel_1 = mixture*channel_1_mask
        classifier_output0 = self.classifier0(channel_0)
        #classifier_output0 = self.classifier0(mixture_1)
        classifier_output1 = self.classifier1(channel_1)
        #classifier_output0 = self.classifier0(mixture)
        #classifier_output1 = self.classifier1(mixture)
        #combined_classifier_output = torch.cat((classifier_output0, classifier_output1), dim=1)
        #return combined_classifier_output
        return classifier_output0,classifier_output1

class mixNet(nn.Module):
    def __init__(self, N, B, H, P, X, R, C, K, norm_type="gLN", causal=False,
                 mask_nonlinear='relu'):
        super(mixNet, self).__init__()
        self.N, self.B, self.H, self.P, self.X, self.R, self.C, self.K = N, B, H, P, X, R, C, K
        self.norm_type = norm_type
        self.causal = causal
        self.mask_nonlinear = mask_nonlinear
        self.net = RepresentLayer('mlp','mlp')
        #self.net = TemporalConvNet(N, B, H, P, X, R, C, norm_type, causal, mask_nonlinear)
        #self.net = mlpNet()
        self.classifier0 = BinaryClassifier(128)
        self.classifier1 = MultiClassifier(0, 128,6)
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)


    def forward(self, mixture):
        #mixture = mixture.unsqueeze(1)
        mixture_1 = self.net(mixture)
        #print(mixture_1.shape)
        #[bs,2,1,128]
        mixture_2 = mixture_1.squeeze(dim=2)
        channel_0 = mixture_2[:, 0, :]
        channel_1 = mixture_2[:, 1, :]
        classifier_output0 = self.classifier0(channel_0)
        #classifier_output0 = self.classifier0(mixture_1)
        #self.confounder = channel_1
        classifier_output1 = self.classifier1(channel_1)
        #classifier_output0 = self.classifier0(mixture)
        #classifier_output1 = self.classifier1(mixture)
        #combined_classifier_output = torch.cat((classifier_output0, classifier_output1), dim=1)
        #return combined_classifier_output
        return classifier_output0,classifier_output1
    
    def cal_confounder(self, mixture):
        mixture_1 = self.net(mixture)
        #print(mixture_1.shape)
        #[bs,2,1,128]
        mixture_2 = mixture_1.squeeze(dim=2)
        channel_0 = mixture_2[:, 0, :]
        channel_1 = mixture_2[:, 1, :]
        return channel_1

class mlpNet(nn.Module):
    def __init__(self):
        super(mlpNet,self).__init__()
        self.fc1 = nn.Linear(128, 2048)
        self.fc2 = nn.Linear(2048, 128)
        self.relu = nn.ReLU()
        
    def forward(self,input):
        output = self.fc1(input)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        return output

class mlpNet1(nn.Module):
    def __init__(self):
        super(mlpNet1,self).__init__()
        self.fc1 = nn.Linear(120000, 2048)
        self.fc2 = nn.Linear(2048, 128)
        self.relu = nn.ReLU()
        
    def forward(self,input):
        output = self.fc1(input)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        return output

#整个表示层     
class RepresentLayer(nn.Module):
    def __init__(self, type1, type2):
        super(RepresentLayer, self).__init__()
        self.type1 = type1
        self.type2 = type2
        self.mlp1 = mlpNet()
        self.mlp2 = mlpNet()
        self.transformer = TransformerLayer(d_model=128,nhead=8)
        self.cnn = Simple1DCNN()
    def forward(self, input):
        if self.type1 == 'mlp':
            output1 = self.mlp1(input)
        elif self.type1 == 'transformer':
            output1 = self.transformer(input)
        elif self.type1 == 'cnn':
            output1 = self.cnn(input)
        
        if self.type2 == 'mlp':
            output_channel1 = self.mlp2(output1)
            output_channel2 = self.mlp2(output1)
        elif self.type2 == 'transformer':
            output_channel1 = self.transformer(output1)
            output_channel2 = self.transformer(output1)
        elif self.type2 == 'cnn':
            output_channel1 = self.cnn(output1)
            output_channel2 = self.cnn(output1)
        
        output = torch.cat((output_channel1.unsqueeze(dim=1),
                            output_channel2.unsqueeze(dim=1)),dim=1)
                            
        return output
            
class Simple1DCNN(nn.Module):
    def __init__(self):
        super(Simple1DCNN, self).__init__()
        
        # 第一个卷积层，输入通道数为1，输出通道数为32，卷积核大小为3
        self.conv1 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=16)
        
        # 第二个卷积层，输入通道数为32，输出通道数为64，卷积核大小为3
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16)
        
        # 池化层，池化窗口大小为2
        self.pool = nn.MaxPool1d(kernel_size=32, stride=2)
        
        # 全连接层，输入特征数为64*4（经过两次池化层），输出特征数为128
        self.fc1 = nn.Linear(1312, 128)
        
        #self.fc2 = nn.Linear(128, 128)
        

    def forward(self, x):
        #[bs,128] bs=16
        
        x = x.unsqueeze(dim=2)
        #print('x.shape',
        x = x.permute(1, 0, 2)
        #[128,bs,1]
        #x = self.conv1(x)
        #print('x.shape',x.shape)
        print('x.shape',x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        #x = self.pool(F.relu(self.conv1(x)))
        x = x.permute(1, 0, 2)
        #x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 1312)
        #print('x.shape',x.shape)
        #print('x.shape',x.shape)
        x = F.relu(self.fc1(x))        
        return x
        
class CNNLayer(nn.Module):
    def __init__(self, input_size=128, num_classes=128):
        super(CNNLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x
        
        
class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerLayer, self).__init__()
        
        # Self-Attention 层
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        
        # 前馈神经网络层
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer Normalization
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-Attention 层
        attn_output, _ = self.self_attn(x, x, x)
        
        # 残差连接和 Layer Normalization
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # 前馈神经网络层
        ff_output = self.feedforward(x)
        
        # 残差连接和 Layer Normalization
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x      
        

        
        
class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)
            #print('perm',perm.shape)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            #print('perm_a',perm_a.shape)
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)
        output = x_normed*sig_mix + mu_mix
        #print('out.shape',output.shape)
        output1 = output[0,:,:]
        output2 = output1.squeeze(dim=0)
        return output2
        
        



        


class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.input_dim = 128
        self.hidden_dim = 256
        self.output_dim = 128
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        #x = x.unsqueeze(1)
        #print(x.shape)
        out, _ = self.lstm(x)
        # out: (batch_size, seq_len, hidden_dim)
        out = self.fc(out.squeeze(1))
        #print(out.shape)
        # out: (batch_size, output_dim)
        return out

# define Biclassifier
class BinaryClassifier(nn.Module):
    def __init__(self,length):
        super(BinaryClassifier, self).__init__()
        # 定义三个全连接层
        # self.fc1 = nn.Linear(2048, 256)
        self.fc1 = nn.Linear(length, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        # self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, 1)
        # self.fc2_1 = nn.Linear(256, 256)
        #self.fc3 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.fc4 = nn.Linear(2048,1)


        # 定义ReLU和Sigmoid激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # group_s = 500
        # x = x.reshape(x.size(0), group_s, -1)
        # # 对第二个维度执行FFT
        # x = torch.fft.fftn(x, dim=2)
        # x = torch.abs(x)
        # # 计算形状为[B,n, 128]的张量的平均值
        # x = torch.mean(x, dim=1)
        # 应用第一个全连接层和ReLU激活函数
        x = self.fc1(x)
        x = self.relu(x)
        # 应用第二个全连接层和ReLU激活函数
        x = self.fc2(x)
        x = self.relu(x)
        # 应用第三个全连接层和ReLU激活函数
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x


# define Multiclassifier
class MultiClassifier(nn.Module):
    def __init__(self, length, fft_length,classNum):
        super(MultiClassifier, self).__init__()
        self.fft_length = fft_length
        self.length = length
        self.group_size = 128
        # 定义三个全连接层
        # self.fc1 = nn.Linear(fft_length, 256)
        self.fc1 = nn.Linear(fft_length, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, classNum)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # # 应用第一个全连接层和ReLU激活函数
        # group_num = self.length // self.group_size
        # num_elements = group_num * self.group_size
        #
        # # 将张量重塑为形状为[B,n, 128]的张量
        # x = x.reshape(x.size(0), group_num, -1)
        #
        # # 对第二个维度执行FFT
        # x = torch.fft.fftn(x, dim=2)
        # x = torch.abs(x)
        # # 计算形状为[B,n, 128]的张量的平均值
        # x = torch.mean(x, dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        # 应用第二个全连接层和ReLU激活函数
        x = self.fc2(x)
        x = self.relu(x)
        # 应用第三个全连接层和ReLU激活函数
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    
    torch.manual_seed(123)
    M, N, L, T = 16, 1, 4, 12
    K = 2*T//L-1
    B, H, P, X, R, C, norm_type, causal = 1, 3, 3, 3, 2, 2, "gLN", False
    # mixture = torch.randint(3, (M, T), dtype=torch.float)
    # # test Encoder
    # encoder = Encoder(L, N)
    # encoder.conv1d_U.weight.data = torch.randint(2, encoder.conv1d_U.weight.size(), dtype=torch.float)
    # with torch.no_grad():
        # mixture_w = encoder(mixture)
    # print('mixture', mixture)
    # print('U', encoder.conv1d_U.weight)
    # print('mixture_w', mixture_w)
    # print('mixture_w size', mixture_w.size())

    num_tokens=100
    bs=16
    #test
    len_sen=64000
    num_layers=6
    input = torch.randint(num_tokens, (bs, len_sen), dtype=torch.float).cuda() #bs,len_sen

    # g = make_dot(output.mean(), params=dict(gmlp.named_parameters()), show_attrs=True, show_saved=True)
    # g = make_dot(output.mean())
    # g.view()
    se_input = torch.randint(100, (bs,128),dtype=torch.float).cuda()
    print(se_input.shape)
    separator = mixNet(N, B, H, P, X, R, C, 128).cuda()
    out = separator(se_input)
    print('out.shape====',out)