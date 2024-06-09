# -*-coding:utf-8-*-
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

__all__ = ["lenet"]

class SelfAttention(nn.Module):
    def __init__(self, last_hidden_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.last_hidden_size = last_hidden_size
        self.hidden_size = hidden_size

        self.wq = nn.Linear(in_features=last_hidden_size, out_features=hidden_size, bias=False)
        self.wk = nn.Linear(in_features=last_hidden_size, out_features=hidden_size, bias=False)
        self.wv = nn.Linear(in_features=last_hidden_size, out_features=hidden_size, bias=False)

        

    def forward(self, h):
       
        q = self.wq(h)
        k = self.wk(h)
        v = self.wv(h)

        dk = q.size(-1)
        z = torch.mm(q, k.t()) / math.sqrt(dk)  
        beta = F.softmax(z, dim=1)
        st = torch.mm(beta, v)  

        
        return st


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv_1 = nn.Conv1d(in_channels=9, out_channels=9,stride = 1, kernel_size=1)
        
        self.attn = SelfAttention(last_hidden_size=180, hidden_size=180)
        
        self.fc_3 = nn.Linear(180, num_classes)

    def forward(self, x):
        
        out = F.relu(self.conv_1(x))
        
        out = out.view(out.size(0), -1)
        
        out = self.attn(out)
        
        out = self.fc_3(out)
        
        return out


def lenet(num_classes):
    return LeNet(num_classes=num_classes)
