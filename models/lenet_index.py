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
        self.conv_1 = nn.Conv2d(5, 6, 5)
        
        self.conv_2 = nn.Conv2d(6, 16, 5)
        self.fc_1 = nn.Linear(16 * 2 * 2, 120)
        
        self.fc_2 = nn.Linear(120, 84)
        
        self.dropout = nn.Dropout(p=0.5)  # dropout训练
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv_1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv_2(out))
        
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        

        out = F.relu(self.fc_1(out))
        
        out = F.relu(self.fc_2(out))
        
        out = self.fc_3(out)
        
        return out


def lenet(num_classes):
    return LeNet(num_classes=num_classes)
