import torch.nn as nn
import torch
import torch.nn.functional as F
from layers import GC_withres, NGCN, NGCNs, FC


class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)

    def forward(self, x):  # shape=[n, d]
        y = torch.matmul(self.Coefficient, x)
        return y


class GCN(nn.Module):
    def __init__(self, nfeat, para3, para4, para5, para6, nclass, dropout, smoo):
        super(GCN, self).__init__()

        self.gc1 = NGCN(nfeat, med_f0=500, med_f1=500, med_f2=500, med_f3=para3, med_f4=para4)
        #        self.gc1 = NGCN(nfeat,med_f0=28,med_f1=1,med_f2=1,med_f3=para3,med_f4=para4)
        self.gc2 = NGCN(1500 + para3 + para4, med_f0=100, med_f1=100, med_f2=100, med_f3=para5, med_f4=para6)
        self.gc3 = NGCN(138, med_f0=28, med_f1=10, med_f2=10, med_f3=para3, med_f4=para4)
        self.gc11 = GC_withres(300 + para5 + para6, nclass, smooth=smoo)
        self.gc12 = GC_withres(1880, nclass, smooth=smoo)
        self.dropout = dropout
        self.self_expression = SelfExpression(2708)

    def forward(self, x, adj, A_tilde, s1_sct, s2_sct, s3_sct, \
                sct_index1, \
                sct_index2):
        x = torch.FloatTensor.abs_(self.gc1(x, adj, A_tilde, \
                                            s1_sct, s2_sct, s3_sct, \
                                            adj_sct_o1=sct_index1, \
                                            adj_sct_o2=sct_index2)) ** 4

        x = F.dropout(x, self.dropout, training=self.training)
        z = self.gc12(x, adj)
        z_recon = self.self_expression(z)
        output = F.log_softmax(z_recon, dim=1)
        return output, z, z_recon

    def fn_loss(self, output, labels, z, z_recon):
        loss_label = F.nll_loss(output, labels)
        loss_selfExp = F.mse_loss(z_recon, z, reduction='sum')
        loss = torch.log(loss_selfExp) + loss_label
        return loss
