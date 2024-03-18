from collections import OrderedDict
from os.path import join
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *

"""

Implement Attention MIL for the unimodal (WSI only) and multimodal setting (pathways + WSI). The combining of modalities 
can be done using bilinear fusion or concatenation. 

Mobadersany, Pooya, et al. "Predicting cancer outcomes from histology and genomics using convolutional networks." Proceedings of the National Academy of Sciences 115.13 (2018): E2970-E2979.

"""

################################
# Attention MIL Implementation #
################################
class ABMIL(nn.Module):
    def __init__(self, omic_input_dim=None, fusion=None, size_arg = "small", dropout=0.25, n_classes=4, df_comp=None, dim_per_path_1=16, dim_per_path_2=64, device="cpu"):
        r"""
        Attention MIL Implementation

        Args:
            omic_input_dim (int): Dimension size of genomic features.
            fusion (str): Fusion method (Choices: concat, bilinear, or None)
            size_arg (str): Size of NN architecture (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(ABMIL, self).__init__()
        self.device = device
        self.fusion = fusion
        self.size_dict_path = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256]}

        ### Deep Sets Architecture Construction
        size = self.size_dict_path[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        self.df_comp = df_comp
        self.dim_per_path_1 = dim_per_path_1
        self.num_pathways = self.df_comp.shape[1]
        self.dim_per_path_2 = dim_per_path_2
        self.input_dim = omic_input_dim

        ### Constructing Genomic SNN
        if self.fusion is not None:
            
            self.num_pathways = self.df_comp.shape[1]
            M_raw = torch.Tensor(self.df_comp.values)
            self.mask_1 = torch.repeat_interleave(M_raw, self.dim_per_path_1, dim=1)
            self.fc_1_weight = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(self.input_dim, self.dim_per_path_1*self.num_pathways)))
            self.fc_1_bias = nn.Parameter(torch.rand(self.dim_per_path_1*self.num_pathways))

            self.fc_2_weight = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(self.dim_per_path_1*self.num_pathways, self.dim_per_path_2*self.num_pathways)))
            self.fc_2_bias = nn.Parameter(torch.rand(self.dim_per_path_2*self.num_pathways))
            self.mask_2 = np.zeros([self.dim_per_path_1*self.num_pathways, self.dim_per_path_2*self.num_pathways])
            for (row, col) in zip(range(0, self.dim_per_path_1*self.num_pathways, self.dim_per_path_1), range(0, self.dim_per_path_2*self.num_pathways, self.dim_per_path_2)):
                self.mask_2[row:row+self.dim_per_path_1, col:col+self.dim_per_path_2] = 1
            self.mask_2 = torch.Tensor(self.mask_2)

            self.upscale = nn.Sequential(
                nn.Linear(self.dim_per_path_2*self.num_pathways, int(256/4)),
                nn.ReLU(),
                nn.Linear(int(256/4), 256)
            )

            if self.fusion == "concat":
                self.mm = nn.Sequential(*[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
            elif self.fusion == 'bilinear':
                self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
            else:
                self.mm = None
            self.activation = nn.ReLU()

            self.fc_1_weight.to(self.device)
            self.fc_1_bias.to(self.device)
            self.mask_1 = self.mask_1.to(self.device)
            self.fc_2_weight.to(self.device)
            self.fc_2_bias.to(self.device)
            self.mask_2 = self.mask_2.to(self.device)
            self.mm = self.mm.to(self.device)

        self.classifier = nn.Linear(size[2], n_classes)
        self.classifier = self.classifier.to(self.device)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')

        if self.fusion is not None:
            self.fc_omic = self.fc_omic.to(device)
            self.mm = self.mm.to(device)

        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)


    def forward(self, **kwargs):
        x_path = kwargs['data_WSI']
        x_path = x_path.squeeze() #---> need to do this to make it work with this set up
        A, h_path = self.attention_net(x_path)  
        A = torch.transpose(A, 1, 0)
        A_raw = A 
        A = F.softmax(A, dim=1) 
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path).squeeze()
        
        if self.fusion is not None:
            
            x_omic = kwargs['data_omics']
            x_omic = x_omic.squeeze()

            out = torch.matmul(x_omic, self.fc_1_weight * self.mask_1) + self.fc_1_bias
            out = self.activation(out)
            out = torch.matmul(out, self.fc_2_weight * self.mask_2) + self.fc_2_bias 

            #---> apply linear transformation to upscale the dim_per_pathway (from 32 to 256) Lin, GELU, dropout, 
            h_omic = self.upscale(out)

            if self.fusion == 'bilinear':
                h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
            elif self.fusion == 'concat':
                h = self.mm(torch.cat([h_path, h_omic], axis=0))
        else:
            h = h_path # [256] vector
        
        logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        # return hazards, S, Y_hat, None, None
        return logits
    
    def captum(self, x_omics, x_wsi):
        x_wsi = x_wsi.squeeze() #---> need to do this to make it work with this set up
        A, h_path = self.attention_net(x_wsi)  
        A = torch.transpose(A, 1, 0)
        A_raw = A 
        A = F.softmax(A, dim=1) 
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path).squeeze()

        if self.fusion is not None:
            x_omics = x_omics.squeeze()

            out = torch.matmul(x_omics, self.fc_1_weight * self.mask_1) + self.fc_1_bias
            out = self.activation(out)
            out = torch.matmul(out, self.fc_2_weight * self.mask_2) + self.fc_2_bias 

            #---> apply linear transformation to upscale the dim_per_pathway (from 32 to 256) Lin, GELU, dropout, 
            h_omic = self.upscale(out)

            if self.fusion == 'bilinear':
                h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
            elif self.fusion == 'concat':
                h = self.mm(torch.cat([h_path, h_omic], axis=0))
        else:
            h = h_path # [256] vector
        
        logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector 
        #---> get risk 
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)

        #---> return risk 
        return risk
