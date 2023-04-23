from torch.nn import ReLU
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from x_transformers import Encoder
from torch.nn import ReLU

"""

Implement Masked MLP model to aggregate the genes into pathways and then pass through a fully connected layer to get the predictions

Elmarakeby, Haitham A., et al. "Biologically informed deep neural network for prostate cancer discovery." Nature 598.7880 (2021): 348-352.

"""


class MaskedOmics(nn.Module):
    def __init__(
        self, 
        device="cpu",
        df_comp=None,
        input_dim=1577,
        dim_per_path_1=8,
        dim_per_path_2=16,
        dropout=0.1,
        num_classes=4,
        ):
        super(MaskedOmics, self).__init__()

        self.df_comp = df_comp
        self.input_dim = input_dim
        self.dim_per_path_1 = dim_per_path_1
        self.dim_per_path_2 = dim_per_path_2
        self.dropout = dropout
        self.num_classes = num_classes

        #---> mask_1
        # df = [genes, pathways]
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

        #---> to_logits 
        self.to_logits = nn.Sequential(
            nn.Linear(self.num_pathways*self.dim_per_path_2, self.num_pathways*self.dim_per_path_2//4), ReLU(), nn.Dropout(self.dropout),    
            nn.Linear(self.num_pathways*self.dim_per_path_2//4, self.num_classes)
        )
        
        #---> manually put on device
        self.fc_1_weight.to(device)
        self.fc_1_bias.to(device)
        self.mask_1 = self.mask_1.to(device)

        self.fc_2_weight.to(device)
        self.fc_2_bias.to(device)
        self.mask_2 = self.mask_2.to(device)

        # self.enc.to(device)
        self.to_logits.to(device)

    def forward(self, **kwargs):

        x = kwargs['data_omics']

        #---> apply mask to fc_1 and apply fc_1
        out = torch.matmul(x, self.fc_1_weight * self.mask_1) + self.fc_1_bias

        #---> apply mask to fc_2 and apply fc_2
        out = torch.matmul(out, self.fc_2_weight * self.mask_2) + self.fc_2_bias

        #---> get logits
        logits = self.to_logits(out)
        return logits
