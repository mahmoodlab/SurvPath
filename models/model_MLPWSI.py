
import torch
import numpy as np 
from x_transformers import CrossAttender

import torch
import torch.nn as nn
from torch import nn
from einops import reduce

from x_transformers import Encoder
from torch.nn import ReLU

from models.layers.cross_attention import FeedForward, MMAttentionLayer

def exists(val):
    return val is not None


class MLPWSI(nn.Module):
    def __init__(
        self, 
        wsi_embedding_dim=1024,
        input_dim_omics=1577,
        dropout=0.1,
        num_classes=4,
        wsi_projection_dim=256,
        device="cpu"
        ):
        super(MLPWSI, self).__init__()

        #---> init self 
        self.num_classes = num_classes

        #---> wsi props
        self.wsi_embedding_dim = wsi_embedding_dim 
        self.wsi_projection_dim = wsi_projection_dim

        self.wsi_projection_net = nn.Sequential(
            nn.Linear(self.wsi_embedding_dim, self.wsi_projection_dim),
            ReLU(),
        )

        #---> omics props
        self.omic_input_dim = input_dim_omics
        self.dropout = dropout
        self.omic_projection_net = nn.Sequential(
            nn.Linear(self.omic_input_dim, self.wsi_projection_dim),
            ReLU(),
            nn.Dropout(p=self.dropout),
        )
        
        self.cross_attender = MMAttentionLayer(
                dim=self.wsi_projection_dim,
                dim_head=self.wsi_projection_dim // 2,
                heads=2,
                residual=False,
                dropout=0.1,
                num_pathways = 1
        )

        self.feed_forward = FeedForward(self.wsi_projection_dim, dropout=dropout)
        self.layer_norm = nn.LayerNorm(self.wsi_projection_dim)

        self.to_logits = nn.Sequential(
            nn.Linear(self.wsi_projection_dim, int(self.wsi_projection_dim/4)),
            nn.ReLU(),
            nn.Linear(int(self.wsi_projection_dim/4), self.num_classes)
        )
        self.device = device
        
    def forward(self, **kwargs):

        omics = kwargs['data_omics']
        wsi = kwargs['data_WSI']
        mask = kwargs['mask']

        #---> project wsi to smaller dimension (same as pathway dimension)
        wsi_embed = self.wsi_projection_net(wsi)

        #---> get pathway embeddings 
        omics_embed = self.omic_projection_net(omics).unsqueeze(1)

        #---> find cross attention between wsi and omics 
        if mask is not None:
            mask = mask.bool()
            add_omics_start = torch.zeros([omics_embed.shape[0], omics_embed.shape[1]]).to(self.device)  # add omics tokens to mask 
            mask = torch.cat([add_omics_start, mask], dim=1).bool()
            mask = ~mask
    
        tokens = torch.cat([omics_embed, wsi_embed], dim=1)
        mm_embed = self.cross_attender(x=tokens, mask=mask if mask is not None else None)

        #---> feedforward and layer norm 
        mm_embed = self.feed_forward(mm_embed)
        mm_embed = self.layer_norm(mm_embed)

        #---> aggregate 
        embedding = torch.mean(mm_embed, dim=1)

        #---> get logits
        logits = self.to_logits(embedding)

        return logits