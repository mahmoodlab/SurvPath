import torch.nn as nn
from torch.nn import ReLU, ELU
import torch

"""

Implement a MLP to handle tabular omics data 

"""

class MLPOmics(nn.Module):
    def __init__(
        self, 
        input_dim,
        n_classes=4, 
        projection_dim = 512, 
        dropout = 0.1, 
        ):
        super(MLPOmics, self).__init__()
        
        # self
        self.projection_dim = projection_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, projection_dim//2), ReLU(), nn.Dropout(dropout),
            nn.Linear(projection_dim//2, projection_dim//2), ReLU(), nn.Dropout(dropout)
        ) 

        self.to_logits = nn.Sequential(
                nn.Linear(projection_dim//2, n_classes)
            )

    def forward(self, **kwargs):
        self.cuda()

        #---> unpack
        data_omics = kwargs["data_omics"].float().cuda().squeeze()
        
        #---> project omics data to projection_dim/2
        data = self.net(data_omics) #[B, n]

        #---->predict
        logits = self.to_logits(data) #[B, n_classes]
        return logits
    
    def captum(self, omics):

        self.cuda()

        #---> unpack
        data_omics = omics.float().cuda().squeeze()
        
        #---> project omics data to projection_dim/2
        data = self.net(data_omics) #[B, n]

        #---->predict
        logits = self.to_logits(data) #[B, n_classes]

        #---> get risk 
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)

        #---> return risk 
        return risk



