import torch
import torch.nn as nn
from enum import Enum

class LossFunctions(Enum):
    #Some common loss functions
    L1LOSS = nn.L1Loss()
    MSELOSS = nn.MSELoss()
    BCELOSS = nn.BCELoss()
    CELOSS = nn.CrossEntropyLoss(reduction="mean") 
    
    def get_loss_obj(self):
        return self.value