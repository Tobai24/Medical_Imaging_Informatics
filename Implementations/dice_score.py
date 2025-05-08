import torch
import torch.nn as nn

class DiceLoss(nn.module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
            
    def forward(self, preds, target):
        preds = torch.sigmoid(preds)
        
        preds = preds.view(-1)
        target = target.view(-1)
        
        intersection = (preds * target).sum()
        
        dice_score = (2 * intersection + self.smooth) / preds.sum() + target.sum() + self.smooth
        
        return 1 - dice_score
        
        