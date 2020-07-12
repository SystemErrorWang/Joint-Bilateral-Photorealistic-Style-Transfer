import torch
import network
import numpy as np

from torch import nn
from torch.nn import functional as F



def gram_matrix(image):
    bs, ch, h, w = image.size() 
    features = image.view(bs * ch, h * w) 
    output = torch.mm(features, features.t()) 
    return output.div(bs * ch * h * w)



def regularize(coeff):
    num_bins = coeff.size()[2]
    total_loss = 0
    for idx in range(num_bins-1):
        feat1 = torch.sqrt(torch.pow(coeff[:, :, idx, :, :], 2))
        feat2 = torch.sqrt(torch.pow(coeff[:, :, idx+1, :, :], 2))
        total_loss += F.mse_loss(feat1, feat2)
    return total_loss


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.extractor = network.Vgg19()
        self.l2_criterion = nn.MSELoss()
        
    def forward(self, output, content):
        feats_o = self.extractor(output)
        feats_c = self.extractor(content)
        '''
        total_loss = 0
        for feat_o, feat_c in zip(feats_o, feats_c):
            total_loss += self.l2_criterion(feat_o, feat_c)
        '''
        total_loss = self.l2_criterion(feats_o[-1], feats_c[-1])
        return total_loss
    

            
class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.extractor = network.Vgg19()
        self.l2_criterion = nn.MSELoss()
        
    def forward(self, output, style):
        total_loss = 0
        feats_o = self.extractor(output)
        feats_s = self.extractor(style)
        for feat_o, feat_s in zip(feats_o, feats_s):
            bs, ch = feat_o.size()[:2]
            feat_o = feat_o.view(bs, ch, -1)
            feat_s = feat_s.view(bs, ch, -1)
            total_loss += self.l2_criterion(feat_o.mean(dim=2), feat_s.mean(dim=2))
            total_loss += self.l2_criterion(feat_o.std(dim=2), feat_s.std(dim=2))
        return total_loss


if __name__ == '__main__':
    pass

