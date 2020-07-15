import torch
import blocks
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from collections import OrderedDict
from adain import adaptive_instance_norm


class ConvBlock(nn.Module):
    def __init__(self, inc , outc, kernel_size=3, padding=1, stride=1, 
                 use_bias=True, activation=nn.ReLU, batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, 
                              padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

class FC(nn.Module):
    def __init__(self, inc , outc, activation=nn.ReLU, batch_norm=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(int(inc), int(outc))
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm1d(outc) if batch_norm else None
        
    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x



class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap): 
        device = bilateral_grid.get_device()

        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)]) # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H-1) # norm to [0,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W-1) # norm to [0,1] NxHxWx1
        hg, wg = hg*2-1, wg*2-1
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1) # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide)
        return coeff.squeeze(2)



class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()
        self.degree = 3

    def forward(self, coeff, full_res_input):

        '''
            Affine:
            r = a11*r + a12*g + a13*b + a14
            g = a21*r + a22*g + a23*b + a24
            ...
        '''

        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]

        return torch.cat([R, G, B], dim=1)
        '''
            Polinomial
        '''
        # res = []
        # bt = list(full_res_input.shape)[0]
        # for batch in range(bt):
        #     r = full_res_input[batch,0:1,:,:]
        #     g = full_res_input[batch,1:2,:,:]
        #     b = full_res_input[batch,2:3,:,:]

        #     pr = []
        #     for i in range(1,self.degree+1):
        #         pr.append(r**i)
        #     pr = torch.cat(pr, dim=0)
        #     pg = []
        #     for i in range(1,self.degree+1):
        #         pg.append(g**i)
        #     pg = torch.cat(pg, dim=0)
        #     pb = []
        #     for i in range(1,self.degree+1):
        #         pb.append(b**i)
        #     pb = torch.cat(pb, dim=0)
        #     yr = torch.sum(pr * coeff[batch,  0:3, :, :], dim=0, keepdim=True) + coeff[batch,  3:4, :, :]
        #     yg = torch.sum(pg * coeff[batch,  4:7, :, :], dim=0, keepdim=True) + coeff[batch,  7:8, :, :]
        #     yb = torch.sum(pb * coeff[batch,  8:11, :, :], dim=0, keepdim=True) + coeff[batch,  11:12, :, :]

        #     res.append(torch.cat([yr,yg,yb], dim=0).repeat(1,1,1,1))
        # x = torch.cat(res, dim=0)
        # return x

class GuideNN(nn.Module):
    def __init__(self, bn=True):
        super(GuideNN, self).__init__()

        self.conv1 = ConvBlock(3, 16, kernel_size=1, padding=0, batch_norm=bn)
        self.conv2 = ConvBlock(16, 1, kernel_size=1, padding=0, activation=nn.Tanh)

    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.conv2(output)

        return output

class Coeffs(nn.Module):

    def __init__(self, nin=3, nout=4, lb=8, cm=1, sb=16, nsize=256, bn=True):
        super(Coeffs, self).__init__()
        self.nin = nin 
        self.nout = nout
        self.lb = lb
        self.cm = cm
        self.sb = sb
        self.bn = bn
        self.nsize = nsize
        self.relu = nn.ReLU()

        # splat features
        n_layers_splat = int(np.log2(nsize/sb))
        self.splat_features = nn.ModuleList()
        prev_ch = nin
        for i in range(n_layers_splat):
            use_bn = bn if i > 0 else False
            self.splat_features.append(ConvBlock(prev_ch, cm*(2**i)*lb, 
                                                 3, stride=2, batch_norm=use_bn))
            prev_ch = splat_ch = cm*(2**i)*lb

        # global features
        n_layers_global = int(np.log2(sb/4))
        self.global_features_conv = nn.ModuleList()
        self.global_features_fc = nn.ModuleList()
        for i in range(n_layers_global):
            self.global_features_conv.append(ConvBlock(prev_ch, cm*8*lb, 3, stride=2, batch_norm=bn))
            prev_ch = cm*8*lb

        n_total = n_layers_splat + n_layers_global
        prev_ch = prev_ch * (nsize/2**n_total)**2
        self.global_features_fc.append(FC(prev_ch, 32*cm*lb, batch_norm=bn))
        self.global_features_fc.append(FC(32*cm*lb, 16*cm*lb, batch_norm=bn))
        self.global_features_fc.append(FC(16*cm*lb, 8*cm*lb, activation=None, batch_norm=bn))

        # local features
        self.local_features = nn.ModuleList()
        self.local_features.append(ConvBlock(splat_ch, 8*cm*lb, 3, batch_norm=bn))
        self.local_features.append(ConvBlock(8*cm*lb, 8*cm*lb, 3, activation=None, use_bias=False))
        
        # predicton
        self.conv_out = ConvBlock(8*cm*lb, lb*nout*nin, 1, padding=0, activation=None)

   
    def forward(self, lowres_input):
        bs = lowres_input.shape[0]
        x = lowres_input
        for layer in self.splat_features:
            x = layer(x)
        splat_features = x
        
        for layer in self.global_features_conv:
            x = layer(x)
        x = x.view(bs, -1)
        for layer in self.global_features_fc:
            x = layer(x)
        global_features = x

        x = splat_features
        for layer in self.local_features:
            x = layer(x)        
        local_features = x

        fusion_grid = local_features
        fusion_global = global_features.view(bs, 8*self.cm*self.lb, 1, 1)
        fusion = self.relu( fusion_grid + fusion_global )

        x = self.conv_out(fusion)
        s = x.shape
        x = x.view(bs, self.nin*self.nout, self.lb, self.sb, self.sb) 
        # B x Coefs x Luma x Spatial x Spatial
        return x


class HDRPointwiseNN(nn.Module):

    def __init__(self, params):
        super(HDRPointwiseNN, self).__init__()
        self.coeffs = Coeffs(params=params)
        self.guide = GuideNN(params=params)
        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()

    def forward(self, lowres, fullres):
        coeffs = self.coeffs(lowres)
        guide = self.guide(fullres)
        slice_coeffs = self.slice(coeffs, guide)
        out = self.apply_coeffs(slice_coeffs, fullres)
        return out
    
    

class StyleNetwork(nn.Module):
    def __init__(self, size=256):
        super(StyleNetwork, self).__init__()
        self.size = size
        self.extractor = blocks.Vgg19()
        self.splat1 = blocks.SplatBlock(64, 8)
        self.splat2 = blocks.SplatBlock(8, 16)
        self.splat3 = blocks.SplatBlock(16, 32)
        self.feat_conv = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True))

        self.local_layer = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1))

        self.global_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.global_fc = nn.Sequential(
            nn.Linear(1024, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64))

        self.final_conv = nn.Conv2d(64, 96, 1, stride=1, padding=0, bias=False)


    def forward(self, content, style):
        lr_c = F.interpolate(content, (self.size, self.size),
                            mode='bilinear', align_corners=True)
        lr_s = F.interpolate(style, (self.size, self.size),
                            mode='bilinear', align_corners=True)
        feats_c = self.extractor(lr_c)
        feats_s = self.extractor(lr_s)
        feat_adain1 = adaptive_instance_norm(feats_c[1], feats_s[1])
        out_c1, out_s1 = self.splat1(feats_c[0], feats_s[0], feat_adain1)
        feat_adain2 = adaptive_instance_norm(feats_c[2], feats_s[2])
        out_c2, out_s2 = self.splat2(out_c1, out_s1, feat_adain2)
        feat_adain3 = adaptive_instance_norm(feats_c[3], feats_s[3])
        out_c3, out_s3 = self.splat3(out_c2, out_s2, feat_adain3)
        out_feat = self.feat_conv(out_c3)

        local_feat = self.local_layer(out_feat)
        global_feat = self.global_conv(out_feat)
        batch_size = global_feat.size()[0]
        global_feat = global_feat.view(batch_size, -1)
        global_feat = self.global_fc(global_feat)
        fuse_feat = local_feat + global_feat.view(batch_size, -1, 1, 1)
        output = self.final_conv(fuse_feat)
        output = output.view(batch_size, 12, 8, 16, 16) 
        return output




class BilateralNetwork(nn.Module):
    def __init__(self, size=256):
        super(BilateralNetwork, self).__init__()
        self.size = size
        self.stylenet = StyleNetwork(size)
        self.guide = GuideNN()
        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()

    def forward(self, content, style):
        coeffs = self.stylenet(content, style)
        guide = self.guide(content)
        slice_coeffs = self.slice(coeffs, guide)
        output = self.apply_coeffs(slice_coeffs, content)
        output = torch.sigmoid(output)
        return output, coeffs




class Vgg19(torch.nn.Module):
    def __init__(self, weight_path='vgg19.pth'):
        super(Vgg19, self).__init__()
        vgg19 = models.vgg19(pretrained=False)
        vgg19.load_state_dict(torch.load(weight_path))
        
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(1):
            self.slice1.add_module(str(x), vgg19.features[x])
        for x in range(1, 6):
            self.slice2.add_module(str(x), vgg19.features[x])
        for x in range(6, 11):
            self.slice3.add_module(str(x), vgg19.features[x])
        for x in range(11, 20):
            self.slice4.add_module(str(x), vgg19.features[x])
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        conv1_1 = self.slice1(inputs)
        conv2_1 = self.slice2(conv1_1)
        conv3_1 = self.slice3(conv2_1)
        conv4_1 = self.slice4(conv3_1)
        
        return conv1_1, conv2_1, conv3_1, conv4_1
        
