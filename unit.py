import torch.nn as nn
import torch
from torch.nn import functional as F
import pywt
import numpy as np

class ResUnit(nn.Module):
    def __init__(self, ksize=3, wkdim=64):
        super(ResUnit, self).__init__()
        self.conv1 = nn.Conv2d(wkdim, wkdim, ksize, 1, int(ksize/2))  
        self.active = nn.PReLU()
        self.conv2 = nn.Conv2d(wkdim, wkdim, ksize, 1, int(ksize/2))

    def forward(self, input):
        current = self.conv1(input)
        current = self.active(current)
        current = self.conv2(current)
        current = input + current
        return current


class UPN(nn.Module):
    def __init__(self, indim=64, scale=2):
        super(UPN, self).__init__()
        self.conv = nn.Conv2d(indim, indim*(scale**2), 3, 1, 1)
        self.Upsample = nn.PixelShuffle(scale)
        self.active = nn.PReLU()

    def forward(self, input):
        current = self.conv(input)
        current = self.Upsample(current)
        current = self.active(current)
        return current


class SRRes(nn.Module):
    def __init__(self, wkdim=64, num_block=16):
        super(SRRes, self).__init__()
        self.head = nn.Sequential(nn.Conv2d(3, wkdim, 3, 1, 1),
                                  nn.PReLU(),
                                  nn.Conv2d(wkdim, wkdim, 3, 1,1))
        self.resblock = self._make_resblocks(wkdim, num_block)
        self.gate = nn.Conv2d(wkdim, wkdim, 3, 1, 1)
        self.up_1 = UPN(wkdim)
        self.up_2 = UPN(wkdim)

        self.comp = nn.Conv2d(wkdim*2, wkdim, 3, 1, 1)

        self.tail = nn.Sequential(nn.Conv2d(wkdim, wkdim, 3, 1, 1),
                                  nn.PReLU(),
                                  nn.Conv2d(wkdim, 3, 3, 1, 1))
            
    def _make_resblocks(self, wkdim, num_block):
        layers = []
        for i in range(1, num_block+1):
            layers.append(ResUnit(wkdim=wkdim))
        return nn.Sequential(*layers)

    def forward(self, input):
        F_0 = self.head(input)
        current = self.resblock(F_0)
        current = self.gate(current)
        current = F_0 + current
        UP_1 = self.up_1(current)
        UP_2 = self.up_2(UP_1)
        
        current = self.tail(UP_2)
        return current


class Stander_Discriminator(nn.Module):
    def __init__(self):
        super(Stander_Discriminator,self).__init__()
        self.layer_1 = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1),
                                     nn.PReLU(),
                                     nn.Conv2d(64, 64, 4, 2, 1),
                                     nn.InstanceNorm2d(64),
                                     nn.PReLU())
        self.layer_2 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1),
                                     nn.InstanceNorm2d(128),
                                     nn.PReLU(),
                                     nn.Conv2d(128, 128, 4, 2, 1),
                                     nn.InstanceNorm2d(128),
                                     nn.PReLU())
        self.layer_3 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1),
                                     nn.InstanceNorm2d(256),
                                     nn.PReLU(),
                                     nn.Conv2d(256, 256, 4, 2, 1),
                                     nn.InstanceNorm2d(256),
                                     nn.PReLU())
        self.layer_4 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1),
                                     nn.InstanceNorm2d(512),
                                     nn.PReLU(),
                                     nn.Conv2d(512, 512, 4, 2, 1),
                                     nn.InstanceNorm2d(512),
                                     nn.PReLU())
        self.layer_5 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1),
                                     nn.InstanceNorm2d(512),
                                     nn.PReLU(),
                                     nn.Conv2d(512, 512, 4, 2, 1),
                                     nn.InstanceNorm2d(512),
                                     nn.PReLU())
        self.tail    = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1),
                                     nn.PReLU(),
                                     nn.Conv2d(512, 512, 3, 1, 1),
                                     nn.PReLU(),
                                     nn.Conv2d(512, 512, 3, 1, 1),
                                     nn.AdaptiveAvgPool2d((1,1)),
                                     nn.Conv2d(512, 512, 1, 1, 0),
                                     nn.PReLU(),
                                     nn.Conv2d(512, 1, 1, 1, 0))
    def forward(self, input):
        current = self.layer_1(input)
        current = self.layer_2(current)
        current = self.layer_3(current)
        current = self.layer_4(current)
        current = self.layer_5(current) 
        current = self.tail(current) 
        return current

class Stander_Ranker(nn.Module):
    def __init__(self):
        super(Stander_Ranker,self).__init__()
        self.layer_1 = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1),
                                     nn.PReLU(),
                                     nn.Conv2d(64, 64, 4, 2, 1),
                                     nn.InstanceNorm2d(64),
                                     nn.PReLU())
        self.layer_2 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1),
                                     nn.InstanceNorm2d(128),
                                     nn.PReLU(),
                                     nn.Conv2d(128, 128, 4, 2, 1),
                                     nn.InstanceNorm2d(128),
                                     nn.PReLU())
        self.layer_3 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1),
                                     nn.InstanceNorm2d(256),
                                     nn.PReLU(),
                                     nn.Conv2d(256, 256, 4, 2, 1),
                                     nn.InstanceNorm2d(256),
                                     nn.PReLU())
        self.layer_4 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1),
                                     nn.InstanceNorm2d(512),
                                     nn.PReLU(),
                                     nn.Conv2d(512, 512, 4, 2, 1),
                                     nn.InstanceNorm2d(512),
                                     nn.PReLU())
        self.layer_5 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1),
                                     nn.InstanceNorm2d(512),
                                     nn.PReLU(),
                                     nn.Conv2d(512, 512, 4, 2, 1),
                                     nn.InstanceNorm2d(512),
                                     nn.PReLU())
        self.tail    = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1),
                                     nn.PReLU(),
                                     nn.Conv2d(512, 512, 3, 1, 1),
                                     nn.PReLU(),
                                     nn.Conv2d(512, 512, 3, 1, 1),
                                     nn.AdaptiveAvgPool2d((1,1)),
                                     nn.Conv2d(512, 512, 1, 1, 0),
                                     nn.PReLU(),
                                     nn.Conv2d(512, 1, 1, 1, 0))
    def forward(self, input):
        current = self.layer_1(input)
        current = self.layer_2(current)
        current = self.layer_3(current)
        current = self.layer_4(current)
        current = self.layer_5(current)
        current = self.tail(current)
        current = current.squeeze(1)
        return current

class Feature_Ranker(nn.Module):
    def __init__(self, inchannel=512):
        super(Feature_Ranker, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(inchannel, inchannel, 3, 1, 1),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(inchannel, inchannel, 3, 1, 1),
                                    nn.InstanceNorm2d(inchannel),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(inchannel, inchannel, 4, 2, 1),
                                    nn.InstanceNorm2d(inchannel),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(inchannel, inchannel*2, 3, 1, 1),
                                    nn.InstanceNorm2d(inchannel*2),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(inchannel*2, inchannel*2, 4, 2, 1),
                                    nn.InstanceNorm2d(inchannel*2),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(inchannel*2, inchannel*2, 3, 1, 1),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(inchannel*2, inchannel*2, 3, 1, 1),
                                    nn.LeakyReLU(),
                                    nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Conv2d(inchannel*2, inchannel*2, 1, 1, 0),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(inchannel*2, 1, 1, 1, 0))

    def forward(self, input):
        rank = self.layers(input)
        rank = rank.squeeze(1)
        return rank

class PT_DWT(nn.Module):
    def __init__(self, wave_model='haar', POOL=False, Drop_LL=False, VH_Only=False, LL_Only=False):
        super(PT_DWT, self).__init__()
        self.wave = pywt.Wavelet(wave_model)
        self.Drop_LL = Drop_LL
        self.VH_Only = VH_Only
        self.LL_Only = LL_Only
        self.filter = self._init_filter()
        if POOL:
            self.stride = [1,2,2]
        else:
            self.stride = 1
    def _init_filter(self):
        ll = np.outer(self.wave.dec_lo, self.wave.dec_lo)
        lh = np.outer(self.wave.dec_hi, self.wave.dec_lo)
        hl = np.outer(self.wave.dec_lo, self.wave.dec_hi)
        hh = np.outer(self.wave.dec_hi, self.wave.dec_hi)

        if self.Drop_LL is False and self.VH_Only is False and self.LL_Only is False:
            d_temp = np.zeros([ll.shape[0], ll.shape[1], 1, 4])

            d_temp[::-1,::-1,0,0] = ll
            d_temp[::-1,::-1,0,1] = lh
            d_temp[::-1,::-1,0,2] = hl
            d_temp[::-1,::-1,0,3] = hh
        elif self.Drop_LL is True:
            d_temp = np.zeros([ll.shape[0], ll.shape[1], 1, 3])

            d_temp[::-1,::-1,0,0] = lh
            d_temp[::-1,::-1,0,1] = hl
            d_temp[::-1,::-1,0,2] = hh
        elif self.VH_Only is True:
            d_temp = np.zeros([ll.shape[0], ll.shape[1], 1, 2])

            d_temp[::-1,::-1,0,0] = lh
            d_temp[::-1,::-1,0,1] = hl
        elif self.LL_Only is True:
            d_temp = np.zeros([ll.shape[0], ll.shape[1], 1, 1])
            d_temp[::-1,::-1,0,0] = ll

        filts = d_temp.astype('float32')

        filts = filts[None,:,:,:,:]
        filts = np.transpose(filts, [4,3,0,1,2])

        filts = (torch.Tensor(filts)).cuda()
        return filts
    def forward(self, input):
        with torch.no_grad():
            current= input.unsqueeze(2)
            current = torch.split(current, [1]*int(current.size()[1]),1)
            current = torch.cat([x for x in current], 2)

            current = F.pad(current, (0,1,0,1), 'constant', 0.)
            output_3d = F.conv3d(current, self.filter, stride=self.stride)
            output = torch.split(output_3d, [1]*int(output_3d.size()[2]), 2)
            output = torch.cat([x for x in output], 1)
            output = output.view(output.size()[0], output.size()[1],
                                 output.size()[3], output.size()[4])
            
            return output

class Self_Match(nn.Module):
    def __init__(self, ksize=5, stride=5, Model='DotP'):
        super(Self_Match,self).__init__()
        self.ksize = ksize
        self.stride= stride
        self.unfold= nn.Unfold(ksize,1,0,stride)
        self.Model = Model
        

    def reduce_sum(self, x, axis=None, keepdim=False):
        if not axis:
            axis = range(len(x.shape))
        for i in sorted(axis, reverse=True):
            x = torch.sum(x, dim=i, keepdim=keepdim)
        return x

    def _Dot_Point(self, wi, xi_t, sigmoid):
        max_wi = torch.max(torch.sqrt(self.reduce_sum(torch.pow(wi, 2),axis=[1, 2, 3],keepdim=True)))
        wi_t   = wi / max_wi
        match  = F.conv2d(xi_t,
                              wi_t,
                              stride=self.stride,
                              padding=0)

        if sigmoid:
            match  = torch.sigmoid(match)
        return match

    def _L1(self, wi, xi):
        xi_t = self.unfold(xi)
        xi_t = xi_t.permute(0,2,1)
        
        wi_t = wi.view(wi.size()[0], -1)
        wi_t = wi_t.unsqueeze(1)

        match = torch.abs(xi_t - wi_t)
        match = torch.sum(match, -1)

        match = match.view(match.size()[0], int(match.size()[-1]**0.5),-1)
        match = match.unsqueeze(0)

        return match

    def forward(self, input, target=None, sigmoid=True):
        
        n,c,h,w = input.size()
        current = input
        n,c,h,w = current.size()
        wi = self.unfold(current)
        wi = wi.permute(0,2,1)
        wi = wi.view(n,-1,c,self.ksize,self.ksize)
        wi = torch.split(wi,1,0)

        if target is None:
            xi = torch.split(current, 1, 0)
        else:
            xi = torch.split(target, 1, 0)

        score = []
        for i in range(n):
            if self.Model == 'DotP':
                match = self._Dot_Point(wi[i].squeeze(0), xi[i], sigmoid)
            elif self.Model == 'L1':
                match = self._L1(wi[i].squeeze(0), xi[i])
            score.append(match)

        score = torch.cat(score, 0)
        return score

class DWT_MATCH(nn.Module):
    def __init__(self, ksize=3, stride=3, Model='DotP'):
        super(DWT_MATCH, self).__init__()
        self.kszie = ksize
        self.stride = stride
        self.DWT = PT_DWT(POOL=True, VH_Only=True)
        self.SELF_MATCH = Self_Match(ksize, stride, Model)

    def forward(self, input, target=None, VHSplit=True):
        if VHSplit:
            current = self.DWT(input)
            current = torch.split(current,1,1)
            V_c = torch.cat([current[0], current[2], current[4]], 1)
            H_c = torch.cat([current[1], current[3], current[5]], 1)
            if target is not None:
                label = self.DWT(target)
                label = torch.split(label,1,1)
                V_l   = torch.cat([label[0], label[2], label[4]], 1)
                H_l   = torch.cat([label[1], label[3], label[5]], 1)

            if target is None:
                V_score = self.SELF_MATCH(V_c)
                H_score = self.SELF_MATCH(H_c)

            else:
                V_score = self.SELF_MATCH(V_c, V_l)
                H_score = self.SELF_MATCH(H_c, H_l)

            return V_score, H_score

        else:
            VH_c = self.DWT(input)
            if target is not None:
                VH_l = self.DWT(target)
            if target is None:
                VH_score = self.SELF_MATCH(VH_c)
            else:
                VH_score = self.SELF_MATCH(VH_c, VH_l)
            return VH_score