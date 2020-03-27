from __future__ import division
import torch
# import torchfile
# from torch.utils.serialization import load_lua
import torchvision.transforms as transforms
import numpy as np
import argparse
import time
import os
from PIL import Image
from modelsNIPS import decoder1,decoder2,decoder3,decoder4,decoder5
from modelsNIPS import encoder1,encoder2,encoder3,encoder4,encoder5
import torch.nn as nn
from models import feature_invertor_conv1_1, feature_invertor_conv2_1, \
    feature_invertor_conv3_1, feature_invertor_conv4_1, feature_invertor_conv5_1, \
    vgg_normalised_conv1_1, vgg_normalised_conv2_1, vgg_normalised_conv3_1, \
    vgg_normalised_conv4_1, vgg_normalised_conv5_1



class WCT(nn.Module):
    def __init__(self,args):
        super(WCT, self).__init__()
        # load pre-trained network
        vgg1 = vgg_normalised_conv1_1.vgg_normalised_conv1_1
        vgg1.load_state_dict(torch.load(args.vgg1))


        decoder1_torch = feature_invertor_conv1_1.feature_invertor_conv1_1
        decoder1_torch.load_state_dict(torch.load(args.decoder1))


        vgg2 = vgg_normalised_conv2_1.vgg_normalised_conv2_1
        vgg2.load_state_dict(torch.load(args.vgg2))

        decoder2_torch = feature_invertor_conv2_1.feature_invertor_conv2_1
        decoder2_torch.load_state_dict(torch.load(args.decoder2))

        vgg3 = vgg_normalised_conv3_1.vgg_normalised_conv3_1
        vgg3.load_state_dict(torch.load(args.vgg3))

        decoder3_torch = feature_invertor_conv3_1.feature_invertor_conv3_1
        decoder3_torch.load_state_dict(torch.load(args.decoder3))

        vgg4 = vgg_normalised_conv4_1.vgg_normalised_conv4_1
        vgg4.load_state_dict(torch.load(args.vgg4))

        decoder4_torch = feature_invertor_conv4_1.feature_invertor_conv4_1
        decoder4_torch.load_state_dict(torch.load(args.decoder4))

        vgg5 = vgg_normalised_conv5_1.vgg_normalised_conv5_1
        vgg5.load_state_dict(torch.load(args.vgg5))

        decoder5_torch = feature_invertor_conv5_1.feature_invertor_conv5_1
        decoder5_torch.load_state_dict(torch.load(args.decoder5))

        # vgg1.eval()
        # decoder1_torch.eval()
        # vgg2.eval()
        # decoder2_torch.eval()
        # vgg3.eval()
        # decoder3_torch.eval()
        # vgg4.eval()
        # decoder4_torch.eval()
        # vgg5.eval()
        # decoder5_torch.eval()

        # decoder1_torch = torchfile.load(args.decoder1, force_8bytes_long=True)
        # vgg2 = torchfile.load(args.vgg2, force_8bytes_long=True)
        # decoder2_torch = torchfile.load(args.decoder2, force_8bytes_long=True)
        # vgg3 = torchfile.load(args.vgg3, force_8bytes_long=True)
        # decoder3_torch = torchfile.load(args.decoder3, force_8bytes_long=True)
        # vgg4 = torchfile.load(args.vgg4, force_8bytes_long=True)
        # decoder4_torch = torchfile.load(args.decoder4, force_8bytes_long=True)
        # vgg5 = torchfile.load(args.vgg5, force_8bytes_long=True)
        # decoder5_torch = torchfile.load(args.decoder5, force_8bytes_long=True)

        # print(vgg1.modules[0].weight.float())

        self.e1 = encoder1(vgg1)
        self.d1 = decoder1(decoder1_torch)
        self.e2 = encoder2(vgg2)
        self.d2 = decoder2(decoder2_torch)
        self.e3 = encoder3(vgg3)
        self.d3 = decoder3(decoder3_torch)
        self.e4 = encoder4(vgg4)
        self.d4 = decoder4(decoder4_torch)
        self.e5 = encoder5(vgg5)
        self.d5 = decoder5(decoder5_torch)

    def whiten_and_color(self,cF,sF):
        cFSize = cF.size()
        c_mean = torch.mean(cF,1) # c x (h x w)
        c_mean = c_mean.unsqueeze(1).expand_as(cF)
        cF = cF - c_mean

        contentConv = torch.mm(cF,cF.t()).div(cFSize[1]-1) + torch.eye(cFSize[0]).double()
        c_u,c_e,c_v = torch.svd(contentConv,some=False)

        k_c = cFSize[0]
        for i in range(cFSize[0]):
            if c_e[i] < 0.00001:
                k_c = i
                break

        sFSize = sF.size()
        s_mean = torch.mean(sF,1)
        sF = sF - s_mean.unsqueeze(1).expand_as(sF)
        styleConv = torch.mm(sF,sF.t()).div(sFSize[1]-1)
        s_u,s_e,s_v = torch.svd(styleConv,some=False)

        k_s = sFSize[0]
        for i in range(sFSize[0]):
            if s_e[i] < 0.00001:
                k_s = i
                break

        c_d = (c_e[0:k_c]).pow(-0.5)
        step1 = torch.mm(c_v[:,0:k_c],torch.diag(c_d))
        step2 = torch.mm(step1,(c_v[:,0:k_c].t()))
        whiten_cF = torch.mm(step2,cF)

        s_d = (s_e[0:k_s]).pow(0.5)
        targetFeature = torch.mm(torch.mm(torch.mm(s_v[:,0:k_s],torch.diag(s_d)),(s_v[:,0:k_s].t())),whiten_cF)
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
        return targetFeature

    def transform(self,cF,sF,csF,alpha):
        cF = cF.double()
        sF = sF.double()
        C,W,H = cF.size(0),cF.size(1),cF.size(2)
        _,W1,H1 = sF.size(0),sF.size(1),sF.size(2)
        cFView = cF.view(C,-1)
        sFView = sF.view(C,-1)

        targetFeature = self.whiten_and_color(cFView,sFView)
        targetFeature = targetFeature.view_as(cF)
        ccsF = alpha * targetFeature + (1.0 - alpha) * cF
        ccsF = ccsF.float().unsqueeze(0)
        with torch.no_grad():
            csF.resize_(ccsF.size()).copy_(ccsF)
        return csF
