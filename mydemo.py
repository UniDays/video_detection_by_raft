import sys
sys.path.append('core')

import torch
from torch import nn
from torchvision.models import resnet50
import torch.nn.functional as F

import argparse
import os
import cv2
import glob
import numpy as np

from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def feature_warp(f_k : torch.Tensor, flow : torch.Tensor):
    n, c, h, w = f_k.shape
    kernel_size = 2
    f_i = torch.zeros_like(f_k)
    flo = - F.interpolate(flow, size=(h,w), mode='bilinear', align_corners=False)

    for px in range(w):
        for py in range(h):
            dpx = flo[:, 0:1, py, px]
            dpy = flo[:, 1:, py, px]
            i, j = torch.floor(py + dpy), torch.floor(px + dpx)
            di, dj = py + dpy - i, px + dpx - j
            G = torch.concat([di * dj, di * (1 - dj), (1 - di) * dj, (1 - di) * (1 - dj)], dim=1).reshape(n, 1, kernel_size, kernel_size)
            # n, c, kernel, kernel
            G = G.repeat(1, c, 1, 1).to(DEVICE)
            grid = torch.zeros(n, kernel_size, kernel_size, 2).to(DEVICE)
            for gy in range(kernel_size):
                for gx in range(kernel_size):
                    grid[:, gy, gx, 0:1] = 2 * (j + gx) / (w - 1) - 1
                    grid[:, gy, gx, 1:] = 2 * (i + gy) / (h - 1) - 1
            # n, c, kernel, kernel
            patch = F.grid_sample(f_k, grid,  mode='bilinear', padding_mode='zeros', align_corners=True)
            f_i[:,:, py, px] = torch.sum(G * patch, dim=(2, 3))

    return f_i

def feature_aggregation(frames : torch.Tensor, feature_maps : torch.Tensor, raft : nn.Module, feature_embedding : nn.Module, K = 10):
    N, C, H, W = frames.shape
    n, c, h, w = feature_maps.shape
    f_i_aggregation_list = []
    for i in range(N):
        w_list = []
        f_list = []
        for j in range(max(0, i - K), min(N, i + K + 1)):
            flow_ji = raft(frames[j:j+1], frames[i:i+1])
            # 1, c, h, w
            f_ji = feature_warp(feature_maps[j:j+1], flow_ji)
            # 1, emb
            f_ji_emb, f_i_emb = feature_embedding(f_ji, feature_maps[i:i+1])
            # 1
            w_ji = torch.exp(torch.sum(f_ji_emb * f_i_emb) / (torch.norm(f_ji_emb, p = 2) *  torch.norm(f_i_emb, p = 2)))
            w_ji.reshape(1, 1, 1, 1)
            # 1, c, 1, 1
            w_ji.repeat(1, c, 1, 1)
            f_list.append(f_ji)
            w_list.append(w_ji)
        # 2K, c, h, w
        f = torch.concatenate(f_list, dim=0)
        # 2K, c, 1, 1
        w = torch.concatenate(w_list, dim=0)
        # 1, c, h, w
        f_i_aggregation = torch.sum(f * w / torch.sum(w), dim = 0, keepdim=True)
        f_i_aggregation_list.append(f_i_aggregation)
    feature_map_aggregation = torch.concatenate(f_i_aggregation_list)

    return feature_map_aggregation

        

class FeatureExtractor(nn.Module):
    def __init__(self, model : nn.Module) -> None:
        super(FeatureExtractor, self).__init__()
        self.feature = nn.Sequential(*list(model.children())[:-2])

    def forward(self, x):
        x = self.feature(x)
        return x
    
class Feature2Class(nn.Module):
    def __init__(self, model : nn.Module) -> None:
        super(Feature2Class, self).__init__()
        self.avgpool = model.avgpool
        self.fc = model.fc
    
    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    resnet = resnet50(pretrained = True)
    feature_extractor = FeatureExtractor(resnet)
    feature_extractor.to(DEVICE)
    feature_extractor.eval()

    feature2class = Feature2Class(resnet)
    feature2class.to(DEVICE)
    feature2class.eval()

    image1 = load_image("demo-frames/frame_0016.png")
    image2 = load_image("demo-frames/frame_0017.png")

    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)

    flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

    f_1 = feature_extractor(image1)
    f_2 = feature_warp(f_1, flow_up)

    prob_pred = feature2class(f_2)

    prob = resnet(image2)

    print(torch.argmax(prob_pred, dim=1))
    print(torch.argmax(prob, dim=1))