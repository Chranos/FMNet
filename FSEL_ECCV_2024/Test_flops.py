import os, argparse
import cv2
from lib.Network_VIM_test import Network
import torch
from thop import profile


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

print('==> Building model..')
input_features = torch.randn(1, 3, 416, 416)
model = Network(128)
# 统计参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")



