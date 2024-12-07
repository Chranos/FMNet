
import timm
import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.modules import  PFAE, MFM, FRD_1, FRD_2, FRD_3
from transformers import AutoModel
from PIL import Image
from timm.data.transforms_factory import create_transform
import requests





class Network(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=128):
        super(Network, self).__init__()
        self.shared_encoder = AutoModel.from_pretrained("nvidia/MambaVision-S-1K", trust_remote_code=True)
        
        base_d_state = 4
        base_H_W = 13

        self.dePixelShuffle = torch.nn.PixelShuffle(2)
       
        self.up = nn.Sequential(
            nn.Conv2d(channels//4, channels, kernel_size=1),nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),nn.BatchNorm2d(channels),nn.ReLU(True)
        )
        self.MFM_5 = MFM(
                dim=int(512+channels),
                out_channel=channels,
                input_resolution = (base_H_W,base_H_W),
                mlp_ratio=4,
                num_heads = 8,
                sr_ratio=1,
            )
        self.MFM_4 = MFM(
                dim=int(256+channels),
                out_channel=channels,
                input_resolution = (base_H_W*2,base_H_W*2),
                mlp_ratio=4,
                num_heads = 8,
                sr_ratio=1,
            )
        self.MFM_3 = MFM(
                dim=int(128+channels),
                out_channel=channels,
                input_resolution = (base_H_W*4,base_H_W*4),
                mlp_ratio=8,
                num_heads = 8,
                sr_ratio=1,
            )
        self.MFM_2 = MFM(
                dim=int(64+channels),
                out_channel=channels,
                input_resolution = (base_H_W*8,base_H_W*8),
                mlp_ratio=8,
                num_heads = 8,
                sr_ratio = 1,
            )



        self.PFAE = PFAE(512, channels)


        self.FRD_1 = FRD_1(channels, channels)
        self.FRD_2 = FRD_2(channels, channels)
        self.FRD_3 = FRD_3(channels,channels)


    def forward(self, x):
        image = x
        _, _, H, W = image.shape

        model = self.shared_encoder

        model.cuda().train()

        out_avg_pool, en_feats = model(image)
        x1, x2, x3, x4 = en_feats


        p1 = self.PFAE(x4)
        x5_4 = p1
        x5_4_1 = x5_4.expand(-1, 128, -1, -1)

        x4   = self.MFM_5(torch.cat((x4,x5_4_1),1))
        x4_up = self.up(self.dePixelShuffle(x4))

        x3   = self.MFM_4(torch.cat((x3,x4_up),1))
        x3_up = self.up(self.dePixelShuffle(x3))

        x2   = self.MFM_3(torch.cat((x2,x3_up),1))
        x2_up = self.up(self.dePixelShuffle(x2))


        x1   = self.MFM_2(torch.cat((x1,x2_up),1))


        x4 = self.FRD_1(x4,x5_4)
        x3 = self.FRD_1(x3,x4)
        x2 = self.FRD_2(x2,x3,x4)
        x1 = self.FRD_3(x1,x2,x3,x4)


        p0 = F.interpolate(p1, size=image.size()[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(x4, size=image.size()[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(x3, size=image.size()[2:], mode='bilinear', align_corners=True)
        f2 = F.interpolate(x2, size=image.size()[2:], mode='bilinear', align_corners=True)
        f1 = F.interpolate(x1, size=image.size()[2:], mode='bilinear', align_corners=True)


        return p0, f4, f3, f2, f1
 

