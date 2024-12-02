
import timm
import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.FSEL_modules import DRP_1, DRP_2, DRP_3, JDPM, ETB , PFAFM , FSFMB, DRD_1, DRD_2, DRD_3
from transformers import AutoModel
from PIL import Image
from timm.data.transforms_factory import create_transform
import requests


'''
backbone: resnet50
'''


class Network(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=128):
        super(Network, self).__init__()
       # self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=False, in_chans=3, features_only=True)
        self.shared_encoder = AutoModel.from_pretrained("nvidia/MambaVision-B-1K", trust_remote_code=True)
        
        base_d_state = 4
        base_H_W = 13
        # self.dePixelShuffle = torch.nn.PixelShuffle(2)

        # self.up = nn.Sequential(
        #     nn.Conv2d(channels//4, channels, kernel_size=1),nn.BatchNorm2d(channels),
        #     nn.Conv2d(channels, channels, kernel_size=3, padding=1),nn.BatchNorm2d(channels),nn.ReLU(True)
        # )
        # self.channel = channels
        # self.ETB_5 = ETB(2048+channels, channels)
        # self.ETB_4 = ETB(1024+channels, channels)
        # self.ETB_3 = ETB(512+channels, channels)
        # self.ETB_2 = ETB(256+channels, channels)

        # self.JDPM = JDPM(2048, channels)

        # self.DRP_1 = DRP_1(channels, channels)
        # self.DRP_2 = DRP_2(channels, channels)
        # self.DRP_3 = DRP_3(channels,channels)
        self.dePixelShuffle = torch.nn.PixelShuffle(2)
       
        self.up = nn.Sequential(
            nn.Conv2d(channels//4, channels, kernel_size=1),nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),nn.BatchNorm2d(channels),nn.ReLU(True)
        )

        self.FSFMB_5 = FSFMB(
                hidden_dim=int(1024+channels),
                out_channel=channels,
                norm_layer=nn.LayerNorm,
                H_W = base_H_W,
            )
        self.FSFMB_4 = FSFMB(
                hidden_dim=int(512+channels),
                out_channel=channels,
                norm_layer=nn.LayerNorm,
                H_W = base_H_W*2,
            )
        self.FSFMB_3 = FSFMB(
                hidden_dim=int(256+channels),
                out_channel=channels,
                norm_layer=nn.LayerNorm,
                H_W = base_H_W*4,
            )
        self.FSFMB_2 = FSFMB(
                hidden_dim=int(128+channels),
                out_channel=channels,
                norm_layer=nn.LayerNorm,
                H_W = base_H_W*8,
            )

        # self.ETB_5 = ETB(1024+channels, channels)
        # self.ETB_4 = ETB(512+channels, channels)
        # self.ETB_3 = ETB(256+channels, channels)
        # self.ETB_2 = ETB(128+channels, channels)

        self.PFAFM = PFAFM(1024, channels)

        # self.DRP_1 = DRP_1(channels, channels)
        # self.DRP_2 = DRP_2(channels, channels)
        # self.DRP_3 = DRP_3(channels,channels)

        self.DRD_1 = DRD_1(channels, channels)
        self.DRD_2 = DRD_2(channels, channels)
        self.DRD_3 = DRD_3(channels,channels)


    def forward(self, x):
        image = x
        _, _, H, W = image.shape
        # Feature Extraction
        # en_feats = self.shared_encoder(x)
        # # eval mode for inference
        # # model.cuda().eval()

        # # train mode for inference
        # en_feats.cuda().train()
        model = self.shared_encoder

        # train mode for inference
        model.cuda().train()

        # input_resolution = (3, 416, 416)  # MambaVision supports any input resolutions

        # transform = create_transform(input_size=input_resolution,
        #                             is_training=True,
        #                             mean=model.config.mean,
        #                             std=model.config.std,
        #                             crop_mode=model.config.crop_mode,
        #                             crop_pct=model.config.crop_pct)
        # inputs = transform(image).unsqueeze(0).cuda()
        # model inference
        out_avg_pool, en_feats = model(image)
        x1, x2, x3, x4 = en_feats


        p1 = self.PFAFM(x4)
        x5_4 = p1
        x5_4_1 = x5_4.expand(-1, 128, -1, -1)

        x4   = self.FSFMB_5(torch.cat((x4,x5_4_1),1))
        x4_up = self.up(self.dePixelShuffle(x4))

        x3   = self.FSFMB_4(torch.cat((x3,x4_up),1))
        x3_up = self.up(self.dePixelShuffle(x3))

        x2   = self.FSFMB_3(torch.cat((x2,x3_up),1))
        x2_up = self.up(self.dePixelShuffle(x2))


        x1   = self.FSFMB_2(torch.cat((x1,x2_up),1))


        x4 = self.DRD_1(x4,x5_4)
        x3 = self.DRD_1(x3,x4)
        x2 = self.DRD_2(x2,x3,x4)
        x1 = self.DRD_3(x1,x2,x3,x4)


        p0 = F.interpolate(p1, size=image.size()[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(x4, size=image.size()[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(x3, size=image.size()[2:], mode='bilinear', align_corners=True)
        f2 = F.interpolate(x2, size=image.size()[2:], mode='bilinear', align_corners=True)
        f1 = F.interpolate(x1, size=image.size()[2:], mode='bilinear', align_corners=True)


        return p0, f4, f3, f2, f1
 


        # p1 = self.JDPM(x4)
        # x5_4 = p1
        # x5_4_1 = x5_4.expand(-1, self.channel, -1, -1)

        # x4   = self.ETB_5(torch.cat((x4,x5_4_1),1))
        # x4_up = self.up(self.dePixelShuffle(x4))

        # x3   = self.ETB_4(torch.cat((x3,x4_up),1))
        # x3_up = self.up(self.dePixelShuffle(x3))

        # x2   = self.ETB_3(torch.cat((x2,x3_up),1))
        # x2_up = self.up(self.dePixelShuffle(x2))

        # x1   = self.ETB_2(torch.cat((x1,x2_up),1))

        # x4 = self.DRP_1(x4,x5_4)
        # x3 = self.DRP_1(x3,x4)
        # x2 = self.DRP_2(x2,x3,x4)
        # x1 = self.DRP_3(x1,x2,x3,x4)

        # p0 = F.interpolate(p1, size=image.size()[2:], mode='bilinear', align_corners=True)
        # f4 = F.interpolate(x4, size=image.size()[2:], mode='bilinear', align_corners=True)
        # f3 = F.interpolate(x3, size=image.size()[2:], mode='bilinear', align_corners=True)
        # f2 = F.interpolate(x2, size=image.size()[2:], mode='bilinear', align_corners=True)
        # f1 = F.interpolate(x1, size=image.size()[2:], mode='bilinear', align_corners=True)

        # return p0, f4, f3, f2, f1