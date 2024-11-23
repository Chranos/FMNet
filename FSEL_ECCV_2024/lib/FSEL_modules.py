import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from einops import rearrange
import numbers
from torch.nn import Softmax


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

    def initialize(self):
        weight_init(self)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

    def initialize(self):
        weight_init(self)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim, bias=bias)
        self.dwconv2 = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim*4, dim, kernel_size=1, bias=bias)
        self.weight = nn.Sequential(
            nn.Conv2d(dim, dim // 16, 1, bias=True),
            nn.BatchNorm2d(dim // 16),
            nn.ReLU(True),
            nn.Conv2d(dim // 16, dim, 1, bias=True),
            nn.Sigmoid())
        self.weight1 = nn.Sequential(
            nn.Conv2d(dim*2, dim // 16, 1, bias=True),
            nn.BatchNorm2d(dim // 16),
            nn.ReLU(True),
            nn.Conv2d(dim // 16, dim*2, 1, bias=True),
            nn.Sigmoid())
    def forward(self, x):

        x_f = torch.abs(self.weight(torch.fft.fft2(x.float()).real)*torch.fft.fft2(x.float()))
        x_f_gelu = F.gelu(x_f) * x_f

        x_s   = self.dwconv1(x)
        x_s_gelu = F.gelu(x_s) * x_s

        x_f = torch.fft.fft2(torch.cat((x_f_gelu,x_s_gelu),1))
        x_f = torch.abs(torch.fft.ifft2(self.weight1(x_f.real) * x_f))

        x_s = self.dwconv2(torch.cat((x_f_gelu,x_s_gelu),1))
        out = self.project_out(torch.cat((x_f,x_s),1))

        return out

    def initialize(self):
        weight_init(self)

def custom_complex_normalization(input_tensor, dim=-1):
    real_part = input_tensor.real
    imag_part = input_tensor.imag
    norm_real = F.softmax(real_part, dim=dim)
    norm_imag = F.softmax(imag_part, dim=dim)

    normalized_tensor = torch.complex(norm_real, norm_imag)

    return normalized_tensor

class Attention_F(nn.Module):
    def __init__(self, dim, num_heads, bias,):
        super(Attention_F, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)
        self.weight = nn.Sequential(
            nn.Conv2d(dim, dim // 16, 1, bias=True),
            nn.BatchNorm2d(dim // 16),
            nn.ReLU(True),
            nn.Conv2d(dim // 16, dim, 1, bias=True),
            nn.Sigmoid())
    def forward(self, x):
        b, c, h, w = x.shape

        q_f = torch.fft.fft2(x.float())
        k_f = torch.fft.fft2(x.float())
        v_f = torch.fft.fft2(x.float())

        q_f = rearrange(q_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_f = rearrange(k_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_f = rearrange(v_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_f = torch.nn.functional.normalize(q_f, dim=-1)
        k_f = torch.nn.functional.normalize(k_f, dim=-1)
        attn_f = (q_f @ k_f.transpose(-2, -1)) * self.temperature
        attn_f = custom_complex_normalization(attn_f, dim=-1)
        out_f = torch.abs(torch.fft.ifft2(attn_f @ v_f))
        out_f = rearrange(out_f, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_f_l = torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(x.float()).real)*torch.fft.fft2(x.float())))
        out = self.project_out(torch.cat((out_f,out_f_l),1))
        return out

class Attention_S(nn.Module):
    def __init__(self, dim, num_heads, bias,):
        super(Attention_S, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv1conv_1 = nn.Conv2d(dim,dim,kernel_size=1)
        self.qkv2conv_1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.qkv3conv_1 = nn.Conv2d(dim, dim, kernel_size=1)


        self.qkv1conv_3 = nn.Conv2d(dim, dim//2, kernel_size=3, stride=1, padding=1, groups=dim//2, bias=bias)
        self.qkv2conv_3 = nn.Conv2d(dim, dim//2, kernel_size=3, stride=1, padding=1, groups=dim//2, bias=bias)
        self.qkv3conv_3 = nn.Conv2d(dim, dim//2, kernel_size=3, stride=1, padding=1, groups=dim//2, bias=bias)

        self.qkv1conv_5 = nn.Conv2d(dim, dim // 2, kernel_size=5, stride=1, padding=2, groups=dim//2, bias=bias)
        self.qkv2conv_5 = nn.Conv2d(dim, dim // 2, kernel_size=5, stride=1, padding=2, groups=dim//2, bias=bias)
        self.qkv3conv_5 = nn.Conv2d(dim, dim // 2, kernel_size=5, stride=1, padding=2, groups=dim//2, bias=bias)


        self.conv_3      = nn.Conv2d(dim, dim//2, kernel_size=3, stride=1, padding=1, groups=dim//2, bias=bias)
        self.conv_5      = nn.Conv2d(dim, dim // 2, kernel_size=5, stride=1, padding=2, groups=dim//2, bias=bias)
        self.project_out = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q_s = torch.cat((self.qkv1conv_3(self.qkv1conv_1(x)),self.qkv1conv_5(self.qkv1conv_1(x))),1)
        k_s = torch.cat((self.qkv2conv_3(self.qkv2conv_1(x)),self.qkv2conv_5(self.qkv2conv_1(x))),1)
        v_s = torch.cat((self.qkv3conv_3(self.qkv3conv_1(x)),self.qkv3conv_5(self.qkv3conv_1(x))),1)

        q_s = rearrange(q_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_s = rearrange(k_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_s = rearrange(v_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_s = torch.nn.functional.normalize(q_s, dim=-1)
        k_s = torch.nn.functional.normalize(k_s, dim=-1)
        attn_s = (q_s @ k_s.transpose(-2, -1)) * self.temperature
        attn_s = attn_s.softmax(dim=-1)
        out_s = (attn_s @ v_s)
        out_s = rearrange(out_s, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_s_l = torch.cat((self.conv_3(x),self.conv_5(x)),1)
        out = self.project_out(torch.cat((out_s,out_s_l),1))

        return out

    def initialize(self):
        weight_init(self)

class Module1(nn.Module):
    def __init__(self, mode='dilation', dim=128, num_heads=8, ffn_expansion_factor=4, bias=False,
                 LayerNorm_type='WithBias'):
        super(Module1, self).__init__()
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn_S = Attention_S(dim, num_heads, bias)
        self.attn_F = Attention_F(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + torch.add(self.attn_F(self.norm1(x)),self.attn_S(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))
        return x


class ETB(nn.Module): # ETB (Entanglement Transformer Block)
    def __init__(self, in_channel, out_channel):
        super(ETB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1), nn.BatchNorm2d(out_channel),nn.ReLU(True)
        )
        self.reduce  = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, 1),nn.BatchNorm2d(out_channel),nn.ReLU(True)
        )
        self.relu = nn.ReLU(True)
        self.Module1 = Module1(dim=out_channel)

    def forward(self, x):
        x0 = self.conv1(x)
        x_FT = self.Module1(x0)
        x    = self.reduce(torch.cat((x0,x_FT),1))+x0
        return x

class PFAFM(nn.Module): # Pyramid Frequency Attention Fusion Module
    def __init__(self, dim,in_dim):
        super(PFAFM, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim,in_dim , 3,padding=1),nn.BatchNorm2d(in_dim),
             nn.ReLU(True))
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.ReLU(True)
        )


        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.ReLU(True)
        )
        self.query_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma2 = nn.Parameter(torch.zeros(1))


        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.ReLU(True)
        )
        self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.value_conv3 =nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = nn.Parameter(torch.zeros(1))


        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.ReLU(True)
        )
        self.query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = nn.Parameter(torch.zeros(1))


        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1),nn.BatchNorm2d(down_dim),  nn.ReLU(True)  #如果batch=1 ，进行batchnorm会有问题
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.ReLU(True)
        )

        self.out = nn.Sequential(
            nn.Conv2d(down_dim, down_dim//2, kernel_size=3, padding=1), nn.BatchNorm2d(down_dim//2), nn.ReLU(True),
            nn.Conv2d(down_dim//2, 1, kernel_size=1)
        )


        self.weight = nn.Sequential(
            nn.Conv2d(down_dim, down_dim // 16, 1, bias=True),
            nn.BatchNorm2d(down_dim // 16),
            nn.ReLU(True),
            nn.Conv2d(down_dim // 16, down_dim, 1, bias=True),
            nn.Sigmoid())

        self.softmax = Softmax(dim=-1)
        self.norm = nn.BatchNorm2d(down_dim)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)

       
        conv2 = self.conv2(x)
        m_batchsize, C, height, width = conv2.size()
        proj_query2 = self.query_conv2(conv2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key2 = self.key_conv2(conv2).view(m_batchsize, -1, width * height)
        energy2 = torch.bmm(proj_query2, proj_key2)
        attention2 = self.softmax(energy2)
        proj_value2 = self.value_conv2(conv2).view(m_batchsize, -1, width * height)
        out2 = torch.bmm(proj_value2, attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)
        out2 = self.gamma2*out2
        out2_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(out2.float()).real)*torch.fft.fft2(out2.float())))))
      # F1_3_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(F13s.float()).real)*torch.fft.fft2(F13s.float())))))
       
        F_2 = torch.add(out2_f , conv2)
        #out2 = self.gamma2* out2 + conv2
        #F1_2_s = 


        conv3 = self.conv3(x)
        m_batchsize, C, height, width = conv3.size()
        proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)
        energy3 = torch.bmm(proj_query3, proj_key3)
        attention3 = self.softmax(energy3)
        proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        out3 = out3.view(m_batchsize, C, height, width)
        out3 = self.gamma3*out3
        out3_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(out3.float()).real)*torch.fft.fft2(out3.float())))))
        F_3 = torch.add(out3_f, conv3)
        #out3 = self.gamma3 * out3 + conv3

        conv4 = self.conv4(x)
        m_batchsize, C, height, width = conv4.size()
        proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)
        energy4 = torch.bmm(proj_query4, proj_key4)
        attention4 = self.softmax(energy4)
        proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        out4 = out4.view(m_batchsize, C, height, width)
        out4 = self.gamma4*out4
        out4_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(out4.float()).real)*torch.fft.fft2(out4.float())))))
        F_4 = torch.add(out4_f, conv4)
        #out4 = self.gamma4 * out4 + conv4

        conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear') # 如果batch设为1，这里就会有问题。

        F_out = self.out(self.fuse(torch.cat((conv1, F_2, F_3,F_4, conv5), 1)))

        return F_out



class JDPM(nn.Module): # JDPM (Joint Domain Perception Module)
    def __init__(self, channels, in_channels):
        super(JDPM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, in_channels, 1), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )

        self.Dconv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 3, padding=3,dilation=3), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )

        self.Dconv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 3, padding=5,dilation=5), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )
        self.Dconv7 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 3, padding=7,dilation=7), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )
        self.Dconv9 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 3, padding=9,dilation=9), nn.BatchNorm2d(in_channels),nn.ReLU(True)
        )

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 5, in_channels, 1), nn.BatchNorm2d(in_channels),nn.ReLU(True)
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1), nn.BatchNorm2d(in_channels//2), nn.ReLU(True),
            nn.Conv2d(in_channels//2, 1, kernel_size=1)
        )

        self.weight = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1, bias=True),
            nn.BatchNorm2d(in_channels // 16),
            nn.ReLU(True),
            nn.Conv2d(in_channels // 16, in_channels, 1, bias=True),
            nn.Sigmoid())

        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(True)

    def forward(self, F1):

       F1_input  = self.conv1(F1)

       F1_3_s = self.Dconv3(F1_input)
       F1_3_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(F1_3_s.float()).real)*torch.fft.fft2(F1_3_s.float())))))
       F1_3 = torch.add(F1_3_s,F1_3_f)

       F1_5_s = self.Dconv5(F1_input + F1_3)
       F1_5_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(F1_5_s.float()).real)*torch.fft.fft2(F1_5_s.float())))))
       F1_5 = torch.add(F1_5_s, F1_5_f)

       F1_7_s = self.Dconv7(F1_input + F1_5)
       F1_7_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(F1_7_s.float()).real)*torch.fft.fft2(F1_7_s.float())))))
       F1_7 = torch.add(F1_7_s, F1_7_f)

       F1_9_s = self.Dconv9(F1_input + F1_7)
       F1_9_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(F1_9_s.float()).real)*torch.fft.fft2(F1_9_s.float())))))
       F1_9 = torch.add(F1_9_s, F1_9_f)

       F_out = self.out(self.reduce(torch.cat((F1_3,F1_5,F1_7,F1_9,F1_input),1)) + F1_input )

       return F_out

class DRP_1(nn.Module): # DRP (Dual-domain Reverse Parser)
    def __init__(self, in_channels, mid_channels):
        super(DRP_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, kernel_size=3, padding=1), nn.BatchNorm2d(mid_channels),nn.ReLU(True),
            nn.Conv2d(mid_channels, 1, kernel_size=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),nn.ReLU(True),
        )

        self.weight = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1, bias=True),
            nn.BatchNorm2d(in_channels // 16),
            nn.ReLU(True),
            nn.Conv2d(in_channels // 16, in_channels, 1, bias=True),
            nn.Sigmoid())

        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(in_channels)

    def forward(self, X, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=X.size()[2:], mode='bilinear',align_corners=True)

        FI  = X

        yt = self.conv(torch.cat([FI, prior_cam.expand(-1, X.size()[1], -1, -1)], dim=1))

        yt_s = self.conv3(yt)
        yt_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(yt.float()).real)*torch.fft.fft2(yt.float())))))
        yt_out = torch.add(yt_s,yt_f)

        r_prior_cam_f = torch.abs(torch.fft.fft2(prior_cam))
        r_prior_cam_f = -1 * (torch.sigmoid(r_prior_cam_f)) + 1
        r_prior_cam_s = -1 * (torch.sigmoid(prior_cam)) + 1
        r_prior_cam = r_prior_cam_s + r_prior_cam_f

        y_ra = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        out = torch.cat([y_ra, yt_out], dim=1)  # 2,128,48,48

        y = self.out(out)
        y = y + prior_cam
        return y

class DRP_2(nn.Module): # DRP (Dual-domain Reverse Parser)
    def __init__(self, in_channels, mid_channels):
        super(DRP_2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),nn.ReLU(True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),nn.ReLU(True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, kernel_size=3, padding=1), nn.BatchNorm2d(mid_channels),nn.ReLU(True),
            nn.Conv2d(mid_channels, 1, kernel_size=1)
        )

        self.weight = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1, bias=True),
            nn.BatchNorm2d(in_channels // 16),
            nn.ReLU(True),
            nn.Conv2d(in_channels // 16, in_channels, 1, bias=True),
            nn.Sigmoid())

        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(True)

    def forward(self, X, x1, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=X.size()[2:], mode='bilinear',align_corners=True)
        x1_prior_cam = F.interpolate(x1, size=X.size()[2:], mode='bilinear', align_corners=True)
        FI = X

        yt = self.conv(torch.cat([FI, prior_cam.expand(-1, X.size()[1], -1, -1), x1_prior_cam.expand(-1, X.size()[1], -1, -1)],dim=1))

        yt_s = self.conv3(yt)
        yt_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(yt.float()).real) * torch.fft.fft2(yt.float())))))
        yt_out = torch.add(yt_s, yt_f)

        r_prior_cam_f = torch.abs(torch.fft.fft2(prior_cam))
        r_prior_cam_f = -1 * (torch.sigmoid(r_prior_cam_f)) + 1
        r_prior_cam_s = -1 * (torch.sigmoid(prior_cam)) + 1
        r_prior_cam = r_prior_cam_s + r_prior_cam_f

        r1_prior_cam_f = torch.abs(torch.fft.fft2(x1_prior_cam))
        r1_prior_cam_f = -1 * (torch.sigmoid(r1_prior_cam_f)) + 1
        r1_prior_cam_s = -1 * (torch.sigmoid(x1_prior_cam)) + 1
        r1_prior_cam = r1_prior_cam_s + r1_prior_cam_f

        r_prior_cam = r_prior_cam + r1_prior_cam

        y_ra = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        out = torch.cat([y_ra, yt_out], dim=1)

        y = self.out(out)
        y = y + prior_cam + x1_prior_cam
        return y

class DRP_3(nn.Module): # DRP (Dual-domain Reverse Parser)
    def __init__(self, in_channels, mid_channels):
        super(DRP_3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),nn.ReLU(True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),nn.ReLU(True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, kernel_size=3, padding=1), nn.BatchNorm2d(mid_channels),nn.ReLU(True),
            nn.Conv2d(mid_channels, 1, kernel_size=1)
        )

        self.weight = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1, bias=True),
            nn.BatchNorm2d(in_channels // 16),
            nn.ReLU(True),
            nn.Conv2d(in_channels // 16, in_channels, 1, bias=True),
            nn.Sigmoid())

        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(True)

    def forward(self, X, x1,x2, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=X.size()[2:], mode='bilinear',align_corners=True)  #
        x1_prior_cam = F.interpolate(x1, size=X.size()[2:], mode='bilinear', align_corners=True)
        x2_prior_cam = F.interpolate(x2, size=X.size()[2:], mode='bilinear', align_corners=True)
        FI = X

        yt = self.conv(torch.cat([FI, prior_cam.expand(-1, X.size()[1], -1, -1), x1_prior_cam.expand(-1, X.size()[1], -1, -1),x2_prior_cam.expand(-1, X.size()[1], -1, -1)],dim=1))

        yt_s = self.conv3(yt)
        yt_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(yt.float()).real) * torch.fft.fft2(yt.float())))))
        yt_out = torch.add(yt_s, yt_f)

        r_prior_cam_f = torch.abs(torch.fft.fft2(prior_cam))
        r_prior_cam_f = -1 * (torch.sigmoid(r_prior_cam_f)) + 1
        r_prior_cam_s = -1 * (torch.sigmoid(prior_cam)) + 1
        r_prior_cam = r_prior_cam_s + r_prior_cam_f

        r1_prior_cam_f = torch.abs(torch.fft.fft2(x1_prior_cam))
        r1_prior_cam_f = -1 * (torch.sigmoid(r1_prior_cam_f)) + 1
        r1_prior_cam_s = -1 * (torch.sigmoid(x1_prior_cam)) + 1
        r1_prior_cam1 = r1_prior_cam_s + r1_prior_cam_f

        r2_prior_cam_f = torch.abs(torch.fft.fft2(x2_prior_cam))
        r2_prior_cam_f = -1 * (torch.sigmoid(r2_prior_cam_f)) + 1
        r2_prior_cam_s = -1 * (torch.sigmoid(x2_prior_cam)) + 1
        r1_prior_cam2 = r2_prior_cam_s + r2_prior_cam_f

        r_prior_cam = r_prior_cam + r1_prior_cam1 + r1_prior_cam2

        y_ra = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        out = torch.cat([y_ra, yt_out], dim=1)

        y = self.out(out)

        y = y + prior_cam + x1_prior_cam + x2_prior_cam

        return y
























