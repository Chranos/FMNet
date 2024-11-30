import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
import numbers
from torch.nn import Softmax
import math
import torch.utils.checkpoint as checkpoint
from functools import partial
from typing import Optional, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
from thop import profile
from torch.nn import init as init
from pdb import set_trace as stx
from pytorch_wavelets import DWTForward, DWTInverse  # (or import DWT, IDWT)
import time

# from model_archs.TTST_arc import Attention as TSA
# from model_archs.layer import *
# from model_archs.comm import *
# from model_archs.uvmb import UVMB
NEG_INF = -1000000

device_id0 = 'cuda:0'
device_id1 = 'cuda:1'
device_id2 = 'cuda:2'

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
        self.dwconv3 = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
       
        self.dwconv5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=bias)
        self.dwconv7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=bias)
       

    
        self.project_out = nn.Conv2d(dim*6, dim, kernel_size=1, bias=bias)
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

        tepx = torch.fft.fft2(x.float())
        x_f = torch.abs(self.weight(tepx.real)*tepx)
        x_f_gelu = F.gelu(x_f) * x_f

        x_s = self.dwconv1(x)
        x_s_gelu = F.gelu(x_s) * x_s

        x_5 = self.dwconv5(x)
        x_5_gelu = F.gelu(x_5) * x_5

        x_7 = self.dwconv7(x)
        x_7_gelu = F.gelu(x_7) * x_7


        x_f = torch.fft.fft2(torch.cat((x_f_gelu,x_s_gelu),1))
        x_f = torch.abs(torch.fft.ifft2(self.weight1(x_f.real) * x_f))

        x_s = self.dwconv3(torch.cat((x_f_gelu,x_s_gelu),1))
        out = self.project_out(torch.cat((x_f,x_s,x_5_gelu,x_7_gelu),1))

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
        tepqkv = torch.fft.fft2(x.float())

        q_f = rearrange(q_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_f = rearrange(k_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_f = rearrange(v_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_f = torch.nn.functional.normalize(q_f, dim=-1)
        k_f = torch.nn.functional.normalize(k_f, dim=-1)
        attn_f = (q_f @ k_f.transpose(-2, -1)) * self.temperature
        attn_f = custom_complex_normalization(attn_f, dim=-1)
        out_f = torch.abs(torch.fft.ifft2(attn_f @ v_f))
        out_f = rearrange(out_f, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_f_l = torch.abs(torch.fft.ifft2(self.weight(tepqkv.real)*tepqkv))
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


class SS2D_F(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    

    def local_scan(self, x, H=14, W=14, w=7, flip=False, column_first=False):
      """Local windowed scan in LocalMamba
      Input: 
          x: [B, C, H, W]
          H, W: original width and height
          column_first: column-wise scan first (the additional direction in VMamba)
      Return: [B, C, L]
      """
      B, C, _, _ = x.shape
      x = x.view(B, C, H, W)
      Hg, Wg = math.floor(H / w), math.floor(W / w)
      if H % w != 0 or W % w != 0:
          newH, newW = Hg * w, Wg * w
          x = x[:,:,:newH,:newW]
      if column_first:
          x = x.view(B, C, Hg, w, Wg, w).permute(0, 1, 4, 2, 5, 3).reshape(B, C, -1)
      else:
          x = x.view(B, C, Hg, w, Wg, w).permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1)
      if flip:
          x = x.flip([-1])
      return x

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        w = H//4
        x1 = self.local_scan(x, H, W, w)
        x2 = self.local_scan(x, H, W, w, column_first = True)

        x_hwwh = torch.stack([x1,x2],dim=1).view(B, 2, -1, L)
        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
 

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
    


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=12):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


## Multi-Scale Feed-Forward Network (MSFN)
class MSFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(MSFN, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv3d(dim, hidden_features*3, kernel_size=(1,1,1), bias=bias)

        self.dwconv1 = nn.Conv3d(hidden_features, hidden_features, kernel_size=(3,3,3), stride=1, dilation=1, padding=1, groups=hidden_features, bias=bias)
        # self.dwconv2 = nn.Conv3d(hidden_features, hidden_features, kernel_size=(3,3,3), stride=1, dilation=2, padding=2, groups=hidden_features, bias=bias)
        # self.dwconv3 = nn.Conv3d(hidden_features, hidden_features, kernel_size=(3,3,3), stride=1, dilation=3, padding=3, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3,3), stride=1, dilation=2, padding=2, groups=hidden_features, bias=bias)
        self.dwconv3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3,3), stride=1, dilation=3, padding=3, groups=hidden_features, bias=bias)


        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=(1,1,1), bias=bias)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.project_in(x)
        x1,x2,x3 = x.chunk(3, dim=1)
        x1 = self.dwconv1(x1).squeeze(2)
        x2 = self.dwconv2(x2.squeeze(2))
        x3 = self.dwconv3(x3.squeeze(2))
        # x1 = self.dwconv1(x1)
        # x2 = self.dwconv2(x2)
        # x3 = self.dwconv3(x3)
        x = F.gelu(x1)*x2*x3
        x = x.unsqueeze(2)
        x = self.project_out(x)
        x = x.squeeze(2)      
        return x



class HybridGate(nn.Module):
    '''
    hybrid feed-forward with channel attention and MLP layer
    '''
    def __init__(self, dim, mlp_ratio):# mlp_ratio = 2.
        super(HybridGate, self).__init__()
        expand_dim = dim * 2
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim)
        self.ca = CAB(num_feat=dim)

        self.expand = nn.Conv2d(in_channels=dim, out_channels=expand_dim, kernel_size=1, padding=0, stride=1, bias=True)
        # self.soomth = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
   
    def forward(self, x):
        b, c, h, w = x.shape

        x = self.expand(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.ca(x1)

        x2 = x2.view(b, -1, c)
        x2 = self.mlp(x2)
        x2 = x2.view(b, c, h, w)

        out = x1 * x2
        # out = self.soomth(out)
        return out


class FSFMB(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            out_channel: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            LayerNorm_type = 'WithBias',
            H_W: float = 416,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.norm1 = LayerNorm(hidden_dim, LayerNorm_type)
        self.mambascan = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))

        expand_dim = hidden_dim * 2
        mlp_hidden_dim = int(hidden_dim * 2.)

        self.mambascan_f = SS2D_F(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.hybridgate = HybridGate(dim=hidden_dim, mlp_ratio=2.)
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, out_features=hidden_dim)
        self.ca = CAB(num_feat=hidden_dim)

        self.expand = nn.Conv2d(in_channels=hidden_dim, out_channels=expand_dim, kernel_size=1, padding=0, stride=1, bias=True)

        self.ffn = FeedForward(hidden_dim, 4, False)
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        self.idwt = DWTInverse(mode='zero', wave='haar')

        self.weight = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 16, 1, bias=True),
            nn.BatchNorm2d(hidden_dim // 16),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim // 16, hidden_dim, 1, bias=True),
            nn.Sigmoid())

        self.softmax = Softmax(dim=-1)
        self.norm = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(True)
        self.num_heads = 8

        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.catout = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=False)
        self.project_out = nn.Conv2d(hidden_dim , out_channel, kernel_size=1, bias=False)
        self.mkh = nn.Parameter(torch.zeros(torch.Size([3,hidden_dim,H_W,H_W])))


    def combine_subbands(self, Yl, Yh):
        """
        Combines the low-frequency and 3 high-frequency subbands into one region (4 blocks), 
        forming a 2x2 subband structure.

        Parameters:
            Yl: Low-frequency subband, shape [B, C, H/2, W/2]
            Yh: High-frequency subband list, containing 3 directions, shape [B, C, 3, H/2, W/2]

        Returns:
            region: Combined tensor, shape [B, C, H, W]
        """
        B, C, H, W = Yl.size(0), Yl.size(1), Yl.size(2), Yl.size(3)
        Hh, Wh = Yh[0].size(3), Yh[0].size(4)


        H_out, W_out = H + Hh, W + Wh
        region = torch.zeros(B, C, H_out, W_out, device=Yl.device)  # Output region with 2x size

        # Place the low-frequency subband in the top-left corner
        region[:, :, :H, :W] = Yl

        # Place the horizontal high-frequency subband in the top-right corner
        region[:, :, :H, W:] = Yh[0][:, :, 0, :, :]

        # Place the vertical high-frequency subband in the bottom-left corner
        region[:, :, H:, :W] = Yh[0][:, :, 1, :, :]

        # Place the diagonal high-frequency subband in the bottom-right corner
        region[:, :, H:, W:] = Yh[0][:, :, 2, :, :]

        return region
    

    def inverse_wavelet_transform(self, h00):
        """
        使用逆小波变换将两次分解后的子带还原为原始图像。

        参数:
            h00: 包含两次小波分解结果的张量，形状 [B, C, H, W]

        返回:
            recons: 重建后的原始图像，形状与分解前相同
        """
        # 提取分辨率信息
        B, C, H, W = h00.shape
        H2, W2 = H // 2, W // 2  # 第一次分解的子带高度和宽度
        H4, W4 = H2 // 2, W2 // 2  # 第二次分解的子带高度和宽度

        # 提取第二次分解的低频和高频子带
        Yl2 = h00[:, :, :H4, :W4]  # 左上角：最低频子带
        Yh_lf = torch.zeros(B, C, 3, H4, W4, device=h00.device)  # 第二次分解的高频子带
        Yh_lf[:, :, 0, :, :] = h00[:, :, :H4, W4:W2]  # 水平高频
        Yh_lf[:, :, 1, :, :] = h00[:, :, H4:H2, :W4]  # 垂直高频
        Yh_lf[:, :, 2, :, :] = h00[:, :, H4:H2, W4:W2]  # 对角线高频

        # 使用逆小波变换恢复第一次分解的低频部分
        Yl = self.idwt((Yl2, [Yh_lf]))

        # 提取第一次分解的高频子带
        Yh_hf = torch.zeros(B, C, 3, H2, W2, device=h00.device)  # 第一次分解的高频子带

        # 水平高频部分（两次分解）
        Yh_hf_l2 = h00[:, :, :H4, W2:W2 + W4]  # 水平高频的第二次分解
        Yh_hf_hf = torch.zeros(B, C, 3, H4, W4, device=h00.device)
        Yh_hf_hf[:, :, 0, :, :] = h00[:, :, :H4, W2 + W4:]  # 水平的水平高频
        Yh_hf_hf[:, :, 1, :, :] = h00[:, :, H4:H2, W2:W2 + W4]  # 水平的垂直高频
        Yh_hf_hf[:, :, 2, :, :] = h00[:, :, H4:H2, W2 + W4:]  # 水平的对角线高频
        Yh_hf[:, :, 0, :, :] = self.idwt((Yh_hf_l2, [Yh_hf_hf]))  # 恢复水平高频

        # 垂直高频部分（两次分解）
        Yh_vf_l2 = h00[:, :, H2:H2 + H4, :W4]  # 垂直高频的第二次分解
        Yh_vf_hf = torch.zeros(B, C, 3, H4, W4, device=h00.device)
        Yh_vf_hf[:, :, 0, :, :] = h00[:, :, H2:H2 + H4, W4:W2]  # 垂直的水平高频
        Yh_vf_hf[:, :, 1, :, :] = h00[:, :, H2 + H4:, :W4]  # 垂直的垂直高频
        Yh_vf_hf[:, :, 2, :, :] = h00[:, :, H2 + H4:, W4:W2]  # 垂直的对角线高频
        Yh_hf[:, :, 1, :, :] = self.idwt((Yh_vf_l2, [Yh_vf_hf]))  # 恢复垂直高频

        # 对角线高频部分（两次分解）
        Yh_df_l2 = h00[:, :, H2:H2 + H4, W2:W2 + W4]  # 对角线高频的第二次分解
        Yh_df_hf = torch.zeros(B, C, 3, H4, W4, device=h00.device)
        Yh_df_hf[:, :, 0, :, :] = h00[:, :, H2:H2 + H4, W2 + W4:]  # 对角线的水平高频
        Yh_df_hf[:, :, 1, :, :] = h00[:, :, H2 + H4:, W2:W2 + W4]  # 对角线的垂直高频
        Yh_df_hf[:, :, 2, :, :] = h00[:, :, H2 + H4:, W2 + W4:]  # 对角线的对角线高频
        Yh_hf[:, :, 2, :, :] = self.idwt((Yh_df_l2, [Yh_df_hf]))  # 恢复对角线高频

        # 使用第一次分解的低频和高频子带恢复原始图像
        recons = self.idwt((Yl, [Yh_hf]))

        return recons
    
    def expand_to_match_dwt(self, x):
        """
        扩展输入张量 tepx2，使其满足小波分解后的分辨率匹配要求。

        参数:
            tepx2: 输入张量，形状 [B, C, H, W]

        返回:
            tepx2_expanded: 扩展后的输入张量
            pad_sizes: 扩展的边界大小，用于后续恢复原始形状
        """
        B, C, H, W = x.size()

        # 确保分辨率是 2 的倍数
        pad_h = (2 - H % 2) % 2  # 高度需要扩展的大小
        pad_w = (2 - W % 2) % 2  # 宽度需要扩展的大小

        # 扩展边界（对称填充）
        x_expanded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        return x_expanded
 

    def forward(self, input):

        b, c, h, w = input.shape

        input = self.norm1(input)

        x1 = input
        x2 = input

        tepx2 = x2
        # First wavelet decomposition
        Yl, Yh = self.dwt(tepx2)  # Yl: First-level low-frequency, Yh[0]: First-level high-frequency
        
        #rem = tepx2.size(2) % 4
        
        tepx2 = F.pad(tepx2,(0,4,0,4),mode="reflect")
        # Create the output tensor h00 with the same shape as the input x
        h00 = torch.zeros_like(tepx2)

        # Perform another DWT on the first-level low-frequency Yl
        Yl2, Yh_lf = self.dwt(Yl)
        Yl =  self.expand_to_match_dwt(Yl)
        # a = Yl.size(2) 
        # b = Yl2.size(2)
        # c = Yh_lf[0].size(3)
        # print("a = " + str(Yl.size(2)) + "b = " + str(Yl2.size(2)) +"c = " + str(Yh_lf[0].size(3)))
        HW = Yl.size(2)
        h00[:, :, :HW, :HW] = self.combine_subbands(Yl2, Yh_lf)

        # Perform another DWT on the first-level horizontal high-frequency Yh[0][:, :, 0]
        Yl_hf, Yh_hf = self.dwt(Yh[0][:, :, 0, :, :])
        # Yh[0][:, :, 0, :, :] = self.expand_to_match_dwt(Yh[0][:, :, 0, :, :])

        h00[:, :, :HW, HW:HW*2] = self.combine_subbands(Yl_hf, Yh_hf)

        # Perform another DWT on the first-level vertical high-frequency Yh[0][:, :, 1]
        Yl_vf, Yh_vf = self.dwt(Yh[0][:, :, 1, :, :])
        #Yh[0][:, :, 1, :, :] = self.expand_to_match_dwt(Yh[0][:, :, 1, :, :])

        h00[:, :, HW:HW*2, :HW] = self.combine_subbands(Yl_vf, Yh_vf)

        # Perform another DWT on the first-level diagonal high-frequency Yh[0][:, :, 2]
        Yl_df, Yh_df = self.dwt(Yh[0][:, :, 2, :, :])
        # Yh[0][:, :, 2, :, :] = self.expand_to_match_dwt(Yh[0][:, :, 2, :, :])

        h00[:, :, Yl.size(2):HW*2, HW:HW*2] = self.combine_subbands(Yl_df, Yh_df)

        h00 = h00[:,:,:HW*2,:HW*2]
        # print("h00")
        # print(h00.shape)


        h00 = rearrange(h00, "b c h w -> b h w c").contiguous()
        h11 = h00+ self.drop_path(self.mambascan_f(h00))
        h11 = rearrange(h11, "b h w c -> b c h w").contiguous()

        # print("h11")
        # print(h11.shape)

        f_out = self.inverse_wavelet_transform(h11)
        f_out = f_out[:,:,:h,:w].contiguous()
        # print("fout")
        # print(f_out.shape)

        # f_x2 = torch.fft.fft2(x2.float())
        # x_2_res = torch.abs(torch.fft.ifft2(self.weight(f_x2.real)*f_x2))

        # f_x2 = rearrange(f_x2, "b c h w -> b h w c").contiguous()
        # f_x2_o = self.drop_path(self.mambascan_f(f_x2.real))
        # f_x2_o = rearrange(f_x2_o, "b h w c -> b c h w").contiguous()

        # f_x2_o = torch.abs(torch.fft.ifft2(f_x2_o))
        # f_out = self.catout(torch.cat((f_x2_o,x_2_res),1))

        s_x1 = rearrange(x1, "b c h w -> b h w c").contiguous()
        s_x1_o = self.drop_path(self.mambascan(s_x1))
        s_x1_o = rearrange(s_x1_o, "b h w c -> b c h w").contiguous()
        s_out = s_x1_o + x1

        f_a = f_out.view(b, -1, c)
        f_a = self.mlp(f_a)
        f_a = f_a.view(b, c, h, w)

        s_a = s_out.view(b, -1, c)
        s_a = self.mlp(s_a)
        s_a = s_a.view(b, c, h, w)

        q_s = s_a
        k_s = s_a
        v_s = s_a

        q_f = f_a
        k_f = f_a
        v_f = f_a

        q_s = rearrange(q_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_s = rearrange(k_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_s = rearrange(v_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_f = rearrange(q_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_f = rearrange(k_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_f = rearrange(v_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_s = torch.nn.functional.normalize(q_s, dim=-1)
        k_s = torch.nn.functional.normalize(k_s, dim=-1)

        q_f = torch.nn.functional.normalize(q_f, dim=-1)
        k_f = torch.nn.functional.normalize(k_f, dim=-1)

        attn_s = (q_f @ k_s.transpose(-2, -1)) * self.temperature
        attn_f = (q_s @ k_f.transpose(-2, -1)) * self.temperature

        attn_s = attn_s.softmax(dim=-1)
        attn_f = attn_f.softmax(dim=-1)

        out_s = (attn_s @ v_s)
        out_f = (attn_f @ v_s)

        out_s = rearrange(out_s, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_f = rearrange(out_f, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.catout(torch.cat((out_s,out_f),1))

        layer1 = out + input

        x11 = self.norm1(layer1)
        x22 = layer1 # B C H W
        #  x = x*self.skip_scale2 + fft + self.hybridgate(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x11 = self.ffn(x11)

        # tepx22 = torch.fft.fft2(x22.float())
        # x22 = torch.abs(torch.fft.ifft2(self.weight(tepx22.real)*tepx22))

        out_final = x11
        out_final = self.project_out(out_final)
        
        return out_final





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
            nn.Conv2d(down_dim, down_dim, kernel_size=3, dilation=3, padding=3), nn.BatchNorm2d(down_dim), nn.ReLU(True)
        )
        self.query_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma2 = nn.Parameter(torch.zeros(1))


        self.conv3 = nn.Sequential(
            nn.Conv2d(down_dim, down_dim, kernel_size=3, dilation=5, padding=5), nn.BatchNorm2d(down_dim), nn.ReLU(True)
        )
        self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.value_conv3 =nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = nn.Parameter(torch.zeros(1))


        self.conv4 = nn.Sequential(
            nn.Conv2d(down_dim, down_dim, kernel_size=3, dilation=7, padding=7), nn.BatchNorm2d(down_dim), nn.ReLU(True)
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


        self.temperature = nn.Parameter(torch.ones(8, 1, 1))
        self.project_out = nn.Conv2d(down_dim*2, down_dim, kernel_size=1, bias=False)

        self.weight = nn.Sequential(
            nn.Conv2d(down_dim, down_dim // 16, 1, bias=True),
            nn.BatchNorm2d(down_dim // 16),
            nn.ReLU(True),
            nn.Conv2d(down_dim // 16, down_dim, 1, bias=True),
            nn.Sigmoid())

        self.softmax = Softmax(dim=-1)
        self.norm = nn.BatchNorm2d(down_dim)
        self.relu = nn.ReLU(True)
        self.num_heads = 8

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)

       
        conv2 = self.conv2(conv1)
        b, c, h, w = conv2.shape

        q_f_2 = torch.fft.fft2(conv2.float())
        k_f_2 = torch.fft.fft2(conv2.float())
        v_f_2 = torch.fft.fft2(conv2.float())
        tepqkv = torch.fft.fft2(conv2.float())

        q_f_2 = rearrange(q_f_2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_f_2 = rearrange(k_f_2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_f_2 = rearrange(v_f_2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_f_2 = torch.nn.functional.normalize(q_f_2, dim=-1)
        k_f_2 = torch.nn.functional.normalize(k_f_2, dim=-1)
        attn_f_2 = (q_f_2 @ k_f_2.transpose(-2, -1)) * self.temperature
        attn_f_2 = custom_complex_normalization(attn_f_2, dim=-1)
        out_f_2 = torch.abs(torch.fft.ifft2(attn_f_2 @ v_f_2))
        out_f_2 = rearrange(out_f_2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_f_l_2 = torch.abs(torch.fft.ifft2(self.weight(tepqkv.real)*tepqkv))
        out_2 = self.project_out(torch.cat((out_f_2,out_f_l_2),1))
        F_2 = torch.add(out_2, conv2)



        conv3 = self.conv3(conv1+F_2)
        b, c, h, w = conv3.shape

        q_f_3 = torch.fft.fft2(conv3.float())
        k_f_3 = torch.fft.fft2(conv3.float())
        v_f_3 = torch.fft.fft2(conv3.float())
        tepqkv = torch.fft.fft2(conv3.float())

        q_f_3 = rearrange(q_f_3, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_f_3 = rearrange(k_f_3, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_f_3 = rearrange(v_f_3, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_f_3 = torch.nn.functional.normalize(q_f_3, dim=-1)
        k_f_3 = torch.nn.functional.normalize(k_f_3, dim=-1)
        attn_f_3 = (q_f_3 @ k_f_3.transpose(-2, -1)) * self.temperature
        attn_f_3 = custom_complex_normalization(attn_f_3, dim=-1)
        out_f_3 = torch.abs(torch.fft.ifft2(attn_f_3 @ v_f_3))
        out_f_3 = rearrange(out_f_3, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_f_l_3 = torch.abs(torch.fft.ifft2(self.weight(tepqkv.real)*tepqkv))
        out_3 = self.project_out(torch.cat((out_f_3,out_f_l_3),1))
        F_3 = torch.add(out_3, conv3)




        conv4 = self.conv4(conv1+F_3)
        b, c, h, w = conv4.shape

        q_f_4 = torch.fft.fft2(conv4.float())
        k_f_4 = torch.fft.fft2(conv4.float())
        v_f_4 = torch.fft.fft2(conv4.float())
        tepqkv = torch.fft.fft2(conv4.float())

        q_f_4 = rearrange(q_f_4, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_f_4 = rearrange(k_f_4, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_f_4 = rearrange(v_f_4, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_f_4 = torch.nn.functional.normalize(q_f_4, dim=-1)
        k_f_4 = torch.nn.functional.normalize(k_f_4, dim=-1)
        attn_f_4 = (q_f_4 @ k_f_4.transpose(-2, -1)) * self.temperature
        attn_f_4 = custom_complex_normalization(attn_f_4, dim=-1)
        out_f_4 = torch.abs(torch.fft.ifft2(attn_f_4 @ v_f_4))
        out_f_4 = rearrange(out_f_4, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_f_l_4 = torch.abs(torch.fft.ifft2(self.weight(tepqkv.real)*tepqkv))
        out_4 = self.project_out(torch.cat((out_f_4,out_f_l_4),1))
        F_4 = torch.add(out_4, conv4)



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




class DRD_1(nn.Module): # DRP (Dual-domain Reverse Parser)
    def __init__(self, in_channels, mid_channels):
        super(DRD_1, self).__init__()
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
        yt_out = yt_s

        r_prior_cam_f = torch.abs(torch.fft.fft2(prior_cam))
        r_prior_cam_f = -1 * (torch.sigmoid(r_prior_cam_f)) + 1
        r_prior_cam_s = -1 * (torch.sigmoid(prior_cam)) + 1
        r_prior_cam = r_prior_cam_s + r_prior_cam_f

        y_ra = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        out = torch.cat([y_ra, yt_out], dim=1)  # 2,128,48,48

        y = self.out(out)
        y = y + prior_cam
        return y

class DRD_2(nn.Module): # DRP (Dual-domain Reverse Parser)
    def __init__(self, in_channels, mid_channels):
        super(DRD_2, self).__init__()
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
        yt_out = yt_s

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

class DRD_3(nn.Module): # DRP (Dual-domain Reverse Parser)
    def __init__(self, in_channels, mid_channels):
        super(DRD_3, self).__init__()
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
        yt_out = yt_s

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
























