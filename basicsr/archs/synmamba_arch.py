import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import savemat
from einops import rearrange
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from pdb import set_trace as stx
from typing import Optional, Callable
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from basicsr.utils.registry import ARCH_REGISTRY
from torchvision.utils import save_image 
from PIL import Image
from datetime import datetime
# from .spatialmamba import StructureAwareSSM

def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch,int(in_channel/(r**2)), r * in_height, r * in_width
    x1 = x[:, :out_channel, :, :] / 2
    x2 = x[:,out_channel:out_channel * 2, :, :] / 2
    x3 = x[:,out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:,out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  

    def forward(self, x):
        return dwt_init(x)
    
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class MambalAttention(nn.Module):
    def __init__(self,dim):
        super(MambalAttention, self).__init__()
        self.channel_attention = ChannelAttention(dim, squeeze_factor=4)
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3) 
        # self.act=SiLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        _,C,_,_ = x.shape
        
        x = self.channel_attention(x)
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        attention_map = self.sigmoid(self.conv2d(out))
        out = attention_map * x
        return attention_map,out


class SS2D_Spaital(nn.Module):
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
        self.d_model = d_model    # 输入特征维度
        self.d_state = d_state   # 状态维度
        self.d_conv = d_conv    
        self.expand = expand  # 内部特征维度的扩展倍数。 D_inner =expand × D_model
        self.d_inner = int(self.expand * self.d_model)  # 内部特征维度
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank  # 对应公式中的 D_rank,时间步长的秩
        self.mambal_attention = MambalAttention(self.d_inner)

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
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=1, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=1, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=1, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=1, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=1, D, N)

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
            nn.init.constant_(dt_proj.weight, dt_init_std)  # 把dt_proj.weight初始化为常数dt_init_std
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)  # 把dt_proj.weight初始化为均匀分布， [−dt_init_std,dt_init_std) 
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
        K = 1
        xs = torch.stack([x.view(B, -1, L)], dim=1).view(B, 1, -1, L)
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

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)
        # print(x.shape)
        mambal_attention,x = self.mambal_attention(x)
        x = self.act(x)
        
        # print(mambal_attention.shape)
        # print(x.shape)
        attention_weights_flatten = mambal_attention.view(B, -1)
        # print(attention_weights_flatten.shape)
        _, sorted_indices = torch.sort(attention_weights_flatten, dim=1, descending=True)
        _, inverse_indices = torch.sort(sorted_indices, dim=1)

        # act attention
        x_flatten = x.view(B, self.d_inner, -1)
        # print("x_flatten.shape ",x_flatten.shape)
        expanded_indices = sorted_indices.unsqueeze(1).expand(-1, self.d_inner, -1)
        x = torch.gather(x_flatten, dim=2, index=expanded_indices).view(B, self.d_inner, H, W)

        y = self.forward_core(x)
        assert y.dtype == torch.float32

        expanded_inverse = inverse_indices.unsqueeze(1).expand(-1, self.d_inner, -1)
        y = torch.gather(y, dim=2, index=expanded_inverse)

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class SS2D_Fourier(nn.Module):
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
        self.d_model = d_model    # 输入特征维度
        self.d_state = d_state   # 状态维度
        self.d_conv = d_conv    
        self.expand = expand  # 内部特征维度的扩展倍数。 D_inner =expand × D_model
        self.d_inner = int(self.expand * self.d_model)  # 内部特征维度
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank  # 对应公式中的 D_rank,时间步长的秩
        self.mambal_attention = MambalAttention(self.d_inner)

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
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=2, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                        **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                        **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=2, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=2, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)  # (K=2, D, N)
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True)  # (K=2, D, N)

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
            nn.init.constant_(dt_proj.weight, dt_init_std)  # 把dt_proj.weight初始化为常数dt_init_std
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)  # 把dt_proj.weight初始化为均匀分布， [−dt_init_std,dt_init_std) 
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
        B, C, L = x.shape
        K = 2
        x_hwwh = torch.stack([x], dim=1).view(B, 1, -1, L)
        xs = torch.stack([x_hwwh,torch.flip(x_hwwh, dims=[-1])], dim=1).view(B, 2, -1, L)
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

        inv_y = torch.flip(out_y[:, 1], dims=[-1]).view(B, 1, -1, L)

        return out_y[:, 0]+inv_y[:,0]

    def forward(self, x: torch.Tensor,freq_indices, **kwargs):
        B, H, W, C = x.shape
        # print(x.shape)
        # print(self.d_model)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)

        # L = H * W
        x_flatten = x.reshape(B, self.d_inner, -1)  # [B, C, L]
        x_sorted = torch.gather(x_flatten, dim=2, index=freq_indices.expand(B, self.d_inner, -1)).view(B, self.d_inner, -1)

        y = self.forward_core(x_sorted)
        assert y.dtype == torch.float32
        inv_indices = torch.argsort(freq_indices)
        y = torch.gather(y, dim=2, index=inv_indices.expand(B, self.d_inner, -1))  # [B, C, L]

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class LowProcessBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            attn_drop_rate: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            d_state: int = 16,
            ss2d_expand: float = 2,
            ffn_scale=4,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)
        self.ln_3 = norm_layer(hidden_dim)
        
        # Mamba Branch
        self.ss2d = SS2D_Spaital(d_model=hidden_dim, d_state=d_state,expand=ss2d_expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale_1= nn.Parameter(torch.ones(hidden_dim))
        self.skip_scale_2= nn.Parameter(torch.ones(hidden_dim))
        self.skip_scale_3= nn.Parameter(torch.ones(hidden_dim))
        # self.skip_scale_4= nn.Parameter(torch.ones(hidden_dim))
        self.ffn = FFN(num_feat = hidden_dim,ffn_expand = ffn_scale)
        self.fourier = FourierFFN(dim = hidden_dim,ffn_scale = ffn_scale)
        # self.fourier = FourierBlock(hidden_dim)
        self.cat = nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=1, bias=False)

    def forward(self, input, x_size):
        # x [B,HW,C]
        # B, L, C = input.shape
        # input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x_ori = input.permute(0, 3, 1, 2).contiguous()
        
        # mamba
        x_mamba = input*self.skip_scale_1 + self.drop_path(self.ss2d(self.ln_1(input)))# B H W C
        # fourier
        x_fourier = (input*self.skip_scale_3).permute(0, 3, 1, 2).contiguous() + self.fourier(self.ln_3(input).permute(0, 3, 1, 2).contiguous())

        x = self.cat(torch.cat([x_mamba.permute(0, 3, 1, 2).contiguous(), x_fourier], dim=1))
        # x = F.gelu(x)
        x = x + x_ori
        x = x.permute(0, 2, 3, 1).contiguous()

        
        x = x*self.skip_scale_2 + self.ffn(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous() # B H W C
        return x

class SpatialProcessBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            attn_drop_rate: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            d_state: int = 16,
            ss2d_expand: float = 2,
            ffn_scale=4,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)
        
        # Mamba Branch
        self.ss2d = SS2D_Spaital(d_model=hidden_dim, d_state=d_state,expand=ss2d_expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale_1= nn.Parameter(torch.ones(hidden_dim))
        self.skip_scale_2= nn.Parameter(torch.ones(hidden_dim))
        # self.skip_scale_4= nn.Parameter(torch.ones(hidden_dim))
        self.ffn = FFN(num_feat = hidden_dim,ffn_expand = ffn_scale)


    def forward(self, input):
        # mamba
        x = input*self.skip_scale_1 + self.drop_path(self.ss2d(self.ln_1(input)))# B H W C
        x = x*self.skip_scale_2 + self.ffn(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous() # B H W C
        return x
    
class FourierProcessBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            attn_drop_rate: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            d_state: int = 16,
            ss2d_expand: float = 2,
            ffn_scale=4,
            **kwargs,
    ):
        super().__init__()
        self.ln_2 = norm_layer(hidden_dim)
        self.ln_3 = norm_layer(hidden_dim)
        self.drop_path = DropPath(drop_path)
        self.skip_scale_1= nn.Parameter(torch.ones(hidden_dim))
        self.skip_scale_2= nn.Parameter(torch.ones(hidden_dim))
        self.ffn = FFN(num_feat = hidden_dim,ffn_expand = ffn_scale)
        self.fourier = FourierFFN(dim = hidden_dim,ffn_scale = ffn_scale)

    def forward(self, input):
        # x [B,HW,C]
        # B, L, C = input.shape
        # fourier
        x = (input*self.skip_scale_1).permute(0, 3, 1, 2).contiguous() + self.fourier(self.ln_3(input).permute(0, 3, 1, 2).contiguous())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x*self.skip_scale_2 + self.ffn(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous() # B H W C

        return x



# H,W  ——>rfft——> H,W//2+1
def getFourierSort(B, C, H, W):
    freqs_x = torch.linspace(-H//2, H//2 - 1, H)  
    freqs_y = torch.linspace(0, W//2, W//2 + 1)
    freq_grid_x, freq_grid_y = torch.meshgrid(freqs_x, freqs_y, indexing='ij')
    freq_grid = freq_grid_x**2 + freq_grid_y**2
    freq_grid_flat = freq_grid.reshape(-1)

    freq_indices = torch.argsort(freq_grid_flat)
    return freq_indices.to("cuda:0")




class FourierBlock(nn.Module):
    # processing in frequency domain
    def __init__(self,
                hidden_dim: int = 0,
                drop_path: float = 0,
                attn_drop_rate: float = 0,
                norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                d_state: int = 16,
                ss2d_expand: float = 2,
                **kwargs,):
        super(FourierBlock, self).__init__()
        # self.freq_preprocess = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        # process Amplitude
        self.ln_mag_1 = norm_layer(hidden_dim)
        self.ln_pha_1 = norm_layer(hidden_dim)
        self.process_amp = SS2D_Fourier(d_model=hidden_dim, d_state=d_state,expand=ss2d_expand,dropout=attn_drop_rate, **kwargs)
        # # process Phase
        self.process_pha = SS2D_Fourier(d_model=hidden_dim, d_state=d_state,expand=ss2d_expand,dropout=attn_drop_rate, **kwargs)
        # self.conv = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        # Frequency domain processing
        # x_freq = self.freq_preprocess(x)
        x_freq = torch.fft.rfft2(x, norm='backward')
        x_freq = torch.fft.fftshift(x_freq, dim=-2)
        
        mag = (torch.abs(x_freq)).permute(0, 2, 3, 1).contiguous()  # B H W//2+1 C
        pha = (torch.angle(x_freq)).permute(0, 2, 3, 1).contiguous()  # B H W//2+1 C
        mag_ori = mag
        pha_ori = pha
        mag = self.ln_mag_1(mag)
        pha = self.ln_pha_1(pha)

        freq_indices = getFourierSort(B, C, H, W)
        mag = self.process_amp(mag,freq_indices)
        pha = self.process_pha(pha,freq_indices)
        mag = mag_ori + mag
        pha = pha_ori + pha
        # print("mag",mag.shape)
        mag = mag.permute(0, 3, 1, 2).contiguous()  # B H W//2+1 C
        pha = pha.permute(0, 3, 1, 2).contiguous()  # B H W//2+1 C
        # print("mag",mag.shape)

        # Reconstruct frequency domain features
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)+1e-9
        x_out = torch.fft.ifftshift(x_out, dim=-2)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')+1e-9
        x_out = torch.abs(x_out)+1e-9
        # x_out = self.conv(x_out)
        # print("x_out",x_out.shape)
        # print("x",x.shape)
        x_out = x_out + x
        return x_out
        
class FourierFFN(nn.Module):
    def __init__(self, dim, ffn_scale=1):
        super(FourierFFN, self).__init__()

        hidden_features = int(dim)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1)
        self.fourier = FourierBlock(hidden_dim=hidden_features)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1)

    def forward(self, x):
        x = self.project_in(x)
        x = self.fourier(x)
        x = self.project_out(x)
        return x
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        x = self.body(x)
        return x
    
class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        x = self.body(x)
        return x




class FFN(nn.Module):
    def __init__(self, num_feat, ffn_expand=2,bias=False):
        super(FFN, self).__init__()

        hidden_features = int(num_feat * ffn_expand//2)
        self.project_in = nn.Conv2d(num_feat, hidden_features, kernel_size=1, padding=0, stride=1)
        self.dw_3x3 = nn.Conv2d(hidden_features, hidden_features, 3,1,padding=2,groups=hidden_features,dilation=2)
        self.pw_3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, padding=0, stride=1)
        
        self.global_adative3= ChannelAttention(hidden_features,squeeze_factor=6)

        self.project_out = nn.Conv2d(hidden_features//2, num_feat, kernel_size=1, padding=0, stride=1)
        
    def forward(self, x):
        # B,C,H,W = x.shape
        x = self.project_in(x)  # B C H W -> B 4C H W 
        x1 = self.dw_3x3(self.pw_3x3(x))
        x1 = self.global_adative3(x1)
        x1_3, x2_3 = x1.chunk(2, dim=1)
        x1 = F.silu(x1_3)*x2_3

        x = self.project_out(x1)
        
        return x

class FourierProcessStage(nn.Module):
    def __init__(self, dim, n_blocks=1,  ss2d_expand=1.5,ffn_scale=4):
        super().__init__()
        self.l_process = nn.Sequential(*[FourierProcessBlock(hidden_dim=dim, ss2d_expand=ss2d_expand,ffn_scale= ffn_scale ) for _ in range(n_blocks)])
    
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c").contiguous()
        for l_layer in self.l_process:
            x = l_layer(x)
        x = rearrange(x, "b h w c -> b c h w").contiguous()

        return x

class SpatialProcessStage(nn.Module):
    def __init__(self, dim, n_blocks=1,  ss2d_expand=1.5,ffn_scale=4):
        super().__init__()
        self.l_process = nn.Sequential(*[SpatialProcessBlock(hidden_dim=dim, ss2d_expand=ss2d_expand,ffn_scale= ffn_scale ) for _ in range(n_blocks)])
    
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c").contiguous()
        for l_layer in self.l_process:
            x = l_layer(x)
        x = rearrange(x, "b h w c -> b c h w").contiguous()

        return x




class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3) 
        # self.act=SiLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out*x

class ChannelAttention(nn.Module):

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0))

        self.attention_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0))

        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.sig(self.attention_avg(x)+self.attention_max(x))
        return x * y

class FourierNet(nn.Module):
    def __init__(self, in_chn=3, wf=32, n_blocks=[2,2,4],ss2d_expand=2, ffn_scale=4,high_heads=[4,4,4],high_blocks=[2,2,4]):
        super(FourierNet, self).__init__()
        
        self.in_conv = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.out_conv = nn.Conv2d(wf, in_chn , 3, 1, 1)
        self.low_down_group1 = FourierProcessStage(wf, n_blocks=n_blocks[0], ss2d_expand=ss2d_expand,ffn_scale=ffn_scale)
        self.low_down_group2 = FourierProcessStage(wf*2**1, n_blocks=n_blocks[1], ss2d_expand=ss2d_expand,ffn_scale=ffn_scale)
        self.low_down_group3 = FourierProcessStage(wf*2**2, n_blocks=n_blocks[2], ss2d_expand=ss2d_expand,ffn_scale=ffn_scale)
        self.low_up_group2 = FourierProcessStage(wf*2**1, n_blocks=n_blocks[1], ss2d_expand=ss2d_expand,ffn_scale=ffn_scale)
        self.low_up_group1 = FourierProcessStage(wf, n_blocks=n_blocks[0], ss2d_expand=ss2d_expand,ffn_scale=ffn_scale)

        self.low_down_sample1 = Downsample(wf)
        self.low_down_sample2 = Downsample(wf*2**1)
        self.low_up_sample2 = Upsample(wf*2**2)
        self.low_up_sample1 = Upsample(wf*2**1)

        self.low_cat_conv1 = nn.Sequential(nn.Conv2d(wf*2**1, wf, kernel_size=1, bias=False))
        self.low_cat_conv2 = nn.Sequential(nn.Conv2d(wf*2**2, wf*2**1, kernel_size=1, bias=False))
        
    
    def forward(self, x):

        B,C,H,W = x.shape
        
        x_ori = x
        x = self.in_conv(x) # B 3 H W -> B C H W
        
        x_L_down_process_1 = self.low_down_group1(x) # B C H//2 W//2
        x_L_down_process_1_cat = x_L_down_process_1

        x_L_down_sample_1 = self.low_down_sample1(x_L_down_process_1) # B 2C H//4 W//4
        x_L_down_process_2 = self.low_down_group2(x_L_down_sample_1) # B 2C H//4 W//4
        x_L_down_process_2_cat = x_L_down_process_2



        x_L_down_sample_2 = self.low_down_sample2(x_L_down_process_2) # B 4C H//8 W//8
        x_L_down_process_3 = self.low_down_group3(x_L_down_sample_2) # B 4C H//8 W//8
        
 
        x_L_up_sample2 = self.low_up_sample2(x_L_down_process_3) # B 2C H//4 W//4
        x_L_up_sample2 = torch.cat([x_L_up_sample2, x_L_down_process_2_cat], 1) # B 4C H//4 W//4
        x_L_up_sample2 = self.low_cat_conv2(x_L_up_sample2) # B 2C H//4 W//4
        x_L_up_process_2 = self.low_up_group2(x_L_up_sample2) # B 2C H//4 W//4
 
        x_L_up_sample1 = self.low_up_sample1(x_L_up_process_2) # B C H//2 W//2
        x_L_up_sample1 = torch.cat([x_L_up_sample1, x_L_down_process_1_cat], 1) # B 2C H//2 W//2
        x_L_up_sample1 = self.low_cat_conv1(x_L_up_sample1) # B C H//2 W//2
        x_L_up_process_1 = self.low_up_group1(x_L_up_sample1) # B C H//2 W//2
    
        out = self.out_conv(x_L_up_process_1)+x_ori # B 3 H W

        return out

class SpatialNet(nn.Module):
    def __init__(self, in_chn=3, wf=32, n_blocks=[2,2,4],ss2d_expand=2, ffn_scale=4,high_heads=[4,4,4],high_blocks=[2,2,4]):
        super(SpatialNet, self).__init__()
        
        self.in_conv = nn.Conv2d(in_chn, wf, 3, 1, 1)

        self.out_conv = nn.Conv2d(wf, in_chn , 3, 1, 1)
        self.low_down_group1 = SpatialProcessStage(wf, n_blocks=n_blocks[0], ss2d_expand=ss2d_expand,ffn_scale=ffn_scale)
        self.low_down_group2 = SpatialProcessStage(wf*2**1, n_blocks=n_blocks[1], ss2d_expand=ss2d_expand,ffn_scale=ffn_scale)
        self.low_down_group3 = SpatialProcessStage(wf*2**2, n_blocks=n_blocks[2], ss2d_expand=ss2d_expand,ffn_scale=ffn_scale)
        self.low_up_group2 = SpatialProcessStage(wf*2**1, n_blocks=n_blocks[1], ss2d_expand=ss2d_expand,ffn_scale=ffn_scale)
        self.low_up_group1 = SpatialProcessStage(wf, n_blocks=n_blocks[0], ss2d_expand=ss2d_expand,ffn_scale=ffn_scale)

        self.low_down_sample1 = Downsample(wf)
        self.low_down_sample2 = Downsample(wf*2**1)
        self.low_up_sample2 = Upsample(wf*2**2)
        self.low_up_sample1 = Upsample(wf*2**1)

        self.low_cat_conv1 = nn.Sequential(nn.Conv2d(wf*2**1, wf, kernel_size=1, bias=False))
        self.low_cat_conv2 = nn.Sequential(nn.Conv2d(wf*2**2, wf*2**1, kernel_size=1, bias=False))
        
    
    def forward(self, x):

        B,C,H,W = x.shape
        
        x_ori = x
        x = self.in_conv(x) # B 3 H W -> B C H W
        
        x_L_down_process_1 = self.low_down_group1(x) # B C H//2 W//2
        x_L_down_process_1_cat = x_L_down_process_1

        x_L_down_sample_1 = self.low_down_sample1(x_L_down_process_1) # B 2C H//4 W//4
        x_L_down_process_2 = self.low_down_group2(x_L_down_sample_1) # B 2C H//4 W//4
        x_L_down_process_2_cat = x_L_down_process_2

        x_L_down_sample_2 = self.low_down_sample2(x_L_down_process_2) # B 4C H//8 W//8
        x_L_down_process_3 = self.low_down_group3(x_L_down_sample_2) # B 4C H//8 W//8
 
        x_L_up_sample2 = self.low_up_sample2(x_L_down_process_3) # B 2C H//4 W//4
        x_L_up_sample2 = torch.cat([x_L_up_sample2, x_L_down_process_2_cat], 1) # B 4C H//4 W//4
        x_L_up_sample2 = self.low_cat_conv2(x_L_up_sample2) # B 2C H//4 W//4
        x_L_up_process_2 = self.low_up_group2(x_L_up_sample2) # B 2C H//4 W//4
   

 
        x_L_up_sample1 = self.low_up_sample1(x_L_up_process_2) # B C H//2 W//2
        x_L_up_sample1 = torch.cat([x_L_up_sample1, x_L_down_process_1_cat], 1) # B 2C H//2 W//2
        x_L_up_sample1 = self.low_cat_conv1(x_L_up_sample1) # B C H//2 W//2
        x_L_up_process_1 = self.low_up_group1(x_L_up_sample1) # B C H//2 W//2
    
        out = self.out_conv(x_L_up_process_1) + x_ori  # B 3 H W

        return out

class UNet(nn.Module):
    def __init__(self, in_chn=3, wf=32, n_blocks=[2,2,4],ss2d_expand=2, ffn_scale=4,high_heads=[4,4,4],high_blocks=[2,2,4]):
        super(UNet, self).__init__()
        self.four = FourierNet(in_chn=in_chn,wf=wf,n_blocks=n_blocks,ss2d_expand=ss2d_expand,ffn_scale=ffn_scale)
        self.spa = SpatialNet(in_chn=in_chn,wf=wf,n_blocks=n_blocks,ss2d_expand=ss2d_expand,ffn_scale=ffn_scale)
        
        
    
    def forward(self, x):

        B,C,H,W = x.shape
        x_llu = self.four(x)
        x_raw = x_llu*x + x
        x_out = self.spa(x_raw)
        return x_raw,x_out




# @ARCH_REGISTRY.register()
class SynMamba(nn.Module):
    def __init__(self,
                 *,
                 in_chn,
                 wf,
                 n_blocks=[1,1,2,4],
                 ffn_scale=2.0, 
                 fla_scale=384,
                 train_size=128,
                 high_heads=[1,2,4],
                 high_blocks=[2,2,4],
                 **ignore_kwargs):
        super().__init__()
        self.restoration_network = UNet(in_chn=in_chn, wf=wf, n_blocks=n_blocks, ffn_scale=ffn_scale,high_heads=high_heads,high_blocks=high_blocks)

    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))

    def encode_and_decode(self, input, current_iter=None):
        x_raw,x_out = self.restoration_network(input)
        return x_raw,x_out

    def check_image_size(self, x, window_size=8):
        _, _, h, w = x.size()
        mod_pad_h = (window_size - h % (window_size)) % (
            window_size)
        mod_pad_w = (window_size - w % (window_size)) % (
            window_size)
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    @torch.no_grad()
    def test(self, input):
        _, _, h_old, w_old = input.shape
        x_raw,x_out = self.encode_and_decode(input)
        return x_out

    def forward(self, input):

        x_raw,x_out= self.encode_and_decode(input)

        return x_raw,x_out


if __name__== '__main__': 
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # x = torch.randn(1, 3, 600, 400).to(device)
    save_dir = "./feature_results"
    os.makedirs(save_dir, exist_ok=True)
    
    x = torch.randn(1, 3, 256, 256).to(device)
    model = UNet(in_chn=3,
             wf=24,
             n_blocks=[1,1,1,1],
             ffn_scale=4,
             high_heads=[2,4,8],
             high_blocks=[4,6,6]).to(device)

    inp_shape=(3,256, 256)
    from ptflops import get_model_complexity_info
    FLOPS = 0
    macs, params = get_model_complexity_info(model, inp_shape, verbose=False, print_per_layer_stat=True)

    params = float(params[:-4])
    # print('mac', macs)
    print(params)
    macs = float(macs[:-4]) + FLOPS / 1024**3

    print('FLOPs： ', macs,' GMac')
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))/(1000000)}M')