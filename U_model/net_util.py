
import torch as th
import torch.nn.functional as F
from torch import nn

from .size_adapter import SizeAdapter
# from size_adapter import SizeAdapter
import torch
from .arches import *
# from arches import *
from einops import rearrange
import numbers

def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out=self.sigmoid(out)
        return out



class Weight_Fusion(nn.Module):
    def __init__(self, in_planes, ratio=16,L=32,M=2):
        super(Weight_Fusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.in_planes=in_planes
        self.M=M
        d = max(in_planes // ratio, L)

        self.fc1=nn.Sequential(nn.Conv2d(in_planes,d,1,bias=False),
                               nn.ReLU(inplace=True))
        self.fc2=nn.Conv2d(d,in_planes*2,1,1,bias=False)
        self.softmax=nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1,x2):
        # 自适应计算x1 和 x2的嘉禾权重
        x=x1+x2
        avg_out=self.avg_pool(x)
        max_out=self.max_pool(x)
        out = avg_out + max_out

        out = self.fc1(out)
        out_two = self.fc2(out)

        batch_size = x.size(0)

        out_two=out_two.reshape(batch_size,self.M,self.in_planes,-1)
        # out_two = self.softmax(out_two)
        out_two = self.sigmoid(out_two)

        w_1, w_2 = out_two[:, 0:1, :, :], out_two[:, 1:2, :, :]

        w_1 = w_1.reshape(batch_size, self.in_planes, 1, 1)
        w_2 = w_2.reshape(batch_size, self.in_planes, 1, 1)
        out = w_1 * x1 + w_2 * x2
        return out




# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y



## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, in_feat,kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(in_feat, in_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(in_feat, in_feat, kernel_size, bias=bias))

        self.CA = CALayer(in_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        res = self.CA(x)
        res += x
        return res


class EN_Block(nn.Module):

    def __init__(self, in_channels, out_channels,BIN,kernel_size=3, reduction=4, bias=False):
        super(EN_Block, self).__init__()
        self.BIN=BIN
        act = nn.ReLU(inplace=True)
        self.conv=conv(in_channels, out_channels, 3, bias=bias)
        self.CABs = [CAB(out_channels,kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.CABs = nn.Sequential(*self.CABs)


    def forward(self, x):
        x=self.conv(x)
        x=self.CABs(x)
        return x
    
class EN_Trans_Block(nn.Module):

    def __init__(self, in_channels, out_channels, head, ffn_expansion_factor=2.66, LayerNorm_type="WithBias", bias=False):
        super(EN_Trans_Block, self).__init__()

        self.conv=conv(in_channels, out_channels, 3, bias=bias)
        self.trans = nn.Sequential(
            *[
                TransformerBlock_JT(
                    dim=out_channels,
                    num_heads=head,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    mode='intra'
                )
                for i in range(4)
            ]
        )


    def forward(self, x):
        x=self.conv(x)
        x=self.trans(x)
        return x
    

class EN_Trans_Block_Evs(nn.Module):

    def __init__(self, in_channels, out_channels, head, ffn_expansion_factor=2.66, LayerNorm_type="WithBias", bias=False):
        super(EN_Trans_Block_Evs, self).__init__()

        self.conv=conv(in_channels, out_channels, 3, bias=bias)
        self.trans = nn.Sequential(
            *[
                TransformerBlock_Evs(
                    dim=out_channels,
                    num_heads=head,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(4)
            ]
        )


    def forward(self, x):
        x=self.conv(x)
        x=self.trans(x)
        return x
    

# 是最后的decode的组合组件
class DE_Block(nn.Module):

    def __init__(self, in_planes, planes, kernel_size=3, reduction=4, bias=False):
        super(DE_Block, self).__init__()
        act = nn.ReLU(inplace=True)

        self.up=SkipUpSample(in_planes, planes)
        self.decoder = [CAB(planes, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder = nn.Sequential(*self.decoder)
        self.skip_attn = CAB(planes, kernel_size, reduction, bias=bias, act=act)


    def forward(self, x, skpCn):
        # x4是小尺寸 skpCn是目标尺寸
        # x做双线性插值 并与self.skip_attn(skpCn)做和
        x = self.up(x, self.skip_attn(skpCn))
        x = self.decoder(x)

        return x


class DE_Trans_Block(nn.Module):

    def __init__(self, in_planes, planes, kernel_size=3, reduction=4, bias=False):
        super(DE_Trans_Block, self).__init__()
        act = nn.ReLU(inplace=True)

        self.up=SkipUpSample(in_planes, planes)
        # self.decoder = [CAB(planes, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        # self.decoder = nn.Sequential(*self.decoder)
        self.skip_attn = CAB(planes, kernel_size, reduction, bias=bias, act=act)

        self.decoder = nn.Sequential(
            *[
                TransformerBlock_Evs(
                    dim=planes,
                    num_heads=4,
                    ffn_expansion_factor=2.66,
                    bias=bias,
                    LayerNorm_type="WithBias",
                )
                for i in range(4)
            ]
        )


    def forward(self, x, skpCn):
        # x4是小尺寸 skpCn是目标尺寸
        # x做双线性插值 并与self.skip_attn(skpCn)做和
        x = self.up(x, self.skip_attn(skpCn))
        x = self.decoder(x)

        return x

class DE_VGGT_Block(nn.Module):

    def __init__(self, in_planes, planes, kernel_size=3, reduction=4, bias=False):
        super(DE_VGGT_Block, self).__init__()
        act = nn.ReLU(inplace=True)

        self.up=SkipUpSample(in_planes, planes)
        # self.decoder = [CAB(planes, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        # self.decoder = nn.Sequential(*self.decoder)
        self.skip_attn = CAB(planes, kernel_size, reduction, bias=bias, act=act)

        self.decoder = nn.Sequential(
            *[
                TransformerBlock_JT(
                    dim=planes,
                    num_heads=4,
                    ffn_expansion_factor=2.66,
                    bias=bias,
                    LayerNorm_type="WithBias",
                    mode='intra'
                ),
                TransformerBlock_JT(
                    dim=planes,
                    num_heads=4,
                    ffn_expansion_factor=2.66,
                    bias=bias,
                    LayerNorm_type="WithBias",
                    mode='inter'
                ),
                TransformerBlock_JT(
                    dim=planes,
                    num_heads=4,
                    ffn_expansion_factor=2.66,
                    bias=bias,
                    LayerNorm_type="WithBias",
                    mode='intra'
                ),
                TransformerBlock_JT(
                    dim=planes,
                    num_heads=4,
                    ffn_expansion_factor=2.66,
                    bias=bias,
                    LayerNorm_type="WithBias",
                    mode='inter'
                ),
                TransformerBlock_JT(
                    dim=planes,
                    num_heads=4,
                    ffn_expansion_factor=2.66,
                    bias=bias,
                    LayerNorm_type="WithBias",
                    mode='intra'
                ),
                TransformerBlock_JT(
                    dim=planes,
                    num_heads=4,
                    ffn_expansion_factor=2.66,
                    bias=bias,
                    LayerNorm_type="WithBias",
                    mode='inter'
                )
            ]
        )


    def forward(self, x, skpCn):
        # x4是小尺寸 skpCn是目标尺寸
        # x做双线性插值 并与self.skip_attn(skpCn)做和
        x = self.up(x, self.skip_attn(skpCn))
        x = self.decoder(x)

        return x

class DE_Trans_Block_S(nn.Module):

    def __init__(self, in_planes, planes, kernel_size=3, reduction=4, bias=False):
        super(DE_Trans_Block_S, self).__init__()
        act = nn.ReLU(inplace=True)

        self.up=SkipUpSample(in_planes, planes)
        # self.decoder = [CAB(planes, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        # self.decoder = nn.Sequential(*self.decoder)
        self.skip_attn = CAB(planes, kernel_size, reduction, bias=bias, act=act)

        self.decoder = nn.Sequential(
            *[
                TransformerBlock(
                    dim=planes,
                    num_heads=4,
                    ffn_expansion_factor=2.66,
                    bias=bias,
                    LayerNorm_type="WithBias",
                )
                for i in range(4)
            ]
        )


    def forward(self, x, skpCn):
        # x4是小尺寸 skpCn是目标尺寸
        # x做双线性插值 并与self.skip_attn(skpCn)做和
        x = self.up(x, self.skip_attn(skpCn))
        x = self.decoder(x)

        return x

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




class Spatio_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Spatio_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, inp):

        b, c, h, w = inp.shape

        q = self.q(inp)  # image
        k = self.k(inp)  # event
        v = self.v(inp)  # event

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = ( k.transpose(-2, -1)@q ) * self.temperature
        attn = attn.softmax(dim=-1)

        return attn,v

## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out
    

class AttentionConv3D(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(AttentionConv3D, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Conv3d: input shape [B, C, T, H, W]
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv3d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # x shape: [B, C, H, W]
        b, c, h, w = x.shape

        # Expand x to [B, C, 1, H, W] to simulate T=1
        x = x.unsqueeze(2)

        # Apply Conv3d
        qkv = self.qkv_dwconv(self.qkv(x))  # [B, 3C, 1, H, W]
        q, k, v = qkv.chunk(3, dim=1)

        # Flatten spatial+temporal dimensions
        q = rearrange(q, "b (head c) t h w -> b head c (t h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) t h w -> b head c (t h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) t h w -> b head c (t h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        # Reshape back
        out = rearrange(out, "b head c (t h w) -> b (head c) t h w", head=self.num_heads, t=1, h=h, w=w)
        out = self.project_out(out)

        # Squeeze temporal dim -> [B, C, H, W]
        out = out.squeeze(2)
        return out


class FusionAttentionConv3D(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(FusionAttentionConv3D, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Q from evs
        self.q_proj = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, bias=bias),
            nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias)
        )

        # K, V from img
        self.kv_proj = nn.Sequential(
            nn.Conv3d(dim, dim * 2, kernel_size=1, bias=bias),
            nn.Conv3d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=bias)
        )

        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, img, evs):
        # img / evs shape: [B, C, H, W]
        b, c, h, w = img.shape

        # Expand to [B, C, 1, H, W] to simulate T=1
        img = img.unsqueeze(2)
        evs = evs.unsqueeze(2)

        # Q from evs
        q = self.q_proj(evs)  # [B, C, 1, H, W]

        # K, V from img
        kv = self.kv_proj(img)  # [B, 2C, 1, H, W]
        k, v = kv.chunk(2, dim=1)

        # Rearranging for multi-head attention
        q = rearrange(q, "b (head c) t h w -> b head c (t h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) t h w -> b head c (t h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) t h w -> b head c (t h w)", head=self.num_heads)

        # L2 normalize along last dim
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        # Reshape back
        out = rearrange(out, "b head c (t h w) -> b (head c) t h w", head=self.num_heads, t=1, h=h, w=w)
        out = self.project_out(out)

        # Squeeze temporal dimension -> [B, C, H, W]
        out = out.squeeze(2)
        return out
    
class AttentionConv3D_VGG(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True, mode='both'):
        super().__init__()
        assert mode in ['intra', 'inter', 'both'], "mode must be 'intra', 'inter', or 'both'"
        self.mode = mode
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        if self.mode in ['inter', 'both']:
            self.qkv_inter = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=bias)
            self.qkv_inter_dw = nn.Conv3d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
            self.proj_inter = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)

        if self.mode in ['intra', 'both']:
            self.qkv_intra = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
            self.qkv_intra_dw = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
            self.proj_intra = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # Input: [B, T, H, W]
        if x.dim() == 4:
            x = x.unsqueeze(2)  # [B, 1, T, H, W]
        B, C, T, H, W = x.shape
        out = 0

        if self.mode in ['inter', 'both']:
            qkv_inter = self.qkv_inter_dw(self.qkv_inter(x))  # [B, 3C, T, H, W]
            q, k, v = qkv_inter.chunk(3, dim=1)
            q = rearrange(q, 'b (head c) t h w -> b head c (t h w)', head=self.num_heads)
            k = rearrange(k, 'b (head c) t h w -> b head c (t h w)', head=self.num_heads)
            v = rearrange(v, 'b (head c) t h w -> b head c (t h w)', head=self.num_heads)

            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)

            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = attn.softmax(dim=-1)
            out_inter = attn @ v
            out_inter = rearrange(out_inter, 'b head c (t h w) -> b (head c) t h w', head=self.num_heads, t=T, h=H, w=W)
            out_inter = self.proj_inter(out_inter)
            out = out + out_inter

        if self.mode in ['intra', 'both']:
            # B, C, T, H, W = x.shape
            x_2d = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # [B*T, C, H, W]
            # x_2d = x.reshape(B * C, T, H, W)  # [B*T, C, H, W]
            qkv = self.qkv_intra_dw(self.qkv_intra(x_2d))  # [B*T, 3C, H, W]
            q, k, v = qkv.chunk(3, dim=1)
            q = rearrange(q, 'bt (head c) h w -> bt head c (h w)', head=self.num_heads)
            k = rearrange(k, 'bt (head c) h w -> bt head c (h w)', head=self.num_heads)
            v = rearrange(v, 'bt (head c) h w -> bt head c (h w)', head=self.num_heads)

            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)

            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = attn.softmax(dim=-1)
            out_intra = attn @ v
            out_intra = rearrange(out_intra, 'bt head c (h w) -> bt (head c) h w', head=self.num_heads, h=H, w=W)
            out_intra = self.proj_intra(out_intra)
            out_intra = out_intra.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
            out = out + out_intra

        return out.squeeze(2) if out.shape[2] == 1 else out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
    

class TransformerBlock_Evs(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_Evs, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = AttentionConv3D(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
    

class TransformerBlock_Cross(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_Cross, self).__init__()

        self.norm0 = LayerNorm(dim, LayerNorm_type)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = FusionAttentionConv3D(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, img, evs):
        x = img + self.attn(self.norm1(img), self.norm0(evs))
        x = x + self.ffn(self.norm2(x))

        return x
    

class TransformerBlock_JT(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, mode):
        super(TransformerBlock_JT, self).__init__()
        assert mode in ['intra', 'inter'], "mode must be 'intra' or 'inter'"
        self.mode = mode
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = AttentionConv3D_VGG(dim, num_heads, bias, self.mode)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # print(x.shape) # torch.Size([1, 128, 64, 64])
        # y = self.attn(self.norm1(x)) # torch.Size([1, 128, 4, 64, 64])
        # print(y.shape)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class EN_Trans_Block_Fus(nn.Module):

    def __init__(self, in_channels, out_channels, head, ffn_expansion_factor=2.66, LayerNorm_type="WithBias", bias=False):
        super(EN_Trans_Block_Fus, self).__init__()

        self.conv=conv(in_channels, out_channels, 3, bias=bias)
        self.trans = nn.Sequential(
            *[
                TransformerBlock_JT(
                    dim=out_channels,
                    num_heads=head,
                    ffn_expansion_factor=2.66,
                    bias=bias,
                    LayerNorm_type="WithBias",
                    mode='intra'
                ),
                TransformerBlock_JT(
                    dim=out_channels,
                    num_heads=head,
                    ffn_expansion_factor=2.66,
                    bias=bias,
                    LayerNorm_type="WithBias",
                    mode='inter'
                ),
                TransformerBlock_JT(
                    dim=out_channels,
                    num_heads=head,
                    ffn_expansion_factor=2.66,
                    bias=bias,
                    LayerNorm_type="WithBias",
                    mode='intra'
                ),
                TransformerBlock_JT(
                    dim=out_channels,
                    num_heads=head,
                    ffn_expansion_factor=2.66,
                    bias=bias,
                    LayerNorm_type="WithBias",
                    mode='inter'
                ),
                TransformerBlock_JT(
                    dim=out_channels,
                    num_heads=head,
                    ffn_expansion_factor=2.66,
                    bias=bias,
                    LayerNorm_type="WithBias",
                    mode='intra'
                ),
                TransformerBlock_JT(
                    dim=out_channels,
                    num_heads=head,
                    ffn_expansion_factor=2.66,
                    bias=bias,
                    LayerNorm_type="WithBias",
                    mode='intra'
                ),
                TransformerBlock_JT(
                    dim=out_channels,
                    num_heads=head,
                    ffn_expansion_factor=2.66,
                    bias=bias,
                    LayerNorm_type="WithBias",
                    mode='intra'
                ),
                TransformerBlock_JT(
                    dim=out_channels,
                    num_heads=head,
                    ffn_expansion_factor=2.66,
                    bias=bias,
                    LayerNorm_type="WithBias",
                    mode='intra'
                )
            ]
        )


    def forward(self, x):
        x=self.conv(x)
        x=self.trans(x)
        return x