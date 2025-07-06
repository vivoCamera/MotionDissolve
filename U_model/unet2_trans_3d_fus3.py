
from torch import nn
import scipy.stats as st
import torch
from .net_util import *
# from net_util import *
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((1, 1, kernlen, kernlen))
    out_filter = np.repeat(out_filter, channels, axis = 0)
    return out_filter





class EEC(nn.Module):
    def __init__(self, dim, bias=False):
        super(EEC, self).__init__()


        self.Conv=nn.Conv2d(2*dim, dim, kernel_size=1, bias=bias)
        self.CA=ChannelAttention(dim)


    def forward(self, f_img, f_event,Mask):


        assert f_img.shape == f_event.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = f_img.shape

        # transfer finer edges from events to image
        # 这里的mask是用模糊卷积核提取的边缘 对应的是Eq 3
        F_event=f_event*Mask
        # 边缘权重加深
        F_event=f_event+F_event

        # 下面对应的是Eq4
        F_cat = torch.cat([F_event, f_img], dim=1)
        F_conv=self.Conv(F_cat)
        # 计算一个时间维度的注意力
        w_c=self.CA(F_conv)
        F_event=F_event*w_c
        # f_img 是img branch对应的特征
        F_out=F_event+f_img

        return F_out

class EEC_depth(nn.Module):
    def __init__(self, dim, bias=False):
        super(EEC_depth, self).__init__()


        self.Conv=nn.Conv2d(2*dim, dim, kernel_size=1, bias=bias)
        self.Conv2=nn.Conv2d(2*dim, dim, kernel_size=1, bias=bias)
        self.CA=ChannelAttention(dim)
        self.ConvDepth = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=dim),
        )


    def forward(self, f_img, f_event, Mask):


        assert f_img.shape == f_event.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = f_img.shape

        # transfer finer edges from events to image
        # 这里的mask是用模糊卷积核提取的边缘 对应的是Eq 3
        F_event=f_event*Mask
        # 边缘权重加深
        F_event=f_event + F_event

        # 下面对应的是Eq4
        F_cat = torch.cat([F_event, f_img], dim=1)
        F_conv = self.Conv(F_cat)
        F_conv = self.ConvDepth(F_conv) + F_conv

        # 计算一个时间维度的注意力
        w_c=self.CA(F_conv)
        F_event=F_event*w_c
        F_event = self.Conv2(torch.cat([F_event, F_conv], dim=1)) + F_event

        # f_img 是img branch对应的特征
        F_out = F_event + f_img

        return F_out

class ISC(nn.Module):
    def __init__(self, dim, num_heads=4,  bias=False, LayerNorm_type='WithBias'):
        super(ISC, self).__init__()
        self.num_heads = num_heads
        self.norm = LayerNorm(dim, LayerNorm_type)
        self.SA= Spatio_Attention(dim, num_heads, bias)
        self.CA=ChannelAttention(dim)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(2*dim, dim // 16, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(dim // 16, dim, 1, bias=False)
        self.sigmoid= nn.Sigmoid()
    def forward(self, f_img, f_event):

        assert f_img.shape == f_event.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = f_img.shape
        SA_att,V=self.SA(f_img)
        F_img=(V@SA_att)
        F_img = rearrange(F_img, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        CA_att=self.CA(f_img)
        F_img=F_img*CA_att
        F_img=F_img+f_img

        # GSE 和semantic injection部分 这部分focus on img 所以w*img + (1-w)event
        w_i=self.avg_pool(F_img)
        w_e=self.avg_pool(f_event)
        w=torch.cat([w_i,w_e],dim=1)
        w= self.fc2(self.relu1(self.fc1(w)))
        w=self.sigmoid(w)
        F_img=F_img*w
        F_event=f_event*(1-w)

        F_event=F_event+F_img

        return F_event

class ISC_Fus(nn.Module):
    def __init__(self, dim, num_heads=4,  bias=False, LayerNorm_type='WithBias'):
        super(ISC_Fus, self).__init__()

        self.spa_trans = TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    ffn_expansion_factor=2.66,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )

        self.CA=ChannelAttention(dim)
        self.SDL = SDL_attention(dim, dim)
        self.trans = nn.Conv2d(2*dim, dim, kernel_size=1, bias=bias)

    def forward(self, f_img, f_event):

        assert f_img.shape == f_event.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = f_img.shape

        F_img = self.spa_trans(f_img)

        CA_att = self.CA(f_img)
        F_img = F_img*CA_att
        F_img = F_img+f_img

        # SDL fusion
        F_event = torch.cat([f_event, F_img],dim=1)
        # 将其通道维度split一半 然后输入SDL fusion
        F1, F2 = torch.chunk(F_event, chunks=2, dim=1)
        F1, F2 = self.SDL(F1, F2)
        F_event =  self.trans(torch.cat([F1, F2],dim=1))

        F_event = F_event + F_img + f_event

        return F_event
    
def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

# spatial-spectral domain attention learning(SDL)
class SDL_attention(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(SDL_attention, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size-1)//2

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    # HR spatial attention
    def spatial_attention(self, x):
        input_x = self.conv_v_right(x)
        batch, channel, height, width = input_x.size()

        input_x = input_x.view(batch, channel, height * width)
        context_mask = self.conv_q_right(x)
        context_mask = context_mask.view(batch, 1, height * width)
        context_mask = self.softmax_right(context_mask)

        context = torch.matmul(input_x, context_mask.transpose(1,2))
        context = context.unsqueeze(-1)
        context = self.conv_up(context)

        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out
    # HR spectral attention
    def spectral_attention(self, x):

        g_x = self.conv_q_left(x)
        batch, channel, height, width = g_x.size()

        avg_x = self.avg_pool(g_x)
        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)
        context = torch.matmul(avg_x, theta_x)
        context = self.softmax_left(context)
        context = context.view(batch, 1, height, width)

        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out

    def forward(self, x_1, x_2):
        context_spectral = self.spectral_attention(x_1)
        context_spatial = self.spatial_attention(x_2)
        # out = context_spatial + context_spectral
        return context_spectral, context_spatial
    

class Decoder_Trans(nn.Module):
    """Modified version of Unet from SuperSloMo.
    """

    def __init__(self, channels):
        super(Decoder_Trans, self).__init__()
        ######Decoder
        self.up1 = DE_Trans_Block(channels[3], channels[2])
        self.up2 = DE_Trans_Block(channels[2], channels[1])
        self.up3 = DE_Trans_Block(channels[1], channels[0])

    def forward(self, input):
        x4=input[3]
        # 256 —— 128
        # input[2]经过CAB组件 和 x4的双线性插值+通道改变的和
        x3 = self.up1(x4, input[2])
        # 128 —— 64
        x2 = self.up2(x3, input[1])
        # 64 —— 64 逐级实现的fusion 然后event和img各一个decoder套件
        x1 = self.up3(x2, input[0])
        return x1




class Restoration(nn.Module):
    """Modified version of Unet from SuperSloMo.

    """

    def __init__(self, 
                 inChannels_img, 
                 inChannels_event,
                 outChannels, 
                 args,
                 ends_with_relu=False,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type="WithBias"  ## Other option 'BiasFree'
                 ):
        super(Restoration, self).__init__()
        self._ends_with_relu = ends_with_relu
        self.num_heads=4
        self.act = nn.ReLU(inplace=True)

        self.channels = [64, 64, 128, 256]
        self.encoder_img_1 = EN_Trans_Block(inChannels_img, self.channels[0], heads[0], ffn_expansion_factor=ffn_expansion_factor, LayerNorm_type=LayerNorm_type, bias=bias )
        self.encoder_img_2 = EN_Trans_Block_Fus(self.channels[0], self.channels[1], heads[1], ffn_expansion_factor=ffn_expansion_factor, LayerNorm_type=LayerNorm_type, bias=bias )
        self.encoder_img_3 = EN_Trans_Block_Fus(self.channels[1], self.channels[2], heads[2], ffn_expansion_factor=ffn_expansion_factor, LayerNorm_type=LayerNorm_type, bias=bias )
        self.encoder_img_4 = EN_Trans_Block(self.channels[2], self.channels[3], heads[3], ffn_expansion_factor=ffn_expansion_factor, LayerNorm_type=LayerNorm_type, bias=bias )

        self.encoder_event_1 = EN_Trans_Block_Evs(inChannels_event, self.channels[0], heads[0], ffn_expansion_factor=ffn_expansion_factor, LayerNorm_type=LayerNorm_type, bias=bias )
        self.encoder_event_2 = EN_Trans_Block_Evs(inChannels_event, self.channels[0], heads[0], ffn_expansion_factor=ffn_expansion_factor, LayerNorm_type=LayerNorm_type, bias=bias )
        self.encoder_event_22 = EN_Trans_Block_Evs(self.channels[0], self.channels[1], heads[1], ffn_expansion_factor=ffn_expansion_factor, LayerNorm_type=LayerNorm_type, bias=bias )
        self.encoder_event_3 = EN_Trans_Block_Fus(self.channels[1], self.channels[2], heads[2], ffn_expansion_factor=ffn_expansion_factor, LayerNorm_type=LayerNorm_type, bias=bias )
        self.encoder_event_4 = EN_Trans_Block_Fus(self.channels[2], self.channels[3], heads[3], ffn_expansion_factor=ffn_expansion_factor, LayerNorm_type=LayerNorm_type, bias=bias )


        self.down = DownSample()
        self.up = SingleUpSample()

        # eec 用evt的edge信息 加深img的高频信息
        self.EEC_1=EEC_depth(self.channels[0])
        self.EEC_2=EEC_depth(self.channels[1])

        # 用img的fea加深evt的semantic representation
        self.ISC_3=ISC_Fus(self.channels[2])
        self.ISC_4=ISC_Fus(self.channels[3])


        self.decoder_img = Decoder_Trans(self.channels)
        self.decoder_event = Decoder_Trans(self.channels)

        self.fus_de = EN_Trans_Block_Fus(self.channels[0], self.channels[0], heads[0], ffn_expansion_factor=ffn_expansion_factor, LayerNorm_type=LayerNorm_type, bias=bias )

        # self.weight_fusion=Weight_Fusion(self.channels[0])


        self.out = nn.Conv2d(self.channels[0], outChannels, 3, stride=1, padding=1)
        self.trans = nn.Conv2d(self.channels[0]*2, self.channels[0], 3, stride=1, padding=1)
        self.trans2 = nn.Conv2d(self.channels[0]*2, self.channels[0], 3, stride=1, padding=1)

    def blur(self, x, kernel = 21, channels = 3, stride = 1, padding = 'same'):
        # 生成高斯核 并用卷积模拟高斯模糊
        kernel_var = torch.from_numpy(gauss_kernel(kernel, 3, channels)).to(device).float()
        return torch.nn.functional.conv2d(x, kernel_var, stride = stride, padding = int((kernel-1)/2), groups = channels)

    def forward(self, input_img, input_event):
        '''
        input_img: [1, 3, 256, 256]
        input_event: [1, 8, 128, 128]
        当输入input_event修改后，需要改变整体网络的输入，不需要上采样 下采样 只需要对应的尺度就行
        '''

        M0 = torch.clamp(self.blur(torch.sum(torch.abs(input_event), axis = 1, keepdim = True), 
                                             kernel = 7, channels = 1), 0, 1)
        
        # M0.shape:torch.Size([16, 1, 128, 128])

        img_encoder_list = []
        event_encoder_list = []

        ####  feature extraction
        # TODO mapping img: 3 - 64; 
        img_1=self.encoder_img_1(input_img) # [1, 64, 256, 256]

        # event: 8 - 64 [1 64 128 128]
        event_1 = self.encoder_event_1(input_event)

        # 针对256 256
        # branch 1: 原始event先上采样在经过encoder 8 - 64
        # branch 2: 原始event encoder 8-64 上采样
        input_event_up2 = self.up(input_event)
        event_2 = self.encoder_event_2(input_event_up2) 
        event_up2 = self.up(event_1)
        event_2 = self.trans(torch.cat([event_2, event_up2], 1)) + event_2

        img_encoder_list.append(img_1)
        event_encoder_list.append(event_2)

        # 这里的插值用的双线性插值 通道数不变
        down_img_1=self.down(img_1) # [1, 64, 256, 256]
        event_1 = self.trans(torch.cat([event_1, self.down(event_2)], 1)) + event_1 
        
        # 用event加深img的edge信息
        fuse_img_1=self.EEC_1(down_img_1, event_1, M0) # [1, 64, 128, 128]

        event_1 = self.encoder_event_22(event_1)

        # TODO mapping img_fea: 64 - 64; evt_fea: 64 - 64
        img_2=self.encoder_img_2(fuse_img_1) # [1, 64, 128, 128]

        img_encoder_list.append(img_2)
        event_encoder_list.append(event_1)

        down_img_2=self.down(img_2) # [1, 64, 64, 64]
        down_event_2=self.down(event_1) # [1, 64, 64, 64]
        # downsample了 所以这里的高斯模糊核的大小在逐级减小
        M1 = self.blur(M0, kernel=5, channels=1, padding=2, stride=2) # [1, 1, 64, 64]

        fuse_img_2=self.EEC_2(down_img_2,down_event_2, M1) # [1, 64, 64, 64]

        # TODO mapping img_fea: 64 - 128; evt_fea: 64 - 128
        img_3=self.encoder_img_3(fuse_img_2) # [1, 128, 64, 64]
        event_3=self.encoder_event_3(down_event_2) # [1, 128, 64, 64]
        img_encoder_list.append(img_3)
        event_encoder_list.append(event_3)

        # TODO 从这里开始就是ISC 也就是用img来fuse event了
        # 前面是fuse_img经过encoder_img_3 后面是fuse_event经过encoder_event_3
        down_img_3=self.down(img_3) # [1, 128, 32, 32]
        down_event_3=self.down(event_3) # [1, 128, 32, 32]
        fuse_event_3= self.ISC_3(down_img_3,down_event_3) # [1, 128, 32, 32]

        # TODO mapping img_fea: 128 - 256; evt_fea: 128 - 256
        img_4=self.encoder_img_4(down_img_3) # [1, 256, 32, 32]
        event_4=self.encoder_event_4(fuse_event_3) 
        event_4= self.ISC_4(img_4,event_4) # [1, 256, 32, 32]
        img_encoder_list.append(img_4)
        event_encoder_list.append(event_4)

        de_img=self.decoder_img(img_encoder_list) # [1, 64, 256, 256]
        de_event=self.decoder_event(event_encoder_list) # [1, 64, 256, 256]

        # de_fuse=self.weight_fusion(de_img,de_event) # [1, 64, 256, 256]
        de_fuse = self.trans2(torch.cat([de_img,de_event],1) )
        de_fuse = self.fus_de(de_fuse) + de_fuse

        out=self.out(de_fuse) # [1, 3, 256, 256]

        out=out+input_img
        return out


if __name__ == "__main__":
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from thop import profile  # pip install thop

    class DummyOpt:
        def __init__(self):
            self.stage = 9         # 网络迭代次数
            # self.bands = 28        # 波段数
            # self.size = 256        # 输入图像宽度/高度

    opt = DummyOpt()

    model_restoration = Restoration(1, 8, 1, opt).cuda()
    
    # model_restoration.cuda()
    model_restoration.eval()
    # events ([1, 8, 360, 640)
    # RGB ([1, 3, 720, 1280])

    input_img = torch.rand(1, 1, 256, 256).cuda()
    input_event = torch.rand(1, 8, 128, 128).cuda()

    output = model_restoration(input_img, input_event)

    print(output.shape) # model_restoration

    macs, params = profile(model_restoration, inputs=(input_img, input_event))
    print(f"\nTotal MACs: {macs / 1e9:.2f} G, Params: {params / 1e6:.2f} M")
    '''
    Total MACs: 120.19 G, Params: 14.91 M 4 stage
    Total MACs: 174.10 G, Params: 20.77 M 【64 64 128 128】

    '''

