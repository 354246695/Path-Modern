import os
import torch
from torch import nn
import torch.nn.functional as F
import math
from layers.RevIN import RevIN
from models.ModernTCN_Layer import series_decomp, Flatten_Head


class LayerNorm(nn.Module):
    """
    实现层归一化，主要用于归一化卷积神经网络中的特征图。

    参数:
    - channels: 特征图的通道数。
    - eps: 用于避免除以零的小值。
    - data_format: 特征图的数据格式，默认为"channels_last"，即通道在最后。
    """

    def __init__(self, channels, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.norm = nn.Layernorm(channels)

    def forward(self, x):
        """
        前向传播函数。

        参数:
        - x: 输入的特征图。

        返回:
        - 归一化并且重新排列后的特征图。
        """
        # 重新排列特征图维度，以便于进行层归一化
        B, M, D, N = x.shape
        x = x.permute(0, 1, 3, 2)

        # 调整特征图形状以适用于层归一化
        x = x.reshape(B * M, N, D)

        # 执行层归一化
        x = self.norm(x)

        # 恢复原始维度排列
        x = x.reshape(B, M, N, D)
        x = x.permute(0, 1, 3, 2)
        return x

def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    """
    创建一个1D卷积层。

    参数:
    - in_channels: 输入通道数。
    - out_channels: 输出通道数。
    - kernel_size: 卷积核大小。
    - stride: 卷积步长。
    - padding: 卷积补齐大小。
    - dilation: 卷积核的扩张率。
    - groups: 分组卷积的组数。
    - bias: 是否使用偏置。

    返回:
    - 一个配置好的1D卷积层。
    """
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                     stride=stride,padding=padding, dilation=dilation, groups=groups, bias=bias)

def get_bn(channels):
    """
    创建一个批归一化层。

    参数:
    - channels: 通道数。

    返回:
    - 一个配置好的批归一化层。
    """
    return nn.BatchNorm1d(channels)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bias=False):
    """
    创建一个包含卷积和批归一化的序列模型。

    参数:
    - in_channels: 输入通道数。
    - out_channels: 输出通道数。
    - kernel_size: 卷积核大小。
    - stride: 卷积步长。
    - padding: 卷积补齐大小。
    - groups: 分组卷积的组数。
    - dilation: 卷积核的扩张率。
    - bias: 卷积层是否使用偏置。

    返回:
    - 一个包含卷积和批归一化的序列模型。
    """
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    result.add_module('bn', get_bn(out_channels))
    return result

def fuse_bn(conv, bn):
    """
    将卷积层和批归一化层融合为一个单一的层。

    参数:
    - conv: 卷积层。
    - bn: 批归一化层。

    返回:
    - 融合后的卷积核和偏置项。
    """
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    # 计算标准化因子和偏移项
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    # 返回融合后的卷积核和偏置项
    return kernel * t, beta - running_mean * gamma / std

class ReparamLargeKernelConv(nn.Module):
    """
    重新参数化具有大核的卷积层类。（实现DWConv）

    通道独立处理是通过以下方式实现的：
    卷积核数量：等于输入 变量 * 通道数（nvars * dmodel），每个卷积核只负责一个通道的卷积。
    分组卷积：通过设置 groups=nvars * dmodel，确保每个通道独立处理，不混合通道之间的信息。
    """
    def __init__(self, in_channels, out_channels, 
                 kernel_size, stride, groups, 
                 small_kernel, small_kernel_merged=False, nvars=7):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel

        # 初始化padding
        # stride=1，padding=kernel_size//2，确保输出长度 L 不变
        padding = kernel_size // 2

        if small_kernel_merged:
            # 使用小卷积核进行重新参数化
            self.lkb_reparam = nn.Conv1d(in_channels=in_channels, 
                                         out_channels=out_channels, 
                                         kernel_size=kernel_size,
                                         stride=stride, 
                                         padding=padding, 
                                         dilation=1, 
                                         groups=groups, 
                                         bias=True)
        else:
            # 2进入
            # 初始化原始大卷积核 和 可选的小卷积核
            self.lkb_origin = conv_bn(in_channels=in_channels, 
                                      out_channels=out_channels, 
                                      kernel_size=kernel_size,
                                        stride=stride, 
                                        padding=padding, 
                                        dilation=1, 
                                        groups=groups,
                                        bias=False)
            #print("期望的输入通道数（lkb_origin）:", in_channels)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, 
                                          out_channels=out_channels,
                                          kernel_size=small_kernel,
                                          stride=stride, 
                                          padding=small_kernel // 2, 
                                          groups=groups, 
                                          dilation=1,
                                          bias=False)
                #print("期望的输入通道数（small_conv）:", in_channels)

    def forward(self, inputs):
        """
        前向传播方法。

        参数:
        - inputs: 输入特征。

        返回:
        - out: 经过卷积处理的输出特征。
        """
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
            
        else: # 1进入           small_kernel_merged 为 False
            # 使用一个大卷积核（lkb_origin）和一个小卷积核（small_conv），并在前向传播时将它们的输出相加。
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def PaddingTwoEdge1d(self, x, pad_length_left, pad_length_right, pad_values=0):
        """

        """
        D_out, D_in, ks = x.shape
        if pad_values == 0:
            pad_left = torch.zeros(D_out, D_in, pad_length_left)
            pad_right = torch.zeros(D_out, D_in, pad_length_right)
        else:
            pad_left = torch.ones(D_out, D_in, pad_length_left) * pad_values
            pad_right = torch.ones(D_out, D_in, pad_length_right) * pad_values
        x = torch.cat([pad_left, x], dims=-1)
        x = torch.cat([x, pad_right], dims=-1)
        return x

    def get_equivalent_kernel_bias(self):
        """
        获取等价的卷积核和偏置，用于重新参数化。

        返回:
        - eq_k: 等价卷积核。
        - eq_b: 等价偏置。
        """
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            # 对小卷积核进行padding，使其与大卷积核等价
            eq_k += self.PaddingTwoEdge1d(small_k, 
                                          (self.kernel_size - self.small_kernel) // 2,
                                          (self.kernel_size - self.small_kernel) // 2, 
                                          0)
        return eq_k, eq_b

    def merge_kernel(self):
        """
        合并卷积核，将小卷积核和大卷积核重新参数化为一个大卷积核。

        无返回值，但会修改self.lkb_reparam以反映合并后的卷积核和偏置。
        """
        eq_k, eq_b = self.get_equivalent_kernel_bias()  # 获取等价的卷积核和偏置。
        # 创建一个新的卷积层 lkb_reparam，并设置其权重和偏置
        self.lkb_reparam = nn.Conv1d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, 
                                     stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, 
                                     dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, 
                                     bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        # 删除原始大卷积核和小卷积核
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')

class Block(nn.Module):
    """
    DWConv

    ffn1pw1

    ffn1pw2
    """

    def __init__(self, large_size, small_size, dmodel, dff, nvars, small_kernel_merged=False, drop=0.1):
        super(Block, self).__init__()

        # 初始化大卷积核和归一化层（DWConv）
        self.dw = ReparamLargeKernelConv(in_channels=nvars * dmodel, 
                                         out_channels=nvars * dmodel,
                                         kernel_size=large_size,        # default=[31,29,27,13], help='大核大小')
                                         stride=1, 
                                         groups=nvars * dmodel,
                                         small_kernel=small_size,       # default=[5,5,5,5], help='小核大小，用于结构重参数化
                                         small_kernel_merged=small_kernel_merged, 
                                         nvars=nvars)
        # 打印期望的输入通道数
        #print(f"期望的输入通道数:{nvars}*{dmodel}={nvars * dmodel}")
        
        self.norm = nn.BatchNorm1d(dmodel)

        """
        初始化第一个feedforward（ConvFFN1）网络的层:
        即: ConvFFN1 
            = ConvFFN_PWConv + GELU + ConvFFN_PWConv.
        提取每个变量的跨特征表示.
        """
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel,    # dmodel，default=[256,256,256,256], help='每个阶段的DW卷积的维度'
                                                                # nvars = enc_in ,('--enc_in', type=int, default=7, help='编码器输入尺寸')
                                 out_channels=nvars * dff,      # dff = d_ffn = dmodel * ffn_ratio = 256 * 2
                                 kernel_size=1, 
                                 stride=1,
                                 padding=0, 
                                 dilation=1, 
                                 groups=nvars)                  # 此处分组不一样，其余一样
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, 
                                 out_channels=nvars * dmodel, 
                                 kernel_size=1, 
                                 stride=1,
                                 padding=0, 
                                 dilation=1, 
                                 groups=nvars)
        
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        """
        初始化第二个feedforward（ConvFFN2）网络的层:
        即: ConvFFN2 
            = ConvFFN_PWConv + GELU + ConvFFN_PWConv.
        提取每个特征的跨变量表示.
        """
        self.ffn2pw1 = nn.Conv1d(in_channels=nvars * dmodel, 
                                 out_channels=nvars * dff, 
                                 kernel_size=1, 
                                 stride=1,
                                 padding=0, 
                                 dilation=1, 
                                 groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=nvars * dff, 
                                 out_channels=nvars * dmodel, 
                                 kernel_size=1, 
                                 stride=1,
                                 padding=0, 
                                 dilation=1, 
                                 groups=dmodel)
        
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

        self.ffn_ratio = dff // dmodel

    def forward(self, x):
        """
        1-1-1       具体的模型 DWConv + ConvFFN1 + ConvFFN2
        """
        #print(f'\nM   Block--------')

        input = x
        #print(f'\nM  4.0 x 进入Block大小:\n      {x.shape}')
        
        # 重塑输入特征以便处理
        B, M, D, N = x.shape  # 分别为 32、61、256、101
        x = x.reshape(B, M * D, N)
        #print(f'\nM  4.1 x reshape:\n      {x.shape}')
        '''

        ↑ 

        DWConv来建模时间上的关系，但又不希望它参与到通道间和变量间的建模上。

        因此，将M和D这两个表示变量和通道的维度 reshape在一起，再进行深度可分离卷积。

        '''


        #print('\nM       DWConv模块开始===')
        # 应用大卷积核和相应的变换（DWConv）
        x = self.dw(x)
        #print(f'\nM  4.2 x 经过DWConv:\n      {x.shape}')
    
        # 重塑和归一化特征
        x = x.reshape(B, M, D, N)
        x = x.reshape(B * M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B, M * D, N)
        #print(f'\nM  4.3 x 经过 重塑和归一化:\n      {x.shape}')


        #print(f'\nM       ConvFFN模块开始===')
        #print(f'        ConvFFN1 模块开始==')
        # 应用第一个feedforward网络
        x = self.ffn1drop1(self.ffn1pw1(x))
        #print(f'\nM  4.4 x 经过 ffn1pw1、ffn1drop1:\n      {x.shape}')
        x = self.ffn1act(x)  # gelu激活函数
        x = self.ffn1drop2(self.ffn1pw2(x))
        #print(f'\nM  4.5 x 经过gelu、ffn1pw2、ffn1drop2:\n      {x.shape}')

        # 重塑以适配第二个feedforward网络的输入
        x = x.reshape(B, M, D, N)
        # 调整维度顺序以适配第二个feedforward网络
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, D * M, N)
        #print(f'\nM  4.6 x 经过 重塑:\n      {x.shape}')


        #print(f'\nM       ConvFFN2 模块开始==')
        # 应用第二个feedforward网络
        x = self.ffn2drop1(self.ffn2pw1(x))
        #print(f'\nM  4.7 x 经过 ffn2pw1、ffn2drop1:\n      {x.shape}')
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        #print(f'\nM  4.8 x 经过gelu、ffn2pw2、ffn2drop2:\n      {x.shape}')

        # 调整回原始维度顺序
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)
        #print(f'\nM  4.9 x 调整回原始维度顺序:\n      {x.shape}')


        # 结合原始输入和处理后的特征
        x = input + x
        #print(f'\nM  4.10 x 与原始8.0input求和:\n      {x.shape}')

        #print(f'\nM   --------Block end')

        return x


class Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, dw_model, nvars,
                 small_kernel_merged=False, drop=0.1):
        super(Stage, self).__init__()
        d_ffn = dmodel * ffn_ratio      # dmodel, default=[256,256,256,256], help='每个阶段的DW卷积的维度' ;  dff  = '--ffn_ratio', type=int, default=2, help='FFN 比例'
        blks = []

        for i in range(num_blocks):
            # 在该阶段创建并添加num_blocks个Block  -----------------------------------------------------------------
            blk = Block(large_size=large_size,      # --> h 273 
                        small_size=small_size, 
                        dmodel=dmodel, 
                        dff=d_ffn, 
                        nvars=nvars, 
                        small_kernel_merged=small_kernel_merged,        # 不组合 dwconv 的卷积核
                        drop=drop)

            blks.append(blk)

        self.blocks = nn.ModuleList(blks)

    def forward(self, x):
        """
        1-1      嵌套在第1步，进入第 1-1-1 ——blk ↑
        """
        # #print(f'   ==blocks==\n{self.blocks}')
        for blk in self.blocks: # 4 层
            x = blk(x)  # -------------------------------------------- block堆叠处理数据
        return x


class ModernTCN(nn.Module):
    def __init__(self, task_name, patch_size, patch_stride, stem_ratio, 
                 downsample_ratio, ffn_ratio, num_blocks, large_size,
                #  small_size, dims, dw_dims, nvars, period,
                 small_size, dims, dw_dims, nvars, 
                 small_kernel_merged=False, backbone_dropout=0.1, head_dropout=0.1, use_multi_scale=True, 
                 revin=True, affine=True, subtract_last=False, freq=None, seq_len=512, c_in=7,
                 individual=False, target_window=96, class_drop=0., class_num=10):
        super(ModernTCN, self).__init__()

        # 设置任务名称、类别Dropout比率与类别数量属性
        self.task_name = task_name
        self.class_drop = class_drop
        self.class_num = class_num

        # 初始化ReVIN层
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # 初始化stem层和下采样层
        self.downsample_layers = nn.ModuleList()

        # 创建Stem层（序列开始部分）
        stem = nn.Sequential(
                            nn.Conv1d(1,  #---------------------------------------------------------------------------
                                      dims[0], 
                                      kernel_size=patch_size, 
                                      stride=patch_stride),  # 原来 输入  1 , 替换为本实验 patch_stride
                            nn.BatchNorm1d(dims[0]) # dims=[256,256,256,256] ;; patch_size = 16 ;; patch_stride = 8
                            )  
        
        self.downsample_layers.append(stem)

        # 计算阶段数
        self.num_stage = len(num_blocks)

        # 若存在多个阶段，构建下采样层
        if self.num_stage > 1:
            for i in range(self.num_stage - 1):  # 本次实验设置为 4 层
                downsample_layer = nn.Sequential(
                                                nn.BatchNorm1d(dims[i]),
                                                nn.Conv1d(dims[i], 
                                                          dims[i + 1], 
                                                          kernel_size=downsample_ratio, 
                                                          stride=downsample_ratio),
                                                )
                self.downsample_layers.append(downsample_layer)

        # 存储补丁大小、步长及下采样比率
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio

        # 初始化各阶段（Stage）
        self.num_stage = len(num_blocks)        # num_blocks = [1,1,1,1]

        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            # ----------------------------------------------------------------- Stage
            layer = Stage(ffn_ratio,  
                          num_blocks[stage_idx], 
                          large_size[stage_idx], 
                          small_size[stage_idx], 
                          dmodel=dims[stage_idx],
                          dw_model=dw_dims[stage_idx], 
                            #nvars=64,  # 缝合 1 原来 64===========================================================================
                          nvars=nvars,  
                          small_kernel_merged=small_kernel_merged, 
                          drop=backbone_dropout)
            self.stages.append(layer)

        # 初始化头部模块
        patch_num = seq_len // patch_stride
        self.n_vars = c_in
        self.individual = individual
        d_model = dims[self.num_stage - 1]

        # 根据是否使用多尺度特征融合设置头部模块输入维度
        if use_multi_scale:
            self.head_nf = d_model * patch_num
            self.head = Flatten_Head(self.individual, 
                                     self.n_vars, 
                                     self.head_nf, 
                                     target_window,
                                     head_dropout=head_dropout)
        else:
            if patch_num % pow(downsample_ratio,(self.num_stage - 1)) == 0:
                self.head_nf = d_model * patch_num // pow(downsample_ratio,(self.num_stage - 1))
            else:
                self.head_nf = d_model * (patch_num // pow(downsample_ratio, (self.num_stage - 1))+1)

            self.head = Flatten_Head(self.individual, 
                                     self.n_vars, 
                                     self.head_nf, 
                                     target_window,
                                     head_dropout=head_dropout)

        if self.task_name == 'classification':
            self.act_class = F.gelu
            self.class_dropout = nn.Dropout(self.class_drop)
            # self.head_class = nn.Linear(self.n_vars[0]*self.head_nf, self.class_num)  # 原
            self.head_class = nn.Linear(32*self.head_nf, self.class_num)  # 两个block
            self.head_class = nn.Linear(114688, self.class_num)  # 4个block

            # print(f'self.head_class = nn.Linear(self.n_vars[0]*self.head_nf, self.class_num)')
            # print(f'n_vars  = {self.n_vars}')
            # print(f'n_vars[0]  = {self.n_vars[0]}')
            # print(f'head_nf  = {self.head_nf}')
            # print(f'class_num  = {self.class_num}')

    def forward_feature(self, x, te=None):
        """
        1.     首先通过此步骤，嵌套 1-1 步（num_stage层 = 4） ↑                                                                                            
        """
        # 修改
        # 输入 (批次大小, 特征数, 序列长度)，最后两个维度和timesNet相反
        B,M,L=x.shape  # B表示批次大小，M表示中间维度-特征数，L表示序列长度

        # 修改
        x = x.unsqueeze(-2)  # 在x的最后一个维度前添加一个维度。
        #print(f'\nM  3.1 x 大小,添加一个维度(进入模型后第一个操作):\n      {x.shape}')

        for i in range(self.num_stage):    # 本次实验设置为 4 层
            #print(f'\nM             ---------ModernTCN Stage {i+1} ---------')

            B, M, D, N = x.shape  # 原 分别为: 32、61、1、405  【8.64.31.2
            x = x.reshape(B * M, D, N)  # x 为 ( 32*61 = 1952, 1, 405)
            #print(f'\nM  3.2 x 大小,reshape:\n      {x.shape}')

            if i==0:  # 针对首个阶段，检查并处理卷积核尺寸与步长不匹配的问题
                if self.patch_size != self.patch_stride:
                    # 对stem层进行padding以处理不匹配的卷积核大小和步幅

                    # 计算需要复制的像素长度
                    # 改，加了if判断：
                    if N+self.patch_size - self.patch_stride< self.patch_size:  #改
                        pad_len = self.patch_size - N # 修改成 计算 patch_size-N                    
                    else: pad_len = self.patch_size - self.patch_stride # 原来
                    # pad_len = self.patch_size - self.patch_stride # 原来

                    # 复制特征图x的最后一列像素，长度为pad_len，并扩展到原来的尺寸
                    pad = x[:,:,-1:].repeat(1,1,pad_len)

                    # 在特征图x的末尾拼接复制的像素
                    x = torch.cat([x,pad],dim=-1)
                    #print(f'\nM  3.3.0 x 大小,第1层：patch_size!=patch_stride，padding后 :\n      {x.shape}')
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]],dim=-1)
                    #print(f'\nM  3.3.{i+1} x 大小,patch_size=patch_stride，padding后 :\n      {x.shape}')    #[1952, 1, 409]

            x = self.downsample_layers[i](x)  
            #print(f'\nM  3.4 x 大小,downsample_layers后 :\n      {x.shape}')  # x 下采样后:torch.Size([1952, 256, 101])

            _, D_, N_ = x.shape
            x = x.reshape(B, M, D_, N_)
            # x = x.reshape(B, D_, M, N_)
            #print(f'\nM  3.5 x 大小,reshape（进入stage前最后一步处理）:\n      {x.shape}')  
            #print(f'\nM  ----forward_feature end')

            # 进入block
            x = self.stages[i](x)
        return x

    def classification(self,x):
        """
        0.3     进入 分类任务，接着先通过 1. forward_feature ↑
        """

        # print('\nM  forward_feature----')
        x = self.forward_feature(x, te=None)

        # x = self.act_class(x)
        # #print(f'\nM  5 x 大小, after gelu、class_dropout、head_class Linear:\n      {x.shape}')

        x = self.class_dropout(x)
        # print(f'\nM  6 x 大小, after class_dropout:\n      {x.shape}')

        # x = x.reshape(x.shape[0], -1)
        # #print(f'\nM  7 x 大小, reshape:\n      {x.shape}')

        # x = self.head_class(x)
        # #print(f'\nM  8 x 大小, after head_class Linear:\n      {x.shape}')

        # print(f'\nM  -------结束 ModernTCN 任务\n')

        return x


    def forward(self, x, te=None):
        """
        0.2     进入任务，进入第0.3步 ↑
        """
        if self.task_name == 'classification':
            x = self.classification(x)
        return x

    def structural_reparam(self):
        # """
        # 对模型进行结构重参数化，主要用于合并卷积核。
        # """
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()


class Model(nn.Module):

    # def __init__(self, configs, period):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.configs = configs
        # self.period = period

        # 从配置中初始化各种参数
        self.task_name = configs.task_name
        self.stem_ratio = configs.stem_ratio
        self.downsample_ratio = configs.downsample_ratio
        self.ffn_ratio = configs.ffn_ratio
        self.num_blocks = configs.num_blocks
        self.large_size = configs.large_size
        self.small_size = configs.small_size
        self.dims = configs.dims
        self.dw_dims = configs.dw_dims

        self.nvars = configs.enc_in     # ('--enc_in', type=int, default=7, help='编码器输入尺寸')
        self.small_kernel_merged = configs.small_kernel_merged
        self.drop_backbone = configs.dropout
        self.drop_head = configs.head_dropout
        self.use_multi_scale = configs.use_multi_scale
        self.revin = configs.revin
        self.affine = configs.affine
        self.subtract_last = configs.subtract_last

        self.freq = configs.freq
        self.seq_len = configs.seq_len
        self.c_in = self.nvars,
        self.individual = configs.individual
        self.target_window = configs.pred_len

        self.kernel_size = configs.kernel_size
        # self.patch_size = configs.patch_size      # 原初始化 代码，参数冲突重新设置 
        self.patch_size = 16
        self.patch_stride = configs.patch_stride

        # 分类任务相关配置
        self.class_dropout = configs.class_dropout
        self.class_num = configs.num_class

        # 解构任务相关配置
        self.decomposition = configs.decomposition

        # 初始化ModernTCN实例
        self.model = ModernTCN(
            task_name=self.task_name,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            stem_ratio=self.stem_ratio,
            downsample_ratio=self.downsample_ratio,
            ffn_ratio=self.ffn_ratio,
            num_blocks=self.num_blocks,
            large_size=self.large_size,
            small_size=self.small_size,
            dims=self.dims,
            dw_dims=self.dw_dims,
            nvars=self.nvars,
            small_kernel_merged=self.small_kernel_merged,
            backbone_dropout=self.drop_backbone,
            head_dropout=self.drop_head,
            use_multi_scale=self.use_multi_scale,
            revin=self.revin,
            affine=self.affine,
            subtract_last=self.subtract_last,
            freq=self.freq,
            seq_len=self.seq_len,
            c_in=self.c_in,
            individual=self.individual,
            target_window=self.target_window,
            class_drop=self.class_dropout,
            class_num=self.class_num,
            # period = self.period
        )
    # def _build_model(self):
    #     """
    #     似乎 源代码没有，timesnet-moderntcn 实验加的？？
    #     构建模型及其相关配置。
        
    #     返回:
    #     - model: 根据配置初始化的模型实例。
    #     """
    #     # 获取训练和测试数据集及其加载器，并配置序列长度、预测长度和编码器输入维度
    #     train_data, train_loader = self._get_data(flag='TRAIN')
    #     test_data, test_loader = self._get_data(flag='TEST')
    #     self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
    #     self.args.pred_len = 0
    #     self.args.enc_in = train_data.feature_df.shape[1]  # 设置编码器输入维度
    #     self.args.num_class = len(train_data.class_names)  # 设置类别数量
        
    #     # 初始化模型，并根据需要使用多GPU
    #     model = self.model_dict[self.args.model].Model(self.args).float()
    #     # model2 = self.model_dict['ModernTCN'].Model(self.args).float()


    #     if self.args.use_multi_gpu and self.args.use_gpu:
    #         model = nn.DataParallel(model, device_ids=self.args.device_ids)
    #         # model2 = nn.DataParallel(model, device_ids=self.args.device_ids)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        0.1     从各个forward看
        """
        #print(f'\n\nM  进入 ModernTCN 任务-------')

        # print(f'\nM  1.x 大小，原始数据:\n      {x.shape}')     # torch.Size([8, 405, 61, 4])

        # 对输入x进行维度置换，以适配模型的输入要求
        x = x.permute(0, 2, 1)        

        # # mine 修改 添加维度变换：([8, 405, 61, 4]) ---> ([8, 61, 1, 405])
        # x = x.permute(0, 2, 1)

        te = None  # 初始化一个临时变量，目前未使用，可能为未来扩展预留

        #print(f'\nM  2.x 大小,维度置换后(进入模型前):\n      {x.shape}')

        # 调用模型的前向传播方法处理输入x
        x = self.model(x, te)
        return x