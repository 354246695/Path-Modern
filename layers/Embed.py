import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    """
    实现位置嵌入（Positional Embedding），用于增加模型对位置信息的感知能力。
    
    参数:
    - d_model: 模型的维度，即嵌入向量的大小。
    - max_len: 最大的序列长度，默认为5000。
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        # 初始化位置嵌入张量，并设置其不需要梯度更新
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        # 计算位置编码
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        # 应用正弦和余弦函数到位置编码上
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数索引应用正弦函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数索引应用余弦函数

        # 将位置嵌入张量设置为模型的buffer，不会被优化器更新
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播函数，根据输入的张量x（序列）返回对应位置嵌入。
        
        参数:
        - x: 输入的张量，其shape为(batch_size, seq_len, d_model)。
        
        返回:
        - 与输入x序列长度相匹配的位置嵌入张量。
        """
        # print(f'\n1.2 操作position_embedding.\n   {self.pe[:, :x.size(1)].shape}')
        return self.pe[:, :x.size(1)]  # 返回与输入序列长度相匹配的位置嵌入


class TokenEmbedding(nn.Module):
    """
    TokenEmbedding 类用于将输入的 token 进行嵌入处理。
    
    参数:
    - c_in: 输入通道数
    - d_model: 输出嵌入维度
    
    方法:
    - forward: 前向传播方法
    """
    def __init__(self, c_in, d_model):
        """
        初始化 TokenEmbedding 模型。
        
        参数:
        - c_in: 输入的通道数
        - d_model: 输出的嵌入维度
        """
        super(TokenEmbedding, self).__init__()
        # 根据 PyTorch 版本选择合适的 padding 值
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        
        # 初始化一个一维卷积层，用于 token 的嵌入
        # print(f'\n\n\nout_channels=d_model  {d_model}')
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        # 对模型中的所有卷积层进行初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """
        定义前向传播的操作。
        
        参数:
        - x: 输入的张量
        
        返回:
        - 经过嵌入处理后的张量
        """
        # 对输入进行维度转换，卷积操作，并调整回原来维度
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        # print(f'\n1.1 操作value_embedding, TokenEmbedding.1d卷积\n   {x.shape}')
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    """
    数据嵌入类，用于将输入数据转换为嵌入表示。

    参数:
    - c_in: 输入通道数。
    - d_model: 嵌入维度。
    - embed_type: 嵌入类型，'fixed'表示固定嵌入，'timeF'表示时间特征嵌入。
    - freq: 用于时间特征嵌入的频率。
    - dropout: Dropout比例。
    """

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        # 初始化值嵌入、位置嵌入和临时嵌入（根据embed_type选择不同的嵌入类型）
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, 
                                                    embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        前向传播函数。

        参数:
        - x: 输入数据。
        - x_mark: 标记数据，用于 temporal embedding，如果为None，则只使用value和position embedding。

        返回值:
        - 嵌入后的数据。
        """
        # 根据是否提供x_mark，选择不同的嵌入方式
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x) 

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()  # 初始化父类
        # 定义补丁（patch）的长度
        self.patch_len = patch_len
        # 定义步长
        self.stride = stride
        # 使用复制填充来处理图像边界，使得每个补丁的大小为 patch_len
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # 定义骨干网络，输入编码：将特征向量投影到 d_model 维的向量空间
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # 定义位置嵌入
        self.position_embedding = PositionalEmbedding(d_model)

        # 定义残差丢弃，用于正则化和防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 对输入进行分块处理
        n_vars = x.shape[1]  # 获取变量的数量
        x = self.padding_patch_layer(x)  # 对输入进行边界填充
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # 将输入展开成多个小块
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # 重塑张量的形状

        # 输入编码
        x = self.value_embedding(x) + self.position_embedding(x)  # 将值嵌入和位置嵌入相加
        return self.dropout(x), n_vars  # 返回经过丢弃处理的嵌入和变量数量
