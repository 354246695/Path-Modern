
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
# from models import ModernTCN
# from ModernTCN import ModernTCN
# from models.ModernTCN import ModernTCN
from models import ModernTCN

def FFT_for_Period(x, k=2):
    """
    使用FFT算法寻找序列x的周期。

    参数:
    - x: 输入序列，格式为[B, T, C]，其中B是批次大小，T是序列长度，C是通道数(特征数)。
    - k: 需要找到的周期数，默认为2。

    返回值:
    - period: 找到的周期数组，长度为k。
    - abs_xf_top_list: 对应于找到的周期的频域幅度均值。
    """
    # print("\n------- FFT -------\n")
    # print("\nx.shape:\n     ", x.shape)

    # 对输入 第2个维度 序列进行傅里叶变换
    xf = torch.fft.rfft(x, dim=1)
    # print("\nxf.shape:\n     ", xf.shape)

    # 计算幅度平均值
    frequency_list = abs(xf).mean(0).mean(-1)
    # print("\nfrequency_list.shape:\n     ", frequency_list.shape)

    # 排除直流分量
    frequency_list[0] = 0
    # print("\n排除直流分量 frequency_list.shape:\n     ", frequency_list.shape)

    # 找到k个主要周期
    _, top_list = torch.topk(frequency_list, k)
    # print(f"\ntop_list.shape(k = {k}):\n     ", top_list.shape)

    # 转换为numpy数组
    top_list = top_list.detach().cpu().numpy()
    # print("\ntop_list_numpy.shape(k = 2):\n     ", top_list.shape)

    # 计算周期
    period = x.shape[1] // top_list
    # print("\nperiod.shape(return 1):\n     ", period.shape)
    # print("\nabs(xf).mean(-1)[:, top_list].shape(return 2):\n     ", abs(xf).mean(-1)[:, top_list].shape)
    # print("\n------- FFT end -------\n")
          
    # 返回 period_list, period权重(频率)
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    """
    TimesBlock 类，用于实现一个时间序列分析的模块。

    参数:
    - configs: 一个配置对象，包含序列长度(seq_len), 预测长度(pred_len), 以及顶部k值(top_k)等配置。

    属性:
    - seq_len: 输入序列的长度。
    - pred_len: 预测序列的长度。
    - k: 选择的顶部k个周期。
    - conv: 一个由Inception Block组成的序列，用于对周期性特征进行转换。
    """

    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        # 存入参数configs
        self.configs = configs


        # 使用参数效率高的设计
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, 
                               configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, 
                               configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x, padding_mask):
        """
        前向传播函数。

        参数:
        - x: 输入的张量，尺寸为(B, T, N)，其中B是批次大小，T是序列长度，N是特征维度。

        返回:
        - 经过处理后的张量，具有相同的输入尺寸(B, T, N)。
        """
        # 获取输入张量的尺寸
        B, T, N = x.size()
        
        # 计算周期列表和对应的权重，用于后续的自适应聚合
        period_list, period_weight = FFT_for_Period(x, self.k)
        p_list_shape = period_list.shape
        p_weight_shape = period_weight.shape
        # print(f'\nperiod_list    =   {period_list}')
        # period_list = [81 ,405]
        # print(f'\nperiod_list修改  =   {period_list}')

        # 存储不同周期的卷积结果
        res = []

        for i in range(self.k):  # 选取top k个周期（k = 1）
            print(f'\n2.0.k{i+1} 初始的 x大小\n   {x.shape}')

            period = period_list[i]
            print(f'\nperiod 大小 =   {period}')
            # print(f'B      大小 =   {B}')
            # print(f'N      大小 =   {N}')
            
            # 零填充以确保序列长度可以被卷积核的周期整除
            if (self.seq_len + self.pred_len) % period != 0:
                # 计算需要填充的长度，并创建相应大小的零张量
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                
                # 在输入序列后添加零填充
                out = torch.cat([x, padding], dim=1)
            else:
                # 如果不需要填充，直接使用输入序列
                length = (self.seq_len + self.pred_len)
                out = x
            print(f'length 大小 =   {length}')

            print(f'\n2.1.k{i+1} 根据周期填充后 x大小\n   {out.shape}')
                
            # 调整张量的形状以适应2D卷积操作，1d转2d
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            print(f'\n2.2.k{i+1} reshape x大小\n   {out.shape}')

            # # 应用2D卷积，inception层  ------------------------- 插入timesnet --------------------------------
            # out = self.conv(out)
            # # 加入 modernTCN
            # dec_out = model2(dec_out, x_mark_enc, None, None)

            print(f'\n2.3.k{i+1} 不进行2d卷积 x大小\n   {out.shape}')
            
            # 加入 modernTCN
            device = torch.device('cuda:{}'.format(self.configs.gpu))
            # 加入 modernTCN
            self.model2 = ModernTCN.Model(self.configs, length // period).float().to(device)
            out = self.model2(out, padding_mask, None, None)
            print(f'\n2.3.5.k{i+1} ModernTCN结束 x大小\n   {out.shape}')

            # 将卷积后的结果恢复到原始维度
            # out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            out = out.permute(0, 3, 1, 2).reshape(B, -1, N)
            print(f'\n2.4.k{i+1} 维度恢复 x大小\n   {out.shape}')

            # 融合实验添加 ：如果 out第二个维度比，期望维度少，进行填充
            if (self.seq_len + self.pred_len) > out.shape[1]:
                # 计算需要填充的张量大小
                padding_size = (self.seq_len + self.pred_len) - out.shape[1]
                # 创建填充张量，填充值设为0
                padding = torch.zeros([out.shape[0], padding_size, out.shape[2]]) 
                # 将原始张量out移动到和padding相同的设备
                padding = padding.to(out.device)
                # 在输入序列后添加零填充
                out = torch.cat([out, padding], dim=1)

                res.append(out[:, :(self.seq_len + self.pred_len), :])
                print(f'\n2.5.k{i+1} 保存的结果 x大小\n   {out[:, :(self.seq_len + self.pred_len), :].shape}')  # ??是否可以加线性层

            else:
                # 保存当前周期的卷积结果
                res.append(out[:, :(self.seq_len + self.pred_len), :])
                print(f'\n2.5.k{i+1} 保存的结果 x大小\n   {out[:, :(self.seq_len + self.pred_len), :].shape}')  # ??是否可以加线性层
            print(' ---- ')


        # 沿着周期轴堆叠不同周期的卷积结果 
        res = torch.stack(res, dim=-1)
        print(' ===== ')
        print(f'\n2.6 x处理堆叠后 res结果\n   {res.shape}')

        # 自适应聚合，使用权重对不同周期的结果进行加权求和
        # 使用softmax对权重进行归一化
        period_weight = F.softmax(period_weight, dim=1)
        # 广播权重以应用于所有时间步和特征
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)

        # 对不同周期的结果进行加权求和
        res = torch.sum(res * period_weight, -1)
        #print(f'\n2.7 x，对不同周期加权（权重即2.6后fft得到的频率）\n   {res.shape}')
        
        # 使用残差连接，将原始输入与加权求和的结果相加
        res = res + x
        # print(f'\n2.8 x，加权后与原始输入残差求和\n   {res.shape}')

        #print(f'\nFFT得到 原始\n     period_list, size{p_list_shape}\n     period_weight, size{p_weight_shape}')
        #print(f'\nFFT得到 period_weight处理后\n     period_weight 2, size{period_weight.shape}\n')
        
        return res

class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        """
        初始化模型。

        参数:
        - configs: 一个配置对象，包含模型训练和构建所需的全部配置。

        属性:...
        """
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # # 加入 modernTCN
        # self.model2 = ModernTCN.Model(self.configs).float()
        # # self.x_mark_enc = x_mark_enc

        # 创建由TimesBlock组成的模型层
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])

        # 初始化嵌入层
        self.enc_embedding = DataEmbedding(configs.enc_in, 
                                           configs.d_model, 
                                           configs.embed, 
                                           configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # 分类任务的配置
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, 
                                        configs.num_class)  # 原程序 存在猜测，节点骤降效果不好
            
            # 以下两行 为自主修改内容：节点缓和下降
            self.projection1 = nn.Linear(configs.d_model * configs.seq_len, 
                                         math.ceil(math.sqrt(configs.d_model * configs.seq_len + configs.num_class)))
            self.projection2 = nn.Linear(math.ceil(math.sqrt(configs.d_model * configs.seq_len + configs.num_class)), 
                                         configs.num_class)
        
    def classification(self, x_enc, x_mark_enc):
        """
        进行分类任务的函数。

        参数:
        - x_enc: 输入编码序列，用于TimesNet模型的输入。
        - x_mark_enc: 输入标记编码序列，用于指示padding位置，形状与x_enc相同。

        返回:
        - output: 经过Transformer编码器、TimesNet层和投影层处理后的输出，形状为(batch_size, num_classes)。
        """
        print('\n进入 timesNet 1->2d 任务-------')
        print(f'\n0. 输入数据x_enc大小\n   {x_enc.shape}')
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        print(f'\n1.3 编码后enc_out大小= token + position编码\n   {enc_out.shape}')
       
        # TimesNet处理 ========================================================
        for i in range(self.layer):
            print(f'\n      block{i+1} start:\n')
            # print(f'\n打印子模块:')
            # print(list(self.model[i].children()))
            enc_out = self.layer_norm(self.model[i](enc_out, x_mark_enc))

        print(f'\n3.1 TimesNet  后enc_out大小\n   {enc_out.shape}')

        # 输出处理
        # 应用非线性激活函数
        output = self.act(enc_out)
        output = self.dropout(output)
        # 通过标记编码零化填充嵌入
        output = output * x_mark_enc.unsqueeze(-1)
        print(f'\n3.2 mark后\n   {output.shape}')

        # 重塑输出以适配投影层
        output = output.reshape(output.shape[0], -1)
        print(f'\n3.3 reshape \n   {output.shape}')

        output = self.projection(output)  #原来 投影到类别数维度
        # output = self.projection1(output)  # 改
        # output = self.act(output) # 改
        # output = self.projection2(output)  # 改
        # print(f'\n3.4 最后，投影到类别数维度 \n   {output.shape}')

        print(f'\n-------结束 分类')

        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        根据不同的任务名称，调用相应的处理函数。

        参数:
        - x_enc: 输入序列的编码，shape为[B, L_enc, D]，其中B为批次大小，L_enc为输入序列的长度，D为特征维度。
        - x_mark_enc: 输入序列的标记编码，shape与x_enc相同。
        - x_dec: 解码序列的输入，shape为[B, L_dec, D]，L_dec为解码序列的长度。
        - x_mark_dec: 解码序列的标记编码，shape与x_dec相同。
        - mask: 可选，用于指定哪些位置是有效的，常用于处理不同长度的输入序列。shape为[B, L_enc, L_dec]。

        返回值:
        - 根据不同的任务返回不同的结果，具体形状取决于具体任务的处理函数。
        """
        if self.task_name == 'classification':

            # model2 = ModernTCN()
            # 假设 args 是一个已经定义好的对象，包含了所有需要的参数

            # 进行分类任务
            dec_out = self.classification(x_enc, x_mark_enc)
            
            # model2 = ModernTCN.Model(self.configs).float()
            # # 加入 modernTCN
            # dec_out = model2(dec_out, x_mark_enc, None, None)

            return dec_out  # 返回分类的预测概率或标签
        
        return None  # 如果没有匹配到任何任务，则返回None

