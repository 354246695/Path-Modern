import math
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from layers.AMS import AMS
from layers.Layer import WeightGenerator, CustomLinear
from layers.RevIN import RevIN
from functools import reduce
from operator import mul
from models import ModernTCN


# 定义模型类，继承自PyTorch的nn.Module
class Model(nn.Module):
    def __init__(self, configs): 
        super(Model, self).__init__()
        self.layer_nums = configs.layer_nums  # 设置路径层数                    # 设置pathway的层数
        self.num_nodes = configs.num_nodes  # 节点数量
        self.pre_len = configs.pred_len  # 预测长度
        self.seq_len = configs.seq_len  # 序列长度
        self.k = configs.k  # k值，用于AMS层
        self.num_experts_list = configs.num_experts_list  # 专家数量列表
        self.patch_size_list = configs.patch_size_list  # 补丁大小列表
        self.d_model = configs.d_model  # 模型维度  ('--d_model', type=int, default=16)
        self.d_ff = configs.d_ff  # 前馈网络维度
        self.residual_connection = configs.residual_connection  # 是否使用残差连接
        self.revin = configs.revin  # 是否使用RevIN层
        if self.revin:
            self.revin_layer = RevIN(num_features=configs.num_nodes, affine=False, subtract_last=False)  # 初始化RevIN层

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)  # 初始化第一个全连接层 ('--d_model', type=int, default=16)
        self.AMS_lists = nn.ModuleList()  # AMS层列表
        self.device = torch.device('cuda:{}'.format(configs.gpu))  # 设备设置
        self.batch_norm = configs.batch_norm  # 是否使用批量归一化
        
        # 报错测试 
        # for i in range(len(self.patch_size_list)):
        #     print(i)
        #     print(f'patch_size : {self.patch_size_list[i]}')

        # 循环创建AMS层
        for num in range(self.layer_nums):
            self.AMS_lists.append(
                # ('--num_experts_list', type=list, default=[4, 4, 4])
                # 直接修改这里ams层内部的 transformer
                AMS(self.seq_len, self.seq_len, self.num_experts_list[num], self.device, configs,        # 加入configs，#padding_mask
                    k=self.k,
                    # num_nodes=self.num_nodes, patch_size_ams=self.patch_size_list[num], noisy_gating=True,
                    num_nodes=self.num_nodes, patch_size=self.patch_size_list[num], noisy_gating=True,
                    d_model=self.d_model, d_ff=self.d_ff, layer_number=num + 1, 
                    residual_connection=self.residual_connection, batch_norm=self.batch_norm
                    )
            )
        self.projections = nn.Sequential(nn.Linear(self.seq_len * self.d_model, self.pre_len))  # 投影层

        # =======  1. 尝试，套用timesnet =======
        self.projection = nn.Linear(configs.d_model * configs.seq_len * configs.num_nodes, configs.num_class)

        # 【3.4 ds建议】
        # self.modern_tcn = ModernTCN.Model(self.configs).float().to(device)
        self.tcn = ModernTCN.Model(configs)  # 添加最终TCN层

    # 前向传播函数
    def forward(self, x, padding_mask):
        balance_loss = 0  # 平衡损失初始化
        #print('\n------------ start ------------')
        #print(f'\n0. 输入x 大小\n   {x.shape}')

        # 归一化处理
        if self.revin:
            x = self.revin_layer(x, 'norm')
        #print(f'\n\n1.1 归一化后 x\n   {x.shape}')
        out = self.start_fc(x.unsqueeze(-1))  # 通过第一个全连接层
        #print(f'\n1.2 首个全连接层start_fc之后 out大小\n   {out.shape}\n')

        batch_size = x.shape[0]  # 批量大小

        # 循环通过AMS层     (加入了 moderntcn)
        nums_ams = 0
        for layer in self.AMS_lists:
            nums_ams += 1   # 记录当前ams层数
            #print(' ------ ')
            #print(f'\n2.第{nums_ams}层ams 初始 out大小\n   {out.shape}')

            out, aux_loss = layer(out, padding_mask)  # 通过AMS层，获取输出和辅助损失
            balance_loss += aux_loss  # 累加辅助损失

            #print(f'\n2.第{nums_ams}层ams 之后 out大小\n   {out.shape}')
            #print(' ------ ')

        # 【3.4 ds】
        # 统一时序建模 (关键修改点)
        print(f'\nout ============ \n      {out.shape}')     #  torch.Size([32, 405, 61, 2])
        out = torch.stack(out, dim=-1)  # 为了匹配 维度
        out = self.tcn(
            out.permute(0, 3, 1, 2),  # [B, L, D, S] -> [B, S, D, L]
            padding_mask,
            None, None)

        # 3.4以前
        # out = out.permute(0,2,1,3).reshape(batch_size, self.num_nodes, -1)  # 调整输出形状
        # print(f'\n\n3. ams后调整输出大小 \n   {out.shape}')

        # =======  0. 原方案，预测任务 =======
        # out = self.projections(out).transpose(2, 1)  # 通过投影层并转置
        # print(f'\n\n4. 通过投影层并转置 \n   {out.shape}')

        # # 反归一化处理
        # if self.revin:
        #     out = self.revin_layer(out, 'denorm')
        #     print(f'\n\n5. 反归一化处理 \n   {out.shape}')

        # print(f'\n\n6. 最后输出尺寸 \n   {out.shape}')
        # print('\n+++++++++++++ end +++++++++++++')


        # =======  1. 尝试，套用timesnet =======
        # # 重塑输出以适配投影层
        # out = out.reshape(out.shape[0], -1)
        # #print(f'\n       reshape \n   {out.shape}')
        # out = self.projection(out)  # 投影到类别数维度
        # #print(f'\n      最后，投影到类别数维度 \n   {out.type} : {out.shape}')


        return out, balance_loss  # 返回输出和平衡损失