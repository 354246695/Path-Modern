import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from layers.Layer import Transformer_Layer
from utils.Other import SparseDispatcher, FourierLayer, series_decomp_multi, MLP
from models import ModernTCN


class AMS(nn.Module):
    def __init__(self, input_size, output_size, num_experts, device, configs,
                 num_nodes=1, d_model=32, d_ff=64, dynamic=False,
                #  patch_size_ams=[8, 6, 4, 2], noisy_gating=True, k=4, 
                # layer_number=1, residual_connection=1, batch_norm=False):
                 patch_size=[8, 6, 4, 2], noisy_gating=True, k=4, 
                 layer_number=1, residual_connection=1, batch_norm=False, ):
        super(AMS, self).__init__()
        self.num_experts = num_experts  # 专家数量
        self.output_size = output_size  # 输出的特征维度大小
        self.input_size = input_size  # 输入的特征维度大小
        self.k = k

        # 存入参数configs
        self.configs = configs
        self.d_model = d_model

        self.start_linear = nn.Linear(in_features=num_nodes, out_features=1)  # 起始的线性层，将节点数量维度转换为1维
        self.seasonality_model = FourierLayer(pred_len=0, k=3)  # 季节性模型（傅里叶层），用于处理季节性相关信息，这里预测长度设为0，k参数设为3
        self.trend_model = series_decomp_multi(kernel_size=[4, 8, 12])  # 趋势模型，通过多尺寸的核来分解趋势相关信息

        self.experts = nn.ModuleList()  
        self.MLPs = nn.ModuleList()  
        # for patch in patch_size_ams:
        patch_size=[8, 6, 4, 2]
        for patch in patch_size:
            patch_nums = int(input_size / patch)        # patch 生成
            self.experts.append(Transformer_Layer(device=device, d_model=d_model, d_ff=d_ff,
                                                    dynamic=dynamic, num_nodes=num_nodes, 
                                                    patch_nums=patch_nums, patch_size=patch, 
                                                    # patch_nums=patch_nums, patch_size_ams=patch, 
                                                    factorized=True, layer_number=layer_number, 
                                                    batch_norm=batch_norm, 
                                                    data_len= 405)) # ==========换数据 手动设置为数据长度============
            
            # 添加Transformer_Layer类型的专家模块，每个模块有对应的参数设置，例如设备、模型维度、前馈网络维度等

        # self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        # self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Linear(input_size, num_experts)  
        self.w_gate = nn.Linear(input_size, num_experts)  

        self.residual_connection = residual_connection  # 是否使用残差连接，1表示使用，0表示不使用
        self.end_MLP = MLP(input_size=input_size, output_size=output_size)  # 最后的MLP层，用于将处理后的特征转换为最终输出维度

        self.noisy_gating = noisy_gating  # 是否使用带噪声的门控机制
        self.softplus = nn.Softplus()  # Softplus激活函数
        self.softmax = nn.Softmax(1)  # Softmax激活函数，在维度1上进行操作
        self.register_buffer("mean", torch.tensor([0.0]))  # 注册一个均值张量作为缓冲区，初始值为0.0
        self.register_buffer("std", torch.tensor([1.0]))  # 注册一个标准差张量作为缓冲区，初始值为1.0
        assert (self.k <= self.num_experts)  # 断言，确保k值小于等于专家数量

    def cv_squared(self, x):
        """
        计算变异系数的平方（Coefficient of Variation squared）
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """
        将门控值转换为负载（Load）信息，计算每个专家被选中的次数总和（按维度0求和）
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """
        计算在top-k情况下的概率值，判断干净值（clean_values）是否在带噪声的top-k值范围内，并据此计算概率

        1.标准化:
        代码利用正态分布的特性，将带有噪声的比较问题转化为标准正态分布下的问题，
        通过标准化 (clean_values - threshold_if_in) / noise_stddev：
        反映了信号相对于阈值的强度（考虑噪声的标准差）

        2.累积分布函数计算概率:
        然后使用正态分布的累积分布函数来计算概率，从而在存在噪声干扰的情况下，评估 clean_values 处于 top-k 内的概率。
        通过 CDF 计算出的概率可以告诉我们在这种噪声水平下，信号强度超过阈值的概率，即 clean_values 处于 top-k 内的概率。
        """
        batch = clean_values.size(0)  # 获取批次大小
        m = noisy_top_values.size(1)  # 获取top-k中的k值
        top_values_flat = noisy_top_values.flatten()  # 展平带噪声的前k大值张量，方便后续按一维索引取值

        # 若干净值在top-k内时的阈值位置索引计算
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        # 依索引取阈值，加一维便于后续比较
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)  # 判断干净值加噪后是否大于阈值，得布尔张量

        # 若干净值不在top-k内时的阈值位置索引计算，简单在前述基础上减 1
        threshold_positions_if_out = threshold_positions_if_in - 1
        # 依新索引取阈值，同样加一维
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(self.mean, self.std)  # 按类中预设均值标准差构建正态分布对象
        # 干净值在top-k内时的概率计算，用正态分布累积分布函数，考虑噪声标准差标准化
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        # 干净值不在top-k内时的概率计算，同理
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        # 按是否在top-k内选择对应概率，组合成最终概率张量
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def seasonality_and_trend_decompose(self, x):
        """
        对输入数据进行季节性和趋势分解，
        先取输入数据的特定维度（这里是第0个维度），
        然后分别通过趋势模型和季节性模型处理，最后返回组合后的结果
        """
        x = x[:, :, :, 0]
        _, trend = self.trend_model(x)
        seasonality, _ = self.seasonality_model(x)
        return x + seasonality + trend

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """
        #multi-scale router 多尺度路由器？
        R(X_trans) = Softmax(X_trans * W_r + ? ? Softplus(X_trans * W_noise))
        """
        #print(f'\n         2.0 初始x大小\n             {x.shape}')
        x = self.start_linear(x).squeeze(-1)
        #print(f'\n         2.1 x 过start_linear线性层后\n             {x.shape}')

        # clean_logits = x @ self.w_gate
        # 通过线性层 = 乘上可学习参数？ ，即 X_trans * W_r
        clean_logits = self.w_gate(x)
        #print(f'\n         2.2 x 过w_gate线性层 后得到clean_logits\n             {clean_logits.shape}')

        if self.noisy_gating and train:
            # raw_noise_stddev = x @ self.w_noise
            raw_noise_stddev = self.w_noise(x)
            # 计算误差项 ? ? Softplus(X_trans * W_noise)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))  # 误差项
            # X_trans * W_r + ? ? Softplus(X_trans * W_noise)
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
            #print(f'\n         2.3 引入噪声 clean_logits --> logits\n             {logits.shape}')
        else:
            logits = clean_logits
            #print(f'\n         2.3 不引入噪声 clean_logits -->等于 logits\n             {logits.shape}')

        # calculate topk + 1 that will be needed for the noisy gates
        # 对路径权重进行topK选择，保持前K个路径权重并将其余权重设置为0
        # 沿维度1找到，min(...)个最大值和他们的索引
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)

        # 获取、选取topk
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        #print(f'\n         2.4 获取topk的权重、索引\n             {top_k_logits.shape}\n             {top_k_indices.shape}')
        top_k_gates = self.softmax(top_k_logits)
        #print(f'\n         2.5 top_k_logits 通过softmax --> top_k_gates，维度不变')

        zeros = torch.zeros_like(logits, requires_grad=True)
        # 依据 top_k_indices 所指定的位置，将 top_k_gates 里的值填充到 zeros 中，得到 gates 张量
        gates = zeros.scatter(1, top_k_indices, top_k_gates) 
        #print(f'\n         2.6 scatter函数保存路径， 最后getes大小：\n             {gates.shape}')

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        #print(f'\n         2.7 load大小：\n             {gates.shape}')

        return gates, load

    def forward(self, x, padding_mask, loss_coef=1e-2, ):
        """
        先对输入进行季节性和趋势分解，
        然后通过带噪声的top-k门控机制将数据分配给不同专家进行处理，
        最后组合专家输出，同时计算平衡损失并返回结果和损失值
        """
        new_x = self.seasonality_and_trend_decompose(x)
        #print(f'\n    ams——1. 季节趋势分解，求和后\n   {new_x.shape}')

        #multi-scale router
        # 多尺度路由器（含topk操作）
        #print(f'\n    ams——2. 进入routing函数')
        gates, load = self.noisy_top_k_gating(new_x, self.training)
        #print(f'\n        routing函数结果：\n        gate size = {gates.shape}\n        load size = {load.shape}')

        # calculate balance loss
        importance = gates.sum(0)
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)
        balance_loss *= loss_coef
        dispatcher = SparseDispatcher(self.num_experts, gates)
        # ？
        #print(f'\n    ams——3. 双注意力模块')
        # print(f'\n    ams——3.1 调整输入')
        expert_inputs = dispatcher.dispatch(x)

        # print(f'\n            最后的expert_inputs 输入：\n                 {expert_inputs.shape}')
        # print(f'\n            最后的expert_inputs 输入：')
        # for element in expert_inputs:
        #     if isinstance(element, torch.Tensor):
        #         print(f'\n                   {element.shape}')
        #     elif isinstance(element, list):
        #         for sub_element in element:
        #             if isinstance(sub_element, torch.Tensor):
        #                 print(f'\n                   {sub_element.shape}')

        # # 多尺度 transformer
        # expert_outputs = [self.experts[i](expert_inputs[i])[0] for i in range(self.num_experts)]  # 不加上注意力

        # 输出 通过tranformer后的维度
        # print(f'\n    ams——3.2 通过tranformer后')
        # print(f'\n              expert_outputs 的维度')
        # for element in expert_outputs:
        #     if isinstance(element, torch.Tensor):
        #         print(f'\n                   {element.shape}')
        #     elif isinstance(element, list):
        #         for sub_element in element:
        #             if isinstance(sub_element, torch.Tensor):
        #                 print(f'\n                   {sub_element.shape}')

        # output = dispatcher.combine(expert_outputs)     # 多尺度聚合器 （注意力之后
        #print(f'\n    ams——4. 通过多尺度聚合器，得到output\n            {output.shape}')

        output = dispatcher.combine(expert_inputs)     # 多尺度聚合器 （不加注意力，直接调节聚合器的位置
        #print(f'\n    ams——3.2 通过多尺度聚合器，得到output\n            {output.shape}')

        # 假设 output 是一个包含多个元素的列表，这些元素可能是张量或张量列表
        # for i in range(self.configs.d_model):
            #print(f'\n                 {output[:,:,:,i].shape}')


        # ======== 加入 moderntcn：d_model层特征，分别投入 各个moderntcn =======
        # 存储不同周期的结果
        res = []
        for i in range(self.configs.d_model):
            #print(f'\n    modernTCN----{i}')
            # 加入 modernTCN
            device = torch.device('cuda:{}'.format(self.configs.gpu))

            # 【加入 modernTCN】
            self.model2 = ModernTCN.Model(self.configs).float().to(device)
            # output  = self.model2(output, None, None)
            out  = self.model2(output[:, :, :, i], padding_mask, None, None)

            #print('\n modernTCN----', {i}, ' 结果： ',{out.shape})
            # 保存结果
            res.append(out)

        # 沿着周期轴堆叠不同周期的卷积结果 
        res = torch.stack(res, dim=-1)
        #print(' ===== ')
        #print(f'\n res结果\n   {res.shape}')

        # if self.residual_connection:
        #     output = res + x
        #print(f'\n    ams——5. 残差和，加上原来的x后，output\n            {output.shape}')
        return output, balance_loss