import torch
import torch.nn as nn

class Inception_Block_V1(nn.Module):
    """
    实现Inception Block V1的结构。

    参数:
    - in_channels: 输入通道数。
    - out_channels: 输出通道数。
    - num_kernels: 卷积核的数量，默认为6。
    - init_weight: 是否初始化权重，默认为True。

    方法:
    - forward: 前向传播方法。
    - _initialize_weights: 权重初始化方法。
    """
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        
        # 初始化不同大小卷积核的列表
        kernels = []

        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        """
        使用kaiming_normal_方法初始化卷积层的权重，对偏差项初始化为0。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        定义前向传播的操作。

        参数:
        - x: 输入特征。

        返回:
        - res: 经过多个卷积核处理后，通道叠加并取平均的特征结果。
        """
        res_list = []
        # 对输入应用多个不同大小的卷积核
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))

        # 将多个卷积结果在通道维度上叠加，然后在最后一个维度上取平均
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Inception_Block_V2(nn.Module):
    """
    实现Inception Block V2的一个简化版本。
    
    参数:
    - in_channels: 输入通道数
    - out_channels: 输出通道数
    - num_kernels: 用于构建不同大小卷积核的个数，默认为6
    - init_weight: 是否初始化权重，默认为True
    
    方法:
    - forward: 前向传播方法
    """
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        # 构建不同大小的卷积核
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        """
        使用kaiming_normal_方法初始化卷积层的权重，并将偏置初始化为0。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        定义前向传播过程。
        
        参数:
        - x: 输入特征数据
        
        返回:
        - res: 经过不同大小卷积核处理后的特征数据的平均值
        """
        res_list = []
        # 对输入应用不同大小的卷积核
        for i in range(self.num_kernels + 1):
            res_list.append(self.kernels[i](x))

        # 将所有结果沿通道维度堆叠，并在最后一个维度上取平均值
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
