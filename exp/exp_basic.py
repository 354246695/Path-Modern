import os
import torch
from models import TimesNet, ModernTCN
from models import Transformer, DLinear, LightTS, PatchTST, MICN, PathFormer

class Exp_Basic(object):
    """
    Exp_Basic 类提供了一个基础的实验框架，用于初始化模型、设备，并提供构建模型、获取数据、验证、训练和测试的方法。

    参数:
    - args: 包含实验配置的参数对象，例如是否使用GPU、使用的GPU编号等。
    """
    def __init__(self, args):
        # 初始化 Exp_Basic 实例
        self.args = args
        # 定义支持的模型字典，键为模型名称，值为模型类
        self.model_dict = {
            'TimesNet': TimesNet,  # TimesNet模型的类
            'ModernTCN': ModernTCN,  # ModernTCN模型的类
            'Transformer': Transformer,  # Transformer模型的类
            'DLinear': DLinear,  # DLinear模型的类
            'LightTS': LightTS,  # LightTS模型的类
            'PatchTST': PatchTST,  # PatchTST模型的类
            'MICN': MICN,  # MICN模型的类
            'PathFormer': PathFormer,  # MICN模型的类
        }
        # 获取用于训练/测试的设备
        self.device = self._acquire_device()
        # 根据配置构建模型并移动到指定设备上
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        """
        构建模型的方法，应该在子类中实现具体的模型构建逻辑。
        这里抛出 NotImplementedError 表示这个方法是抽象的，需要在子类中重写。
        """
        raise NotImplementedError
        return None

    def _acquire_device(self):
        """
        获取用于训练/测试的设备。
        根据用户指定的参数决定使用CPU还是GPU，并配置相应的环境。
    
        如果指定使用GPU，还会根据是否指定了多个GPU来决定如何配置CUDA环境。
        """
        # 根据是否使用GPU以及是否使用多个GPU来配置环境变量和选择设备
        if self.args.use_gpu:
            # 如果使用单个GPU，设置CUDA可见设备；如果使用多个GPU，设置为用户指定的设备列表
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            # 使用CPU
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):

        pass