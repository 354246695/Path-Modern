import argparse
import os
import torch
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np
from utils.str2bool import str2bool
from utils.print_args import print_args


"""
终端命令行，参数设置

python run.py --task_name classification --is_training 1 --model_id True --model TimesNet

"""


if __name__ == '__main__':
    # 设置随机种子，确保实验可复现
    fix_seed = 2024
    random.seed(fix_seed) # python内置随机数种子
    torch.manual_seed(fix_seed) # pytorch随机数种子
    np.random.seed(fix_seed) # numpy随机数种子

    # 初始化命令行参数解析器
    parser = argparse.ArgumentParser(description='TimesNet')

    # 基本配置参数
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='任务名称，选项：[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='状态')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='模型ID')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='模型名称，选项：[Autoformer, Transformer, TimesNet]')

    # 数据加载器参数
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='数据集类型')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='数据文件的根路径')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='数据文件')  # 默认为ETTh1.csv,此处需要修改  ???
    parser.add_argument('--features', type=str, default='M',
                        help='预测任务选项：[M, S, MS]；M：多变量预测多变量，S：单变量预测单变量，MS：多变量预测单变量')
    parser.add_argument('--target', type=str, default='OT', help='在S或MS任务中的目标特征')
    parser.add_argument('--freq', type=str, default='h',
                        help='时间特征编码的频率，选项：[s:秒, t:分钟, h:小时, d:天, '
                             'b:工作日, w:周, m:月]，你可以使用更详细的频率如15min或3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='模型检查点的位置')
    
    # 预测任务参数配置
    parser.add_argument('--seq_len', type=int, default=96, help='输入序列长度')
    parser.add_argument('--label_len', type=int, default=48, help='起始标记长度')
    parser.add_argument('--pred_len', type=int, default=96, help='预测序列长度')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='M4子集')
    parser.add_argument('--inverse', action='store_true', help='是否反转输出数据', default=False)

    # 缺失值填充任务参数配置
    parser.add_argument('--mask_rate', type=float, default=0.25, help='缺失值比例')

    # 异常检测任务参数配置
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='先验异常比例（%）')

    # 模型定义参数
    parser.add_argument('--top_k', type=int, default=5, help='TimesBlock中保留的特征数量')
    parser.add_argument('--num_kernels', type=int, default=6, help='Inception模块中使用的核数量')
    parser.add_argument('--enc_in', type=int, default=7, help='编码器输入尺寸')
    parser.add_argument('--dec_in', type=int, default=7, help='解码器输入尺寸')
    parser.add_argument('--c_out', type=int, default=7, help='输出尺寸')
    # parser.add_argument('--d_model', type=int, default=512, help='模型维度')            # 与 pathformer 冲突参数
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--e_layers', type=int, default=2, help='编码器层数')
    parser.add_argument('--d_layers', type=int, default=1, help='解码器层数')
    # parser.add_argument('--d_ff', type=int, default=2048, help='全连接层维度')          # 与 pathformer 冲突参数
    parser.add_argument('--moving_avg', type=int, default=25, help='移动平均窗口大小')
    parser.add_argument('--factor', type=int, default=1, help='注意力因子')
    parser.add_argument('--distil', action='store_false',
                        help='是否在编码器中使用蒸馏，使用该参数意味着不使用蒸馏', default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout比例')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='时间特征编码方式，选项：[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='激活函数')
    parser.add_argument('--output_attention', action='store_true', help='是否输出编码器中的注意力')
    parser.add_argument('--channel_independence', type=int, default=0, help='0：通道独立 1：通道依赖，用于FreTS模型')

    # 参数配置解析
    # 优化参数
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')  # 数据加载器的worker数量
    parser.add_argument('--itr', type=int, default=1, help='experiments times')  # 实验次数
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')  # 训练轮数
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')  # 批量大小
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')  # 早停的耐心值
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')  # 优化器学习率
    parser.add_argument('--des', type=str, default='test', help='exp description')  # 实验描述
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')  # 损失函数
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')  # 调整学习率的方法
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)  # 是否使用自动混合精度训练

    # GPU配置参数
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')  # 是否使用GPU
    parser.add_argument('--gpu', type=int, default=0, help='gpu')  # 使用的GPU设备ID
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)  # 是否使用多个GPU
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')  # 多个GPU的设备ID

    # 去稳态投影器参数
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128], help='hidden layer dimensions of projector (List)')  # 投影器隐藏层维度
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')  # 投影器中隐藏层的数量


    #--------------------------
    # ModernTCN 参数配置
    parser.add_argument('--stem_ratio', type=int, default=6, help='stem 比例')
    parser.add_argument('--downsample_ratio', type=int, default=2, help='降采样比例')
    parser.add_argument('--ffn_ratio', type=int, default=2, help='FFN 比例')
    # parser.add_argument('--patch_size', type=int, default=16, help='补丁大小')           # 与 pathformer 冲突参数
    parser.add_argument('--patch_stride', type=int, default=8, help='补丁步长')

    # ModernTCN 各阶段配置
    parser.add_argument('--num_blocks', nargs='+',type=int, default=[1,1,1,1], help='每个阶段的块数量')  # 原 default=[1,1,1,1]
    # parser.add_argument('--num_blocks', nargs='+',type=int, default=[1,1], help='每个阶段的块数量')  # 原 default=[1,1,1,1]
    parser.add_argument('--large_size', nargs='+',type=int, default=[31,29,27,13], help='大核大小')
    parser.add_argument('--small_size', nargs='+',type=int, default=[5,5,5,5], help='小核大小，用于结构重参数化')
    parser.add_argument('--dims', nargs='+',type=int, default=[256,256,256,256], help='每个阶段的模型维度')   #  moderntcn 前向传播特征提取 下采样层
    parser.add_argument('--dw_dims', nargs='+',type=int, default=[256,256,256,256], help='每个阶段的DW卷积的维度')

    # ModernTCN 特有选项
    parser.add_argument('--small_kernel_merged', type=str2bool, default=False, help='小核是否已合并')
    parser.add_argument('--call_structural_reparam', type=bool, default=False, help='训练后是否调用结构重参数化')
    parser.add_argument('--use_multi_scale', type=str2bool, default=True, help='是否使用多尺度融合')

    # PatchTST 参数配置
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='全连接层dropout比例')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='头部dropout比例')
    parser.add_argument('--patch_len', type=int, default=16, help='补丁长度')
    parser.add_argument('--stride', type=int, default=8, help='步长')
    parser.add_argument('--padding_patch', default='end', help='None: 不进行填充； end: 在末尾进行填充')
    parser.add_argument('--revin', type=int, default=1, help='RevIN；True为1，False为0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-仿射；True为1，False为0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0：减去均值；1：减去最后一个')
    parser.add_argument('--decomposition', type=int, default=0, help='分解；True为1，False为0')
    parser.add_argument('--kernel_size', type=int, default=25, help='分解核大小')
    parser.add_argument('--individual', type=int, default=0, help='独立头部；True为1，False为0')

    # 参数配置：定义模型和优化器的参数
    parser.add_argument('--embed_type', type=int, default=0, 
                        help='嵌入类型：0为默认，1为值嵌入+时间嵌入+位置嵌入，2为值嵌入+时间嵌入，3为值嵌入+位置嵌入，4为值嵌入')
    # parser.add_argument('--enc_in', type=int, default=7, help='编码器输入大小')
    # parser.add_argument('--dec_in', type=int, default=7, help='解码器输入大小')
    # parser.add_argument('--c_out', type=int, default=7, help='输出大小')
    # parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    # parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
    # parser.add_argument('--e_layers', type=int, default=2, help='编码器层数')
    # parser.add_argument('--d_layers', type=int, default=1, help='解码器层数')
    # parser.add_argument('--d_ff', type=int, default=2048, help='fcn维度')
    # parser.add_argument('--moving_avg', type=int, default=25, help='移动平均窗口大小')
    # parser.add_argument('--factor', type=int, default=1, help='注意力因子')
    # parser.add_argument('--distil', action='store_false',
    #                     help='是否在编码器中使用蒸馏，使用该参数意味着不使用蒸馏', default=True)
    # parser.add_argument('--dropout', type=float, default=0.05, help='dropout比例')

    # parser.add_argument('--activation', type=str, default='gelu', help='激活函数')
    # parser.add_argument('--output_attention', action='store_true', help='是否输出编码器中的注意力')
    # parser.add_argument('--do_predict', action='store_true', help='是否预测未见的未来数据')           # 与 pathformer 冲突参数
    
    # 优化器参数
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start值')
     
    # 分类任务参数
    parser.add_argument('--class_dropout', type=float, default=0.05, help='分类任务的dropout比例')

    #--------------------------

    # pathformer 模型参数--------------------------
    # 模型
    parser.add_argument('--d_model', type=int, default=16, help='模型维度')     
    parser.add_argument('--d_ff', type=int, default=64, help='前馈网络维度')
    # parser.add_argument('--num_nodes', type=int, default=21, help='节点数')       # 为输入数据最后一个维度？手动设置
    parser.add_argument('--num_nodes', type=int, default=61, help='节点数')         # ================== 换数据 手动设置为数据最后一个维度===================
    parser.add_argument('--layer_nums', type=int, default=1, help='层数')           # 原来是3
    parser.add_argument('--k', type=int, default=2, help='每层选择的Top K补丁大小')
    parser.add_argument('--num_experts_list', type=list, default=[4, 4, 4], help='专家数量列表')
    parser.add_argument('--patch_size_list', nargs='+', type=int, default=[16,12,8,32,12,8,6,4,8,6,4,2], help='补丁大小列表')
    # parser.add_argument('--do_predict', action='store_true', help='是否预测未见的未来数据')
    # parser.add_argument('--revin', type=int, default=1, help='是否应用RevIN')
    parser.add_argument('--drop', type=float, default=0.1, help='dropout比率')
    # parser.add_argument('--embed', type=str, default='timeF', help='时间特征编码，选项：[timeF, fixed, learned]')
    parser.add_argument('--residual_connection', type=int, default=0, help='残差连接')
    parser.add_argument('--metric', type=str, default='mae', help='度量标准')
    parser.add_argument('--batch_norm', type=int, default=0, help='是否使用批量归一化')

    # 解析参数
    # args = parser.parse_args([
    #     '--task_name', 'classification',
    #     '--is_training', '1',

        # # 心跳分类
        # '--root_path', 'F:/0_czl/yan2/0-1_code/2nd_s_2nd/timesNet/dataset/all_datasets/Heartbeat/',
        # '--model_id', 'Heartbeat',

        # # 日语原音分类任务
        # '--root_path', "F:/0_czl/yan2/0-1_code/2nd_s_2nd/timesNet/dataset/all_datasets/JapaneseVowels/",  
        # '--model_id',  'JapaneseVowels',
        # '--dims', '256','512', # 1.通道维度  原：'256','512'
        # '--ffn_ratio', '4', # 2.FFN 率  原：'2'
        # '--num_blocks', '1', '1', # 3.模型层数，需要与1、3、5统一; 原 '1', '1',
        # '--patch_size', '1', # 4.patch大小、步长
        # '--patch_stride' ,'1',
        # '--large_size', '21', '19', # 5.核大小设置 原：'21', '19'  31,29,27
        # '--small_size', '5', '5', # 原：'5', '5'

        # # 面部分类  无法运行
        # '--root_path', 'F:/0_czl/yan2/0-1_code/2nd_s_2nd/timesNet/dataset/all_datasets/FaceDetection/',
        # '--model_id', 'FaceDetection ',

        # # PEMS-SF交通状况
        # '--root_path', 'F:/0_czl/yan2/0-1_code/2nd_s_2nd/timesNet/dataset/all_datasets/PEMS-SF/' ,
        # '--model_id', 'PEMS-SF',
        # # '--dims', '32', # 1.通道维度  原：'32',
        # # '--ffn_ratio', '4', # 2.FFN 率  原：'4'
        # # '--num_blocks', '2', # 3.模型层数，需要与1、3、5统一; 原 '2',
        # # '--patch_size', '48', # 4.patch大小、步长
        # # '--patch_stride' ,'24',
        # # '--large_size', '91', # 5.核大小设置 原：'91'
        # # '--small_size', '5', # 原：'5',

        # # SCP1 
        # '--root_path', "F:/0_czl/yan2/0-1_code/2nd_s_2nd/timesNet/dataset/all_datasets/SelfRegulationSCP1/",  
        # '--model_id',  'SelfRegulationSCP1',
        # # timesNet参数：
        # '--e_layers', '2',      # 1.    原 4
        # '--d_model', '64',      # 2.1   原 32
        # '--d_ff', '64',         # 2.2   原 32
        # '--top_k', '2',         # 3     原 4
        # # modernTCN参数：
        # '--dims', '32',         # 1.    原：'32',   通道维度    
        # '--ffn_ratio', '1',     # 2.    原：'4',    FFN 率  
        # '--num_blocks', '1',    # 3.    原 '1',     模型层数，需要与1、3、5统一;    
        # '--patch_size', '1',    # 4.1   原 '1',     patch大小 
        # '--patch_stride' ,'1',  # 4.2   原 '1',     patch步长 
        # '--large_size', '13',   # 5.1   原 '13',    大核大小设置 
        # '--small_size', '5',    # 5.2   原 '5',     小核大小        


        # # spoken数字 
        # '--root_path', "F:/0_czl/yan2/0-1_code/2nd_s_2nd/timesNet/dataset/all_datasets/SpokenArabicDigits/",  
        # '--model_id',  'SpokenArabicDigits',

        # # timesNet参数： 设置pathfomer时注释掉冲突参数==========================
        # '--e_layers', '2',      # 1.    原 2
        # '--d_model', '64',      # 2.1   原 32
        # '--d_ff', '64',         # 2.2   原 32
        # '--top_k', '2',         # 3.    原 2
        # modernTCN参数：
        # '--dims', '32','64','128',  # 1.    原：'32','64','128',    通道维度                        # 与 pathformer 冲突参数
        # '--ffn_ratio', '16',         # 2.    原：'4',                FFN 率  
        # '--num_blocks', '1','1',#'1',# 3.    原 '1','1','1',         模型层数，需要与1、3、5统一;   # 与 pathformer 冲突参数
        # '--patch_size', '16',       # 4.1   原 '16',                patch大小                       # 与 pathformer 冲突参数
        # '--patch_stride' ,'16',     # 4.2   原 '16',                patch步长                        # 与 pathformer 冲突参数
        # '--large_size', '5','5','5',# 5.1   原 '1','1','1',         大核大小设置                      # 与 pathformer 冲突参数
        # '--small_size', '1','1','1',# 5.2   原 '5','5','5',         小核大小                          # 与 pathformer 冲突参数
        
        # # 面部识别 
        # '--root_path', "F:/0_czl/yan2/0-1_code/2nd_s_2nd/timesNet/dataset/all_datasets/FaceDetection/",  
        # '--model_id',  'FaceDetection',
        # # timesNet参数：
        # '--e_layers', '2',      # 1.    原 3
        # '--d_model', '64',      # 2.1   原 64
        # '--d_ff', '256',         # 2.2   原 256
        # '--top_k', '3',         # 3     原 3
        # # modernTCN参数：
        # '--dims', '32','64','128',  # 1.    原：'32','64','128',    通道维度    
        # '--ffn_ratio', '1',         # 2.    原：'1',                FFN 率  
        # '--num_blocks', '1','1','1',# 3.    原 '1','1','1',         模型层数，需要与1、3、5统一;    
        # '--patch_size', '16',       # 4.1   原 '32',                patch大小 
        # '--patch_stride' ,'8',     # 4.2   原 '16',                patch步长 
        # '--large_size', '9','9','9',# 5.1   原 '9','9','9',         大核大小设置 
        # '--small_size', '5','5','5',# 5.2   原 '5','5','5',         小核大小        
       
        # # 手势识别 
        # '--root_path', "F:/0_czl/yan2/0-1_code/2nd_s_2nd/timesNet/dataset/all_datasets/UWaveGestureLibrary/",  
        # '--model_id',  'UWaveGestureLibrary',
        # # timesNet参数：
        # '--e_layers', '2',      # 1.    原 2
        # '--d_model', '64',      # 2.1   原 32
        # '--d_ff', '256',         # 2.2   原 64
        # '--top_k', '3',         # 3     原 3
        # # modernTCN参数：
        # '--dims', '128','256',      # 1.    原：'128','256'    通道维度    
        # '--ffn_ratio', '1',         # 2.    原：'1',                FFN 率  
        # '--num_blocks', '1','1',    # 3.    原 '1','1',         模型层数，需要与1、3、5统一;    
        # '--patch_size', '1',        # 4.1   原 '1',                patch大小 
        # '--patch_stride' ,'1',      # 4.2   原 '1',                patch步长 
        # '--large_size', '51','49',  # 5.1   原 '51','49',        大核大小设置 
        # '--small_size', '5','5',    # 5.2   原 '5','5',        小核大小


        # '--model', 'TimesNet', # TimesNet(mine)_0

        # '--model', 'MICN', # MICN_1

        # '--model', 'DLinear',  # DLinear_2

        # '--model', 'LightTS',  # LightTS_3

        # '--model', 'Transformer', # Transformer_4'

        # '--model', 'PatchTST',  # PatchTST_5

        # '--e_layers', '3',  # 6. 原 心跳 2；面部 3；日语 2; 交通 3；scp1：4
        # '--batch_size', '8',  # 7. 原 16
        # '--d_model', '64',  # 8.1 原 64；交况 32；scp1: 32
        # '--d_ff', '64',  # 8.2 原64 全连接层维度；交况 32; scp1: 32
        # '--top_k', '8',  # 9 原 1；面部 3；日语 3; scp1：4

        # '--batch_size', '8',    # 原 16
        # '--patience', '15',  # 原 10
        # '--train_epochs', '50',

        # '--data', 'UEA',
        # '--des', 'Exp',
        # '--itr', '1',
        # '--learning_rate', '0.001', 
        # '--num_workers', '0', 
        # '--c', '2',
        # '--model2', "ModernTCN"  # 添加第二个模型

        # ==========================PathFormer参数设置；存在重复、冲突参数，需要单独设置==========================
        # '--model', 'PathFormer', # PathFormer(mine)_0 直接修改了原来的代码
        # '--d_model', '4',    
    # ])

    # args = parser.parse_args([
    #     '--batch_size', '16',  
    #     '--data', 'UEA',
    #     '--d_ff', '64',
    #     '--d_model', '64',
    #     '--des', 'Exp',
    #     '--e_layers', '2',
    #     '--is_training', '1',
    #     '--itr', '1',
    #     '--learning_rate', '0.001',
    #     '--model', 'TimesNet',
    #     '--model_id', 'Heartbeat',
    #     '--num_workers', '0',
    #     '--patience', '10',
    #     '--root_path', 'E:/yan2/0-1_code/2nd_s_2nd/timesNet/dataset/all_datasets/Heartbeat/',
    #     '--task_name', 'classification',
    #     '--top_k', '1',
    #     '--train_epochs', '30',
    # ])

    args = parser.parse_args()  # 解析命令行参数
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False  # 根据是否可用及用户设置决定是否使用GPU
    print(f'GPU:{torch.cuda.is_available()}')

    # 根据是否使用GPU和多个GPU来配置设备
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # 打印实验参数
    print('Args in experiment:')
    print_args(args)

    # 超参数打印
    print(f"--num_blocks:   {args.num_blocks}")
    print(f"--large_size:   {args.large_size}")
    print(f"--small_size:   {args.small_size}")
    print(f"--dims:         {args.dims}")
    print('')
    
    # 根据任务名称选择对应的实验类
    if args.task_name == 'classification':
        Exp = Exp_Classification

    # 根据是否处于训练模式来执行不同的操作
    if args.is_training:
        for ii in range(args.itr):
            # 设置实验记录
            exp = Exp(args)  # 初始化实验

            # 格式化生成实验设置字符串
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            # 开始训练并测试
            print('\n>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('\n>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()  # 清理GPU缓存
    else:
        ii = 0
        # 生成实验设置字符串，用于测试
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # 初始化实验
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)  # 仅进行测试
        torch.cuda.empty_cache()  # 清理GPU缓存
