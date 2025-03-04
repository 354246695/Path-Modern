from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader  # 心跳------
}


def data_provider(args, flag):
    """
    根据提供的参数和标志，为不同任务提供数据集和数据加载器。

    参数:
    - args: 包含各种数据加载和处理配置的参数对象。
    - flag: 标志符，用于指定数据集的加载模式（如'test'或'train'）。

    返回:
    - data_set: 数据集对象，根据任务类型和参数配置。
    - data_loader: 数据加载器对象，用于批处理数据集。
    """
    # 根据参数确定是否使用时间编码
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    # 根据flag确定数据加载的配置
    if flag == 'test':
        # 测试集的配置
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':  # classification 实验用=========
            batch_size = args.batch_size
        else:
            batch_size = 1  # 评估任务使用批大小为1
        freq = args.freq
    else:
        # 训练集或验证集的配置
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    # 根据任务类型分别处理数据集 和 数据加载器的创建
    if args.task_name == 'anomaly_detection':
        drop_last = False  # 异常检测任务不丢弃最后一个批次
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        
    elif args.task_name == 'classification':  # classification 实验用=============
        drop_last = False  # 分类任务不丢弃最后一个批次
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
    else:
        # 默认处理方式，适用于除异常检测和分类外的任务
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
    return data_set, data_loader

