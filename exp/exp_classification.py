from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.optim import lr_scheduler
import pdb

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        """
        初始化Exp_Classification类的一个实例。
        
        参数:
        - args: 一个包含实验配置的参数对象。
        """
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        """
        构建模型及其相关配置。
        
        返回:
        - model: 根据配置初始化的模型实例。
        """
        # 获取训练和测试数据集及其加载器，并配置序列长度、预测长度和编码器输入维度
        # 这里假设_get_data方法根据flag参数返回对应数据集和数据加载器
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        # 设置序列长度为训练集和测试集中最大序列长度
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        # 预测长度设置为0，具体值根据实际任务和模型需求确定
        self.args.pred_len = 0
        # 设置编码器输入维度为训练数据特征DataFrame的列数
        self.args.enc_in = train_data.feature_df.shape[1]
        # print('in _build_model enc_in = ', self.args.enc_in)

        # 设置类别数量为训练数据类别名称列表的长度
        self.args.num_class = len(train_data.class_names)

        
        # 根据配置的模型名称初始化模型实例
        # 这里假设模型类包含一个名为Model的静态方法或类方法，用于创建模型实例
        model = self.model_dict[self.args.model].Model(self.args).float()
        # #ModernTCN模型
        # model2 = self.model_dict['ModernTCN'].Model(self.args).float()

        # 如果配置了使用多个GPU并且使用了GPU，则使用DataParallel包装模型以实现多GPU训练
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = torch.nn.DataParallel(model, device_ids=self.args.device_ids)
            ## 下面的代码被注释掉了，如果需要使用ModernTCN模型并且使用多GPU，可以取消注释
            # model2 = torch.nn.DataParallel(model, device_ids=self.args.device_ids)

        # 返回构建的模型实例
        return model  # 如果有model2，这里可以返回model和model2的元组 (model, model2)

    def _get_data(self, flag):
        """
        根据标志获取数据集及其加载器。
        
        参数:
        - flag: 字符串，指定获取训练数据集还是测试数据集。
        
        返回:
        - data_set: 数据集实例。
        - data_loader: 数据加载器。
        """
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """
        选择并初始化优化器。
        
        返回:
        - model_optim: 优化器实例。
        """
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """
        选择损失函数。
        
        返回:
        - criterion: 损失函数实例。
        """
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        """
        在验证集上评估模型性能。
        
        参数:
        - vali_data: 验证数据集实例。
        - vali_loader: 验证数据加载器。
        - criterion: 损失函数实例。
        
        返回:
        - total_loss: 验证集上的平均损失。
        - accuracy: 验证集上的准确率。
        """
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # 通过模型进行前向传播
                # outputs = self.model(batch_x, padding_mask, None, None)       # 原来的 timesnet + moderntcn
                outputs, _ = self.model(batch_x, padding_mask)      # pathfomer

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds, dim=1)  # 计算每个样本每个类别的预测概率
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # 获取每个样本的预测类别
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)  # 计算准确率

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        """
        训练模型。
        
        参数:
        - setting: 训练设置字符串，用于指定训练的配置。
        
        返回:
        - 训练好的模型。
        """
        # 获取训练、验证和测试数据
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        # 检查并创建模型检查点目录
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # 初始化时间和训练步数
        time_now = time.time()
        train_steps = len(train_loader)
        
        # 设置早停和学习率调度器，防止过拟合 
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # modern TCN特有
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        # 开始训练循环
        for epoch in range(self.args.train_epochs):
            
            # 初始化迭代计数器和训练损失列表
            iter_count = 0
            train_loss = []

            # 将模型设置为训练模式
            self.model.train()
            # 记录当前epoch开始的时间
            epoch_time = time.time()

            # 开始每轮的迭代
            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1  # 增加迭代计数器
                model_optim.zero_grad()  # 清空模型的梯度

                # 将数据移动到指定的设备上
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # 通过模型进行前向传播
                # outputs = self.model(batch_x, padding_mask, None, None)       # 原来的 timesnet + moderntcn
                outputs, balance_loss = self.model(batch_x, padding_mask)       # pathfomer
                
                # 计算损失
                # loss = criterion(outputs, label.long().squeeze(-1))           # 原来的
                loss = criterion(outputs, label.long().squeeze(-1)) + balance_loss  # 加入balanceloss
                train_loss.append(loss.item())  # 记录训练损失

                # 每100次迭代打印一次日志
                if (i + 1) % 100 == 0:
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # 计算损失函数的反向传播，用于更新网络参数
                loss.backward()

                # 对梯度进行规范化限制，防止梯度爆炸
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)

                # 更新模型参数，应用学习率等优化策略
                model_optim.step()

            # 计算每轮的平均损失，并进行验证
            print("\nEpoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            # 打印和记录每轮的结果
            print("Epoch: {}, Steps: {} | "
                "Train Loss: {:.3f}, Vali Loss: {:.3f}, Vali Acc: {:.3f}, "
                "Test Loss: {:.3f}, Test Acc: {:.3f}"
                .format(epoch + 1, train_steps, 
                        train_loss, vali_loss, val_accuracy, 
                        test_loss, test_accuracy))
                        
            # 使用早停策略判断当前模型性能是否改善，如果未改善，则停止训练
            early_stopping(-val_accuracy, self.model, path)  # 保存模型

            # 检查是否需要提前停止训练
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # 每隔5个周期调整学习率
            if (epoch + 1) % 5 == 0:
                # 调整模型优化器的学习率
                adjust_learning_rate(model_optim, 
                                     scheduler, 
                                     epoch + 1, 
                                     self.args)

        # 加载最佳模型
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path), strict=False) #, strict=False
        print(f'\n读取 最佳模型 -> {best_model_path}')

        return self.model

    def test(self, setting, test=0):
        """
        测试模型。
        
        参数:
        - setting: 测试设置字符串，用于指定测试的配置。
        - test: 一个标志，决定是否加载已经训练好的模型。
        
        返回:
        - None
        """
        # 获取测试数据
        test_data, test_loader = self._get_data(flag='TEST')

        if test:
            # 如果test标志为真，则加载已经训练好的模型
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 
                                                               'checkpoint.pth')))

        # 初始化用于存储预测结果和真实标签的列表
        preds = []
        trues = []
        
        # 设置测试结果的文件夹路径
        folder_path = './test_results/' + setting + '/'
        # 如果文件夹不存在，则创建该文件夹
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 将模型设置为评估模式
        self.model.eval()
        with torch.no_grad():
            # 遍历测试数据加载器中的数据
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                # 将输入数据和填充掩码转换为浮点数并移动到指定的设备（如GPU）
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # 通过模型进行预测
                # outputs = self.model(batch_x, padding_mask, None, None)       # 原来的 timesnet + moderntcn
                outputs, balance_loss = self.model(batch_x, padding_mask)       # pathfomer

                # 将预测结果和对应的真实标签分别添加到preds和trues列表中
                preds.append(outputs.detach())
                trues.append(label)

        # 将预测结果和真实标签转换为一个完整的张量
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print(f'test shape:\n  preds {preds.shape}\n  trues {trues.shape}')  # 打印测试数据的形状

        # 计算预测概率，并找到最大概率对应的预测类别
        probs = torch.nn.functional.softmax(preds)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()

        # 将真实标签展平并转换为NumPy数组
        trues = trues.flatten().cpu().numpy()
        # 计算准确率
        accuracy = cal_accuracy(predictions, trues)

        # 设置结果保存的文件夹路径
        folder_path = './results/' + setting + '/'
        # 如果结果文件夹不存在，则创建文件夹
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 打印准确率
        print('accuracy:{}'.format(accuracy))
        # 打开或创建文件用于追加测试结果
        f = open("result_classification.txt", 'a')
        # 将测试配置和准确率写入文件
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('\n')
        # 关闭文件
        f.close()