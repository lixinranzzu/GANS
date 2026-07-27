# coding: utf-8
# @email  : enoche.chow@gmail.com

import os  # 导入操作系统接口模块，用于处理文件和目录路径
import numpy as np  # 导入 NumPy 库，用于数值计算和数组操作
import torch  # 导入 PyTorch 库，用于深度学习
import torch.nn as nn  # 导入 PyTorch 的神经网络模块

class AbstractRecommender(nn.Module):
    r"""Base class for all models
    """
    def pre_epoch_processing(self):
        pass  # 占位符方法，子类可以重写以实现每个训练周期开始前的处理逻辑

    def post_epoch_processing(self):
        pass  # 占位符方法，子类可以重写以实现每个训练周期结束后的处理逻辑

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError  # 抛出未实现错误，子类必须实现此方法以计算损失

    def predict(self, interaction):
        r"""Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError  # 抛出未实现错误，子类必须实现此方法以进行预测

    def full_sort_predict(self, interaction):
        r"""full sort prediction function.
        Given users, calculate the scores between users and all candidate items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
            shape: [n_batch_users * n_candidate_items]
        """
        raise NotImplementedError  # 抛出未实现错误，子类必须实现此方法以进行全排序预测

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = self.parameters()  # 获取模型的所有参数
        params = sum([np.prod(p.size()) for p in model_parameters])  # 计算所有可训练参数的数量
        return super().__str__() + '\nTrainable parameters: {}'.format(params)  # 返回模型的字符串表示，包括可训练参数数量


class GeneralRecommender(AbstractRecommender):
    """This is a abstract general recommender. All the general model should implement this class.
    The base general recommender class provide the basic dataset and parameters information.
    """
    def __init__(self, config, dataloader):
        super(GeneralRecommender, self).__init__()  # 调用父类的构造函数以初始化基类部分

        # load dataset info
        self.USER_ID = config['USER_ID_FIELD']  # 从配置中获取用户 ID 字段
        self.ITEM_ID = config['ITEM_ID_FIELD']  # 从配置中获取项目 ID 字段
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID  # 构建负样本 ID
        self.n_users = dataloader.dataset.get_user_num()  # 获取用户数量
        self.n_items = dataloader.dataset.get_item_num()  # 获取项目数量

        # load parameters info
        self.batch_size = config['train_batch_size']  # 从配置中获取批量大小
        self.device = config['device']  # 从配置中获取设备信息（CPU 或 GPU）

        # load encoded features here
        self.v_feat, self.t_feat = None, None  # 初始化视觉特征和文本特征为 None
        if not config['end2end'] and config['is_multimodal_model']:  # 检查是否为多模态模型且不是端到端模型
            dataset_path = os.path.abspath(config['data_path'] + config['dataset'])  # 构建数据集路径
            # if file exist?
            v_feat_file_path = os.path.join(dataset_path, config['vision_feature_file'])  # 构建视觉特征文件路径
            t_feat_file_path = os.path.join(dataset_path, config['text_feature_file'])  # 构建文本特征文件路径
            if os.path.isfile(v_feat_file_path):  # 检查视觉特征文件是否存在
                self.v_feat = torch.from_numpy(np.load(v_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    self.device)  # 加载视觉特征并转换为 PyTorch 张量，移动到指定设备
            if os.path.isfile(t_feat_file_path):  # 检查文本特征文件是否存在
                self.t_feat = torch.from_numpy(np.load(t_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    self.device)  # 加载文本特征并转换为 PyTorch 张量，移动到指定设备

            assert self.v_feat is not None or self.t_feat is not None, 'Features all NONE'  # 确保至少加载了视觉特征或文本特征
