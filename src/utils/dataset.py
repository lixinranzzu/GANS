from logging import getLogger  # 从 logging 模块导入 getLogger，用于获取日志记录器
from collections import Counter  # 从 collections 模块导入 Counter，用于计数
import os  # 导入 os 模块，用于处理文件和目录路径
import pandas as pd  # 导入 pandas，用于数据处理和分析
import numpy as np  # 导入 numpy，用于数值计算
import torch  # 导入 PyTorch 库，用于深度学习
#from utils.data_utils import (ImageResize, ImagePad, image_to_tensor, load_decompress_img_from_lmdb_value)  # 从自定义工具模块导入图像处理相关的函数

# 定义推荐数据集类
class RecDataset(object):
    def __init__(self, config, df=None):
        self.config = config  # 保存配置字典
        self.logger = getLogger()  # 获取日志记录器

        # 数据路径和文件
        self.dataset_name = config['dataset']  # 从配置中获取数据集名称
        self.dataset_path = os.path.abspath(config['data_path'] + self.dataset_name)  # 构建数据集的绝对路径
        #E:\project\my_rec\.venv\data\baby

        # 数据框架字段
        self.uid_field = self.config['USER_ID_FIELD']  # 用户 ID 字段名,及UserID
        self.iid_field = self.config['ITEM_ID_FIELD']  # 项目 ID 字段名,及ItemID
        self.splitting_label = self.config['inter_splitting_label']  # 用于划分数据集的标签,及x_label
        self.timestamp_field = self.config['TIME_FIELD']  # 时间戳字段名,及Timestamp

        if df is not None:  # 如果提供了 DataFrame
            self.df = df  # 使用提供的 DataFrame
            return  # 结束初始化

        # 检查所有文件是否存在
        check_file_list = [self.config['inter_file_name']]  # 要检查的文件列表,及baby.inter
        for i in check_file_list:  # 遍历文件列表
            file_path = os.path.join(self.dataset_path, i)  # 构建文件的完整路径
            if not os.path.isfile(file_path):  # 如果文件不存在
                raise ValueError('File {} not exist'.format(file_path))  # 抛出错误

        # 从数据路径加载评分文件
        self.load_inter_graph(config['inter_file_name'])  # 加载交互数据
        self.item_num = int(max(self.df[self.iid_field].values)) + 1  # 计算项目数量
        self.user_num = int(max(self.df[self.uid_field].values)) + 1  # 计算用户数量

    def load_inter_graph(self, file_name):
        inter_file = os.path.join(self.dataset_path, file_name)  # 构建交互文件的完整路径
        cols = [self.uid_field, self.iid_field, self.splitting_label]  # 需要加载的列
        self.df = pd.read_csv(inter_file, usecols=cols, sep=self.config['field_separator'])  # 从 CSV 文件加载数据,field_separator及分隔符“\t”
        if not self.df.columns.isin(cols).all():  # 检查是否所有必需的列都存在
            raise ValueError('File {} lost some required columns.'.format(inter_file))  # 抛出错误

    def split(self):
        dfs = []  # 初始化一个列表，用于存储划分后的数据集
        #更像是排序，因为inter中的用户-项目交互中的x_label不是按照从小到大排序的，这个之后的dfs按照x_label=0、1、2的顺序从上到下
        for i in range(3):  # 遍历 0, 1, 2        0为训练集，1为验证集，2为测试集  这是inter中划分好的
            temp_df = self.df[self.df[self.splitting_label] == i].copy()  # 根据 splitting_label 划分数据
            temp_df.drop(self.splitting_label, inplace=True, axis=1)  # 删除不再使用的列
            dfs.append(temp_df)  # 将划分后的 DataFrame 添加到列表中

        if self.config['filter_out_cod_start_users']:  # 如果配置中要求过滤新用户
            train_u = set(dfs[0][self.uid_field].values)  # 获取训练集中的用户 ID
            for i in [1, 2]:  # 遍历验证集和测试集
                dropped_inter = pd.Series(True, index=dfs[i].index)  # 创建一个布尔序列，初始值为 True
                dropped_inter ^= dfs[i][self.uid_field].isin(train_u)  # 如果训练集中没有这个用户，保持为True
                dfs[i].drop(dfs[i].index[dropped_inter], inplace=True)  # dfs[i].index[dropped_inter] 获取 dropped_inter 为 True 的索引，即需要删除的行的索引

        # 将划分后的数据集包装为 RecDataset
        full_ds = [self.copy(_) for _ in dfs]  # 复制每个划分后的 DataFrame
        return full_ds  # 返回划分后的数据集列表

    def copy(self, new_df):
        """给定新的交互特征，返回一个新的 RecDataset 对象，
        其交互特征已更新为 new_df，其他属性保持不变。
        """
        nxt = RecDataset(self.config, new_df)  # 创建一个新的 RecDataset 对象
        nxt.item_num = self.item_num  # 复制项目数量
        nxt.user_num = self.user_num  # 复制用户数量
        return nxt  # 返回新的 RecDataset 对象

    def get_user_num(self):
        return self.user_num  # 返回用户数量

    def get_item_num(self):
        return self.item_num  # 返回项目数量

    def shuffle(self):
        """就地打乱交互记录。
        """
        self.df = self.df.sample(frac=1, replace=False).reset_index(drop=True)  # 随机打乱 DataFrame 的行

    def __len__(self):
        return len(self.df)  # 返回数据集的大小

    def __getitem__(self, idx):
        # 返回指定索引的交互记录
        return self.df.iloc[idx]  # 使用 iloc 获取指定行的数据

    def __repr__(self):
        return self.__str__()  # 返回字符串表示

    def __str__(self):
        info = [self.dataset_name]  # 初始化信息列表，包含数据集名称
        self.inter_num = len(self.df)  # 计算交互记录的数量
        uni_u = pd.unique(self.df[self.uid_field])  # 获取唯一用户 ID
        uni_i = pd.unique(self.df[self.iid_field])  # 获取唯一项目 ID
        tmp_user_num, tmp_item_num = 0, 0  # 初始化用户和项目数量
        if self.uid_field:  # 如果用户 ID 字段存在
            tmp_user_num = len(uni_u)  # 计算用户数量
            avg_actions_of_users = self.inter_num / tmp_user_num  # 计算用户的平均交互次数
            info.extend(['The number of users: {}'.format(tmp_user_num),  # 添加用户数量信息
                         'Average actions of users: {}'.format(avg_actions_of_users)])  # 添加用户平均交互次数信息
        if self.iid_field:  # 如果项目 ID 字段存在
            tmp_item_num = len(uni_i)  # 计算项目数量
            avg_actions_of_items = self.inter_num / tmp_item_num  # 计算项目的平均交互次数
            info.extend(['The number of items: {}'.format(tmp_item_num),  # 添加项目数量信息
                         'Average actions of items: {}'.format(avg_actions_of_items)])  # 添加项目平均交互次数信息
        info.append('The number of inters: {}'.format(self.inter_num))  # 添加交互记录数量信息
        if self.uid_field and self.iid_field:  # 如果用户和项目 ID 字段都存在
            sparsity = 1 - self.inter_num / tmp_user_num / tmp_item_num  # 计算稀疏度
            info.append('The sparsity of the dataset: {}%'.format(sparsity * 100))  # 添加稀疏度信息
        return '\n'.join(info)  # 返回信息列表的字符串表示

    #稀疏度
    def get_user_train_interaction_count(self):
        """
        Count the number of interactions per user in the training set (x_label == 0).
        Return: dict {user_id: interaction_count}
        """
        # 注意：self.df 在原始 dataset 中仍包含 x_label
        if self.splitting_label not in self.df.columns:
            raise RuntimeError("Splitting label not found in dataset.")

        train_df = self.df[self.df[self.splitting_label] == 0]
        user_cnt = train_df.groupby(self.uid_field).size()
        return user_cnt.to_dict()

    def get_sparsity_user_groups(self, bins):
        """
        Group users by training interaction counts.

        Args:
            bins: list of tuples, e.g. [(1,10), (11,20), (21,30)]

        Returns:
            dict: { '1-10': set(user_ids), ... }
        """
        user_cnt = self.get_user_train_interaction_count()
        groups = {}

        for (l, r) in bins:
            key = f'{l}-{r}'
            groups[key] = set(
                u for u, c in user_cnt.items()
                if l <= c <= r
            )

        return groups