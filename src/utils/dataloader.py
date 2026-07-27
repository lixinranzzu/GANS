# coding: utf-8
# @email: enoche.chow@gmail.com
"""
Wrap dataset into dataloader
################################################
"""
import math
import torch
import random
import numpy as np
from logging import getLogger
from scipy.sparse import coo_matrix


class AbstractDataLoader(object):
    """:class:`AbstractDataLoader` is an abstract object which would return a batch of data which is loaded by
        :class:`~recbole.data.interaction.Interaction` when it is iterated.
        And it is also the ancestor of all other dataloader.

        Args:
            config (Config): The config of dataloader.
            dataset (Dataset): The dataset of dataloader.
            batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
            dl_format (InputType, optional): The input type of dataloader. Defaults to
                :obj:`~recbole.utils.enum_type.InputType.POINTWISE`.
            shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.

        Attributes:
            dataset (Dataset): The dataset of this dataloader.
            shuffle (bool): If ``True``, dataloader will shuffle before every epoch.
            real_time (bool): If ``True``, dataloader will do data pre-processing,
                such as neg-sampling and data-augmentation.
            pr (int): Pointer of dataloader.
            step (int): The increment of :attr:`pr` for each batch.
            batch_size (int): The max interaction number for all batch.
        """
    def __init__(self, config, dataset, additional_dataset=None,
                 batch_size=1, neg_sampling=False, shuffle=False):
        self.config = config  # 保存配置字典
        self.logger = getLogger()  # 获取日志记录器
        self.dataset = dataset  # 主数据集
        self.dataset_bk = self.dataset.copy(self.dataset.df)  # 备份数据集
        self.additional_dataset = additional_dataset  # 额外的数据集（可选）
        self.batch_size = batch_size  # 批处理大小
        self.step = batch_size  # 步长，通常与批处理大小相同
        self.shuffle = shuffle  # 是否打乱数据
        self.neg_sampling = neg_sampling  # 是否进行负采样
        self.device = config['device']  # 设备配置（如 CPU 或 GPU）

        # 计算稀疏度
        self.sparsity = 1 - self.dataset.inter_num / self.dataset.user_num / self.dataset.item_num
        self.pr = 0  # 当前处理的记录索引
        self.inter_pr = 0  # 当前交互记录索引

        self.S_nn=config['S_nn']

    def pretrain_setup(self):
        """用于处理一些初始化后的问题，例如在需要负采样时调整批大小等。默认情况下不执行任何操作。"""
        pass

    def data_preprocess(self):
        """用于进行一些数据预处理，例如预负采样和预数据增强。默认情况下不执行任何操作。"""
        pass

    def __len__(self):
        return math.ceil(self.pr_end / self.step)  # 返回数据加载器的长度

    def __iter__(self):
        if self.shuffle:  # 如果需要打乱数据
            self._shuffle()  # 调用打乱方法
        return self  # 返回自身以支持迭代

    def __next__(self):
        if self.pr >= self.pr_end:  # 如果当前索引超过结束索引
            self.pr = 0  # 重置索引
            self.inter_pr = 0  # 重置交互记录索引
            raise StopIteration()  # 停止迭代
        return self._next_batch_data()  # 返回下一个批次的数据

    @property
    def pr_end(self):
        """此属性标记 dataloader.pr 的结束，用于 :meth:`__next__()`。"""
        raise NotImplementedError('Method [pr_end] should be implemented')  # 抛出未实现错误

    def _shuffle(self):
        """打乱数据的顺序，如果 self.shuffle 为 True，将在 :meth:`__iter__()` 中调用。"""
        raise NotImplementedError('Method [shuffle] should be implemented.')  # 抛出未实现错误

    def _next_batch_data(self):
        """组装下一个批次的数据并返回。

        Returns:
            Interaction: 下一个批次的数据。
        """
        raise NotImplementedError('Method [next_batch_data] should be implemented.')  # 抛出未实现错误

class TrainDataLoader(AbstractDataLoader):
    """
    带有负采样的通用数据加载器。
    """
    def __init__(self, config, dataset, batch_size=1, shuffle=False):
        # 调用父类的构造函数，初始化基本参数
        super().__init__(config, dataset, additional_dataset=None,
                         batch_size=batch_size, neg_sampling=True, shuffle=shuffle)

        # 特别为训练数据加载器
        self.history_items_per_u = dict()  # 每个用户的历史项目字典
        self.all_items = self.dataset.df[self.dataset.iid_field].unique().tolist()# 所有项目的唯一列表，作为列表，便于进行随机采样、打乱顺序等操作，这些操作在列表上更为直观和简单
        self.all_uids = self.dataset.df[self.dataset.uid_field].unique()  # 所有用户的唯一 ID
        self.all_items_set = set(self.all_items)  # 所有项目的集合
        self.all_users_set = set(self.all_uids)  # 所有用户的集合
        self.all_item_len = len(self.all_items)  # 项目总数
        self.use_full_sampling = config['use_full_sampling']  # 是否使用全采样,MMGCN中为false

        # 根据配置选择采样方法
        if config['use_neg_sampling']:
            if self.use_full_sampling:
                self.sample_func = self._get_full_uids_sample  # 全用户采样
            else:
                self.sample_func = self._get_neg_sample  # 负采样,MMGCN中采用的是这个 ，返回的是一个张量，第一行的张量为用户id,第二行的张量为其交互的样本，第三行的为负样本
        else:
            self.sample_func = self._get_non_neg_sample  # 非负采样

        self._get_history_items_u()  # 获取每个用户的历史项目
        self.neighborhood_loss_required = config['use_neighborhood_loss']  # 是否需要邻域损失
        if self.neighborhood_loss_required:
            self.history_users_per_i = {}  # 每个项目的历史用户字典
            self._get_history_users_i()  # 获取每个项目的历史用户
            self.user_user_dict = self._get_my_neighbors(self.config['USER_ID_FIELD'])  # 用户邻居字典
            self.item_item_dict = self._get_my_neighbors(self.config['ITEM_ID_FIELD'])  # 项目邻居字典

    def pretrain_setup(self):
        """
        重置数据加载器。输出相同的正负样本以进行每次训练。
        :return:
        """
        # 排序和随机
        if self.shuffle:
            self.dataset = self.dataset_bk.copy(self.dataset_bk.df)  # 如果需要，恢复备份数据集
        self.all_items.sort()  # 对所有项目进行排序
        if self.use_full_sampling:
            self.all_uids.sort()  # 对所有用户进行排序
        random.shuffle(self.all_items)  # 随机打乱项目顺序

    def inter_matrix(self, form='coo', value_field=None):
        """将交互数据转换为稀疏矩阵。

        Args:
            form (str): 矩阵格式（'coo' 或 'csr'）。
            value_field (str): 用于稀疏矩阵的值字段。

        Returns:
            scipy.sparse.coo_matrix: 稀疏矩阵。
        """
        if not self.dataset.uid_field or not self.dataset.iid_field:
            raise ValueError('dataset doesn\'t exist uid/iid, thus can not converted to sparse matrix')
        return self._create_sparse_matrix(self.dataset.df, self.dataset.uid_field,
                                          self.dataset.iid_field, form, value_field)

    def _create_sparse_matrix(self, df_feat, source_field, target_field, form='coo', value_field=None):
        """创建稀疏矩阵。

        Args:
            df_feat (DataFrame): 特征数据框。
            source_field (str): 源字段（用户 ID）。
            target_field (str): 目标字段（项目 ID）。
            form (str): 矩阵格式（'coo' 或 'csr'）。
            value_field (str): 值字段。

        Returns:
            scipy.sparse.coo_matrix: 稀疏矩阵。
        """
        src = df_feat[source_field].values  # 获取源字段的值（用户 ID）
        tgt = df_feat[target_field].values  # 获取目标字段的值（项目 ID）
        if value_field is None:
            data = np.ones(len(df_feat))  # 如果没有值字段，默认值为 1
        else:
            if value_field not in df_feat.columns:
                raise ValueError('value_field [{}] should be one of `df_feat`\'s features.'.format(value_field))
            data = df_feat[value_field].values  # 获取值字段的值
        mat = coo_matrix((data, (src, tgt)), shape=(self.dataset.user_num, self.dataset.item_num))  # 创建 COO 格式的稀疏矩阵

        if form == 'coo':
            return mat  # 返回 COO 格式的矩阵
        elif form == 'csr':
            return mat.tocsr()  # 返回 CSR 格式的矩阵
        else:
            raise NotImplementedError('sparse matrix format [{}] has not been implemented.'.format(form))

    @property
    def pr_end(self):
        """根据采样方式返回 pr 的结束值。"""
        if self.use_full_sampling:
            return len(self.all_uids)  # 如果使用全采样，返回用户数量
        return len(self.dataset)  # 否则返回数据集的长度

    def _shuffle(self):
        """打乱数据集和用户 ID 的顺序。"""
        self.dataset.shuffle()  # 打乱数据集
        if self.use_full_sampling:
            np.random.shuffle(self.all_uids)  # 打乱用户 ID

    def _next_batch_data(self):
        """返回下一个批次的数据。"""
        return self.sample_func()  # 调用采样函数获取下一个批次的数据

    #新的负采样
    def _get_neg_sample(self):
        """获取正样本和对应的多个候选负样本"""
        # 获取当前批次数据
        cur_data = self.dataset[self.pr: self.pr + self.step]
        self.pr += self.step

        # 用户 & 正样本 ID（张量）
        user_tensor = torch.tensor(cur_data[self.config['USER_ID_FIELD']].values).type(torch.LongTensor).to(self.device)
        item_tensor = torch.tensor(cur_data[self.config['ITEM_ID_FIELD']].values).type(torch.LongTensor).to(self.device)

        #print(self.S_nn)
        # 批量采样：每个用户n个负样本
        u_ids = cur_data[self.config['USER_ID_FIELD']]
        neg_ids = self._sample_neg_ids(u_ids, k=self.S_nn).to(self.device)  # shape: [batch_size, 10]
        # 注意：此时不再构造 batch_tensor，而是返回 3 元组
        return user_tensor, item_tensor, neg_ids  # [B], [B], [B, 10]'''


    #原始的负采样
    '''def _get_neg_sample(self):
        """获取负样本。"""
        cur_data = self.dataset[self.pr: self.pr + self.step]  # 获取当前批次的数据
        self.pr += self.step  # 更新当前索引
        # 转换为张量
        user_tensor = torch.tensor(cur_data[self.config['USER_ID_FIELD']].values).type(torch.LongTensor).to(self.device)
        item_tensor = torch.tensor(cur_data[self.config['ITEM_ID_FIELD']].values).type(torch.LongTensor).to(self.device)
        batch_tensor = torch.cat((torch.unsqueeze(user_tensor, 0),
                                  torch.unsqueeze(item_tensor, 0)))  # 合并用户和项目张量
        u_ids = cur_data[self.config['USER_ID_FIELD']]  # 获取用户 ID
        # 仅在数据集中采样负项目
        neg_ids = self._sample_neg_ids(u_ids).to(self.device)  # 获取负样本 ID
        # 对于邻域损失
        if self.neighborhood_loss_required:
            i_ids = cur_data[self.config['ITEM_ID_FIELD']]  # 获取项目 ID
            pos_neighbors, neg_neighbors = self._get_neighborhood_samples(i_ids, self.config['ITEM_ID_FIELD'])  # 获取邻域样本
            pos_neighbors, neg_neighbors = pos_neighbors.to(self.device), neg_neighbors.to(self.device)  # 转换为张量

            batch_tensor = torch.cat((batch_tensor, neg_ids.unsqueeze(0),
                                      pos_neighbors.unsqueeze(0), neg_neighbors.unsqueeze(0)))  # 合并负样本和邻域样本

        # 合并负样本
        else:
            batch_tensor = torch.cat((batch_tensor, neg_ids.unsqueeze(0)))  # 仅合并负样本

        return batch_tensor  # 返回批次张量'''


    def _get_non_neg_sample(self):
        """获取非负样本。"""
        cur_data = self.dataset[self.pr: self.pr + self.step]  # 获取当前批次的数据
        self.pr += self.step  # 更新当前索引
        # 转换为张量
        user_tensor = torch.tensor(cur_data[self.config['USER_ID_FIELD']].values).type(torch.LongTensor).to(self.device)
        item_tensor = torch.tensor(cur_data[self.config['ITEM_ID_FIELD']].values).type(torch.LongTensor).to(self.device)
        batch_tensor = torch.cat((torch.unsqueeze(user_tensor, 0),
                                  torch.unsqueeze(item_tensor, 0)))  # 合并用户和项目张量
        return batch_tensor  # 返回批次张量

    def _get_full_uids_sample(self):
        """获取全用户样本。"""
        user_tensor = torch.tensor(self.all_uids[self.pr: self.pr + self.step]).type(torch.LongTensor).to(self.device)  # 获取用户 ID
        self.pr += self.step  # 更新当前索引
        return user_tensor  # 返回用户张量

    #新的采样方式，为每个用户采样候选负样本10个
    def _sample_neg_ids(self, u_ids, k=5):
        """为每个用户采样 k 个未交互过的负样本 ID"""
        neg_ids = []
        for u in u_ids:
            user_neg = []
            while len(user_neg) < k:
                iid = self._random()
                if iid not in self.history_items_per_u[u]:
                    user_neg.append(iid)
            neg_ids.append(user_neg)
        return torch.tensor(neg_ids).type(torch.LongTensor)#'''



    '''def _sample_neg_ids(self, u_ids):
        """根据用户 ID 采样负项目 ID。"""
        neg_ids = []  # 初始化负样本列表
        for u in u_ids:
            # 随机选择一个项目
            iid = self._random()  # 随机选择项目 ID
            while iid in self.history_items_per_u[u]:  # 如果该项目在用户的历史记录中
                iid = self._random()  # 重新选择
            neg_ids.append(iid)  # 添加负样本
        return torch.tensor(neg_ids).type(torch.LongTensor)  # 返回负样本张量
        #原始采样方式'''




    def _get_my_neighbors(self, id_str):
        """获取用户或项目的邻居。"""
        ret_dict = {}  # 初始化返回字典
        a2b_dict = self.history_items_per_u if id_str == self.config['USER_ID_FIELD'] else self.history_users_per_i  # 根据 ID 字段选择字典
        b2a_dict = self.history_users_per_i if id_str == self.config['USER_ID_FIELD'] else self.history_items_per_u  # 反向字典
        for i, j in a2b_dict.items():
            k = set()  # 初始化邻居集合
            for m in j:
                k |= b2a_dict.get(m, set()).copy()  # 获取邻居
            k.discard(i)  # 移除自身
            ret_dict[i] = k  # 添加到返回字典
        return ret_dict  # 返回邻居字典

    def _get_neighborhood_samples(self, ids, id_str):
        """获取邻域样本。"""
        a2a_dict = self.user_user_dict if id_str == self.config['USER_ID_FIELD'] else self.item_item_dict  # 根据 ID 字段选择字典
        all_set = self.all_users_set if id_str == self.config['USER_ID_FIELD'] else self.all_items_set  # 所有用户或项目集合
        pos_ids, neg_ids = [], []  # 初始化正负样本列表
        for i in ids:
            pos_ids_my = a2a_dict[i]  # 获取正样本
            if len(pos_ids_my) <= 0 or len(pos_ids_my)/len(all_set) > 0.8:  # 如果没有正样本或正样本过多
                pos_ids.append(0)  # 添加占位符
                neg_ids.append(0)  # 添加占位符
                continue
            pos_id = random.sample(pos_ids_my, 1)[0]  # 随机选择一个正样本
            pos_ids.append(pos_id)  # 添加正样本
            neg_id = random.sample(all_set, 1)[0]  # 随机选择一个负样本
            while neg_id in pos_ids_my:  # 如果负样本在正样本中
                neg_id = random.sample(all_set, 1)[0]  # 重新选择
            neg_ids.append(neg_id)  # 添加负样本
        return torch.tensor(pos_ids).type(torch.LongTensor), torch.tensor(neg_ids).type(torch.LongTensor)  # 返回正负样本张量

    def _random(self):
        """随机选择一个项目 ID。"""
        rd_id = random.sample(self.all_items, 1)[0]  # 从所有项目中随机选择
        return rd_id  # 返回随机选择的项目 ID

    def _get_history_items_u(self):
        """获取每个用户的历史项目。"""
        uid_field = self.dataset.uid_field  # 用户 ID 字段
        iid_field = self.dataset.iid_field  # 项目 ID 字段
        # 加载所有用户的可用项目
        uid_freq = self.dataset.df.groupby(uid_field)[iid_field]  # 按用户分组
        for u, u_ls in uid_freq:  # 遍历每个用户及其项目
            self.history_items_per_u[u] = set(u_ls.values)  # 将项目存储为集合
        return self.history_items_per_u  # 返回历史项目字典

    def _get_history_users_i(self):
        """获取每个项目的历史用户。"""
        uid_field = self.dataset.uid_field  # 用户 ID 字段
        iid_field = self.dataset.iid_field  # 项目 ID 字段
        # 加载所有用户的可用项目
        iid_freq = self.dataset.df.groupby(iid_field)[uid_field]  # 按项目分组
        for i, u_ls in iid_freq:  # 遍历每个项目及其用户
            self.history_users_per_i[i] = set(u_ls.values)  # 将用户存储为集合
        return self.history_users_per_i  # 返回历史用户字典

class EvalDataLoader(AbstractDataLoader):
    """
        additional_dataset: training dataset in evaluation
    """
    def __init__(self, config, dataset, additional_dataset=None,
                 batch_size=1, shuffle=False):
        super().__init__(config, dataset, additional_dataset=additional_dataset,
                         batch_size=batch_size, neg_sampling=False, shuffle=shuffle)

        if additional_dataset is None:
            raise ValueError('Training datasets is nan')
        self.eval_items_per_u = []
        self.eval_len_list = []
        self.train_pos_len_list = []

        self.eval_u = self.dataset.df[self.dataset.uid_field].unique()
        # special for eval dataloader
        self.pos_items_per_u = self._get_pos_items_per_u(self.eval_u).to(self.device)
        self._get_eval_items_per_u(self.eval_u)
        # to device
        self.eval_u = torch.tensor(self.eval_u).type(torch.LongTensor).to(self.device)

    @property
    def pr_end(self):
        return self.eval_u.shape[0]

    def _shuffle(self):
        self.dataset.shuffle()

    def _next_batch_data(self):
        inter_cnt = sum(self.train_pos_len_list[self.pr: self.pr+self.step])
        batch_users = self.eval_u[self.pr: self.pr + self.step]
        batch_mask_matrix = self.pos_items_per_u[:, self.inter_pr: self.inter_pr+inter_cnt].clone()
        # user_ids to index
        batch_mask_matrix[0] -= self.pr
        self.inter_pr += inter_cnt
        self.pr += self.step

        return [batch_users, batch_mask_matrix]

    def _get_pos_items_per_u(self, eval_users):
        """
        history items in training dataset.
        masking out positive items in evaluation
        :return:
        user_id - item_ids matrix
        [[0, 0, ... , 1, ...],
         [0, 1, ... , 0, ...]]
        """
        uid_field = self.additional_dataset.uid_field
        iid_field = self.additional_dataset.iid_field
        # load avail items for all uid
        uid_freq = self.additional_dataset.df.groupby(uid_field)[iid_field]
        u_ids = []
        i_ids = []
        for i, u in enumerate(eval_users):
            u_ls = uid_freq.get_group(u).values
            i_len = len(u_ls)
            self.train_pos_len_list.append(i_len)
            u_ids.extend([i]*i_len)
            i_ids.extend(u_ls)
        return torch.tensor([u_ids, i_ids]).type(torch.LongTensor)

    def _get_eval_items_per_u(self, eval_users):
        """
        get evaluated items for each u
        :return:
        """
        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field
        # load avail items for all uid
        uid_freq = self.dataset.df.groupby(uid_field)[iid_field]
        for u in eval_users:
            u_ls = uid_freq.get_group(u).values
            self.eval_len_list.append(len(u_ls))
            self.eval_items_per_u.append(u_ls)
        self.eval_len_list = np.asarray(self.eval_len_list)

    # return pos_items for each u
    def get_eval_items(self):
        return self.eval_items_per_u

    def get_eval_len_list(self):
        return self.eval_len_list

    def get_eval_users(self):
        return self.eval_u.cpu()

