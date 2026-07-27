# coding: utf-8
# @email: enoche.chow@gmail.com

r"""
################################
"""

import os
import itertools
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import matplotlib.pyplot as plt

from time import time
from logging import getLogger

from utils.utils import get_local_time, early_stopping, dict2str
from utils.topk_evaluator import TopKEvaluator


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
   and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model, mg=False,sparsity_groups=False):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.weight_decay = 0.0
        if config['weight_decay'] is not None:
            wd = config['weight_decay']
            self.weight_decay = eval(wd) if isinstance(wd, str) else wd

        self.req_training = config['req_training']

        self.start_epoch = 0
        self.cur_step = 0

        tmp_dd = {}
        for j, k in list(itertools.product(config['metrics'], config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()

        #fac = lambda epoch: 0.96 ** (epoch / 50)
        lr_scheduler = config['learning_rate_scheduler']        # check zero?
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

        self.eval_type = config['eval_type']
        self.evaluator = TopKEvaluator(config)
        self.sparsity_groups = sparsity_groups
        self.evaluator1=TopKEvaluator(config, sparsity_groups=self.sparsity_groups)

        self.item_tensor = None
        self.tot_item_num = None
        self.mg = mg
        self.alpha1 = config['alpha1']
        self.alpha2 = config['alpha2']
        self.beta = config['beta']

    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        # 如果不需要训练，直接返回 0.0 和空列表
        if not self.req_training:
            return 0.0, []

        # 将模型设置为训练模式，启用 dropout 和 batch normalization
        self.model.train()

        # 如果没有提供损失函数，则使用模型的默认损失函数
        loss_func = loss_func or self.model.calculate_loss

        # 初始化总损失和每个批次的损失列表
        total_loss = None
        loss_batches = []

        # 遍历训练数据的每个批次
        for batch_idx, interaction in enumerate(train_data):
            # 清零优化器的梯度，以避免累积
            self.optimizer.zero_grad()

            # 克隆当前批次的交互数据，以便后续使用
            second_inter = interaction

            # 计算当前批次的损失
            losses = loss_func(interaction)

            # 如果损失是一个元组，表示有多个部分的损失
            if isinstance(losses, tuple):
                # 计算总损失
                loss = sum(losses)
                # 将每个部分的损失转换为标量
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                # 更新总损失
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                # 如果损失是单一值，直接赋值
                loss = losses
                # 更新总损失
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()

            # 检查损失是否为 NaN
            if self._check_nan(loss):
                # 记录日志并退出训练
                self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                return loss, torch.tensor(0.0)

            # 如果使用多梯度更新且当前批次满足条件
            if self.mg and batch_idx % self.beta == 0:
                # 计算第一次损失并进行反向传播
                first_loss = self.alpha1 * loss
                first_loss.backward()

                # 更新模型参数
                self.optimizer.step()
                # 清零梯度
                self.optimizer.zero_grad()

                # 使用克隆的交互数据计算第二次损失
                losses = loss_func(second_inter)
                if isinstance(losses, tuple):
                    loss = sum(losses)  # 计算总损失
                else:
                    loss = losses  # 直接赋值

                # 检查第二次损失是否为 NaN
                if self._check_nan(loss):
                    # 记录日志并退出训练
                    self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                    return loss, torch.tensor(0.0)

                # 计算第二次损失并进行反向传播
                second_loss = -1 * self.alpha2 * loss
                second_loss.backward()
            else:
                # 对当前损失进行反向传播
                loss.backward()

            # 如果设置了梯度裁剪，则对模型参数的梯度进行裁剪
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)

            # 更新模型参数
            self.optimizer.step()

            # 将当前批次的损失添加到损失列表中
            loss_batches.append(loss.detach())

            # for test
            # if batch_idx == 0:
            #     break  # 仅用于测试，限制为一个批次

        # 返回总损失和每个批次的损失
        return total_loss, loss_batches

    def _valid_epoch(self, valid_data):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data)
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
        return valid_score, valid_result

    def _valid_epoch1(self, valid_data):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate1(valid_data)
        #valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
        return  valid_result

    def _check_nan(self, loss):
        if torch.isnan(loss):
            #raise ValueError('Training loss is nan')
            return True

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            test_data (DataLoader, optional): None
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            if torch.is_tensor(train_loss):
                # get nan loss
                break
            #for param_group in self.optimizer.param_groups:
            #    print('======lr: ', param_group['lr'])
            self.lr_scheduler.step()

            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)

            # eval: To ensure the test result is the best model under validation data, set self.eval_step == 1
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)
                # test
                test_result = self._valid_epoch1(test_data)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                    self.logger.info('test result: \n' + str(test_result))
                if update_flag:
                    update_output = '██ ' + self.config['model'] + '--Best validation results updated!!!'
                    if verbose:
                        self.logger.info(update_output)
                    self.best_valid_result = valid_result
                    self.best_test_upon_valid = test_result

                if stop_flag:
                    stop_output = '+++++Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        with torch.no_grad():
            print("1111111111111111111111111111111111111111111")
            # 重新跑一次 GCN 拿到最新权重的特征
            v_rep, _ = self.model.v_gcn(self.model.edge_index, self.model.v_feat)
            t_rep, _ = self.model.t_gcn(self.model.edge_index, self.model.t_feat)
            id_rep, _ = self.model.id_gcn(self.model.edge_index, self.model.id_feat)  # 【新增】跑一次 ID 的 GCN

            # 截取纯 Item 侧的特征并转为 numpy
            v_item_feats = v_rep[self.model.num_user:].cpu().numpy()
            t_item_feats = t_rep[self.model.num_user:].cpu().numpy()
            id_item_feats = id_rep[self.model.num_user:].cpu().numpy()  # 【新增】截取 ID 的 Item 侧特征

            # 判断当前是跑“对齐前”还是“对齐后”的实验，并分别保存
            # 通过判断对齐损失权重 (cl_loss) 是否大于 0 来自动命名文件
            if self.model.cl_loss > 0:
                np.save('v_feat_post_align.npy', v_item_feats)
                np.save('t_feat_post_align.npy', t_item_feats)
                np.save('id_feat_post_align.npy', id_item_feats)  # 【新增】保存对齐后的 ID 特征
                self.logger.info("已成功保存【对齐后 (w/ SVD)】的 V, T, ID 特征用于 t-SNE")
            else:
                np.save('v_feat_pre_align.npy', v_item_feats)
                np.save('t_feat_pre_align.npy', t_item_feats)
                np.save('id_feat_pre_align.npy', id_item_feats)  # 【新增】保存对齐前的 ID 特征
                self.logger.info("已成功保存【对齐前 (w/o SVD)】的 V, T, ID 特征用于 t-SNE")
        # =========================================================
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid



    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        r"""Evaluate the model based on the eval data.
        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        self.model.eval()

        # batch full users
        batch_matrix_list = []
        for batch_idx, batched_data in enumerate(eval_data):
            # predict: interaction without item ids
            scores = self.model.full_sort_predict(batched_data)
            masked_items = batched_data[1]
            # mask out pos items
            scores[masked_items[0], masked_items[1]] = -1e10
            # rank and get top-k
            _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)  # nusers x topk
            batch_matrix_list.append(topk_index)
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)
    #分组
    def evaluate1(self, eval_data, is_test=False, idx=0):
        r"""Evaluate the model based on the eval data.
        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        self.model.eval()

        # batch full users
        batch_matrix_list = []
        for batch_idx, batched_data in enumerate(eval_data):
            # predict: interaction without item ids
            scores = self.model.full_sort_predict(batched_data)
            masked_items = batched_data[1]
            # mask out pos items
            scores[masked_items[0], masked_items[1]] = -1e10
            # rank and get top-k
            _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)  # nusers x topk
            batch_matrix_list.append(topk_index)
        return self.evaluator1.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)

    def plot_train_loss(self, show=True, save_path=None):
        r"""Plot the train loss in each epoch

        Args:
            show (bool, optional): whether to show this figure, default: True
            save_path (str, optional): the data path to save the figure, default: None.
                                       If it's None, it will not be saved.
        """
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)

