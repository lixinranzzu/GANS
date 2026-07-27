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

    def __init__(self, config, model):
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

        self.item_tensor = None
        self.tot_item_num = None
        self.split_user= None

        self.use_group=False

        split_uids_path = '/home/server/workspace/lixinran/my_rec/data/baby/baby_split_uids.npy'
        self.split_user = np.load(split_uids_path, allow_pickle=True)

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

        # ===== 1. reset only once =====
        if epoch_idx == 0 and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        if not self.req_training:
            return 0.0, []

        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        loss_batches = []

        for batch_idx, interaction in enumerate(train_data):
            self.optimizer.zero_grad()
            losses = loss_func(interaction)

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()

            if self._check_nan(loss):
                self.logger.info(
                    f'Loss is nan at epoch: {epoch_idx}, batch index: {batch_idx}. Exiting.'
                )
                return loss, torch.tensor(0.0)

            loss.backward()

            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)

            self.optimizer.step()
            loss_batches.append(loss.detach())

        # ===== 2. record peak AFTER epoch =====
        if epoch_idx == 0 and torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            self.logger.info(
                f'[Efficiency] Peak GPU Memory (Training): {peak_mem:.2f} GB'
            )

        return total_loss, loss_batches

    def _valid_epoch(self, valid_data, use_split=False, split_user=None):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """
        if use_split:
            valid_result,recall_group_dict, ndcg_group_dict = self.evaluate(valid_data, use_split=use_split, split_user=split_user)
            valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
            return valid_score, valid_result, recall_group_dict, ndcg_group_dict
        else:
            valid_result = self.evaluate(valid_data, use_split=use_split)
            valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
            return valid_score, valid_result




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

    def fit1(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
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
                if self.use_group:

                    _, test_result, recall_group_dict, ndcg_group_dict= self._valid_epoch(test_data, use_split=True, split_user=self.split_user)
                else:
                    _, test_result = self._valid_epoch(test_data, use_split=False)


                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                    self.logger.info('test result: \n' + dict2str(test_result))
                    if self.use_group:
                        self.logger.info('gorup result Recall@20: \n' + dict2str(recall_group_dict))
                        self.logger.info('gorup result NDCG@20: \n' + dict2str(ndcg_group_dict))

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
        self.model.eval()  # 确保模型处于评估模式
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

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        """
        Train the model based on train_data and valid_data.
        Early stopping is still based on validation data.
        Logs and prints remain unchanged.

        Internal tracking:
            best_test_result_during_training -> 训练期间测试集表现最优模型
        """
        # 内部记录训练期间测试集最优
        best_test_score = -float('inf') if self.valid_metric_bigger else float('inf')
        best_test_result_during_training = None

        for epoch_idx in range(self.start_epoch, self.epochs):
            # ===== 训练 =====
            training_start_time = time()
            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            if torch.is_tensor(train_loss):
                break
            self.lr_scheduler.step()

            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time,
                                                                 train_loss)
            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)

            # ===== 验证 =====
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_score, valid_result = self._valid_epoch(valid_data)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)

                # ===== 测试 =====
                test_result = None
                if test_data is not None:
                    _, test_result = self._valid_epoch(test_data)
                    test_score = test_result[self.valid_metric] if self.valid_metric else test_result['NDCG@20']

                    # 内部记录训练期间测试集最优
                    if (self.valid_metric_bigger and test_score > best_test_score) or \
                            (not self.valid_metric_bigger and test_score < best_test_score):
                        best_test_score = test_score
                        best_test_result_during_training = test_result

                # ===== 更新最优验证模型及对应测试集结果 =====
                if update_flag:
                    self.best_valid_result = valid_result
                    self.best_test_upon_valid = test_result  # 保留原来的逻辑
                    if verbose:
                        self.logger.info('██ Best validation model updated! Corresponding test results recorded.')

                # ===== 原来的日志打印保持不变 =====
                if verbose:
                    valid_result_output = 'valid result: \n' + dict2str(valid_result)
                    valid_score_output = f"epoch {epoch_idx} evaluating [time: {time() - training_end_time:.2f}s, valid_score: {valid_score}]"
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                    if test_result is not None:
                        self.logger.info('test result: \n' + dict2str(test_result))

                if stop_flag:
                    if verbose:
                        self.logger.info(
                            f'+++++ Finished training, best eval result in epoch {epoch_idx - self.cur_step * self.eval_step}')
                    break
        '''with torch.no_grad():
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
        # ========================================================='''


        # 返回值保持原来的结构，不改变打印逻辑
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid, best_test_result_during_training


    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0, use_split=False, split_user=None):
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
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx, use_split= use_split, split_user=split_user)

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