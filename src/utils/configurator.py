# coding: utf-8
# @email: enoche.chow@gmail.com
#
"""
################################
"""

import re
import os
import yaml
import torch
from logging import getLogger


class Config(object):
    """ Configurator module that load the defined parameters.

    Configurator module will first load the default parameters from the fixed properties in RecBole and then
    load parameters from the external input.

    External input supports three kind of forms: config file, command line and parameter dictionaries.

    - config file: It's a file that record the parameters to be modified or added. It should be in ``yaml`` format,
      e.g. a config file is 'example.yaml', the content is:

        learning_rate: 0.001

        train_batch_size: 2048

    - command line: It should be in the format as '---learning_rate=0.001'

    - parameter dictionaries: It should be a dict, where the key is parameter name and the value is parameter value,
      e.g. config_dict = {'learning_rate': 0.001}

    Configuration module allows the above three kind of external input format to be used together,
    the priority order is as following:

    command line > parameter dictionaries > config file

    e.g. If we set learning_rate=0.01 in config file, learning_rate=0.02 in command line,
    learning_rate=0.03 in parameter dictionaries.

    Finally the learning_rate is equal to 0.02.
    """

    def __init__(self, model=None, dataset=None, config_dict=None, mg=False):
        """
        Args:
            model (str/AbstractRecommender): the model name or the model class, default is None, if it is None, config
            will search the parameter 'model' from the external input as the model name or model class.
            dataset (str): the dataset name, default is None, if it is None, config will search the parameter 'dataset'
            from the external input as the dataset name.
            config_file_list (list of str): the external config file, it allows multiple config files, default is None.
            config_dict (dict): the external parameter dictionaries, default is None.
        """
        # 如果没有提供 config_dict，则初始化为空字典
        if config_dict is None:
            config_dict = {}

        # 将模型和数据集名称添加到配置字典中
        config_dict['model'] = model
        config_dict['dataset'] = dataset

        # 得到yaml文件中的内容
        self.final_config_dict = self._load_dataset_model_config(config_dict, mg)
       

        # 更新 final_config_dict，确保包含最新的配置参数
        self.final_config_dict.update(config_dict)

        # 设置默认参数
        self._set_default_parameters()

        # 初始化计算设备（CPU 或 GPU）
        self._init_device()

    def _load_dataset_model_config(self, config_dict, mg):  #得到超参数
        file_config_dict = dict()  # 初始化一个空字典，用于存储文件配置
        file_list = []  # 初始化一个空列表，用于存储配置文件路径

        # 获取当前工作目录
        cur_dir = os.getcwd()
        #print(f"111111111111111111111111111111111111111111111111111111111:{cur_dir}")
        # 将当前目录与 'configs' 目录连接
        cur_dir = os.path.join(cur_dir,'src', 'configs')
        #E:\project\my_rec\.venv\src\configs
        # 添加整体配置文件路径
        file_list.append(os.path.join(cur_dir, "overall.yaml"))
        #E:\project\my_rec\.venv\src\configs\overall.yaml
        # 添加数据集配置文件路径
        file_list.append(os.path.join(cur_dir, "dataset", "{}.yaml".format(config_dict['dataset'])))
        #E:\project\my_rec\.venv\src\configs\dataset\baby.yaml
        # 添加模型配置文件路径
        file_list.append(os.path.join(cur_dir, "model", "{}.yaml".format(config_dict['model'])))
        #E:\project\my_rec\.venv\src\configs\model\MMGCN.yaml

        # 如果 mg 为 True，则添加多图配置文件路径
        if mg:
            file_list.append(os.path.join(cur_dir, "mg.yaml"))

        hyper_parameters = []  # 初始化一个空列表，用于存储超参数
        # 遍历文件列表，读取每个配置文件
        for file in file_list:
            if os.path.isfile(file):  # 检查文件是否存在
                with open(file, 'r', encoding='utf-8') as f:  # 打开文件
                    # 使用自定义的 YAML 加载器读取文件内容
                    fdata = yaml.load(f.read(), Loader=self._build_yaml_loader())
                    # 如果文件中有超参数，则添加到 hyper_parameters 列表中
                    if fdata.get('hyper_parameters'):
                        hyper_parameters.extend(fdata['hyper_parameters'])
                    # 更新文件配置字典
                    file_config_dict.update(fdata)

        # 将超参数列表添加到文件配置字典中
        file_config_dict['hyper_parameters'] = hyper_parameters   
        return file_config_dict  # 返回文件配置字典

    def _build_yaml_loader(self):#读取yaml文件的方法
        loader = yaml.FullLoader  # 使用完整的 YAML 加载器
        # 为浮点数类型添加隐式解析器
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))  # 定义浮点数的合法字符
        return loader  # 返回构建的 YAML 加载器

    def _set_default_parameters(self):
        smaller_metric = ['rmse', 'mae', 'logloss']  # 定义较小指标的列表
        valid_metric = self.final_config_dict['valid_metric'].split('@')[0]  # 获取有效指标
        # 根据有效指标是否在较小指标列表中设置 valid_metric_bigger
        self.final_config_dict['valid_metric_bigger'] = False if valid_metric in smaller_metric else True

        # 如果超参数中没有 'seed'，则将其添加到超参数列表中
        if "seed" not in self.final_config_dict['hyper_parameters']:
            self.final_config_dict['hyper_parameters'] += ['seed']

    def _init_device(self):
        use_gpu = self.final_config_dict['use_gpu']  # 获取是否使用 GPU 的标志
        if use_gpu:  # 如果使用 GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.final_config_dict['gpu_id'])  # 设置可见 GPU
        # 根据 GPU 可用性和配置设置计算设备（CUDA 或 CPU）
        self.final_config_dict['device'] = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    def __setitem__(self, key, value):
        # 检查键是否为字符串类型
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config_dict[key] = value  # 设置配置项

    def __getitem__(self, item):
        # 如果项在 final_config_dict 中，则返回对应的值
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        else:
            return None  # 如果项不存在，则返回 None

    def __contains__(self, key):
        # 检查键是否为字符串类型
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config_dict  # 返回键是否在配置字典中

    def __str__(self):
        args_info = '\n'  # 初始化字符串
        # 将配置字典中的每个项格式化为字符串并连接
        args_info += '\n'.join(["{}={}".format(arg, value) for arg, value in self.final_config_dict.items()])
        args_info += '\n\n'  # 添加换行
        return args_info  # 返回字符串表示

    def __repr__(self):
        return self.__str__()  # 返回对象的字符串表示，通常用于调试