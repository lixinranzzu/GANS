from logging import getLogger  # 从 logging 模块导入 getLogger 函数，用于获取日志记录器
from itertools import product  # 从 itertools 模块导入 product 函数，用于生成参数组合

from utils.dataset import RecDataset# 从 dataset 模块导入 RecDataset 类，用于加载和处理数据集
from utils.dataloader import TrainDataLoader, EvalDataLoader  # 从 dataloader 模块导入训练和评估数据加载器
from utils.logger import init_logger  # 从 logger 模块导入 init_logger 函数，用于初始化日志记录器
from utils.configurator import Config  # 从 configurator 模块导入 Config 类，用于管理配置
from utils.utils import init_seed, get_model, get_trainer, dict2str  # 从 utils 模块导入各种实用函数
import platform  # 导入 platform 模块，用于获取系统信息
import os  # 导入 os 模块，用于处理文件和目录路径

def quick_start(model, dataset, config_dict, save_model=True, mg=False):
    # merge config dict
    config = Config(model, dataset, config_dict, mg)

    # logger
    local_time = init_logger(config)
    logger = getLogger()

    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # ======== Load dataset only once (NO repeated I/O) ========
    dataset = RecDataset(config)
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()

    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # ======== Hyper-parameter preparation ========
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    # Ensure seed exists
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']

    # build hyper-parameter search space
    hyper_ls = []
    for hp in config['hyper_parameters']:
        hyper_ls.append(config[hp] if isinstance(config[hp], list) else [config[hp]])

    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)

    # ========== Start hyper-parameter search ==========
    for hyper_tuple in combinators:
        # assign hyper-parameter values
        for key, val in zip(config['hyper_parameters'], hyper_tuple):
            config[key] = val

        # reset seed
        init_seed(config['seed'])

        logger.info(
            '========={}/{}: Parameters:{}={}======='.format(
                idx + 1, total_loops, config['hyper_parameters'], hyper_tuple
            )
        )

        # ========== Re-create DataLoaders so S_nn works ==========
        train_data = TrainDataLoader(
            config,
            train_dataset,
            batch_size=config['train_batch_size'],
            shuffle=True
        )

        valid_data = EvalDataLoader(
            config,
            valid_dataset,
            additional_dataset=train_dataset,
            batch_size=config['eval_batch_size']
        )

        test_data = EvalDataLoader(
            config,
            test_dataset,
            additional_dataset=train_dataset,
            batch_size=config['eval_batch_size']
        )

        train_data.pretrain_setup()

        # ========== Model initialization ==========
        model = get_model(config['model'])(config, train_data).to(config['device'])
        logger.info(model)

        # ========== Trainer ==========
        trainer = get_trainer()(config, model, mg)

        # run training
        best_valid_score, best_valid_result, best_test_upon_valid,best_test_result = trainer.fit(
            train_data, valid_data=valid_data, test_data=test_data, saved=save_model
        )

        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid, best_test_result))

        # update best test
        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx

        idx += 1

        # Logging
        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        logger.info(
            '████Current BEST████:\nParameters: {}={},\nValid: {},\nTest: {}\n\n\n'.format(
                config['hyper_parameters'],
                hyper_ret[best_test_idx][0],
                dict2str(hyper_ret[best_test_idx][1]),
                dict2str(hyper_ret[best_test_idx][2])
            )
        )

    # ======= Final Summary =======
    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(
            config['hyper_parameters'], p, dict2str(k), dict2str(v)
        ))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info(
        '\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(
            config['hyper_parameters'],
            hyper_ret[best_test_idx][0],
            dict2str(hyper_ret[best_test_idx][1]),
            dict2str(hyper_ret[best_test_idx][2])
        )
    )