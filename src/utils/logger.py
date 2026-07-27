# coding: utf-8
# @email: enoche.chow@gmail.com

"""
###############################
"""

import logging
import os
from utils.utils import get_local_time


def init_logger(config):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
    """
    LOGROOT = './log/'  # 定义日志文件的根目录
    dir_name = os.path.dirname(LOGROOT)  # 获取日志目录的名称
    if not os.path.exists(dir_name):  # 检查日志目录是否存在
        os.makedirs(dir_name)  # 如果不存在，则创建日志目录

    # 根据模型名称、数据集名称和当前时间生成日志文件名
    logfilename = '{}-{}-{}.log'.format(config['model'], config['dataset'], get_local_time())

    logfilepath = os.path.join(LOGROOT, logfilename)  # 生成完整的日志文件路径

    # 定义文件日志格式
    filefmt = "%(asctime)-15s %(levelname)s %(message)s"  # 日志格式，包括时间、级别和消息
    filedatefmt = "%a %d %b %Y %H:%M:%S"  # 文件日志的日期格式
    fileformatter = logging.Formatter(filefmt, filedatefmt)  # 创建文件日志格式化器

    # 定义控制台日志格式
    sfmt = u"%(asctime)-15s %(levelname)s %(message)s"  # 控制台日志格式
    sdatefmt = "%d %b %H:%M"  # 控制台日志的日期格式
    sformatter = logging.Formatter(sfmt, sdatefmt)  # 创建控制台日志格式化器

    # 根据配置中的状态设置日志级别
    if config['state'] is None or config['state'].lower() == 'info':
        level = logging.INFO  # 默认日志级别为 INFO
    elif config['state'].lower() == 'debug':
        level = logging.DEBUG  # 如果状态为 debug，则设置为 DEBUG 级别
    elif config['state'].lower() == 'error':
        level = logging.ERROR  # 如果状态为 error，则设置为 ERROR 级别
    elif config['state'].lower() == 'warning':
        level = logging.WARNING  # 如果状态为 warning，则设置为 WARNING 级别
    elif config['state'].lower() == 'critical':
        level = logging.CRITICAL  # 如果状态为 critical，则设置为 CRITICAL 级别
    else:
        level = logging.INFO  # 其他情况默认设置为 INFO 级别

    # 创建文件处理器，负责将日志写入文件
    fh = logging.FileHandler(logfilepath, 'w', 'utf-8')  # 创建文件处理器，使用 UTF-8 编码
    fh.setLevel(level)  # 设置文件处理器的日志级别
    fh.setFormatter(fileformatter)  # 设置文件处理器的格式化器

    # 创建控制台处理器，负责将日志输出到标准输出
    sh = logging.StreamHandler()  # 创建控制台处理器
    sh.setLevel(level)  # 设置控制台处理器的日志级别
    sh.setFormatter(sformatter)  # 设置控制台处理器的格式化器

    # 配置日志系统的基本设置
    logging.basicConfig(
        level=level,  # 设置日志级别
        #handlers=[sh]  # 仅使用控制台处理器
        handlers=[sh, fh]  # 同时使用控制台和文件处理器
    )


