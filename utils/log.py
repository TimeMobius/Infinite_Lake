#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   log.py
@Contact :   

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------

"""

# Start dance your fingers
import os
import time
from loguru import logger

# 日志的路径
log_path = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(log_path):
    os.mkdir(log_path)

# 日志输出的文件格式
log_path_error = os.path.join(log_path, f'error_{time.strftime("%Y-%m-%d")}.log')

logger.add(log_path_error, rotation="12:00", retention="5 days", enqueue=True,
           format="{time:YYYY-MM-DD HH:mm:ss} {level} From {module}.{function} : {message}")
