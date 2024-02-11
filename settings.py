#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   settings.py
@Contact :   

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------

"""

# Start dance your fingers

import os
import urllib.parse
import datetime
import yaml

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:

    ALLOWED_IPS = []    # IP白名单

class Settings:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self._parse_config()

    def _load_config(self, path):
        """加载YAML配置文件"""
        with open(path, 'r') as file:
            return yaml.safe_load(file)
    
    def _parse_config(self):
        """将配置项解析为属性"""
        for key, value in self.config.items():
            setattr(self, key, self._to_object(value))

    def _to_object(self, value):
        """将字典转换为对象"""
        if isinstance(value, dict):
            return type('ConfigObject', (object,), {k: self._to_object(v) for k, v in value.items()})
        return value