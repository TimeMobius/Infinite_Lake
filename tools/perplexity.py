# -*- coding: utf-8 -*-
from typing import Union
from sentencepiece import SentencePieceProcessor
from models.model import KenlmModel
import kenlm
import re
from config_plugins import KENLM_CONFIG


class PerplexityFilter():
    """过滤器，保留困惑度得分低于特定最大值的样本。"""

    def __init__(self,
                 lang: str = 'zh',
                 max_ppl: Union[int, float] = 2500):
        """
        初始化方法。

        :param lang: 计算哪种语言样本的困惑度。
        :param max_ppl: 此操作中的最大过滤困惑度，如果样本的困惑度超过此参数，则将被过滤。
        """
        self.max_ppl = max_ppl
        self.lang = lang


    def compute_stats(self, text: str):
        model = KenlmModel.from_pretrained(KENLM_CONFIG['zh_model_dir'], KENLM_CONFIG['language'])
        ppl = model.get_perplexity(text)
        #print(ppl)
        return round(ppl, 1)


    def process(self, text: str):
        ppl = self.compute_stats(text)
        if ppl <= self.max_ppl:
            return text


