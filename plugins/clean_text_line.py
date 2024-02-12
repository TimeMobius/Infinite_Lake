# clean_text_line.py
import re
import unicodedata


# 定义一个函数，接受一个字符串作为参数，返回一个经过Unicode Normalization的字符串
def normalize_string(string):
    # 使用unicodedata.normalize函数，指定兼容等价形式，将字符串中的组合字符替换为单个字符
    normalized_string = unicodedata.normalize('NFKC', string)
    # 返回处理后的字符串
    return normalized_string


def clean_text_line(data):
    
    data = normalize_string(data)
    data = re.sub(r'\r', "\n", data)
    data = re.sub(r'\n\n', "\n", data)
    data = re.sub(r"\n\s*\n", "\n", data)
    data = re.sub(r'\n+', '\n', data)
    data = re.sub(r'\s+$', '', data)
    
    return data