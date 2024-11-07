# clean_text_line.py
import re
import string
import unicodedata
import chardet
from tools.clean_copyright_mapper import CleanCopyrightMapper
from tools.remove_sentence import remove_repeat_sentences
from tools.whitespace_normalization_mapper import normalize_whitespace
from tools.special_characters_filter import SpecialCharactersFilter
from tools.perplexity import PerplexityFilter



# 定义一个函数，接受一个字符串作为参数，返回一个经过Unicode Normalization的字符串
def normalize_string(string):
    # 使用unicodedata.normalize函数，指定兼容等价形式，将字符串中的组合字符替换为单个字符
    normalized_string = unicodedata.normalize('NFKC', string)
    # 返回处理后的字符串
    return normalized_string


def check_asc_text(text):
    # 计算文本中乱码字符的数量和总字符数
    if text != None:
        gibberish_count = 0
        total_chars = len(text)
        # 对文本中的每个字符进行遍历
        for char in text:
            # 判断字符是否为 ASCII 范围内的可打印字符或空格
            if ord(char) >= 32 and ord(char) <= 126:
                continue
            else:
                # 如果字符不是ASCII可打印字符，则判断它是否为控制字符
                category = unicodedata.category(char)
                if category == 'Cc':
                    gibberish_count += 1

        # 计算乱码字符占整个文本的比例
        gibberish_ratio = gibberish_count / total_chars

        # 根据乱码字符的比例判断文本是否包含过多的乱码字符
        if gibberish_ratio > 0.03:  # 如果乱码字符比例超过3%
            return False
        else:
            return True

# 乱码字符和非标准的Unicode字符
def remove_garbled_characters(text):
    # 定义正则表达式，匹配非标准的Unicode字符和乱码
    garbled_pattern = re.compile(r'[^\u0020-\u007E\u4e00-\u9fa5]+')

    # 使用正则表达式查找并替换这些字符，替换为空字符串即删除
    return garbled_pattern.sub('', text)

# 删除url
def remove_urls(text):
    # 定义URL的正则表达式
    # 匹配以http://或https://开头，后面跟着任意非空格字符的字符串
    url_pattern = re.compile(r'https?://[^\s]+')

    # 使用正则表达式查找并替换URL，替换为空字符串即删除
    return url_pattern.sub('', text)


# 直接去除，包括较短文本去除，乱码占比较大的文本去除
def direct_drop_line(data):
    if data != None:
        # 乱码检查
        actionCode = check_asc_text(data)
        #print(actionCode)
        if actionCode:
            # 删除过短文本
            if len(data) >= 50:
                return True
            else:
                return False
        else:
            return False

# common处理
def clean_text_line(data):
    # 检查数据是否为字符串
    if not isinstance(data, str):
        return data

    data = normalize_string(data)
    # 替换符号
    data = re.sub(r"\「|\」|\｢|\｣|\『|\』", '\"', data)
    data = re.sub(r'\r', "\n", data)
    data = re.sub(r'\n\n', "\n", data)
    data = re.sub(r"\n\s*\n", "\n", data)
    data = re.sub(r'\n+', '\n', data)
    data = re.sub(r'\s+$', '', data)
    data = re.sub(r'\s+', ' ', data)

    # 使用精准匹配，匹配连续出现的符号，并用空字符替换它
    data = re.sub(r'(\W)\1+', r'\1', data)

    # 只替换汉字后面的空格
    data = re.sub(r'(?<=[\u4e00-\u9fff])\s+', '', data)

    return data



# sft common 处理

def clean_text_line_sft(data):
    # 检查数据是否为字符串
    if not isinstance(data, str):
        return data

    data = normalize_string(data)
    # 替换符号
    data = re.sub(r"\「|\」|\｢|\｣|\『|\』", '\"', data)
    data = re.sub(r'\r', "\n", data)
    data = re.sub(r'\n\n', "", data)
    data = re.sub(r"\n\s*\n", "\n", data)
    data = re.sub(r'\n+', '\n', data)
    data = re.sub(r'\s+$', '', data)
    data = re.sub(r'\s+', ' ', data)

    # 使用精准匹配，匹配连续出现的符号，并用空字符替换它
    data = re.sub(r'(\W)\1+', r'\1', data)

    # 只替换汉字后面的空格
    data = re.sub(r'(?<=[\u4e00-\u9fff])\s+', '', data)

    return data


# mapper处理，对数据样本进行编辑和转换

def mapper_clean_text(data):
    mapper = CleanCopyrightMapper()
    data1 = mapper.process_text(data)
    data2 = remove_repeat_sentences(data1)
    final_data = normalize_whitespace(data2)
    return final_data

def get_cleaned_line(data):
    removed_url_data = remove_urls(data)
    # print(removed_url_data)
    garbled_data = remove_garbled_characters(removed_url_data)
    if garbled_data != None:
        filter_instance = SpecialCharactersFilter(min_ratio=0.0, max_ratio=0.5)
        processed_sample = filter_instance.compute_stats(garbled_data)
        #print(processed_sample)
        processed_result = filter_instance.process(processed_sample)
        #print(processed_result)
        if processed_result:
            processing_data = clean_text_line(garbled_data)
            cleaned_data = mapper_clean_text(processing_data)
            preplexity = PerplexityFilter()
            preplexity_data = preplexity.process(cleaned_data)
            #print(preplexity_data)
            actionCode = direct_drop_line(preplexity_data)
            if actionCode:
                return preplexity_data

# 论文处理,增加困惑度

def get_cleaned_line_paper(data):
    processing_data = clean_text_line(data)
    cleaned_data = mapper_clean_text(processing_data)
    preplexity = PerplexityFilter()
    preplexity_data = preplexity.process(cleaned_data)
    return preplexity_data

