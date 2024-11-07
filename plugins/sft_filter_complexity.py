import re
import jieba
from collections import Counter


def is_chinese(text):
    """
    判断文本是否包含中文字符。
    """
    return any('\u4e00' <= char <= '\u9fff' for char in text)


def extract_terms(text, chinese):
    """
    提取术语，如果是中文则使用jieba分词，否则提取英文单词。
    """
    if chinese:
        terms = jieba.lcut(text)
        # 过滤掉长度小于2的词语
        terms = [term for term in terms if len(term) >= 2]
    else:
        terms = re.findall(r'\b[a-zA-Z]+\b', text)
    return terms


def calculate_complexity(question, answer):
    w1, w2, w3, w4 = 0.2, 0.2, 0.3, 0.3
    # 判断语言
    chinese = is_chinese(question) or is_chinese(answer)

    if chinese:
        # 中文的字数计算
        question_length = len(question)
        answer_length = len(answer)
    else:
        # 英文的字数计算，单词数量作为字数
        question_length = len(re.findall(r'\b[a-zA-Z]+\b', question))
        answer_length = len(re.findall(r'\b[a-zA-Z]+\b', answer))

    # 计算从句数量（假设句子由句号、问号、感叹号分割）
    sentence_delimiters = r'[.?!。？！]' if chinese else r'[.?!]'
    question_sentences = re.split(sentence_delimiters, question)
    answer_sentences = re.split(sentence_delimiters, answer)
    total_sentences = len(question_sentences) + len(answer_sentences) - 2  # 减去分割带来的空字符串

    # 提取术语
    question_terms = extract_terms(question, chinese)
    answer_terms = extract_terms(answer, chinese)

    # 计算术语数量
    if chinese:
        term_counter = Counter(question_terms + answer_terms)
        total_terms = len(term_counter)

        ## 计算术语复杂度
        term_complexity = total_terms / total_sentences if total_sentences > 0 else 0
    else:
        term_complexity = 0

    ## 计算从句复杂度（句子数量）
    syntactic_complexity = total_sentences

    ## 计算每个句子的平均长度
    avg_question_length = question_length / len(question_sentences) if len(question_sentences) > 0 else 0
    avg_answer_length = answer_length / len(answer_sentences) if len(answer_sentences) > 0 else 0

    ## 计算综合复杂度
    if chinese:
        complexity = (w1 * avg_question_length +
                      w2 * avg_answer_length +
                      w3 * syntactic_complexity +
                      w4 * term_complexity)
    else:
        complexity = (w1 * avg_question_length +
                      w2 * avg_answer_length +
                      w3 * syntactic_complexity)

    # 保留一位小数
    avg_question_length = round(avg_question_length, 1)
    avg_answer_length = round(avg_answer_length, 1)
    syntactic_complexity = round(syntactic_complexity, 1)
    term_complexity = round(term_complexity, 1) if chinese else 'N/A'
    complexity = round(complexity, 1)

    return {
        "Question Length": avg_question_length,
        "Answer Length": avg_answer_length,
        "Syntactic Complexity": syntactic_complexity,
        "Term Complexity": term_complexity if chinese else 'N/A',
        "Overall Complexity": complexity
    }


def calculate_complexity_single(text):
    w1, w2, w3 = 0.3, 0.3, 0.4  # 权重比例，可以根据需要调整
    # 判断语言
    chinese = is_chinese(text)
    # 判断语言，按中文或英文计算字数
    if chinese:
        text_length = len(text)  # 中文字数
    else:
        text_length = len(re.findall(r'\b[a-zA-Z]+\b', text))  # 英文的单词数量

    # 计算从句数量（假设句子由句号、问号、感叹号分割）
    sentence_delimiters = r'[。？！]' if chinese else r'[.?!]'
    sentences = re.split(sentence_delimiters, text)
    total_sentences = len(sentences) - 1  # 减去分割带来的空字符串

    # 提取术语
    terms = extract_terms(text, chinese)

    # 计算术语数量
    if chinese:
        term_counter = Counter(terms)
        total_terms = len(term_counter)

        ## 计算术语复杂度
        term_complexity = total_terms / total_sentences if total_sentences > 0 else 0
    else:
        term_complexity = 0

    ## 计算从句复杂度（句子数量）
    syntactic_complexity = total_sentences

    ## 计算每个句子的平均长度
    avg_sentence_length = text_length / len(sentences) if len(sentences) > 0 else 0

    ## 计算综合复杂度
    if chinese:
        complexity = (w1 * avg_sentence_length +
                      w2 * syntactic_complexity +
                      w3 * term_complexity)
    else:
        complexity = (w1 * avg_sentence_length +
                      w2 * syntactic_complexity)

    # 保留一位小数
    avg_sentence_length = round(avg_sentence_length, 1)
    syntactic_complexity = round(syntactic_complexity, 1)
    term_complexity = round(term_complexity, 1) if chinese else 'N/A'
    complexity = round(complexity, 1)

    #return {
        #"Text Length": avg_sentence_length,
        #"Syntactic Complexity": syntactic_complexity,
        #"Term Complexity": term_complexity if chinese else 'N/A',
        #"Overall Complexity": complexity
    #}
    return complexity


