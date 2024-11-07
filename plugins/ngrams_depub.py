#检测连续5个汉字或单词的重复率
def split_text(text):
    """将文本分割成单词或单个汉字/字母的序列"""
    # 如果文本包含空格，则按空格分割
    if ' ' in text:
        return text.split()
    else:
        # 否则逐个字符分割
        return list(text)

def generate_ngrams(sequence, n):
    """生成n-gram序列"""
    ngrams = [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]
    #print(ngrams)
    return ngrams

def check_repeats(text, n):
    """检查n-gram序列的重复率"""
    from collections import defaultdict
    # 分割文本
    sequence = split_text(text)
    ngrams = generate_ngrams(sequence, n)
    if not ngrams:
        return 0.0

    # 使用字典统计n-gram的出现次数
    ngram_counts = defaultdict(int)
    for ngram in ngrams:
        ngram_counts[ngram] += 1

    # 计算重复率
    total_ngrams = len(ngrams)
    repeated_ngrams = sum(count > 1 for count in ngram_counts.values())
    repeat_rate = repeated_ngrams / total_ngrams

    return repeat_rate

# 示例文本
#text = "In addition, the pursuit of quality and diversity tends to trade off with data volume, In addition, the pursuit of quality and diversity tends to trade off with data volume。"

# 分割文本
#sequence = split_text(text)
#print(sequence)

# 检查重复率
#n = 5  # 设定n-gram的大小
#repeat_rate = check_repeats(text)

#print(f"5-gram 的重复率: {repeat_rate:.2}")