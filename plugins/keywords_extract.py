import jieba
import jieba.analyse
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from config_plugins import STOPWORDS_PATH, KEYWORDS_CONFIG


def keywords_extract(text, language):
    if language == "zh":
        # 设置停用词路径
        jieba.analyse.set_stop_words(STOPWORDS_PATH)

        # 提取关键词，使用 config.py 中的配置参数
        keywords = jieba.analyse.extract_tags(
            text,
            topK=KEYWORDS_CONFIG['topK'],
            withWeight=KEYWORDS_CONFIG['withWeight'],
            allowPOS=KEYWORDS_CONFIG['allowPOS']
        )
        # 将关键词列表转换为字符串
        keywords_string = ', '.join(keywords)
        #print(keywords_string)
    else:
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
        freq_dist = nltk.FreqDist(filtered_words)
        keywords = [word for (word, freq) in freq_dist.most_common(5)]
        # print(keywords)
        keywords_string = ', '.join([kw for kw in keywords])
        # print(keywords_string)
    return keywords_string

#text = """
       #Extracting keywords from texts has become a challenge for individuals and organizations as the information grows in complexity and size. The need to automate this task so that texts can be processed in a timely and adequate manner has led to the emergence of automatic keyword extraction tools.
       #"""

#language = "en"
#keywords_extract(text, language)




