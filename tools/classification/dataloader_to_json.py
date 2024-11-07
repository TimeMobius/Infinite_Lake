import jieba.analyse as ana
import re
import json
import jieba
import pandas as pd

def make_data_json(df, outpath):
    def stop_words(path):
        txt = open(path, "r", encoding='utf-8')
        lines = txt.readlines()
        txt.close()
        stop_txt = []
        for line in lines:
            stop_txt.append(line.strip('\n'))
        return stop_txt

    with open(outpath, "w+", encoding='utf-8') as f:

        # with open(output_path, "w") as fw:
        for indexs in df.index:
            dict1 = {}
            dict1['doc_label'] = [str(df.loc[indexs].values[1])]
            doc_token = df.loc[indexs].values[2]
            # 只保留中文、大小写字母和阿拉伯数字
            reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
            doc_token = re.sub(reg, '', doc_token)
            #print(doc_token)
            # 中文分词
            seg_list = jieba.cut(doc_token, cut_all=False)
            # $提取关键词，20个：
            #ana.set_stop_words('./人工智能挑战赛-文本分类/停用词列表.txt')
            keyword = ana.extract_tags(doc_token, topK=20, withWeight=False, )  # True表示显示权重
            # 去除停用词
            content = [x for x in seg_list if x not in stop_words('stopwords.txt')]
            dict1['doc_token'] = content
            dict1['doc_keyword'] = []
            dict1['doc_topic'] = [str(df.loc[indexs].values[0])]

            # 组合成字典
            print(dict1)
            # 将字典转化成字符串
            json_str = json.dumps(dict1, ensure_ascii=False)
            f.write('%s\n' % json_str)

file_path = r"path\train_convert0304.csv"

df = pd.read_csv(file_path)

train_df = df[0:400]
test_df = df[400:450]
val_df = df[450:]

train_outpath = r'path\chinese_news_train.json'
test_outpath = r'path\chinese_news_test.json'
val_outpath = r'path\chinese_news_val.json'

make_data_json(train_df, train_outpath)
make_data_json(test_df, test_outpath)
make_data_json(val_df, val_outpath)