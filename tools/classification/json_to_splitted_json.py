import json
import jieba
import re
import jieba.analyse as ana
import random
from multiprocessing import Pool

def stop_words(path):
    with open(path, "r", encoding='utf-8') as txt:
        lines = txt.readlines()
    stop_txt = [line.strip('\n') for line in lines]
    return stop_txt

def process_data(data, stopwords):
    # Tokenize the "sentence" using jieba
    seg_list = jieba.cut(data['sentence'], cut_all=False)
    content = [x for x in seg_list if x not in stopwords]
    # 去除content中的空格，保留2个字以上的，只保留有中文
    content = [re.sub(r'\s+', '', word) for word in content if len(word) >= 2 and re.search('[\u4e00-\u9fff]+', word)]
    # 转换为新格式
    new_data = {
        'doc_topic': [data['label']],
        'doc_label': [data['label-des']],
        'doc_token': content,
        'doc_keyword': [''],
    }
    #print(new_data)
    return new_data


def split_data(data, train_ratio, val_ratio):
    random.shuffle(data)
    total_samples = len(data)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    return train_data, val_data, test_data

def process_json(input_file, train_file, val_file, test_file):
    stopwords = stop_words("stopwords.txt")
    processed_data = []

    # Read each line of the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            processed_data.append(data)

    # Process each data item
    processed_data = [process_data(data, stopwords) for data in processed_data]

    # Split the data into train, val, and test sets
    train_data, val_data, test_data = split_data(processed_data, 0.8, 0.1)

    # Write train data to a new JSON file
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

    # Write validation data to a new JSON file
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

    # Write test data to a new JSON file
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
