import jieba
import torch
import numpy as np
import codecs
import json
import re
from models.classification.config import Config
from models.classification.dataset.classification_dataset import ClassificationDataset
from models.classification.dataset.collator import ClassificationCollator
from models.classification.dataset.collator import ClassificationType
from models.classification.dataset.collator import FastTextCollator
from models.classification.model.classification.drnn import DRNN
from models.classification.model.classification.fasttext import FastText
from models.classification.model.classification.textcnn import TextCNN
from models.classification.model.classification.textvdcnn import TextVDCNN
from models.classification.model.classification.textrnn import TextRNN
from models.classification.model.classification.textrcnn import TextRCNN
from models.classification.model.classification.transformer import Transformer
from plugins.language_classified import detect_language
from config_plugins import INDUSTRY_CONFIG_FILES


class Predictor(object):
    def __init__(self, config):
        self.config = config
        self.model_name = config.model_name
        self.use_cuda = config.device.startswith("cuda")
        self.dataset_name = "ClassificationDataset"
        self.collate_name = "FastTextCollator" if self.model_name == "FastText" \
            else "ClassificationCollator"
        self.dataset = globals()[self.dataset_name](config, [], mode="infer")
        self.collate_fn = globals()[self.collate_name](config, len(self.dataset.label_map))
        self.model = Predictor._get_classification_model(self.model_name, self.dataset, config)
        Predictor._load_checkpoint(config.eval.model_dir, self.model, self.use_cuda)
        self.model.eval()

    @staticmethod
    def _get_classification_model(model_name, dataset, conf):
        model = globals()[model_name](dataset, conf)
        model = model.cuda(conf.device) if conf.device.startswith("cuda") else model
        return model

    @staticmethod
    def _load_checkpoint(file_name, model, use_cuda):
        if use_cuda:
            checkpoint = torch.load(file_name)
        else:
            checkpoint = torch.load(file_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["state_dict"])

    def predict(self, text):
        """
        输入文本为单篇字符串
        """
        with torch.no_grad():
            input_text = self.dataset._get_vocab_id_list(json.loads(text))
            #print(input_text)
            input_text = self.collate_fn([input_text])
            logits = self.model(input_text)
            #print(logits)
            if self.config.task_info.label_type != ClassificationType.MULTI_LABEL:
                probs = torch.softmax(logits, dim=1)
            else:
                probs = torch.sigmoid(logits)
            probs = probs.cpu().tolist()
            return np.array(probs)



def tokenize_text(input_data, language):
    if language == "zh":
        # 使用jieba进行中文分词
        seg_list = jieba.cut(input_data, cut_all=False)
        # 去除空格并只保留长度大于等于2且包含中文的词语
        return [re.sub(r'\s+', '', word) for word in seg_list if len(word) >= 2 and re.search('[\u4e00-\u9fff]+', word)]
    else:
        # 对英文文本按空格进行切分
        return input_data.split(" ")

def predict_label(predictor, data_json, label_type, top_k, threshold):
    predict_prob = predictor.predict(data_json)
    if label_type == ClassificationType.MULTI_LABEL:
        predict_label_ids = []
        predict_label_idx = np.argsort(-predict_prob)
        for j in range(0, top_k):
            if predict_prob[predict_label_idx[j]] > threshold:
                predict_label_ids.append(predict_label_idx[j])
        return predict_label_ids
    else:
        return predict_prob.argmax()


def get_industry_subject(input_data, language):
    #config_file = '/mnt/data/ML_folder/Infinite_Lake/models/classification/conf/chinese_industry_train_conf.json' if language == "zh" else '/mnt/data/ML_folder/Infinite_Lake/models/classification/conf/chinese_industry_en_train_conf.json'
    # 根据语言选择配置文件路径
    config_file = INDUSTRY_CONFIG_FILES.get(language, INDUSTRY_CONFIG_FILES['en'])  # 默认为英文配置文件
    config = Config(config_file=config_file)
    predictor = Predictor(config)
    tokenized_text = tokenize_text(input_data, language)

    data_dict = {
        "doc_label": [],
        "doc_token": tokenized_text,
        "doc_keyword": [],
        "doc_topic": []
    }

    data_json = json.dumps(data_dict)
    predict_label_id = predict_label(predictor, data_json, config.task_info.label_type, config.eval.top_k,
                                     config.eval.threshold)

    if config.task_info.label_type == ClassificationType.MULTI_LABEL:
        predict_label_name = [predictor.dataset.id_to_label_map[i] for i in predict_label_id]
    else:
        predict_label_name = predictor.dataset.id_to_label_map[predict_label_id]

    data_dict["doc_label"] = "".join(predict_label_name)
    #data_dict["doc_topic"] = str(predict_label_id)
    #print(data_dict["doc_label"])
    return data_dict["doc_label"]


#if __name__ == "__main__":
    #language = "zh"
    #input_text = """
#1）如果输入的为jsonl文件，先判断文件格式是否正确，如编码格式（是否是utf-8）；字段名（role、content）（role对应的值是否为：user、system、assistant、observation）
#"""
    #get_industry_subject(input_text, language)