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
from config_plugins import TOXIC_MODEL_PATHS



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


import torch
from transformers.models.bert import BertTokenizer, BertForSequenceClassification

def get_toxic_subject(input_data, language):
    #text_language = detect_language(str(input_data))

    # 构造用于返回的字典
    data_dict = {
        "doc_label": "",
        "doc_token": [],
        "doc_keyword": [],
        "doc_topic": ""
    }

    if language == "zh":
        # 从config.py中获取模型和tokenizer的路径
        tokenizer_path = TOXIC_MODEL_PATHS['zh']['tokenizer']
        model_path = TOXIC_MODEL_PATHS['zh']['model']

        # 加载模型和tokenizer
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        model.eval()


        texts = [input_data]
        model_input = tokenizer(texts, return_tensors="pt", padding=True)
        model_output = model(**model_input, return_dict=False)
        prediction = torch.argmax(model_output[0].cpu(), dim=-1).item()

        data_dict["doc_label"] = "safety" if prediction == 0 else "unsafety"

    else:
        # 加载英文语言下的配置文件路径
        config_file = TOXIC_MODEL_PATHS['en']['config_file']
        config = Config(config_file=config_file)
        predictor = Predictor(config)

        # 处理英文文本并预测
        tokenized_text = input_data.split(" ")
        data_dict["doc_token"] = tokenized_text
        data_json = json.dumps(data_dict)

        predict_prob = predictor.predict(data_json)

        if config.task_info.label_type == ClassificationType.MULTI_LABEL:
            predict_label_ids = np.argsort(-predict_prob)[:config.eval.top_k]
            predict_label_ids = [idx for idx in predict_label_ids if predict_prob[idx] > config.eval.threshold]
            predict_label_name = "".join([predictor.dataset.id_to_label_map[idx] for idx in predict_label_ids])
            data_dict["doc_topic"] = str(predict_label_ids)
        else:
            predict_label_id = predict_prob.argmax()
            predict_label_name = predictor.dataset.id_to_label_map[predict_label_id]
            data_dict["doc_topic"] = str(predict_label_id)

        data_dict["doc_label"] = "unsafety" if predict_label_name == "0" else "safety"
    #print(data_dict["doc_label"])
    return data_dict["doc_label"]

#if __name__ == "__main__":
    #input_text ="""
    #GRAND RAPIDS, MI - More than 600 third-, fourth- and fifth-grade students from 19 Grand Rapids Public Schools will square off Saturday, April 19, for Jump Jam at DeVos Place.\nFor the second year, GRPS and Spectrum Health are partnering on the no-fee jump rope competition geared toward promoting health and wellness to children in the school district.\nSpectrum Health officials say Jump Jam meets it mission to improve the health of the communities it serves. The initiative is one way the GRPS seeks to promote physical activity.\nIn 2012, GRPS entered into partnership with the Student Advancement Foundation to offer a variety of elementary after-school sports, including basketball and soccer. Students attend gym class once a week for 48 minutes in the district.\nAn amusing video promoting the merits of the event, featuring Superintendent Teresa Weatherall Neal and Steve Heacock, senior vice president for public affairs at Spectrum Health, now has more than 700 views on YouTube. Billed as a \"Hashtag Battle,\" Mayor George Heartwell, referred to by Neal as her BFF, has a cameo.\nNeal and Heacock are filmed in their respective offices typing, sharing out loud what they are saying in back and forth emails about the upcoming event that include a ton hashtags.\nFor example, \"Jump Jam is great way to encourage our children to exercise,\" said Neal, # Healthy Bodies, # Healthy Minds. Heacock replies, \"We could all use more exercise,\" #Healthier Possibilities, #Sitting, Sitting, Sitting, #Spare Tire.\nAbout one-third of U.S. children ages 10 to 17 were found to be either overweight, 15.6 percent or obese, 15.7 percent, according to a 2013 report released by the National Survey of Children's Health from the Data Resource Center for Child and Adolescent Health.\nHowever, there are signs Michigan is making progress on childhood obesity. An August 2013 report released by the Centers for Disease Control and Prevention showed that 18 states, including Michigan, experienced a decline in obesity rates among 2- to 4-year-olds from low-income families between 2008 and 2011.\nOver that period, Michigan's rate fell from 13.9 percent to 13.2 percent, a decrease deemed statistically significant in the CDC analysis. The obesity rates 10- to 17-year-olds was 14.8 percent.\nJump Jam, which drew around [[phy_address]] year, is scheduled from 9 a.m. to 2:30 p.m. downtown in the DeVos Hall Exhibit Hall C. However, event planners say it is not open to the public because in addition to the jumpers, there will be hundreds of family members.
    #"""
    #input_text ="""
#根据 FastText 模型的输出格式，确保语言代码的提取逻辑是正确的。可能需要调整 [9:] 以适应实际的输出格式
    #"""
    #get_toxic_subject(input_text)