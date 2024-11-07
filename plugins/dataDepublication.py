import json
import time
import requests
import random


def get_message(text):
    # 将文本数据放入字典中
    data = {"text_list": text}
    # 调用embedding服务
    response = requests.post("http://192.168.10.58:5003/api/data/depublication", json=data, timeout=10000000)
    try:
        # 尝试手动解析 JSON 数据
        result = json.loads(response.text)
        content = result.get('message')
        #print(content)
        return content
    except json.JSONDecodeError as e:
        # 如果解析出现 JSONDecodeError 异常，则打印错误信息并返回 None
        print("反馈:", response)
        #print("bad_line:", text)
        print("JSON Decode Error:", e)
        return None


def qdrant_storage(text, idlist):
    # 将文本数据放入字典中
    data = {"text_list": text, "lineId_list": idlist}
    # 调用embedding服务
    response = requests.post("http://192.168.10.58:5003/api/data/qdrant_store", json=data, timeout=10000000)
    print(response)
    return response





