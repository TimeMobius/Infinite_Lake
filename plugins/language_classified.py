#import fasttext
from fasttext.FastText import _FastText
from config_plugins import LANGUAGE_MODEL_PATH



language_model = _FastText(model_path=LANGUAGE_MODEL_PATH)


def detect_language(text):
    # 使用 fasttext 进行自然语言识别
    predictions = language_model.predict(text, k=1)
    language_code = predictions[0][0][9:]  # 提取语言代码
    return f"{language_code}"

# 测试文本
#texts = [
    #"Hello, how are you?",       # 英文
    #"你好，你在做什么？",          # 中文
    #"こんにちは、お元気ですか？",  # 日文
    #"print('Hello, World!')"    # Python 代码
#]

# 输出识别结果
#for text in texts:
    #print(f"Text: {text}\n{detect_language(text)}\n")

