import torch
from transformers import AutoModel, AutoTokenizer
from config_plugins import REWARD_MODEL_PATH

# 加载模型和tokenizer
reward_path=REWARD_MODEL_PATH
model = AutoModel.from_pretrained(
    reward_path,
    device_map="cuda",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(reward_path, trust_remote_code=True)

# 封装函数
def get_messages_score(messages):
    """
    计算一组对话 messages 的总得分。

    :param messages: 包含多个对话的列表，每个对话包含 'role' 和 'content' 字段
    :return: 对话的总得分
    """

    score = model.get_score(tokenizer, messages)
    return score
