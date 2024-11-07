# 数据湖存储路径
LAKE_DATA_DIR = "/mnt/data/Lake_Data"


# 数据库连接配置
db_config = {
    'host': '192.168.10.58',
    'user': 'root',
    'password': 'cetc15s',
    'database': 'lakehouse_db'
}

# Qdrant 配置
QDRANT_CONFIG = {
    'host': '192.168.10.58',
    'port': 6333,
    'timeout': 100000000
}

COLLECTION_NAME = 'pt_datasets_inner'  # 默认集合名称



# 向量模型配置
EMBEDDING_MODEL_CONFIG = {
    'model_path': './models/bge-m3',  # 模型路径
}



# ad分类预测模型配置文件路径
AD_CONFIG_FILES = {
    'zh': '/mnt/data/ML_folder/Infinite_Lake/models/classification/conf/chinese_ads_train_conf.json',
    'en': '/mnt/data/ML_folder/Infinite_Lake/models/classification/conf/chinese_ads_en_train_conf.json'
}


# toxic_data分类预测模型配置文件路径
TOXIC_MODEL_PATHS = {
    'zh': {
        'tokenizer': '/mnt/data/ML_folder/Infinite_Lake/models/robert-base-cold',
        'model': '/mnt/data/ML_folder/Infinite_Lake/models/robert-base-cold'
    },
    'en': {
        'config_file': '/mnt/data/ML_folder/Infinite_Lake/models/classification/conf/chinese_toxic_en_train_conf.json'
    }
}

# industry分类预测模型配置文件路径
INDUSTRY_CONFIG_FILES = {
    'zh': '/mnt/data/ML_folder/Infinite_Lake/models/classification/conf/chinese_industry_train_conf.json',
    'en': '/mnt/data/ML_folder/Infinite_Lake/models/classification/conf/chinese_industry_en_train_conf.json'
}

# 停用词路径
STOPWORDS_PATH = '/mnt/data/ML_folder/Infinite_Lake/stopwords.txt'

# 关键词提取配置
KEYWORDS_CONFIG = {
    'topK': 5,  # 提取的关键词数量
    'withWeight': False,  # 是否返回权重
    'allowPOS': ('n', 'ns', 'nt', 'nw', 'nz')  # 允许的词性
}

# 语种分类模型路径
LANGUAGE_MODEL_PATH = '/mnt/data/ML_folder/Infinite_Lake/models/classification/model/lid.176.bin'

# 奖励模型路径
REWARD_MODEL_PATH = '/mnt/data/ML_folder/Infinite_Lake/models/internlm2_reward'

# Kenlm 模型配置
KENLM_CONFIG = {
    'zh_model_dir': '/mnt/data/ML_folder/Infinite_Lake/models/kenlm/wikipedia',  # Kenlm 模型路径
    'language': 'zh'  # 使用的语言
}