import time
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import sys
import aiofiles
import ray
import threading
from plugins.store_text_line import store_text_line
from plugins.clean_text_line import clean_text_line, get_cleaned_line_paper
from plugins.industry_predict import get_industry_subject
from plugins.ad_predict import get_ad_subject
from plugins.toxic_predict import get_toxic_subject
from plugins.language_classified import detect_language
from plugins.keywords_extract import keywords_extract
from plugins.sql import get_engine, get_session_factory, initialize_database
from plugins.dataDepublication import get_message, qdrant_storage
from plugins.ngrams_depub import check_repeats

#ray.shutdown()
# 初始化 Ray
#ray.init()

# 设置TOKENIZERS_PARALLELISM
os.environ["TOKENIZERS_PARALLELISM"] = "true"

JSONL_FILE_SIZE_LIMIT = 0.001 * 1024 * 1024 * 1024  # 1G

#@ray.remote
def process_text_line(text_line_list, other_list, sql_alchemy_path, output_directory, global_index_start):
    # print("开始写入:"+str(global_index_start))
    # 在远程函数内部创建和使用 Session
    engine = get_engine(sql_alchemy_path)
    session_factory = get_session_factory(engine)
    session = session_factory()

    # 获取输出目录的基本名称
    base_output_name = os.path.basename(output_directory)
    # 计算新文件的索引并格式化
    hex_jsonl_number = f'{global_index_start:04x}'
    # 构造新JSONL文件的路径
    jsonl_file_path = os.path.join(output_directory, f'{base_output_name}{hex_jsonl_number}.jsonl')
    jsonl_file = open(jsonl_file_path, 'w', encoding='utf-8')

    line = ''
    current_line_number = 0
    print("======开始去重处理======")
    text_line_nodup = get_message(text_line_list)
    print(text_line_nodup)
    text_lines = []
    id_values = []
    #text_line_nodup = [True] * len(text_line_list)
    for line_id, line_data in enumerate(text_line_list):
        if text_line_nodup[line_id]:
            other = json.dumps(other_list[line_id], ensure_ascii=False)
            print(other)
            title = json.dumps(other_list[line_id]['title'], ensure_ascii=False).strip('\"')
            print("标题：",title)

            # 构造行号的十六进制表示
            hex_line_number = f'{current_line_number:08x}'
            # 构造唯一标识符
            id_value = f'{base_output_name}{hex_jsonl_number}{hex_line_number}'
            # 构造JSONL格式的行并写入文件
            line = json.dumps({'id': id_value, 'text': line_data}, ensure_ascii=False) + '\n'
            jsonl_file.write(line)
            # 存储文本行及其元数据到sql数据库

            language = detect_language(line_data)
            print("语言：", language)
            industry_subject = get_industry_subject(line_data, language)
            print("主题：", industry_subject)
            keywords = keywords_extract(line_data, language)
            print("关键词：", keywords)
            fifth_grams = check_repeats(line_data, n=5)
            print("5grams：", fifth_grams)
            ninth_grams = check_repeats(line_data, n=9)
            print("9grams：", ninth_grams)
            ad_subjecrt = get_ad_subject(line_data, language)
            print("广告判别：", ad_subjecrt)
            line_data_split = line_data[0:510]
            toxic_subject = get_toxic_subject(line_data_split, language)
            print("毒性判别：", toxic_subject)

            store_text_line(session, id_value, line_data, other, industry_subject, language, keywords, title, fifth_grams, ninth_grams, ad_subjecrt, toxic_subject)
            current_line_number += 1
            text_lines.append(line_data)
            id_values.append(id_value)
            #qdrant_store_data(line_data)

    session.commit()
    if len(text_lines) > 0:
        qdrant_storage(text_lines, id_values)


    if jsonl_file is not None:
        jsonl_file.close()

    return current_line_number


def process_json(file_chunk, output_directory, text_key, sql_alchemy_path, global_index_start):
    # 获取数据库引擎
    engine = get_engine(sql_alchemy_path)

    # 初始化数据库
    initialize_database(engine)

    # 获取会话工厂
    session_factory = get_session_factory(engine)

    session = session_factory()
    try:
        text_line_list = []
        other_list = []
        current_jsonl_size = 0

        # 遍历文件块中的每个文件路径
        for file_path in file_chunk:
            with open(file_path, 'r', encoding='utf-8') as json_file:
                # 读取JSON文件内容
                for line in json_file:
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        continue

                    if text_key in data:
                        text = data[text_key]
                        text = clean_text_line(text)
                        #text = get_cleaned_line_paper(text)
                        #print(text)
                        if text is not None:
                            item_copy = data.copy()
                            del item_copy[text_key]
                            size = sys.getsizeof(text)
                            #print(size)

                            if current_jsonl_size + size >= JSONL_FILE_SIZE_LIMIT:
                                # 处理当前批次的文本行
                                process_text_line(text_line_list, other_list, sql_alchemy_path, output_directory, global_index_start)

                                global_index_start += 1  # Prepare for next file if needed
                                current_jsonl_size = 0
                                text_line_list = []
                                other_list = []

                            # 添加当前文本行到批次中
                            text_line_list.append(text)
                            other_list.append(item_copy)
                            current_jsonl_size += size
        # 处理剩余的文本行
        if text_line_list:
            process_text_line(text_line_list, other_list, sql_alchemy_path, output_directory, global_index_start)
        #print(text_line_list)
        session.commit()
    finally:
        session.commit()
        session.close()

    session_factory.remove()


def save_json_to_jsonl(dataset_directory, output_directory, text_key, settings):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    workers = settings.convert.workers
    sql_alchemy_path = settings.database.sql.SQLAlchemy_path

    # 初始化json文件列表
    json_files = []
    for root, dirs, files in os.walk(dataset_directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                json_files.append(file_path)

    total_files = len(json_files)
    if total_files < workers:
        workers = total_files
    file_chunks = [json_files[i::workers] for i in range(workers)]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for i, file_chunk in enumerate(file_chunks):
            global_index_start = math.floor(i * 65536 / workers)
            futures.append(executor.submit(process_json, file_chunk, output_directory, text_key, sql_alchemy_path, global_index_start))

        for future in as_completed(futures):
            future.result()  # Wait for all threads to complete

#ray.shutdown()
