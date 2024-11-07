import os
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datasets import load_dataset
import math
import pymysql
import torch
from transformers import AutoModel, AutoTokenizer
import threading
import multiprocessing
from multiprocessing import Manager
from multiprocessing import Pool
#from sft_plugins.store_text_line_sft import store_text_line
from plugins.clean_text_line import clean_text_line

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy import select
from plugins.sql import TextLine_sft_test
from config_plugins import db_config

from plugins.industry_predict import get_industry_subject
from plugins.language_classified import detect_language
from plugins.keywords_extract import keywords_extract
from plugins.sft_filter_qulity import get_messages_score
from plugins.sft_filter_complexity import calculate_complexity_single
from plugins.sql import get_engine, get_session_factory, initialize_database


# 定义1G的大小限制
JSONL_FILE_SIZE_LIMIT = 0.05 * 1024 * 1024 * 1024  # 1G



# 创建数据库连接
connection = pymysql.connect(**db_config)
cursor = connection.cursor()


def read_jsonl(file_path):
    data_result = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            messages = data.get("messages", [])
            data_result.append({"messages": messages})
    return data_result

def write_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry, ensure_ascii=False) + '\n')

def add_end_key(data):
    for entry in data:
        messages = entry["messages"]
        if len(messages) == 1:
            messages[0]["end"] = True
        else:
            for i, msg in enumerate(messages):
                msg["end"] = (i == len(messages) - 1)
    return data

def is_first_duplicate(role, content, session):
    query = select(TextLine_sft_test.line_id).where(
        TextLine_sft_test.role == role,
        TextLine_sft_test.content == content,
        TextLine_sft_test.frontID.is_(None)
    )
    result = session.execute(query).scalar()
    return str(result) if result else None

def is_subsequent_duplicate(role, content, parent_line_id, session):
    query = select(TextLine_sft_test.line_id).where(
        TextLine_sft_test.role == role,
        TextLine_sft_test.content == content,
        TextLine_sft_test.frontID == parent_line_id
    )
    result = session.execute(query).scalar()
    return str(result) if result else None

def generate_line_id(base_output_name, global_index_start, current_line_number):
    hex_jsonl_number = f'{global_index_start:04x}'
    hex_line_number = f'{current_line_number:08x}'
    line_id = f'{base_output_name}{hex_jsonl_number}{hex_line_number}'
    return line_id



def store_text_line(session, line_id, role_value, role_content, frontID, indexID, end, score, messages_score, language, industry_subject, keywords, text_complexity):
    try:
        length = len(role_content.encode('utf-8')) if role_value else 0
        role_content = role_content[:1000]
        text_line = TextLine_sft_test(
            line_id=line_id,
            role=role_value,
            content=role_content,
            length=length,
            frontID=frontID,
            indexID=indexID,
            end=end,
            score=score,
            messages_score=messages_score,
            language=language,
            industry_subject=industry_subject,
            key_word=keywords,
            text_complexity=text_complexity
        )

        session.add(text_line)

    except SQLAlchemyError as e:
        session.rollback()
        print(f"Error storing text line: {e}")

def fetch_data_from_db(query):
    # Establish a database connection
    connection = pymysql.connect(**db_config)
    cursor = connection.cursor(pymysql.cursors.DictCursor)

    cursor.execute(query)
    rows = cursor.fetchall()

    # Close the database connection
    cursor.close()
    connection.close()
    return rows


def process_json_data(json_data, base_output_name, sql_alchemy_path, global_index_start, current_line_number):
    engine = create_engine(sql_alchemy_path)
    Session = sessionmaker(bind=engine)
    session = Session()
    messages = json_data['messages']
    indexID = None

    # 计算整个messages的score
    messages_score = get_messages_score(messages)


    first_message = messages[0]
    first_message['role'] = clean_text_line(first_message['role'])
    first_message['content'] = clean_text_line(first_message['content'])
    first_message['score'] = first_message.get('score', None)

    # 为第一个message分配总的messages_score
    first_message['messages_score'] = messages_score
    first_duplicate = is_first_duplicate(first_message['role'], first_message['content'], session)

    if first_duplicate:
        indexID = str(first_duplicate)
        last_inserted_id = str(first_duplicate)
    else:
        hex_jsonl_number = f'{global_index_start:04x}'  # 使用全局的 global_index_start 来保证 hex_jsonl_number 一致
        indexID = generate_line_id(base_output_name, global_index_start, current_line_number.value)
        language = detect_language(first_message['content'])
        print("语言：", language)
        industry_subject = get_industry_subject(first_message['content'], language)
        print("主题：", industry_subject)
        keywords = keywords_extract(first_message['content'], language)
        print("关键词：", keywords)
        text_complexity = calculate_complexity_single(first_message['content'])
        print("复杂度：", text_complexity)
        store_text_line(session, indexID, first_message['role'], first_message['content'], None, indexID, 'false', first_message['score'], first_message['messages_score'], language, industry_subject, keywords, text_complexity)
        last_inserted_id = indexID
        current_line_number.value += 1  # 直接增加值

    for i in range(1, len(messages)):
        message = messages[i]
        role = clean_text_line(message['role'])
        content = clean_text_line(message['content'])
        score = message.get('score', None)

        # 为每个message分配总的messages_score
        message['messages_score'] = messages_score
        end = 'true' if i == len(messages) - 1 else 'false'
        duplicate_id = is_subsequent_duplicate(role, content, last_inserted_id, session)
        if duplicate_id:
            last_inserted_id = str(duplicate_id)
            continue

        frontID = str(last_inserted_id)
        line_id = generate_line_id(base_output_name, global_index_start, current_line_number.value)  # 保持一致性
        language = detect_language(content)
        print("语言：", language)
        industry_subject = get_industry_subject(content, language)
        print("主题：", industry_subject)
        keywords = keywords_extract(content, language)
        print("关键词：", keywords)
        text_complexity = calculate_complexity_single(content)
        print("复杂度：", text_complexity)



        store_text_line(session, line_id, role, content, frontID, indexID, end, score, message['messages_score'], language, industry_subject, keywords, text_complexity)
        last_inserted_id = line_id
        current_line_number.value += 1

    session.commit()
    session.close()

    return current_line_number


def process_json(file_chunk, output_directory, dataset_type, sql_alchemy_path, global_index_start, current_line_number):
    current_jsonl_size = 0
    jsonl_file = None
    base_output_name = os.path.basename(output_directory)
    hex_jsonl_number = f'{global_index_start:04x}'  # 每个文件都使用同一个 global_index_start 来保持一致性
    jsonl_file_path = os.path.join(output_directory, f'{base_output_name}{hex_jsonl_number}.jsonl')

    for file_path in file_chunk:
        data = read_jsonl(file_path)
        data_with_end = add_end_key(data)

        for json_data in data_with_end:
            process_json_data(json_data, base_output_name, sql_alchemy_path, global_index_start, current_line_number)

        # 多轮数据处理逻辑
        if dataset_type == "多轮" or dataset_type == "单轮":
            multi_round_query = f"""
            SELECT line_id, role, content, frontID, indexID, end
            FROM text_line_sft_test
            WHERE SUBSTRING(line_id, 1, 4) = '{base_output_name}'
            ORDER BY indexID, line_id
            """

            data_sql = fetch_data_from_db(multi_round_query)

            def build_tree(messages):
                tree = {}
                root_nodes = set()

                for msg in messages:
                    line_id = msg['line_id']
                    frontID = msg['frontID']

                    # Create a new node for this message
                    tree[line_id] = {'node': msg, 'children': [], 'parent': None}

                    # If it has a frontID, it's not a root node
                    if frontID is not None:
                        # Find or create its parent node
                        if frontID not in tree:
                            tree[frontID] = {'node': {'line_id': frontID, 'frontID': None}, 'children': [],
                                             'parent': None}
                        tree[frontID]['children'].append(line_id)
                        tree[line_id]['parent'] = tree[frontID]['node']
                    else:
                        # It's a root node
                        root_nodes.add(line_id)

                return tree, list(root_nodes)

            def dfs(node, path, result):
                # Add current node to the path
                path.append(node['node'])
                # If no more children, we have reached the end of a thread
                if not node['children']:
                    result.append(path.copy())
                else:
                    # Recursively visit all children
                    for child_line_id in node['children']:
                        dfs(tree[child_line_id], path, result)
                # Remove last added node (backtrack)
                path.pop()

            def build_list_representation(tree, root_nodes):
                result = []
                for root_id in root_nodes:
                    dfs(tree[root_id], [], result)
                return result

            tree, root_nodes = build_tree(data_sql)
            list_representation = build_list_representation(tree, root_nodes)

            for lst in list_representation:
                line = json.dumps({'message': lst}, ensure_ascii=False)
                line_size = len(line.encode('utf-8')) + 1

                if jsonl_file is None or (current_jsonl_size + line_size) > JSONL_FILE_SIZE_LIMIT:
                    # 如果文件大小超限，则新建文件
                    hex_jsonl_number = f'{global_index_start:04x}'
                    jsonl_file_path = os.path.join(output_directory, f'{base_output_name}{hex_jsonl_number}.jsonl')
                    jsonl_file = open(jsonl_file_path, 'w', encoding='utf-8')
                    global_index_start += 1
                    current_jsonl_size = 0

                jsonl_file.write(line + '\n')
                current_jsonl_size += line_size


def save_jsonl_to_jsonl(dataset_directory: str, output_directory: str, dataset_type: str, settings):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    workers = settings.convert.workers
    sql_alchemy_path = settings.database.sql.SQLAlchemy_path
    json_files = []

    for root, dirs, files in os.walk(dataset_directory):
        for file in files:
            if file.endswith('.jsonl'):
                json_files.append(os.path.join(root, file))

    total_files = len(json_files)
    if total_files < workers:
        workers = total_files
    file_chunks = [json_files[i::workers] for i in range(workers)]
    #print(file_chunks)

    with Manager() as manager:
        current_line_number = manager.Value('i', 0)  # 使用 Manager 创建共享整数
        with ProcessPoolExecutor(max_workers=1) as executor:
            futures = []
            for i, file_chunk in enumerate(file_chunks):
                #global_index_start = math.floor(i * 65536 / workers)
                global_index_start = 0
                futures.append(executor.submit(
                    process_json, file_chunk, output_directory, dataset_type, sql_alchemy_path, global_index_start, current_line_number))

            for future in as_completed(futures):
                future.result()  # 等待所有进程完成