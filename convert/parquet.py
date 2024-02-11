import os
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datasets import load_dataset
from store_text_line import store_text_line
from sql import get_engine, get_session_factory, initialize_database
import math

# 定义1G的大小限制
JSONL_FILE_SIZE_LIMIT = 1 * 1024 * 1024 * 1024  # 1G


def process_files(file_chunk, output_directory, text_key, sql_alchemy_path, global_index_start):

    # 获取数据库引擎
    engine = get_engine(sql_alchemy_path)

    # 初始化数据库
    initialize_database(engine)

    # 获取会话工厂
    session_factory = get_session_factory(engine)

    session = session_factory()
    try:
        current_jsonl_size = 0
        current_line_number = 0
        jsonl_file = None
        base_output_name = os.path.basename(output_directory)
        for file_path in file_chunk:
            dataset = load_dataset('parquet', data_files=file_path)

            for item in dataset['train']:
                if text_key in item:
                    if jsonl_file is None or current_jsonl_size >= JSONL_FILE_SIZE_LIMIT:
                        if jsonl_file is not None:
                            jsonl_file.close()
                        hex_jsonl_number = f'{global_index_start:04x}'
                        jsonl_file_path = os.path.join(output_directory, f'{base_output_name}{hex_jsonl_number}.jsonl')
                        jsonl_file = open(jsonl_file_path, 'w', encoding='utf-8')
                        current_jsonl_size = 0
                        current_line_number = 0
                        global_index_start += 1  # Prepare for next file if needed
                        session.commit()

                    text = item[text_key]
                    item_copy = item.copy()
                    del item_copy[text_key]
                    other = json.dumps(item_copy, ensure_ascii=False)
                    hex_line_number = f'{current_line_number:08x}'
                    id_value = f'{base_output_name}{hex_jsonl_number}{hex_line_number}'
                    store_text_line(session, id_value, text, other)

                    line = json.dumps({'id': id_value, 'text': text})
                    jsonl_file.write(line + '\n')
                    current_jsonl_size += len(line.encode('utf-8')) + 1
                    current_line_number += 1
            session.commit()
    finally:
        session.commit()
        session.close()

    session_factory.remove()

    if jsonl_file is not None:
        jsonl_file.close()

def save_parquet_to_jsonl(dataset_directory: str, output_directory: str, text_key: str, settings):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    workers = settings.convert.workers
    # 从配置获取SQLAlchemy连接路径
    sql_alchemy_path = settings.database.sql.SQLAlchemy_path
    parquet_files = []
    for root, dirs, files in os.walk(dataset_directory):
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                parquet_files.append(file_path)

    total_files = len(parquet_files)
    if total_files < workers:
        workers = total_files

    file_chunks = [parquet_files[i::workers] for i in range(workers)]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for i, file_chunk in enumerate(file_chunks):
            global_index_start = math.floor(i * 65536 / workers)
            futures.append(executor.submit(process_files, file_chunk, output_directory, text_key, sql_alchemy_path, global_index_start))

        for future in as_completed(futures):
            future.result()  # Wait for all threads to complete
