import os
import json
import pymysql
import random
from config_plugins import db_config, LAKE_DATA_DIR


def get_line_ids(filter_fields, filter_values):
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            # 构建 SQL 查询条件
            conditions = ' AND '.join([f"`{field}` REGEXP %s" for field in filter_fields])
            sql = f"SELECT line_id FROM text_line WHERE {conditions}"

            # 使用 executemany 来执行 SQL 语句，适用于多个条件
            cursor.executemany(sql, [filter_values])

            result = cursor.fetchall()
            return [row[0] for row in result]
    finally:
        connection.close()




def get_storage_path(line_id):
    """
    根据 line_id 生成存储路径。

    :param line_id: 字符串，表示行ID
    :return: 生成的文件路径
    """
    prefix = line_id[:4]
    url = f"{LAKE_DATA_DIR}/{prefix}"
    return url


def get_text_from_jsonl(storage_path, line_id):
    """
    从 JSONL 文件中检索指定 line_id 的文本内容。

    :param storage_path: 存储路径
    :param line_id: 字符串，表示行ID
    :return: 检索到的文本内容或 None
    """
    jsonl_file_path = f"{storage_path}/{line_id[:8]}.jsonl"
    print(jsonl_file_path)

    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                if data.get('id') == line_id:
                    return data.get('text')
    except FileNotFoundError:
        print(f"File not found: {jsonl_file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file {jsonl_file_path}: {e}")
    return None


def write_to_jsonl(data, output_directory, file_size_bytes=None):
    """
    将数据写入 JSONL 文件，根据指定的文件大小分割文件。

    :param data: 要写入的字典列表
    :param output_directory: 输出目录
    :param file_size_bytes: 每个文件的最大大小（字节）
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    file_count = 0
    current_file_path = os.path.join(output_directory, f"data_{file_count}.jsonl")
    current_file_size = 0

    file = open(current_file_path, 'w', encoding='utf-8')

    for item in data:
        line = json.dumps(item, ensure_ascii=False) + "\n"
        line_size = len(line.encode('utf-8'))

        # Check file size constraint if file_size_bytes is specified
        if file_size_bytes is not None and current_file_size + line_size > file_size_bytes:
            file.close()
            file_count += 1
            current_file_path = os.path.join(output_directory, f"data_{file_count}.jsonl")
            file = open(current_file_path, 'w', encoding='utf-8')
            current_file_size = 0

        file.write(line)
        current_file_size += line_size

    file.close()



def main(filter_field, filter_value, total_size_mb=None, file_size_mb=None, output_directory=None):
    """
    主函数，用于执行从数据库中获取数据并写入 JSONL 文件的全过程。

    :param filter_field: 字符串或列表，表示数据库中的字段名
    :param filter_value: 字符串或列表，表示与字段名对应的值
    :param total_size_mb: 总数据大小的限制（MB），可选参数
    :param file_size_mb: 每个文件的最大大小（MB），可选参数
    :param output_directory: 输出目录
    """

    # 将输入的字符串参数转换为浮点数，如果为空则设为无限制
    total_size_bytes = float('inf') if total_size_mb is None else float(total_size_mb) * 1024 * 1024
    file_size_bytes = float('inf') if file_size_mb is None else float(file_size_mb) * 1024 * 1024

    # 根据过滤条件检索和随机化行ID
    line_ids = get_line_ids(filter_field, filter_value)
    random.shuffle(line_ids)

    data = []
    current_size = 0

    # 处理每个行ID
    for line_id in line_ids:
        if current_size >= total_size_bytes:
            break

        storage_path = get_storage_path(line_id)
        text = get_text_from_jsonl(storage_path, line_id)

        if text:
            text_size = len(text.encode('utf-8'))
            if current_size + text_size <= total_size_bytes:
                data.append({"id": line_id, "text": text})
                current_size += text_size
                # print(f"Text for line ID {line_id}: {text[:50]}...")  # 预览前50个字符
            else:
                print(f"由于大小限制，跳过了行ID {line_id}。")

    # 将收集的数据写入 JSONL 文件
    write_to_jsonl(data, output_directory, file_size_bytes)
    processed_size_mb = current_size / 1024 / 1024
    print(f'处理的数据总大小: {processed_size_mb:.2f} MB')
    return f'文件已写入 {output_directory}'



# 示例使用

# Single field with multiple values
#main('field1', ['value1', 'value2'], total_size_mb=100, file_size_mb=10, output_directory='output')

# Multiple fields and values
#main(["key_word", "key_word", "language"], ["本体", "概念", "en"], total_size_mb=100, file_size_mb=10, output_directory='output')