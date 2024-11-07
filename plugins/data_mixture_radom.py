import os
import random
import shutil
import json


def discover_jsonl_files(directory_paths):
    # 递归发现所有的jsonl文件
    jsonl_files = []
    for directory_path in directory_paths:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.jsonl'):
                    jsonl_files.append(os.path.join(root, file))
    return jsonl_files


def mix_jsonl_files(input_folders, output_folder, total_size_mb, file_size_mb):
    # 将输入的字符串参数转换为整数
    total_size_mb = float(total_size_mb)
    file_size_mb = float(file_size_mb)
    # 设定输出文件的总大小限制和每个文件的大小限制（字节）
    total_size_bytes = total_size_mb * 1024 * 1024
    file_size_bytes = file_size_mb * 1024 * 1024
    current_size = 0
    current_file_data = []
    file_count = 0

    # 发现所有的jsonl文件
    all_files = discover_jsonl_files(input_folders)
    random.shuffle(all_files)  # 打乱文件顺序以随机选取

    # 检查输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in all_files:
        if current_size >= total_size_bytes:
            break

        file_size = os.path.getsize(file)
        if file_size > 0 and current_size + file_size <= total_size_bytes:
            with open(file, 'r', encoding='utf-8') as f:
                # 读取jsonl文件中的所有行
                lines = f.readlines()
                for line in lines:
                    line_size = len(line.encode('utf-8'))
                    if current_size + line_size > total_size_bytes:
                        break
                    if current_size + line_size > file_size_bytes:
                        # 保存当前文件
                        save_jsonl_file(output_folder, current_file_data, file_count)
                        # 重置状态
                        current_file_data = []
                        current_size = 0
                        file_count += 1
                    current_file_data.append(line)
                    current_size += line_size

    # 保存最后一个文件
    if current_file_data:
        save_jsonl_file(output_folder, current_file_data, file_count)

    print(f'Total data size copied: {current_size / 1024 / 1024:.2f} MB')
    return f'Files copied to {output_folder}'


def save_jsonl_file(output_folder, data, file_count):
    # 保存单个jsonl文件
    file_path = os.path.join(output_folder, f'data_part_{file_count}.jsonl')
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line)





# 使用示例
#if __name__ == "__main__":
    #input_folders = ['/mnt/data/Lake_Data/folder1', '/mnt/data/Lake_Data/folder2']  # 替换为实际的输入文件夹路径列表
    #output_folder = '/mnt/data/ML_folder/Data_Test/data_mix/folders'  # 替换为希望保存混合文件的输出文件夹路径
    #total_size_mb = 10  # 设定需要获取的数据总大小，以MB为单位
    #file_size_mb = 1  # 设定每个jsonl文件的最大大小，以MB为单位
    #result = mix_jsonl_files(input_folders, output_folder, total_size_mb, file_size_mb)
    #print(result)