# 输入为一个文件夹路径，将该文件夹下及各子文件夹下所包括的.md文件保存到另一个文件夹中

import os
import shutil


def copy_md_files(source_folder, destination_folder):
    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 遍历源文件夹及其所有子文件夹
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.md'):
                source_file_path = os.path.join(root, file)
                destination_file_path = os.path.join(destination_folder, file)

                # 如果目标文件已存在，则添加前缀避免重名
                if os.path.exists(destination_file_path):
                    base, ext = os.path.splitext(file)
                    count = 1
                    while os.path.exists(destination_file_path):
                        new_file_name = f"{base}_{count}{ext}"
                        destination_file_path = os.path.join(destination_folder, new_file_name)
                        count += 1

                shutil.copy2(source_file_path, destination_file_path)
                print(f"Copied: {source_file_path} to {destination_file_path}")


# 示例使用
source_folder = 'path/sp_20240705_result'
destination_folder = 'path/sp_md'
copy_md_files(source_folder, destination_folder)
