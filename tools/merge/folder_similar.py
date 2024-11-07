import os
import shutil

# 设定文件夹路径
folder_a = "path/sp_20240705_result"
folder_b = "path/sp_20240705"
destination_folder = "path/sp_reminer_0708"

# 创建一个字典来存储folder_b中的所有路径
subfolders_in_folder_a = set(os.listdir(folder_a))
#print(subfolders_in_folder_a)


for file in os.listdir(folder_b):
    if file.lower().endswith(".pdf"):
        filename = file.split(".pdf")[0]

        if filename not in subfolders_in_folder_a:
            print(filename)
            source_file = os.path.join(folder_b, file)
            destination_file = os.path.join(destination_folder, file)
            shutil.copy2(source_file, destination_file)
            print(f'文件{file}已被复制到{destination_folder}')

print("处理完成")
