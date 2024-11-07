import os

def txt_to_md(txt_file_path, md_folder_paths):
    # 确保输出文件夹存在
    if not os.path.exists(md_folder_paths):
        os.makedirs(md_folder_paths)

    # 打开TXT文件并读取内容
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 处理每一行
    processed_lines = [f'*{line.strip()}*\n' for line in lines]

    # 获取输入文件的文件名并修改扩展名为.md
    filename = os.path.basename(txt_file_path).replace('.txt', '.md')
    md_file_path = os.path.join(md_folder_paths, filename)

    # 写入到MD文件
    with open(md_file_path, 'w', encoding='utf-8') as file:
        file.writelines(processed_lines)

# 示例使用
#txt_file_path = 'path/to/your/input.txt'
#md_folder_path = 'path/to/your/output/folder'
#txt_to_md(txt_file_path, md_folder_path)
