import os
# 输入为txt文件，输出为json文件
def txt_to_md(txt_file_path, md_file_path):
    # 打开TXT文件并读取内容
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 处理每一行
    processed_lines = [f'*{line.strip()}*\n' for line in lines]

    # 写入到MD文件
    with open(md_file_path, 'w', encoding='utf-8') as file:
        file.writelines(processed_lines)


def txt_folder_to_md_folder(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            txt_file_path = os.path.join(input_folder, filename)
            md_file_path = os.path.join(output_folder, filename.replace('.txt', '.md'))

            # 调用转换函数
            txt_to_md(txt_file_path, md_file_path)




# 调用函数，输入你的txt文件路径和期望的md文件输出路径
#txt_file_path = "/mnt/data/ML_folder/Data_Test/test_txt2.txt"
#md_file_path = "/mnt/data/ML_folder/Data_Test/test_txt2_output.md"
#txt_to_md(txt_file_path, md_file_path)
