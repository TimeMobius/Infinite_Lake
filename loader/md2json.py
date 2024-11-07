#import sys
#sys.path.append('/mnt/data/ML_folder/marker')
import json
import markdown
import os
import html2text

def md_to_json(md_file_path, json_file_path):
    # 读取Markdown文件
    with open(md_file_path, 'r', encoding='utf-8') as file:
        md_content = file.read()

    # 将Markdown转换为HTML
    html_content = markdown.markdown(md_content)
    # 使用html2text将HTML转换为纯文本
    text_maker = html2text.HTML2Text()
    text_maker.ignore_links = True
    text_maker.ignore_images = True
    text_maker.ignore_emphasis = False
    text_maker.ignore_tables = False
    content = text_maker.handle(html_content)
    # 将HTML内容封装成JSON对象
    json_data = {"text": content}

    # 写入JSON文件
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, ensure_ascii=False)


def convert_md_files_in_folder(folder_path, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中所有的文件
    #for filename in os.listdir(folder_path):
    for root, dirs, files in os.walk(folder_path):
        #print(files)
        #print(dirs)
        for filename in files:
            #print(filename)
            if filename.endswith('.md'):
                #md_file_path = os.path.join(folder_path, filename)
                md_file_path = os.path.join(root, filename)
                json_file_name = filename[:-3] + '.json'  # 将.md后缀改为.json
                json_file_path = os.path.join(output_folder, json_file_name)
                #md_to_json(md_file_path, json_file_path)
                # 读取Markdown文件
                with open(md_file_path, 'r', encoding='utf-8') as file:
                    md_content = file.read()

                # 将内容封装成JSON对象
                json_data = {"title": filename[:-3], "text": md_content}

                # 写入JSON文件
                with open(json_file_path, 'w', encoding='utf-8') as json_file:
                    json.dump(json_data, json_file, ensure_ascii=False)



#if __name__ == "__main__":
    #input_paths = "/mnt/data/Raw_Data/zhiku_total_md_json/wiley_md"
    #output_paths = "/mnt/data/Raw_Data/zhiku_total_md_json/wiley_json"
    #convert_md_files_in_folder(input_paths, output_paths)