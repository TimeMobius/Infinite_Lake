import subprocess
import os
import glob
from docx import Document


# doc格式文件转换成docx文件
# 如果存在文件路径中存在doc，则需要转化成docx文件

def convert_doc_to_docx(doc_path):
    output = subprocess.check_output(["soffice",
        "--headless",
        "--invisible",
        "--convert-to",
        "docx",
        f"{doc_path}",
        "--outdir",
        f"{os.path.dirname(doc_path)}"])

    output_directory = f"{os.path.dirname(doc_path)}/{os.path.basename(doc_path).split('.')[0]}.docx"

    print(output_directory)
    return output_directory

def process_files(file_path):
    if os.path.isdir(file_path):
        doc_files = glob.glob(os.path.join(file_path, '*.doc'))
        if doc_files:
            for doc_file in doc_files:
                convert_doc_to_docx(doc_file)
        file_list = []  # 创建一个空列表来存储文件路径
        for root, dirs, files in os.walk(file_path):
            for file in files:
                if file.endswith(('.docx')):
                    file_list.append(os.path.join(root, file))
        print(file_list)
        return file_list

    elif os.path.isfile(file_path):
        if file_path.lower().endswith('.doc'):
            convert_doc_to_docx(file_path)
            return [f'{file_path}' + 'x']
        elif file_path.lower().endswith('.docx'):
            return [file_path]
        else:
            return ["不包含.docx和.doc文件"]

# 将docx文件转换成md文件，包括docx文件中文本和表格的转换

def docx_to_md(docx_path, md_path):
    # 加载文档
    docx = Document(docx_path)
    md_content = []

    for para in docx.paragraphs:
        # 将docx中的文本格式转换为Markdown格式
        text = para.text
        md_content.append(text)

    md_content.append("\n")  # 添加空行以分隔段落和表格

    # 转换文档中的表格
    for table in docx.tables:
        #print(table)
        # 添加Markdown表格的头部
        md_content.append("| " + " | ".join(cell.text for cell in table.rows[0].cells) + " |")
        md_content.append("| " + " | ".join("---" for _ in table.rows[0].cells) + " |")

        # 转换表格的内容行
        for row in table.rows[1:]:
            md_content.append("| " + " | ".join(cell.text for cell in row.cells) + " |")

        md_content.append("\n")  # 在表格后添加空行

    # 将生成的Markdown内容写入文件
    with open(md_path, 'w', encoding='utf-8') as md_file:
        md_file.write("\n".join(md_content))


def docx_folder_to_md_folder(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".docx"):
            docx_path = os.path.join(input_folder, filename)
            md_path = os.path.join(output_folder, filename.replace('.docx', '.md'))

            # 调用转换函数
            docx_to_md(docx_path, md_path)

        elif filename.endswith(".doc"):
            doc_path = os.path.join(input_folder, filename)
            docx_path = convert_doc_to_docx(doc_path)
            md_path = os.path.join(output_folder, filename.replace('.doc', '.md'))
            docx_to_md(docx_path, md_path)



# 使用示例
#docx_path = '/mnt/data/Raw_Data/api_test/太和县人民政府.docx'  # 要转换的docx文件路径
#md_path = '/mnt/data/Raw_Data/api_test_out/太和县人民政府.md'  # Markdown输出文件路径
#process_files(docx_path)
#docx_to_md(docx_path , md_path)
#print(f"Markdown file '{md_path}' has been created.")