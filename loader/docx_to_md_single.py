import os
import subprocess
import glob
from docx import Document

def convert_doc_to_docx(doc_path):
    output = subprocess.check_output(["/usr/bin/libreoffice7.6",
        "--headless",
        "--invisible",
        "--convert-to",
        "docx",
        f"{doc_path}",
        "--outdir",
        f"{os.path.dirname(doc_path)}"])

    output_directory = f"{os.path.dirname(doc_path)}/{os.path.basename(doc_path).split('.')[0]}.docx"
    return output_directory

def process_file(file_path):
    if file_path.lower().endswith('.doc'):
        docx_path = convert_doc_to_docx(file_path)
        return docx_path
    elif file_path.lower().endswith('.docx'):
        return file_path
    else:
        return None

def docx_to_md(docx_path, md_output_folder):
    docx_path = process_file(docx_path)
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
        # 添加Markdown表格的头部
        md_content.append("| " + " | ".join(cell.text for cell in table.rows[0].cells) + " |")
        md_content.append("| " + " | ".join("---" for _ in table.rows[0].cells) + " |")

        # 转换表格的内容行
        for row in table.rows[1:]:
            md_content.append("| " + " | ".join(cell.text for cell in row.cells) + " |")

        md_content.append("\n")  # 在表格后添加空行

    filename = os.path.basename(docx_path).replace('.docx', '.md')
    md_path = os.path.join(md_output_folder, filename)
    # 将生成的Markdown内容写入文件
    with open(md_path, 'w', encoding='utf-8') as md_file:
        md_file.write("\n".join(md_content))



# 示例使用
#input_path = '/nfs/dubhe-prod/dataset/2/versionFile/V0002/origin/无忧考网_2022-2023学年北京海淀四年级上学期期末数学真题及答案(1)_1715737597190_1715932664745.doc'  # 或者 'path/to/your/input.docx'
#output_folder = '/mnt/data/ML_folder/Data_Test/data_mix'
#process_file(input_path)
#docx_to_md(input_path, output_folder)
