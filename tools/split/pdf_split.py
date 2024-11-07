import os
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.errors import PdfReadError

"""

def split_pdf(input_pdf_path, output_folder, max_pages=4):
    try:
        # 打开原始PDF文件
        with open(input_pdf_path, 'rb') as input_pdf:
            reader = PdfReader(input_pdf)
            total_pages = len(reader.pages)

            # 检查是否需要拆分
            if total_pages <= max_pages:
                print(f"PDF文件页数少于或等于{max_pages}页，无需拆分。")
                return

            # 创建输出文件夹（如果不存在）
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # 进行拆分
            for i in range(0, total_pages, max_pages):
                writer = PdfWriter()
                for j in range(i, min(i + max_pages, total_pages)):
                    try:
                        writer.add_page(reader.pages[j])
                    except PdfReadError as e:
                        print(f"跳过无法读取的页面 {j + 1}：{e}")
                        continue

                start_page = i + 1
                end_page = min(i + max_pages, total_pages)
                output_pdf_filename = f"{os.path.splitext(os.path.basename(input_pdf_path))[0]}_pages_{start_page}_to_{end_page}.pdf"
                output_pdf_path = os.path.join(output_folder, output_pdf_filename)
                with open(output_pdf_path, 'wb') as output_pdf:
                    writer.write(output_pdf)

                print(f"生成拆分PDF: {output_pdf_path}")
    except Exception as e:
        print(f"处理文件 {input_pdf_path} 时出错：{e}")


def split_pdfs_in_folder(input_folder_path, output_folder_path, max_pages=10):
    # 遍历文件夹中的所有PDF文件
    for filename in os.listdir(input_folder_path):
        if filename.lower().endswith('.pdf'):
            input_pdf_path = os.path.join(input_folder_path, filename)
            split_pdf(input_pdf_path, output_folder_path, max_pages)
# 使用示例
input_folder_path = '/mnt/EPan/Mpaper/paper'
output_folder_path = '/mnt/EPan/mapper_split_pdf'
split_pdfs_in_folder(input_folder_path, output_folder_path)
"""

# 三、提取制定页数的pdf文件
import os
import shutil


def extract_small_pdfs(input_folder, output_folder, max_pages=10):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(input_folder, filename)
            try:
                with open(pdf_path, 'rb') as pdf_file:
                    reader = PdfReader(pdf_file)
                    num_pages = len(reader.pages)

                    # 如果PDF页数小于等于指定的页数，则将其复制到输出文件夹
                    if num_pages <= max_pages:
                        shutil.copy(pdf_path, output_folder)
                        print(f"复制文件: {filename}, 页数: {num_pages}")
            except Exception as e:
                print(f"处理文件 {filename} 时发生错误: {e}")


input_folder = '/mnt/EPan/Mpaper/paper'  # 替换为您的输入文件夹路径
output_folder = '/mnt/EPan/mpaper_reminer0701'  # 替换为您的输出文件夹路径

extract_small_pdfs(input_folder, output_folder)
