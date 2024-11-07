#  遍历文件夹内各文件的后缀名，根据文件的后缀名，判断文件类型，并调用相应的函数
import os
import multiprocessing
from loader.txt_to_md_single import txt_to_md
from loader.docx_to_md_single import docx_to_md
from loader.convert_pdf_single import paper_single_process
from loader.md2json import convert_md_files_in_folder
from loader.convert_epub_single import extract_and_convert_epub


def process_single_file(file_path, md_output_folder, json_output_folder):
    _, file_extension = os.path.splitext(file_path)
    if file_extension in ['.txt']:
        txt_to_md(file_path, md_output_folder)
    elif file_extension in ['.doc', '.docx']:
        docx_to_md(file_path, md_output_folder)
    elif file_extension in ['.pdf']:
        #process_pdf(file_path, md_output_folder)
        paper_single_process(file_path, md_output_folder)
    elif file_extension in ['.epub']:
        extract_and_convert_epub(file_path, md_output_folder)
    convert_md_files_in_folder(md_output_folder, json_output_folder)


def process_files_in_directory(input_path, md_output_folder, json_output_folder):
    for filename in os.listdir(input_path):
        file_path = os.path.join(input_path, filename)
        if os.path.isfile(file_path):
            process_single_file(file_path, md_output_folder, json_output_folder)



#input_path = "/mnt/data/ML_folder/Data_Test/ocr_data"
#md_output_path = "/mnt/data/ML_folder/Data_Test/ocr_resluts/md"
#json_output_path = "/mnt/data/ML_folder/Data_Test/ocr_resluts"
#process_files_in_directory(input_path, md_output_path, json_output_path)

