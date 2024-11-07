import os
import re
import json
import warnings
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import markdownify

# 忽略来自 ebooklib 库的特定警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="In the future version we will turn default option ignore_ncx to True.")



def contains_only_spaces_and_chars(input_string, char):
    pattern = re.compile(r'^[ \t\n' + char + ']*$')
    return bool(pattern.match(input_string))

def clean_special_characters(text):
    rows = text.split("\n")
    cleaned_text = []
    last_row = ""
    for row in rows:
        row = row.strip()
        if row == last_row or contains_only_spaces_and_chars(row, '<>') or "更多新书、好书扫描下方二维码，关注微信公众号：伴阅读书" in row:
            continue
        if row and row[-1] in "（］":
            row = row.rstrip("（］")
        if row and row[0] == '>':
            row = row[1:].lstrip()
        cleaned_text.append(row)
        last_row = row
    return "\n".join(cleaned_text).strip()

def extract_and_convert_epub(epub_file, output_directory):
        try:
            book = epub.read_epub(epub_file)
            output_text = ""
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                soup = BeautifulSoup(item.content, 'html.parser')
                content = markdownify.markdownify(str(soup), heading_style="ATX")
                output_text += clean_special_characters(content) + "\n\n"

            if output_text:
                file_name = os.path.splitext(os.path.basename(epub_file))[0] + '.md'
                output_file_path = os.path.join(output_directory, file_name)
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(output_text)
                #print(f"文件已保存到：{output_file_path}")
        except Exception as e:
            print(f"处理文件{epub_file}时发生错误：{e}")