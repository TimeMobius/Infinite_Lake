# store_text_line.py
from sqlalchemy.exc import SQLAlchemyError
from plugins.sql import TextLine_sft_test


def store_text_line(session, line_id, role_value, role_content):
    #if user is None:
        # 如果 text 为 None，则跳过处理
        #return
    try:
        length = len(role_content.encode('utf-8')) if role_value else 0

        #context = context[:500]
        role_content = role_content[:500]
        #assistant = assistant[:500]
        #other = other[:255]
        text_line = TextLine_sft_test(
            line_id=line_id,
            role=role_value,
            content=role_content,
            length=length,
            # 其他字段根据需要置空或赋予默认值
        )

        session.add(text_line)
        
    except SQLAlchemyError as e:
        session.rollback()
        print(f"Error storing text line: {e}")

