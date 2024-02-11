# store_text_line.py
from sqlalchemy.exc import SQLAlchemyError
from sql import TextLine

def store_text_line(session, line_id, text, other):
    try:
        length = len(text.encode('utf-8'))

        text_line = TextLine(
            line_id=line_id,
            length=length,
            other=other,
            # 其他字段根据需要置空或赋予默认值
        )

        session.add(text_line)
        
    except SQLAlchemyError as e:
        session.rollback()
        print(f"Error storing text line: {e}")
