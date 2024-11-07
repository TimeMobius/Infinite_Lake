# store_text_line.py
from sqlalchemy.exc import SQLAlchemyError
from plugins.sql import TextLine_test



def store_text_line(session, line_id, text, other, industry_subject, language, key_word, title, fifth_grams, ninth_grams, ad_subjecrt, toxic_subject):
    if text is None:
        # 如果 text 为 None，则跳过处理
        return
    try:
        length = len(text.encode('utf-8'))
        other = other[:255]
        key_word = key_word[:255]
        text_line = TextLine_test(
            line_id=line_id,
            length=length,
            other=other,
            industry_subject=industry_subject,
            language=language,
            key_word=key_word,
            title=title,
            fifth_grams=fifth_grams,
            ninth_grams=ninth_grams,
            ad_subjecrt=ad_subjecrt,
            toxic_subject=toxic_subject
        )

        session.add(text_line)

    except SQLAlchemyError as e:
        session.rollback()
        print(f"Error storing text line: {e}")
