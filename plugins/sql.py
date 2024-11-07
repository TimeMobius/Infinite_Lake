from sqlalchemy import create_engine, Column, Integer, String, BigInteger, SmallInteger, DateTime, Boolean
#from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base
from datetime import datetime
import pytz  # 导入pytz模块

Base = declarative_base()

def get_utc_plus_8():
    # 获取当前时间，并将其转换为UTC+8时区
    utc_plus_8 = pytz.timezone('Asia/Shanghai')  # 你可以根据需要选择适当的时区
    return datetime.now(utc_plus_8)

class TextMeta(Base):
    __tablename__ = 'text_meta'
    dataset_id = Column(SmallInteger, primary_key=True, autoincrement=True)
    dataset_name = Column(String(255))
    dataset_source = Column(String(255))
    dataset_path = Column(String(255))
    original_format = Column(String(255))
    text_key = Column(String(255))
    dataset_type = Column(String(255))
    original_size = Column(BigInteger)
    lake_size = Column(BigInteger)
    language = Column(String(50))
    added_date = Column(DateTime, default=get_utc_plus_8)  # 使用UTC+8时区的时间作为默认值


class TextLine(Base):
    __tablename__ = 'text_line'
    line_id = Column(String(16), primary_key=True)
    title = Column(String(255))
    language = Column(String(50))
    type = Column(String(50))
    source_type = Column(String(50))
    industry_subject = Column(String(255))
    length = Column(BigInteger)
    key_word = Column(String(255))
    fifth_grams = Column(String(10))
    ninth_grams = Column(String(10))
    ad_subjecrt = Column(String(10))
    toxic_subject = Column(String(10))
    other = Column(String(255))
    #added_date = Column(DateTime, default=get_utc_plus_8)  # 使用UTC+8时区的时间作为默认值


class TextLine_sft(Base):
    __tablename__ = 'text_line_sft'
    line_id = Column(String(16), primary_key=True)
    role = Column(String(10))
    content = Column(String(500))
    frontID = Column(String(500))
    indexID = Column(String(500))
    end = Column(String(10))
    language = Column(String(50))
    industry_subject = Column(String(255))
    length = Column(BigInteger)
    key_word = Column(String(255))
    score = Column(BigInteger)
    messages_score = Column(String(10))
    text_complexity = Column(String(10))
    other = Column(String(255))


class TextLine_dpo(Base):
    __tablename__ = 'text_line_dpo'
    line_id = Column(String(16), primary_key=True)
    role = Column(String(10))
    content = Column(String(500))
    frontID = Column(String(500))
    indexID = Column(String(500))
    end = Column(String(10))
    language = Column(String(50))
    industry_subject = Column(String(255))
    length = Column(BigInteger)
    key_word = Column(String(255))
    score = Column(BigInteger)
    messages_score = Column(String(10))
    text_complexity = Column(String(10))
    other = Column(String(255))


class TextMeta_test(Base):
    __tablename__ = 'text_meta_test'
    dataset_id = Column(SmallInteger, primary_key=True, autoincrement=True)
    dataset_name = Column(String(255))
    dataset_source = Column(String(255))
    dataset_path = Column(String(255))
    original_format = Column(String(255))
    text_key = Column(String(255))
    dataset_type = Column(String(255))
    original_size = Column(BigInteger)
    lake_size = Column(BigInteger)
    language = Column(String(50))
    added_date = Column(DateTime, default=get_utc_plus_8)  # 使用UTC+8时区的时间作为默认值

class TextLine_test(Base):
    __tablename__ = 'text_line_test'
    line_id = Column(String(16), primary_key=True)
    title = Column(String(255))
    language = Column(String(50))
    type = Column(String(50))
    source_type = Column(String(50))
    industry_subject = Column(String(255))
    length = Column(BigInteger)
    key_word = Column(String(255))
    fifth_grams = Column(String(10))
    ninth_grams = Column(String(10))
    ad_subjecrt = Column(String(10))
    toxic_subject = Column(String(10))
    other = Column(String(255))
    #added_date = Column(DateTime, default=get_utc_plus_8)  # 使用UTC+8时区的时间作为默认值

class TextLine_dpo_test(Base):
    __tablename__ = 'text_line_dpo_test'
    line_id = Column(String(16), primary_key=True)
    role = Column(String(10))
    content = Column(String(500))
    frontID = Column(String(500))
    indexID = Column(String(500))
    end = Column(String(10))
    language = Column(String(50))
    industry_subject = Column(String(255))
    length = Column(BigInteger)
    key_word = Column(String(255))
    score = Column(BigInteger)
    messages_score = Column(String(10))
    text_complexity = Column(String(10))
    other = Column(String(255))

class TextLine_sft_test(Base):
    __tablename__ = 'text_line_sft_test'
    line_id = Column(String(16), primary_key=True)
    role = Column(String(10))
    content = Column(String(500))
    frontID = Column(String(500))
    indexID = Column(String(500))
    end = Column(String(10))
    language = Column(String(50))
    subject = Column(String(255))
    length = Column(BigInteger)
    key_word = Column(String(255))
    score = Column(BigInteger)
    other = Column(String(255))


def get_engine(sql_alchemy_path):
    return create_engine(sql_alchemy_path)

def get_session_factory(engine):
    return scoped_session(sessionmaker(bind=engine))

def initialize_database(engine):
    Base.metadata.create_all(engine)

