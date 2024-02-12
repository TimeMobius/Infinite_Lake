from sqlalchemy import create_engine, Column, Integer, String, BigInteger, SmallInteger, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
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
    subject = Column(String(255))
    length = Column(BigInteger)
    key_word = Column(String(255))
    other = Column(String(255))

def get_engine(sql_alchemy_path):
    return create_engine(sql_alchemy_path)

def get_session_factory(engine):
    return scoped_session(sessionmaker(bind=engine))

def initialize_database(engine):
    Base.metadata.create_all(engine)
