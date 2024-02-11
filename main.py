#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   app.py
@Contact :   

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------

"""

# Start dance your fingers
import traceback
import time
import random
import string
import sys
import os
import uvicorn
import argparse

from settings import Settings
from sql import get_engine, get_session_factory, initialize_database, TextMeta

from convert.parquet import save_parquet_to_jsonl

from sqlalchemy.exc import SQLAlchemyError
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from utils.log import logger


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'algo_sdk'))


app = FastAPI()
# 全局会话工厂引用
session_factory = None
settings = None

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='启动应用程序并指定配置文件。')
    parser.add_argument('-c', '--config', default='config.yaml', help='配置文件的路径。默认为config.yaml')
    return parser.parse_args()


def main():
    args = parse_args()
    global settings
    config_path = args.config  # 根据命令行参数获取配置文件路径
    settings = Settings(config_path)

    # # 通过属性访问配置信息
    # print(f"Server Port: {settings.server.port}")
    # print(f"Data Lake Path: {settings.storage.data_lake_path}")
    # print(f"Text Vectorization API URL: {settings.text_vectorization.api_url}")
    # print(f"Truncation Length: {settings.text_vectorization.truncation_length}")

@app.on_event("startup")
def startup_event():
    global session_factory
    main()

    # 从配置获取SQLAlchemy连接路径
    sql_alchemy_path = settings.database.sql.SQLAlchemy_path

    # 获取数据库引擎
    engine = get_engine(sql_alchemy_path)

    # 初始化数据库
    initialize_database(engine)

    # 获取会话工厂
    session_factory = get_session_factory(engine)
    print("Database initialized and session factory created successfully.")



@app.on_event("shutdown")
def shutdown_event():
    global session_factory
    if session_factory:
        session_factory.remove()  # 关闭会话工厂中的所有会话
        print("Session factory closed successfully.")
    print("Application is shutting down")





# # ------log-------
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# ch = logging.StreamHandler()
# fh = logging.FileHandler(filename='logs/server.log')
# formatter = logging.Formatter(
#     "%(asctime)s - %(module)s - %(funcName)s - line:%(lineno)d - %(levelname)s - %(message)s"
# )
# ch.setFormatter(formatter)
# fh.setFormatter(formatter)
# logger.addHandler(ch)  # 将日志输出至屏幕
# logger.addHandler(fh)  # 将日志输出至文件
#
#


@app.middleware("http")
async def log_requests(request, call_next):
    idem = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    logger.info(f"rid={idem} client_host={request.client.host} start request path={request.url.path}")
    start_time = time.time()

    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000
    formatted_process_time = '{0:.2f}'.format(process_time)
    logger.info(f"rid={idem} completed_in={formatted_process_time}ms status_code={response.status_code}")

    return response


# --------API---------


@app.get("/")
def read_root(request: Request):
    return {"message": "Hello World", "client_ip": request.client.host}


class Import_Text_Item(BaseModel):
    dataset_name: str
    dataset_source: str
    dataset_path: str
    original_format: str
    text_key: str

@app.post("/api/text/import")
async def text_logging(item: Import_Text_Item):
    session = session_factory()
    try:
        start_at = time.time()
        
        # 使用会话添加到text_meta表
        new_text_meta = TextMeta(
            dataset_name=item.dataset_name,
            dataset_source=item.dataset_source,
            dataset_path=item.dataset_path,
            original_format=item.original_format,
            text_key=item.text_key
        )
        session.add(new_text_meta)
        session.commit()
        dataset_id = new_text_meta.dataset_id
        
        # 将dataset_id转换为十六进制并填充至四位字符串
        hex_dataset_id = hex(dataset_id)[2:]  # 转换为十六进制并去除'0x'
        hex_dataset_id_padded = hex_dataset_id.zfill(4)  # 用0填充至四位

        # 获取设置中的基础路径
        base_path = settings.export.base_path

        # 创建以此字符串命名的文件夹
        folder_path = os.path.join(base_path, hex_dataset_id_padded)
        os.makedirs(folder_path, exist_ok=True)  # 创建文件夹，如果已存在则不报错

        # 判断 original_format 是否为 parquet
        if new_text_meta.original_format.lower() == 'parquet':
            save_parquet_to_jsonl(item.dataset_path, folder_path, item.text_key, settings)


        end_at = time.time()
        time_length = (end_at - start_at)
        response_data = {"resultCode": '00', "resultMessage": "操作成功!",
                         "timeConsuming": "{:.2f}秒".format(time_length),
                         "result": 'done'}
    except SQLAlchemyError as e:
        session.rollback()
        logger.info(traceback.format_exc())
        response_data = {"resultCode": "01", "resultMessage": "数据库操作失败",
                         "reason": str(e)}
        raise HTTPException(status_code=500, detail=response_data)
    except Exception as e:
        session.rollback()
        logger.info(traceback.format_exc())
        response_data = {"resultCode": "02", "resultMessage": "失败",
                         "reason": str(e)}
        raise HTTPException(status_code=500, detail=response_data)
    finally:
        session.close()
    return response_data




if __name__ == "__main__":
    main()
    uvicorn.run(app='main:app', host=settings.server.host, port=settings.server.port,
                workers=settings.server.workers, log_level='info', loop="asyncio")