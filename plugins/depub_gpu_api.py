from __future__ import division
from qdrant_client import QdrantClient, async_qdrant_client
from qdrant_client.http import models
from chunking import get_chunks
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from FlagEmbedding import BGEM3FlagModel
import numpy as np
import uvicorn
import asyncio
from qdrant_client.models import PointStruct
from uuid import uuid4
import uuid
import json
import time
import os
import requests
import hashlib
from config_plugins import QDRANT_CONFIG, EMBEDDING_MODEL_CONFIG, COLLECTION_NAME


# Initialize Qdrant client

client = QdrantClient(
    host=QDRANT_CONFIG['host'],
    port=QDRANT_CONFIG['port'],
    timeout=QDRANT_CONFIG['timeout']
)


collection_name = COLLECTION_NAME


# 全局变量
model = None


# 设置TOKENIZERS_PARALLELISM
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI()


class Import_Text_Item(BaseModel):
    text_list: list


class Import_qdrant_Item(BaseModel):
    text_list: list
    lineId_list: list

def load_model():
    global model
    modelPath = EMBEDDING_MODEL_CONFIG['model_path']
    model = BGEM3FlagModel(modelPath, use_fp16=True)
# 在程序启动时调用此函数
load_model()


async def get_embedding(text):
    # 使用全局变量
    global model
    chunk_list = get_chunks(text)
    embeddings_data = model.encode(chunk_list, batch_size=4096, max_length=8192)['dense_vecs']
    # Release GPU memory
    torch.cuda.empty_cache()
    embeddings_data_list = embeddings_data.tolist()
    embeddings_normalized = np.apply_along_axis(l2_normalize, axis=1, arr=embeddings_data_list)
    return embeddings_normalized


# 归一化，向量的长度缩放到1
# L2范数归一化（将向量长度缩放到1）
def l2_normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


@app.post("/api/data/depublication")
async def similar_calculator(input_data: Import_Text_Item):
    # def similar_calculator(input_data):
    # 文本切块
    text_block_info = []
    for text_index, text in enumerate(input_data.text_list):
        blocks = get_chunks(text)
        for block_index, block in enumerate(blocks):
            text_block_info.append({'text_index': text_index, 'block_index': block_index, 'block_content': block})

    #print(text_block_info)

    # 生成头尾块列表
    first_last_info = []
    # 遍历数据列表
    for item in text_block_info:
        # 获取同一文本的所有块
        blocks_same_text = [d for d in text_block_info if d['text_index'] == item['text_index']]
        # 获取同一文本的最大block_index
        max_block_index = max([d['block_index'] for d in blocks_same_text])

        # 检查当前块的block_index是否为0或者最大值
        if item['block_index'] == 0 or item['block_index'] == max_block_index:
            # 检查该块是否已经在结果列表中，避免重复添加
            if item not in first_last_info:
                first_last_info.append(item)

    # print(first_last_info)
    query_vectors_with_info = await qdrant_search(first_last_info)
    # print(query_vectors_with_info)
    # 先确保每个元素都有 'vector_search' 键，并设置默认值为 []
    for item in query_vectors_with_info:
        item.setdefault("vector_search", [])

    sum_dict = defaultdict(list)
    # 遍历数据，将相同text_index的vector_search值加入对应的列表中
    for item in query_vectors_with_info:
        sum_dict[item["text_index"]].extend(item["vector_search"])

    # 将列表中的空字符串去除
    for text_index in sum_dict:
        sum_dict[text_index] = [x for x in sum_dict[text_index] if x]

    first_last_result_list = [{"text_index": key, "vector_search": value} for key, value in sum_dict.items()]
    # print(first_last_result_list)

    # 向量碰撞
    updated_result_list = []  # 用于存储text_line和insert_to_qdrant关系
    insert_to_qdrant_by_text_index = defaultdict(lambda: None)

    for item in first_last_result_list:
        vector_search = item['vector_search']
        # print(len(vector_search))
        if len(vector_search) == 0:
            item['insert_to_qdrant'] = True
            insert_to_qdrant_by_text_index[item['text_index']] = True
        elif len(vector_search) >= 1:
            # 如果相同text_line下的block_index小于2，item['insert_to_qdrant'] = True
            blocks_by_text_index = defaultdict(list)
            for block_item in text_block_info:
                blocks_by_text_index[block_item['text_index']].append(block_item['block_index'])


            text_index = item['text_index']
            block_indexes = blocks_by_text_index[text_index]
            # 检查同一text_index对应的vector_search的值前16位是否有相同的
            vector_hashes = [hash(bytes(v[:8], 'utf-8')) for v in vector_search]

            if len(set(vector_hashes)) < len(vector_search):
                # insert_to_qdrant_by_text_index[text_index] = False
                insert_to_qdrant_by_text_index[text_index] = False
            elif len(block_indexes) == 1 and len(set(vector_hashes)) == len(vector_search):
                insert_to_qdrant_by_text_index[text_index] = False
            else:
                if 1 < len(block_indexes) <= 2:
                    insert_to_qdrant_by_text_index[text_index] = True
                else:
                    # 取第二个文本块
                    second_info = [{'text_index': text_index, 'block_index': 1, 'block_content': item['block_content']}
                                   for item in text_block_info if
                                   item['text_index'] == text_index and item['block_index'] == 1]

                    second_query_vectors_with_info = await qdrant_search(second_info)

                    #print(second_query_vectors_with_info)

                    for second_block in second_query_vectors_with_info:
                        second_block.setdefault("vector_search", [])

                        # 检查vector_search是否为空列表
                        if all(not vector for vector in second_block['vector_search']):
                            # 如果所有vector_search都是空列表，则设置insert_to_qdrant_by_text_index[text_index]为True
                            insert_to_qdrant_by_text_index[second_block['text_index']] = True
                        else:
                            # 如果vector_search不为空，进行比对
                            text_index = second_block['text_index']
                            second_block_ids = set(
                                map(lambda item: str(item)[:8], second_block['vector_search']))
                            first_block_ids = set(map(lambda item: str(item)[:8],
                                                      [item['vector_search'] for item in first_last_result_list
                                                       if item['text_index'] == text_index]))

                            # 比较second_block_ids和first_block_ids是否有重复元素
                            if second_block_ids.intersection(first_block_ids):
                                # 如果有重复元素，则设置insert_to_qdrant_by_text_index[text_index]为False
                                insert_to_qdrant_by_text_index[text_index] = False
                            else:
                                # 否则，设置为True
                                insert_to_qdrant_by_text_index[text_index] = True

    # 将结果整合到 updated_result_list
    for text_index, insert_to_qdrant in insert_to_qdrant_by_text_index.items():
        updated_result_list.append({'text_index': text_index, 'insert_to_qdrant': insert_to_qdrant})
    # print(updated_result_list)
    # 按照 text_index 升序排序 updated_result_list
    sorted_result_list = sorted(updated_result_list, key=lambda x: x['text_index'])
    # 创建一个新的列表存储 insert_to_qdrant 的值
    stateCodes = [item['insert_to_qdrant'] for item in sorted_result_list]
    # print(stateCodes)
    return {"message": stateCodes}


async def qdrant_search(first_last_info):
    # 生成向量：将每个block——content生成向量
    first_last_vector = []
    for block in first_last_info:
        # 生成向量
        block_vector = await get_embedding(block['block_content'])
        if block_vector is not None:
            first_last_vector.append({
                'text_index': block['text_index'],
                'block_index': block['block_index'],
                'block_content': block['block_content'],
                'block_vector': block_vector
            })
    # print(first_last_vector)
    # 获取向量检索结果：将每个block——content生成向量
    query_vectors_with_info = [
        {
            'text_index': block['text_index'],
            'block_index': block['block_index'],
            'block_content': block['block_content'],
            'block_vector': block['block_vector'][0]
        }
        for block in first_last_vector
    ]
    # print(query_vectors_with_info)
    search_requests = []
    for vector in query_vectors_with_info:
        query = models.SearchRequest(
            vector=vector['block_vector'],
            limit=5,
            score_threshold=0.87,
            params=models.SearchParams(quantization=models.QuantizationSearchParams(rescore=False)))
        search_requests.append(query)
    # 分批次处理搜索请求
    batch_size = 200  # 每批次的请求数量
    batched_requests = [search_requests[i:i + batch_size] for i in range(0, len(search_requests), batch_size)]

    for batch in batched_requests:
        result = client.search_batch(collection_name=f'{collection_name}', requests=batch, timeout=1000000)
        for idx, res in enumerate(result):
            ids = [point.id for point in res]
            query_vectors_with_info[idx]['vector_search'] = ids
    #print(query_vectors_with_info)
    return query_vectors_with_info


@app.post("/api/data/qdrant_store")
async def qdrant_store_data(input_data: Import_qdrant_Item):
#async def qdrant_store_data(input_data):
    # 创建一个异步任务列表
    tasks = [get_embedding(text) for text in input_data.text_list]
    # 并行执行所有任务并获取结果
    results = await asyncio.gather(*tasks)

    all_points = []
    for text, line_id, embedding_query in zip(input_data.text_list, input_data.lineId_list, results):
        fixed_uuid_str = generate_fixed_uuid(text)[:-8]
        for idx, vector in enumerate(embedding_query):
            point_id = f'{fixed_uuid_str}{idx:08d}'
            point = PointStruct(id=point_id, vector=vector, payload={"line_id": line_id})
            all_points.append(point)

    # 批量上传点
    batch_size = 10240
    for i in range(0, len(all_points), batch_size):
        batch_points = all_points[i:i + batch_size]
        #print(batch_points)
        client.upload_points(
            collection_name=f"{collection_name}",
            points=batch_points,
            batch_size=batch_size,
            parallel=2
        )

    return {"message": "数据成功存储到 Qdrant 中。"}


def generate_fixed_uuid(text):
    hash_object = hashlib.md5(text.encode())
    return hash_object.hexdigest()


if __name__ == "__main__":
    uvicorn.run(app='depub_gpu_api:app', host='0.0.0.0', port=5003, workers=2, loop="asyncio")
