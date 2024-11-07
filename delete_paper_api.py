from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union, List
import pymysql
import uvicorn
import json
import os
from qdrant_client import QdrantClient, models
from config_plugins import QDRANT_CONFIG, COLLECTION_NAME
from config_plugins import db_config

app = FastAPI()


# Qdrant 配置
qdrant_host = QDRANT_CONFIG['host']
qdrant_port = QDRANT_CONFIG['port']
collection_name = COLLECTION_NAME

class DeleteRequest(BaseModel):
    title: Union[str, List[str]]
    dataset_id: Union[str, List[str]]

    def delete_matching_records(self):
        connection = pymysql.connect(**db_config)
        cursor = connection.cursor()
        deleted_line_ids = []
        titles = self.title if isinstance(self.title, list) else [self.title]
        dataset_ids = self.dataset_id.split('_') if isinstance(self.dataset_id, str) else self.dataset_id

        for t in titles:
            for d_id in dataset_ids:
                query = "SELECT line_id FROM text_line WHERE title = %s"
                cursor.execute(query, (t,))
                result = cursor.fetchall()
                for (line_id,) in result:
                    line_id_prefix = line_id[:4]
                    if line_id_prefix in d_id:
                        deleted_line_ids.append(line_id)
                        delete_query = "DELETE FROM text_line WHERE line_id = %s"
                        cursor.execute(delete_query, (line_id,))
                        connection.commit()
        cursor.close()
        connection.close()
        return deleted_line_ids

    def delete_records(self, deleted_line_ids):
        base_url = "path/Lake_Data"
        dataset_urls = [os.path.join(base_url, id[:4]) for id in deleted_line_ids]
        for dataset_url in dataset_urls:
            for id in deleted_line_ids:
                jsonl_url = os.path.join(dataset_url, f"{id[:8]}.jsonl")
                if os.path.exists(jsonl_url):
                    with open(jsonl_url, 'r') as file:
                        data = [json.loads(line.strip()) for line in file]
                    data = [record for record in data if record['id'] not in deleted_line_ids]
                    with open(jsonl_url, 'w') as file:
                        for record in data:
                            file.write(json.dumps(record) + '\n')

    def delete_vectors_by_line_ids(self, deleted_line_ids):
        client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=100000000)
        scroll_filter = {"must": [{"key": "line_id", "match": {"any": deleted_line_ids}}]}
        scroll_result, next_page = client.scroll(collection_name=collection_name, scroll_filter=scroll_filter, limit=100)
        while scroll_result:
            for vector in scroll_result:
                point_id = vector.id
                payload = vector.payload
                line_id = payload.get('line_id')
                if line_id in deleted_line_ids:
                    delete_filter = models.Filter(
                        must=[models.FieldCondition(key="line_id", match=models.MatchValue(value=line_id))]
                    )
                    client.delete(collection_name=collection_name, points_selector=models.FilterSelector(filter=delete_filter))
            if not next_page:
                break
            scroll_result, next_page = client.scroll(collection_name=collection_name, scroll_filter=scroll_filter, limit=100, offset=next_page)
        return "finished"

@app.post("/delete_data/")
def delete_data(request: DeleteRequest):
    try:
        deleted_line_ids = request.delete_matching_records()
        request.delete_records(deleted_line_ids)
        result = request.delete_vectors_by_line_ids(deleted_line_ids)
        #return {"deleted_line_ids": deleted_line_ids, "result": result}
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app='delete_paper_api:app', host="0.0.0.0", port=5005)