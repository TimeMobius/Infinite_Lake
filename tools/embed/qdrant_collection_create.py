from qdrant_client import QdrantClient
from qdrant_client.http import models
from config_plugins import QDRANT_CONFIG, COLLECTION_NAME

def save_qdrant_url():
    client = QdrantClient(
        host=QDRANT_CONFIG['host'],
        port=QDRANT_CONFIG['port']
    )
    collection_name = COLLECTION_NAME

    try:
        table_collection = client.get_collection(collection_name=f"{collection_name}")
        print(table_collection)
        print(f'集合已存在:{collection_name}')

    except:
        print(f'创建新集合：{collection_name}')


        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE, on_disk=True),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    always_ram=True,
                ),
            ),
        )


def updata_qdrant_url():
    client = QdrantClient(
        host=QDRANT_CONFIG['host'],
        port=QDRANT_CONFIG['port']
    )
    collection_name = COLLECTION_NAME
    client.update_collection(
        collection_name=collection_name,
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=100, default_segment_number=8),
        hnsw_config=models.HnswConfigDiff(full_scan_threshold=100),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                always_ram=True,
            ),
        ),
    )


if __name__ == '__main__':
    save_qdrant_url()
    #updata_qdrant_url()

