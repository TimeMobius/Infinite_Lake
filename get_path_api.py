from fastapi import FastAPI
from typing import Dict
import uvicorn
import os
import shutil
from typing import Dict, Optional
from datetime import datetime
#from plugins.json2binidx_tool.tools.preprocess_data_files import process_data
from plugins.megatron.preprocess_data_new import process_data

app = FastAPI()

def create_timestamped_folder(export_path: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    folder_path = os.path.join(export_path, timestamp)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def gather_jsonl_files(ids_list, base_path, folder_path):
    for id in ids_list:
        path = os.path.join(base_path, id)
        if os.path.exists(path):
            if os.path.isfile(path):
                destination_path = os.path.join(folder_path, os.path.basename(path))
                shutil.copy(path, destination_path)
            elif os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.endswith(".jsonl"):
                            file_path = os.path.join(root, file)
                            destination_path = os.path.join(folder_path, file)
                            shutil.copy(file_path, destination_path)


@app.get("/export_paths/")
def get_paths(ids: str, target_dataType: str, tokenizer_type: Optional[str] = None, vocab_file: Optional[str] = None):
    if target_dataType == "jsonl":
        ids_list = ids.split("_")

        base_path = "/mnt/data/Lake_Data"
        export_path = "/nfs/dubhe-prod/dataset/datalake"
        folder_path = create_timestamped_folder(export_path)
        gather_jsonl_files(ids_list, base_path, folder_path)
        #real_output_path = os.path.realpath(folder_path)
        #print(real_output_path)


        #return {"jsonl_path": real_output_path}
        return {"jsonl_path": folder_path}

    if target_dataType == "binidx":
        if tokenizer_type is None or vocab_file is None:
            return {"error": "tokenizer_type 和 vocab_file 是必填项当 target_dataType 为 bin_idx 时"}
        ids_list = ids.split("_")

        base_path = "/mnt/data/Lake_Data"
        export_path = "/nfs/dubhe-prod/dataset/datalake"
        folder_path = create_timestamped_folder(export_path)
        gather_jsonl_files(ids_list, base_path, folder_path)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_path = f"{export_path}/binidx/{timestamp}"
        os.makedirs(output_path, exist_ok=True)
        output_prefix = f"{output_path}/{timestamp}"
        process_data(folder_path, output_prefix, tokenizer_type, vocab_file)
        #print("folder_path", output_prefix)
        #real_output_path = os.path.realpath(output_path)
        #print(real_output_path)
        #return {"binidx_path": real_output_path}
        shutil.rmtree(folder_path)
        return {"binidx_path": output_path}


if __name__ == "__main__":
    uvicorn.run(app='get_path_api:app', host="0.0.0.0", port=22334)
