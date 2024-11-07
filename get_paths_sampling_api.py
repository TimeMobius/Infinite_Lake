from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Union, List
import os
import shutil
from datetime import datetime
import uvicorn

# 引入预处理和混合数据的工具函数
# from plugins.json2binidx_tool.tools.preprocess_data_files import process_data
from plugins.megatron.preprocess_data_new import process_data
from plugins.data_mixture_radom import mix_jsonl_files
from plugins.data_mixture_select import main


app = FastAPI()

class ExportRequest(BaseModel):
    target_dataType: str
    target_sampling_method: Optional[str] = None
    ids: Optional[str] = None
    total_size_mb: Optional[float] = None
    file_size_mb: Optional[float] = None
    filter_field: Optional[Union[str, List[str]]] = None
    filter_value: Optional[Union[str, List[str]]] = None
    tokenizer_type: Optional[str] = None
    vocab_file: Optional[str] = None

def create_timestamped_folder(export_path: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    folder_path = os.path.join(export_path, timestamp)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def gather_jsonl_files(ids_list: List[str], base_path: str, folder_path: str):
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

@app.post("/export_paths/")
def get_paths(request: ExportRequest):
    # 定义基本路径
    base_path = "/mnt/data/Lake_Data"
    export_path = "/mnt/data/Exp_Data"
    folder_path = create_timestamped_folder(export_path)
    # 删除 0 字节文件的函数
    def remove_zero_byte_files(path):
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
                os.remove(file_path)
        # 检查是否文件夹为空
        if not os.listdir(path):
            return None
        return path

    if request.target_dataType == "jsonl":
        if request.target_sampling_method == "random":
            ids_list = request.ids.split("_")
            gather_jsonl_files(ids_list, base_path, folder_path)
            # 移除 0 字节文件
            folder_path = remove_zero_byte_files(folder_path)
            if folder_path is None:
                return {"jsonl_path": None}
            return {"jsonl_path": folder_path}


        elif request.target_sampling_method == "certain_sampling":
            ids_list = request.ids.split("_")
            input_list = [os.path.join(base_path, id) for id in ids_list]
            mix_jsonl_files(input_list, folder_path, request.total_size_mb, request.file_size_mb)
            # 移除 0 字节文件
            folder_path = remove_zero_byte_files(folder_path)
            if folder_path is None:
                return {"jsonl_path": None}
            return {"jsonl_path": folder_path}

        elif request.target_sampling_method in ["select", "subject_sampling"]:
            main(request.filter_field, request.filter_value, request.total_size_mb, request.file_size_mb, folder_path)
            # 移除 0 字节文件
            folder_path = remove_zero_byte_files(folder_path)
            if folder_path is None:
                return {"jsonl_path": None}
            return {"jsonl_path": folder_path}

    elif request.target_dataType == "binidx":
        if not request.tokenizer_type or not request.vocab_file:
            raise HTTPException(status_code=400, detail="当 target_dataType 为 'binidx' 时，tokenizer_type 和 vocab_file 是必填项")
        if request.target_sampling_method == "random":
            ids_list = request.ids.split("_")
            gather_jsonl_files(ids_list, base_path, folder_path)
            # 移除 0 字节文件
            folder_path = remove_zero_byte_files(folder_path)
            if folder_path is None:
                return {"binidx_path": None}
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_path = os.path.join(export_path, "binidx", timestamp)
            os.makedirs(output_path, exist_ok=True)
            output_prefix = os.path.join(output_path, timestamp)
            process_data(folder_path, output_prefix, request.tokenizer_type, request.vocab_file)
            shutil.rmtree(folder_path)
            return {"binidx_path": output_path}
        elif request.target_sampling_method == "certain_sampling":
            ids_list = request.ids.split("_")
            input_list = [os.path.join(base_path, id) for id in ids_list]
            mix_jsonl_files(input_list, folder_path, request.total_size_mb, request.file_size_mb)
            # 移除 0 字节文件
            folder_path = remove_zero_byte_files(folder_path)
            if folder_path is None:
                return {"binidx_path": None}
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_path = os.path.join(export_path, "binidx", timestamp)
            os.makedirs(output_path, exist_ok=True)
            output_prefix = os.path.join(output_path, timestamp)
            process_data(folder_path, output_prefix, request.tokenizer_type, request.vocab_file)
            shutil.rmtree(folder_path)
            return {"binidx_path": output_path}
        elif request.target_sampling_method in ["select", "subject_sampling"]:
            main(request.filter_field, request.filter_value, request.total_size_mb, request.file_size_mb, folder_path)
            # 移除 0 字节文件
            folder_path = remove_zero_byte_files(folder_path)
            if folder_path is None:
                return {"binidx_path": None}
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_path = os.path.join(export_path, "binidx", timestamp)
            os.makedirs(output_path, exist_ok=True)
            output_prefix = os.path.join(output_path, timestamp)
            process_data(folder_path, output_prefix, request.tokenizer_type, request.vocab_file)
            shutil.rmtree(folder_path)
            return {"binidx_path": output_path}


    else:
        raise HTTPException(status_code=400, detail="不支持的 target_dataType 或 target_sampling_method")

if __name__ == "__main__":
    uvicorn.run(app='get_paths_sampling_api:app', host="0.0.0.0", port=22334)
