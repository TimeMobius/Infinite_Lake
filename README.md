# Infinite Lake
## 无湖：LLM专用数据框架
本项目设计目标为实现高效的大语言模型预训练及后续流程所需数据集收集、清洗、归档、成集，当前的数据处理流程和工具难以满足大规模数据处理的需求，本架构借鉴了大数据领域的数据湖思路专门为LLM定制相关pipline。
## 框架结构
框架结构如下：
```
Infinite Lake
|-- loader
|-- convert
|-- tools
|-- plugins
|-- preprocess
|-- postprocess
|-- models
|-- utils
|-- logs
|-- main_api.py
|-- main_api_sft.py
|-- main_api_dpo.py
|-- delete_paper_api.py
|-- get_paths_sampling_api.py
```
框架结构说明：
1. loader：数据加载模块，负责从原始数据源加载数据，并转换为.json或者.md标准格式。
2. convert：数据转换模块，对JSON、JSONL、Parquet标准格式数据提供PipeLine处理。
3. tools：工具模块，负责提供数据清洗、数据切分、数据融合、数据分类等功能函数。
4. plugins：插件模块，负责提供数据清洗、数据去重、关键词抽取等插件。
5. preprocess：预处理模块，负责对原始数据进行预处理。
6. postprocess：后处理模块，包括数据采样、数据质量检测等。
7. models：模型模块，负责提供一些模型，如向量模型、语言模型、分类模型等。
8. utils：工具模块，负责提供日志等通用工具函数。
9. logs：日志模块，负责提供一些日志记录功能。
10. main_api.py：pt数据处理主程序。
11. main_api_sft.py：SFT数据处理主程序。
12. main_api_dpo.py：DPO数据处理主程序。
13. delete_paper_api.py：PT数据删除主程序。
14. get_paths_sampling_api.py：数据出湖路径获取主程序。

