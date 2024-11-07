# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Processing large data for pretraining."""
import argparse
import math
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import gzip
import glob
import torch
import numpy as np
import multiprocessing
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

#from tokenizer import build_tokenizer
#import indexed_dataset


from plugins.megatron.my_tokenizer import build_tokenizer
#import indexed_dataset
from plugins.megatron.indexed_dataset import IndexedDatasetBuilder, DType
import logging

logging.basicConfig(filename='error.log', level=logging.ERROR)
MAX_PROCESSES = 5

# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text #返回原text


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            if os.environ.get("NLTK_DATA"):
                library = os.path.join(os.environ.get("NLTK_DATA"), "tokenizers", "punkt", f"{self.args.lang}.pickle")
                url = f"file:{library}"
            else:
                library = os.path.join("tokenizers", "punkt", f"{self.args.lang}.pickle")
                url = f"nltk:{library}"
            splitter = nltk.load(url)
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text = splitter._params,
                    lang_vars = CustomLanguageVars())
            else:
                Encoder.splitter = splitter

        else:
            Encoder.splitter = IdentitySplitter()

    def split(self, json_line):
        data = json.loads(json_line)
        output = {}
        for key in self.args.json_keys:
            text = data[key]
            max_len = 1000000
            tokens_list = [Encoder.splitter.tokenize(text[i:i+max_len]) for i in range(0, len(text), max_len)] # 没有tokenizer化，而只是处理长度，可能还会进行分句，但不会ID化（输出是[['《王的中殿之凤舞之阙》是连载于红袖添香网的网络小说，作者是流莫星。 都市言情。都市言情 他是朝鲜的王,她是他的妃,在一次的相遇他们在一起,意味可以这样幸福下去,没想到一次...他,从此他便的更冷酷。。。。。多年后,即时再相遇,俩人也成了陌生人。。。。。敬请关注']]）
            output[key] = [tokens for partial in tokens_list for tokens in partial] # 输出是{'text': ['《王的中殿之凤舞之阙》是连载于红袖添香网的网络小说，作者是流莫星。 都市言情。都市言情 他是朝鲜的王,她是他的妃,在一次的相遇他们在一起,意味可以这样幸福下去,没想到一次...他,从此他便的更冷酷。。。。。多年后,即时再相遇,俩人也成了陌生人。。。。。敬请关注']}
        return json.dumps(output), len(json_line) #  输出是json格式{"text": ["\u300a\u7384\u7687\u964d\u4e16\u3

    def encode(self, json_line):
        try:
            data = json.loads(json_line)
        except json.JSONDecodeError as e:
            # 记录错误信息（可选）
            logging.error(f"JSON decode error: {e} - Line: {json_line}")
            return None, None, len(json_line)
        ids = {}
        lens = {}
        for key in self.args.json_keys:
            text = data.get(key, "")
            if isinstance(text, list):
                sentences = text
            else:
                sentences = [text]
            doc_ids = []
            sentence_lens = []
            for sentence in sentences:
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.extend(sentence_ids)
                    sentence_lens.append(len(sentence_ids))
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids.append(Encoder.tokenizer.eod)
                sentence_lens[-1] += 1
            ids[key] = doc_ids
            lens[key] = sentence_lens
        return ids, lens, len(json_line)


class Partition(object):
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers

    def print_processing_stats(self, count, proc_start, total_bytes_processed):
        if count % self.args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {count} documents",
                  f"({count/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    def split_sentences(self, file_name):
        input_file_name, output_file_name = file_name # _ss就是sentence_split后的输出file name
        print("Opening", input_file_name)
        fin = open(input_file_name, 'r', encoding='utf-8')
        fout = open(output_file_name, 'w')

        encoder = Encoder(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        split_docs = pool.imap(encoder.split, fin, 32)
        pool.close()

        proc_start = time.time()
        total_bytes_processed = 0
        for i, (doc, bytes_processed) in enumerate(split_docs, start=1):
            total_bytes_processed += bytes_processed
            fout.write(doc + "\n")
            self.print_processing_stats(i, proc_start, total_bytes_processed)

        pool.join()
        fin.close()
        fout.close()

        if self.args.partitions != 1:
            os.remove(input_file_name)  # 删除输入文件
            print(f"Deleted: {input_file_name}")


    def process_json_file(self, file_name):
        input_file_name, output_prefix = file_name
        print("Opening", input_file_name)
        fin = open(input_file_name, 'r', encoding='utf-8')

        startup_start = time.time()
        encoder = Encoder(self.args)
        tokenizer = build_tokenizer(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, fin, 32)
        pool.close()

        level = "document"
        if self.args.split_sentences:
            level = "sentence"

        output_bin_files = {}
        output_idx_files = {}
        builders = {}

        for key in self.args.json_keys:
            output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix,
                                                        key, level)
            output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix,
                                                        key, level)
            builders[key] = IndexedDatasetBuilder(
                output_bin_files[key],
                dtype=DType.optimal_dtype(tokenizer.vocab_size),
            )

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)
        for i, (doc, sentence_lens, bytes_processed) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            if doc is None:
                # 如果 doc 为 None，表示该行有问题，跳过
                continue
            for key in doc.keys():
                builders[key].add_document(doc[key], sentence_lens[key])
            self.print_processing_stats(i, proc_start, total_bytes_processed)

        pool.join()
        fin.close()
        for key in builders:
            builders[key].finalize(output_idx_files[key])
        if self.args.partitions != 1 or self.args.split_sentences:
            os.remove(input_file_name)
            print(f"Deleted: {input_file_name}")


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer', 'Llama2Tokenizer',
                                "RWKVTokenizer", "HFTokenizer",
                                'NullTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--tokenizer-model', type=str, default=None,
                       help='YTTM tokenizer model.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--vocab-size', default=786,
                       help='size of vocab for use with NullTokenizer')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help=('Number of worker processes to launch.'
                             'A good default for fast pre-processing '
                             'is: (workers * partitions) = available CPU cores.'))
    group.add_argument('--partitions', type=int, default=1,
                        help='Number of file partitions')
    group.add_argument('--max-processes', type=int, default=6,
                        help='Maximum number of file processes')
    group.add_argument('--merge-partitions', action='store_true',
                        help='Merge partition files into a single output.')
    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval between progress updates')
    group.add_argument('--keep-sequential-samples', action='store_true',
                       help='Ensure ordering of samples in .jsonl files is '
                            'preserved when using partitions>1.')
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert') and not args.split_sentences:
        print("Are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


def get_file_name(args, file_id):
    file_name, extension = os.path.splitext(args.input) # 这块和glob冲突了，因为通配符被当作了文件名，所以不能使用glob里面带通配符，不过问题不大，只是中间的partition文件名字有问题，输出没问题
    if not extension:
        file_name += "/temp"
        extension = ".jsonl"
    input_file_name = file_name + "_" + str(file_id) + extension
    sentence_split_file = file_name + "_ss_" + str(file_id) + extension
    output_prefix = args.output_prefix + "_" + str(file_id)
    file_names = {
        'partition': input_file_name,
        'sentence_split': sentence_split_file,
        'output_prefix': output_prefix}
    return file_names


def check_files_exist(in_ss_out_names, key, num_partitions):
    for i in range(num_partitions):
        if not os.path.exists(in_ss_out_names[i][key]):
            return False
    return True

def get_input_files(input_path):
    """根据输入路径返回文件列表，包括多层目录中的文件"""
    if not os.path.exists(input_path):
        raise ValueError("The provided input path does not exist.")

    if os.path.isdir(input_path):
        file_list = []
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith('.jsonl'):  # 使用endswith并忽略大小写
                    file_list.append(os.path.join(root, file))
        return file_list
    elif os.path.isfile(input_path) and input_path.lower().endswith('.jsonl'):
        return [input_path]  # 确保单个文件也是.jsonl文件
    else:
        raise ValueError("The input path is a file but does not end with .jsonl.")

def merge_files(args, in_ss_out_names):
    level = "document" if not args.split_sentences else "sentence"
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    tokenizer = build_tokenizer(args)

    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                      key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                      key, level)
        builders[key] = IndexedDatasetBuilder(
            output_bin_files[key],
            dtype= DType.optimal_dtype(tokenizer.vocab_size),
            # dtype=np.uint16
        )


        for name in in_ss_out_names:
            parition_output_prefix = name['output_prefix']
            full_partition_output_prefix = "{}_{}_{}".format(parition_output_prefix,
                                                             key, level)                                              
            builders[key].add_index(full_partition_output_prefix)
        builders[key].finalize(output_idx_files[key])

        # 删除分区文件
        for name in in_ss_out_names:
            try:
                partition_bin = f"{name['output_prefix']}_{key}_{level}.bin"
                partition_idx = f"{name['output_prefix']}_{key}_{level}.idx"
                if os.path.exists(partition_bin):
                    os.remove(partition_bin)
                if os.path.exists(partition_idx):
                    os.remove(partition_idx)
                print(f"Deleted: {partition_bin}, {partition_idx}")
            except FileNotFoundError as e:
                print(f"Error deleting files {partition_bin} or {partition_idx}: {str(e)}")

def manage_processes(task_func, task_args_list, max_processes):
    """管理进程的函数，限制最大并发进程数"""
    active_processes = []
    while task_args_list or active_processes:
        # 如果有空闲进程槽且还有任务未启动，则启动新的进程
        while task_args_list and len(active_processes) < max_processes:
            args = task_args_list.pop(0)  # 获取一组待处理参数
            p = multiprocessing.Process(target=task_func, args=(args,))
            p.start()
            active_processes.append(p)
        
        # 检查已启动的进程是否结束，移除已结束的进程
        active_processes = [p for p in active_processes if p.is_alive()]

        time.sleep(0.5)  # 稍微等待一下再次检查

def process_data(input, output_prefix, tokenizer_type, vocab_file, jsonl_keys=["text"], split_sentences=False, keep_newlines=False, tokenizer_model=None, vocab_size=65529, merge_file=None, workers=10, max_processes=10, log_interval=1000, append_eod=True, lang='english', partitions=1, merge_partitions=True, keep_sequential_samples=False):
    args = argparse.Namespace(
        input=input,
        output_prefix=output_prefix,
        tokenizer_type=tokenizer_type,
        vocab_file=vocab_file,
        json_keys=jsonl_keys,
        keep_newlines=keep_newlines,
        split_sentences=split_sentences,
        tokenizer_model=tokenizer_model,
        vocab_size=vocab_size,
        merge_file=merge_file,
        workers=workers,
        max_processes=max_processes,
        log_interval=log_interval,
        append_eod=append_eod,
        lang=lang,
        partitions=partitions,
        merge_partitions=merge_partitions,
        keep_sequential_samples=keep_sequential_samples,
        keep_empty=False,
        rank=1,
        make_vocab_size_divisible_by=128,
        tensor_model_parallel_size=1,
        vocab_extra_ids=0
    )
    if args.tokenizer_type.lower().startswith('bert') and not args.split_sentences:
        print("Are you sure you don't want to split sentences?")
    
    # supported_tokenizers = [
    #     'BertWordPieceLowerCase', 'BertWordPieceCase',
    #     'GPT2BPETokenizer', 'SentencePieceTokenizer',
    #     'GPTSentencePieceTokenizer', 'Llama2Tokenizer',
    #     'RWKVTokenizer', 'HFTokenizer', 'NullTokenizer', 'PretrainedFromHF'
    # ]
    # if tokenizer_type not in supported_tokenizers:
    #     print("Only support:", ', '.join(supported_tokenizers))

    if args.split_sentences:
        if nltk_available:
            # nltk.data.find('tokenizers/punkt')
            nltk.download("punkt", download_dir=os.environ.get("NLTK_DATA"))
        else:
            raise Exception(
                "nltk library required for sentence splitting is not available.")

    in_ss_out_names = []
    if args.partitions == 1:
        in_file_names = get_input_files(args.input)
        if len(in_file_names) == 1:
            file_name, extension = os.path.splitext(in_file_names[0])
            sentence_split_file = file_name + "_ss" + extension
            file_names = {
                'partition': in_file_names[0],
                'sentence_split': sentence_split_file,
                'output_prefix': args.output_prefix}
            in_ss_out_names.append(file_names) # 输入的文件
        else:
            for file in in_file_names:
                file_name, extension = os.path.splitext(file)
                sentence_split_file = file_name + "_ss" + extension
                output_prefix = args.output_prefix + "_" + file_name.split('/')[-1]
                file_names = {
                'partition': file,
                'sentence_split': sentence_split_file,
                'output_prefix': output_prefix}
                in_ss_out_names.append(file_names) # 输入的文件

    else:
        # in_file_names = glob.glob(args.input) # 写明通配符 # 匹配当前目录及所有子目录中的 txt 文件 txt_files = glob.glob('**/*.txt', recursive=True)
        in_file_names = get_input_files(args.input)

        # Count total number of lines across .jsonl files 
        if args.keep_sequential_samples:
            total_sample_count = 0
            for filename in in_file_names:
                with open(filename, "r") as fin:
                    for fc, _ in enumerate(fin):
                        pass
                total_sample_count += (fc + 1)
            partition_size = math.ceil(total_sample_count / args.partitions)

        # create .jsonl parition files
        for idx in range(args.partitions):
            in_ss_out_name = get_file_name(args, idx)
            in_ss_out_names.append(in_ss_out_name)

        # check to see if paritions were already created
        partitions_present = check_files_exist(in_ss_out_names, 'partition', args.partitions)

        # check to see if paritions with split sentences already created
        split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', args.partitions)

        if not partitions_present and not split_sentences_present:
            # populate .jsonl partition files from parent files
            partitioned_input_files = []
            for idx in range(args.partitions):
                partitioned_input_file = open(in_ss_out_names[idx]['partition'], 'w')
                partitioned_input_files.append(partitioned_input_file) # partition文件句柄的列表

            index = 0
            if args.keep_sequential_samples: line_count = 0
            for in_file_name in in_file_names: # 父jsonl文件
                # support for gzip files
                if in_file_name.endswith(".gz"):
                    fin = gzip.open(in_file_name, 'rt')
                else:
                    fin = open(in_file_name, 'r', encoding='utf-8') # 打开父jsonl文件

                for line in fin:
                    partitioned_input_files[index].write(line) # 写入其中一个子partition文件
                    if args.keep_sequential_samples:
                        line_count += 1
                        if line_count % partition_size == 0: # 若sequential，写满一个子文件再写另一个
                            index += 1
                    else:
                        index = (index + 1)%args.partitions # 否则多个子partition文件换着写

                fin.close()

            for idx in range(args.partitions):
                partitioned_input_files[idx].close()

    # 首先，按照原来的思路，每个文件都会独立开一个进程，然后每个进程都会有args.workers//args.partitions子进程
    # 也就意味着，一共有file_num * args.workers//args.partitions 个进程
    # 如果args.partitions=1，则更大
    # 所以现在指定文件的独立进程数，然后一共有max * args.workers//args.partitions 个进程
    assert args.workers % args.partitions == 0
    partition = Partition(args, args.workers//args.partitions) 

    # check to see if paritions with split sentences already created
    split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', args.partitions)

    # split sentences in partition files
    if args.split_sentences and not split_sentences_present:
        task_args_list = [(name['partition'], name['sentence_split']) for name in in_ss_out_names]
        manage_processes(partition.split_sentences, task_args_list, args.max_processes)
        # processes = []
        # for name in in_ss_out_names:
        #     p = multiprocessing.Process(target=partition.split_sentences,
        #                                 args=((name['partition'], name['sentence_split']),))
        #     p.start()
        #     processes.append(p)

        # for p in processes:
        #     p.join()

        # if args.partitions == 1: # 若是单一文件则结束，若不分区且split，则只到split _ss结束，不encode，感觉不对，所以注释掉
        #     return


    # encode partition files in parallel
    input_key = 'sentence_split' if args.split_sentences else 'partition'
    task_args_list = [(name[input_key], name['output_prefix']) for name in in_ss_out_names]
    manage_processes(partition.process_json_file, task_args_list, args.max_processes)
    # processes = []
    # input_key = 'sentence_split' if args.split_sentences else 'partition' 
    # for name in in_ss_out_names: # # 每个输入的文件都有一个独立的进程管理
    #     p = multiprocessing.Process(target=partition.process_json_file,
    #                                 args=((name[input_key], name['output_prefix']),))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()

    if len(in_file_names) == 1 and partitions == 1:
        return

    # merge bin/idx partitions
    if args.merge_partitions:
        merge_files(args, in_ss_out_names)



def main():
    args = get_args()

    if args.split_sentences:
        if nltk_available:
            # nltk.data.find('tokenizers/punkt')
            nltk.download("punkt", download_dir=os.environ.get("NLTK_DATA"))
        else:
            raise Exception(
                "nltk library required for sentence splitting is not available.")

    in_ss_out_names = []
    if args.partitions == 1:
        in_file_names = get_input_files(args.input)
        if len(in_file_names) == 1:
            file_name, extension = os.path.splitext(args.input)
            sentence_split_file = file_name + "_ss" + extension
            file_names = {
                'partition': args.input,
                'sentence_split': sentence_split_file,
                'output_prefix': args.output_prefix}
            in_ss_out_names.append(file_names) # 输入的文件
        else:
            for file in in_file_names:
                file_name, extension = os.path.splitext(file)
                sentence_split_file = file_name + "_ss" + extension
                output_prefix = args.output_prefix + "_" + file_name.split('/')[-1]
                file_names = {
                'partition': file,
                'sentence_split': sentence_split_file,
                'output_prefix': output_prefix}
                in_ss_out_names.append(file_names) # 输入的文件

    else:
        # in_file_names = glob.glob(args.input) # 写明通配符 # 匹配当前目录及所有子目录中的 txt 文件 txt_files = glob.glob('**/*.txt', recursive=True)
        in_file_names = get_input_files(args.input)

        # Count total number of lines across .jsonl files 
        if args.keep_sequential_samples:
            total_sample_count = 0
            for filename in in_file_names:
                with open(filename, "r") as fin:
                    for fc, _ in enumerate(fin):
                        pass
                total_sample_count += (fc + 1)
            partition_size = math.ceil(total_sample_count / args.partitions)

        # create .jsonl parition files
        for idx in range(args.partitions):
            in_ss_out_name = get_file_name(args, idx)
            in_ss_out_names.append(in_ss_out_name)

        # check to see if paritions were already created
        partitions_present = check_files_exist(in_ss_out_names, 'partition', args.partitions)

        # check to see if paritions with split sentences already created
        split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', args.partitions)

        if not partitions_present and not split_sentences_present:
            # populate .jsonl partition files from parent files
            partitioned_input_files = []
            for idx in range(args.partitions):
                partitioned_input_file = open(in_ss_out_names[idx]['partition'], 'w')
                partitioned_input_files.append(partitioned_input_file) # partition文件句柄的列表

            index = 0
            if args.keep_sequential_samples: line_count = 0
            for in_file_name in in_file_names: # 父jsonl文件
                # support for gzip files
                if in_file_name.endswith(".gz"):
                    fin = gzip.open(in_file_name, 'rt')
                else:
                    fin = open(in_file_name, 'r', encoding='utf-8') # 打开父jsonl文件

                for line in fin:
                    partitioned_input_files[index].write(line) # 写入其中一个子partition文件
                    if args.keep_sequential_samples:
                        line_count += 1
                        if line_count % partition_size == 0: # 若sequential，写满一个子文件再写另一个
                            index += 1
                    else:
                        index = (index + 1)%args.partitions # 否则多个子partition文件换着写

                fin.close()

            for idx in range(args.partitions):
                partitioned_input_files[idx].close()

    assert args.workers % args.partitions == 0
    partition = Partition(args, args.workers//args.partitions)

    # check to see if paritions with split sentences already created
    split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', args.partitions)

    # split sentences in partition files
    if args.split_sentences and not split_sentences_present:
        task_args_list = [(name['partition'], name['sentence_split']) for name in in_ss_out_names]
        manage_processes(partition.split_sentences, task_args_list, args.max_processes)
        # processes = []
        # for name in in_ss_out_names:
        #     p = multiprocessing.Process(target=partition.split_sentences,
        #                                 args=((name['partition'], name['sentence_split']),))
        #     p.start()
        #     processes.append(p)

        # for p in processes:
        #     p.join()

        # if args.partitions == 1: # 若是单一文件则结束，若不分区且split，则只到split _ss结束，不encode，感觉不对，所以注释掉
        #     return


    # encode partition files in parallel
    input_key = 'sentence_split' if args.split_sentences else 'partition'
    task_args_list = [(name[input_key], name['output_prefix']) for name in in_ss_out_names]
    manage_processes(partition.process_json_file, task_args_list, args.max_processes)
    # processes = []
    # input_key = 'sentence_split' if args.split_sentences else 'partition' 
    # for name in in_ss_out_names: # # 每个输入的文件都有一个独立的进程管理
    #     p = multiprocessing.Process(target=partition.process_json_file,
    #                                 args=((name[input_key], name['output_prefix']),))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()

    if len(in_file_names) == 1:
        return

    # merge bin/idx partitions
    if args.merge_partitions:
        merge_files(args, in_ss_out_names)
    # level = "document"
    # if args.split_sentences:
    #     level = "sentence"

    # output_bin_files = {}
    # output_idx_files = {}
    # builders = {}
    # tokenizer = build_tokenizer(args)

    # for key in args.json_keys:
    #     output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
    #                                                   key, level)
    #     output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
    #                                                   key, level)
    #     builders[key] = indexed_dataset.IndexedDatasetBuilder(
    #         output_bin_files[key],
    #         dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
    #     )

    #     for name in in_ss_out_names:
    #         parition_output_prefix = name['output_prefix']
    #         full_partition_output_prefix = "{}_{}_{}".format(parition_output_prefix,
    #                                                          key, level)
    #         builders[key].add_index(full_partition_output_prefix)
    #     builders[key].finalize(output_idx_files[key])


if __name__ == '__main__':
    # main()
    start_time = time.time()
    process_data("/mnt/data/Lake_Data/20241028192429","./data/test","RWKVTokenizer",vocab_file="/mnt/data/ML_folder/Infinite_Lake/plugins/megatron/rwkv_vocab_v20230424.txt")
    end_time = time.time()
    elapsed_time = end_time - start_time  # 计算执行时间
    print(f"The function took {elapsed_time} seconds to complete.")
