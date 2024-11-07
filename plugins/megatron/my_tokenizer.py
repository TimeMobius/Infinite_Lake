# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Megatron tokenizers."""

from abc import ABC
from abc import abstractmethod

from plugins.megatron.megatron_tokenizer import MegatronTokenizer
from tokenizers import Tokenizer
import rwkv_tokenizer

from plugins.megatron.bert_tokenization import FullTokenizer as FullBertTokenizer
from plugins.megatron.gpt2_tokenization import GPT2Tokenizer
from typing import List, Optional, Union, Dict
from transformers import AutoTokenizer

def build_tokenizer(args):
    """Initialize tokenizer."""

    if args.rank == 0:
        print(' > building HFTokenizer tokenizer, '
                'loading tokenizer from pre-trained model', flush=True)

    if args.tokenizer_type == "HFTokenizer":
        if args.vocab_file is None:
            raise ValueError("Missing tokenizer path while building Huggingface tokenizer.")

        hf_tokenizer_kwargs = dict()
        if hasattr(args, "tokenizer_kwargs") and args.tokenizer_kwargs:
            if len(args.tokenizer_kwargs) % 2 != 0:
                raise ValueError("The token name and token value must be entered in pairs.")

            for i in range(0, len(args.tokenizer_kwargs), 2):
                hf_tokenizer_kwargs[args.tokenizer_kwargs[i]] = \
                    args.tokenizer_kwargs[i + 1]

        tokenizer = _HFTokenizer(
            args.vocab_file,
            vocab_extra_ids=args.vocab_extra_ids,
            **hf_tokenizer_kwargs
        )

    elif args.tokenizer_type.lower() == "RWKVTokenizer".lower():
        assert args.vocab_file is not None
        tokenizer = RWKVTokenizer(args.vocab_file, vocab_extra_ids=args.vocab_extra_ids)

    else:
        raise NotImplementedError('{} tokenizer is not implemented.'.format(args.tokenizer_type))

    # Add vocab size (if not already set from a checkpoint).
    if getattr(args, "padded_vocab_size", None) is None:
        args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size, args)

    return tokenizer

class RWKVTokenizer():
    """RWKV Tokenizer"""
    def __init__(self, vocab_file, vocab_extra_ids):
        self.tokenizer = rwkv_tokenizer.RWKVTokenizer()
        self.eod_id = 0

    @property
    def vocab_size(self):
        return 65529

    @property
    def vocab(self):
        return self.tokenizer.encode

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def cls(self):
        return -1

    @property
    def sep(self):
        return -1

    @property
    def pad(self):
        return -1

    @property
    def eod(self):
        return self.eod_id

    @property
    def mask(self):
        return -1

    @property
    def additional_special_tokens_ids(self):
        return None

class _HFTokenizer(MegatronTokenizer):
    """_HFTokenizer for Hf Pretrained model loading."""

    def __init__(self, tokenizer_name_or_path, vocab_extra_ids, **kwargs):
        name = tokenizer_name_or_path
        super().__init__(name)
        hf_tokenizer_kwargs = kwargs
        if vocab_extra_ids > 0:
            hf_tokenizer_kwargs["additional_special_tokens"] = [f"<extra_id_{_id}>" for _id in range(vocab_extra_ids)]

        hf_tokenizer_kwargs["trust_remote_code"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **hf_tokenizer_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.encoder = self.tokenizer.get_vocab()
        self.decoder = {v: k for k, v in self.encoder.items()}

    @property
    def vocab_size(self):
        return len(self.tokenizer)  # vocab_size doesn't contain additional tokens

    @property
    def vocab(self):
        return {
            **{special_token: self.tokenizer.convert_tokens_to_ids(special_token)
               for special_token in self.tokenizer.additional_special_tokens},
            **self.tokenizer.vocab,
        }

    @property
    def inv_vocab(self):
        return {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eos

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def cls(self):
        candidate = self.tokenizer.cls_token_id
        return self._check_token_candidate(candidate)

    @property
    def sep(self):
        candidate = self.tokenizer.sep_token_id
        return self._check_token_candidate(candidate)

    @property
    def pad(self):
        candidate = self.tokenizer.pad_token_id

        # just use eos_token_id if pad_token_id is not available, it is reasonable
        # maybe add a new token, and resize embedding layer is better
        if candidate is None:
            candidate = self.tokenizer.eos_token_id
        return self._check_token_candidate(candidate)

    @property
    def mask(self):
        candidate = self.tokenizer.mask_token_id
        return self._check_token_candidate(candidate)

    @property
    def bos(self):
        raise NotImplementedError("Missing <bos>")

    @property
    def eos(self):
        candidate = self.tokenizer.eos_token_id
        return self._check_token_candidate(candidate)

    @property
    def additional_special_tokens_ids(self):
        """ All the additional special tokens you may want to use (list of strings)."""
        return self.tokenizer.additional_special_tokens_ids

    @staticmethod
    def _check_token_candidate(candidate):
        if candidate is None:
            raise AttributeError("Token doesn't exist")
        return candidate

def _vocab_size_with_padding(orig_vocab_size, args):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * \
        args.tensor_model_parallel_size
    while (after % multiple) != 0:
        after += 1
    if args.rank == 0:
        print(' > padded vocab (size: {}) with {} dummy tokens '
              '(new size: {})'.format(
                  orig_vocab_size, after - orig_vocab_size, after), flush=True)
    return after