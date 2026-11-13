# -*- coding: utf-8 -*-
"""数据读取：CSV -> 模型能吃的张量。

约定 CSV 格式（第一行是表头）：
    text,label
    这家店服务态度很好，下次还来,1
    外卖送了两个小时，面都坨了,0

label 含义：1 = 好评（正面），0 = 差评（负面）。
"""
import csv

import torch


def read_csv(path):
    """读 CSV，返回 (句子列表, 标签列表)。"""
    texts, labels = [], []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            t = (row["text"] or "").strip()
            if t:
                texts.append(t)
                labels.append(int(row["label"]))
    if not texts:
        raise ValueError(f"{path} 里没有读到任何数据，检查表头是否为 text,label")
    return texts, labels


def encode(texts, labels, tokenizer, max_len):
    """把句子列表编码成 input_ids / attention_mask / labels 三个张量。"""
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"], torch.tensor(labels, dtype=torch.long)


def load_split(path, tokenizer, max_len):
    texts, labels = read_csv(path)
    input_ids, attention_mask, labels = encode(texts, labels, tokenizer, max_len)
    return input_ids, attention_mask, labels
