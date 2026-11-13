# -*- coding: utf-8 -*-
"""用训练好的模型对任意句子做情感预测（面试演示/录 demo 用）。

用法：
  python predict.py --ckpt runs/full/best.pt
  python predict.py --ckpt runs/full/best.pt --texts "这家店太棒了" "难吃，再也不来了"
"""
import argparse

import torch
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()

from sa.ckpt import read_ckpt
from sa.model import build_model

LABELS = {0: "差评", 1: "好评"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--texts", nargs="*", default=None, help="要预测的句子；不给则用内置演示句子")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    info = read_ckpt(args.ckpt, map_location="cpu")
    model_name = info.get("model_name", "bert-base-chinese")
    tok_name = info.get("tokenizer_name", "bert-base-chinese")

    model = build_model(model_name).to(device)
    model.load_state_dict(info["state_dict"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tok_name)

    texts = args.texts or [
        "这家店服务态度特别好，菜品也很新鲜，下次还来！",
        "等了两个小时才上菜，味道还特别咸，太失望了。",
        "性价比很高，环境干净整洁，推荐给大家。",
        "包装烂得不成样子，送到都洒了，差评。",
        "客服回复很快，问题当天就解决了，体验不错。",
        "用了一周就坏了，售后还找不到人，避雷。",
    ]

    enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).logits
        probs = torch.softmax(logits, dim=-1).cpu()

    print(f"权重: {args.ckpt} | 模型: {model_name}\n")
    for t, pr in zip(texts, probs):
        conf, cls = pr.max(dim=0)
        print(f"[{LABELS[int(cls)]} {conf:.1%}] {t}")


if __name__ == "__main__":
    main()
