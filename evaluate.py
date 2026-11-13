# -*- coding: utf-8 -*-
"""在测试集上评测一个或多个权重，打印可直接粘贴的对照表。

用法：
  python evaluate.py --data_dir data/sample --ckpt runs/_smoke/best.pt
  python evaluate.py --data_dir data/chnsenticorp --ckpt runs/frozen/best.pt runs/full/best.pt
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

# 关掉加载权重时的进度条，否则会和表格挤在一起
hf_logging.disable_progress_bar()

from sa.ckpt import read_ckpt
from sa.data import load_split
from sa.metrics import evaluate
from sa.model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/sample")
    p.add_argument("--ckpt", nargs="+", required=True, help="一个或多个 .pt 权重路径")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=128)
    return p.parse_args()


def display_width(s):
    """中文占 2 列宽，用于对齐表格（否则中文会把表格撑歪）。"""
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def pad(s, width):
    return s + " " * max(0, width - display_width(s))


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 先读第一个权重的元信息，确定分词器
    first = read_ckpt(args.ckpt[0], map_location="cpu")
    tok_name = first.get("tokenizer_name", "bert-base-chinese")

    ids, mask, labels = load_split(Path(args.data_dir) / "test.csv",
                                   AutoTokenizer.from_pretrained(tok_name), args.max_len)
    loader = DataLoader(TensorDataset(ids, mask, labels),
                        batch_size=args.batch_size, shuffle=False)
    print(f"设备: {device} | 测试集 {len(labels)} 条 | 分词器: {tok_name}\n")

    header = [("模型", 34), ("Acc", 8), ("F1", 8), ("参数量(M)", 10)]
    width = sum(w for _, w in header)
    print("".join(pad(h, w) for h, w in header))
    print("-" * width)

    for ck in args.ckpt:
        path = Path(ck)
        info = read_ckpt(path, map_location="cpu")
        model_name = info.get("model_name", "bert-base-chinese")
        model = build_model(model_name).to(device)
        model.load_state_dict(info["state_dict"])

        acc, f1 = evaluate(model, loader, device)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        tag = "冻结编码器" if info.get("freeze") else "全参数微调"
        name = f"{model_name} [{tag}]"
        print("".join([pad(name, 34), pad(f"{acc:.4f}", 8), pad(f"{f1:.4f}", 8), pad(f"{params:.2f}", 10)]))

        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
