# -*- coding: utf-8 -*-
"""checkpoint 读写（兼容 torch 2.6+ 的 weights_only 默认值变化）。

权重里除了 state_dict，还存了模型名、分词器名等元信息，
这样评测时不用手动告诉脚本"这个权重是哪个模型训出来的"。
"""
import torch


def save_ckpt(path, model, meta: dict):
    torch.save({"state_dict": model.state_dict(), **meta}, path)


def read_ckpt(path, map_location="cpu") -> dict:
    """读出完整 checkpoint，返回 dict（含 state_dict 和各元信息）。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        # 老版本 torch 或存了非张量对象时走这里
        return torch.load(path, map_location=map_location)


def load_state_dict(path, model, map_location="cpu"):
    ckpt = read_ckpt(path, map_location)
    model.load_state_dict(ckpt["state_dict"])
    return ckpt
