# -*- coding: utf-8 -*-
"""在测试集上评测一个或多个权重，打印可直接粘贴的对照表。

用法：
  python evaluate.py --data_dir data/sample --ckpt runs/_smoke/best.pt
  python evaluate.py --data_dir data/chnsenticorp --ckpt runs/frozen/best.pt runs/full/best.pt
"""
import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

# 关掉加载权重时的进度条，否则会和表格挤在一起
hf_logging.disable_progress_bar()

from sa.ckpt import read_ckpt
from sa.data import load_split, read_csv

LABEL_NAMES = {0: "差评", 1: "好评"}
from sa.metrics import confusion_matrix, per_class_stats, predict_all
from sa.model import build_model, count_params, count_trainable, freeze_encoder


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

    test_csv = Path(args.data_dir) / "test.csv"
    texts, _ = read_csv(test_csv)  # 原始句子，导出判错样本时用
    ids, mask, labels = load_split(test_csv, AutoTokenizer.from_pretrained(tok_name), args.max_len)
    loader = DataLoader(TensorDataset(ids, mask, labels),
                        batch_size=args.batch_size, shuffle=False)
    print(f"设备: {device} | 测试集 {len(labels)} 条 | 分词器: {tok_name}")

    # 先把所有模型都算完，最后统一打印。
    # 这样加载模型时 torch/transformers 打出的警告不会插进表格中间。
    results = []
    for ck in args.ckpt:
        path = Path(ck)
        info = read_ckpt(path, map_location="cpu")
        model_name = info.get("model_name", "bert-base-chinese")
        freeze = bool(info.get("freeze"))
        model = build_model(model_name).to(device)
        model.load_state_dict(info["state_dict"])

        if freeze:
            freeze_encoder(model)  # 只为统计"训练时可训练的参数量"，不影响推理结果

        pred, gold, conf = predict_all(model, loader, device, return_conf=True)
        stats = per_class_stats(pred, gold)

        # 把判错的样本存下来，做错误分析用（按置信度从高到低：越靠前越是"自信地错了"）
        err_path = path.parent / "test_errors.csv"
        wrong = [i for i in range(len(gold)) if pred[i] != gold[i]]
        wrong.sort(key=lambda i: -conf[i].item())
        with open(err_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["text", "真实", "预测", "置信度"])
            for i in wrong:
                w.writerow([texts[i], LABEL_NAMES[int(gold[i])], LABEL_NAMES[int(pred[i])], f"{conf[i].item():.3f}"])
        results.append({
            "name": f"{model_name} [{'冻结编码器' if freeze else '全参数微调'}]",
            "acc": (pred == gold).float().mean().item(),
            "f1": sum(s["f1"] for s in stats) / len(stats),
            "trainable": count_trainable(model) / 1e6,
            "params": count_params(model) / 1e6,
            "stats": stats,
            "cm": confusion_matrix(pred, gold),
            "err_path": err_path,
        })

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---------- 总表（放简历的就是这张） ----------
    header = [("模型", 34), ("Acc", 9), ("Macro-F1", 10), ("可训练(M)", 11), ("总参数(M)", 10)]
    width = sum(w for _, w in header)
    print("\n" + "".join(pad(h, w) for h, w in header))
    print("-" * width)
    for r in results:
        print("".join([
            pad(r["name"], 34), pad(f"{r['acc']:.4f}", 9), pad(f"{r['f1']:.4f}", 10),
            pad(f"{r['trainable']:.2f}", 11), pad(f"{r['params']:.2f}", 10),
        ]))

    # ---------- 细节：每类指标 + 混淆矩阵（错误分析用） ----------
    names = {0: "差评", 1: "好评"}
    for r in results:
        print(f"\n[{r['name']}]")
        for cls, s in enumerate(r["stats"]):
            print(f"  {names[cls]}：精确率 {s['precision']:.4f} | 召回率 {s['recall']:.4f} | F1 {s['f1']:.4f}")
        cm = r["cm"]
        print("  混淆矩阵（行=真实，列=预测）")
        print(f"              预测差评  预测好评")
        print(f"    真实差评  {cm[0][0]:>8}  {cm[0][1]:>8}")
        print(f"    真实好评  {cm[1][0]:>8}  {cm[1][1]:>8}")
        print(f"  共判错 {cm[0][1] + cm[1][0]} 条：差评误判成好评 {cm[0][1]} 条，好评误判成差评 {cm[1][0]} 条")
        print(f"  判错样本已保存：{r['err_path']}")


if __name__ == "__main__":
    main()
