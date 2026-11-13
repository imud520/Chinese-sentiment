# -*- coding: utf-8 -*-
"""评测指标：准确率 Accuracy 与宏平均 F1。"""
import torch


@torch.no_grad()
def evaluate(model, loader, device):
    """返回 (accuracy, macro_f1)。loader 每个 batch 是 (input_ids, attention_mask, labels)。"""
    model.eval()
    tp = fp = fn = correct = total = 0
    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        pred = logits.argmax(dim=-1)

        correct += (pred == labels).sum().item()
        total += labels.numel()

        for cls in (0, 1):
            tp += ((pred == cls) & (labels == cls)).sum().item()
            fp += ((pred == cls) & (labels != cls)).sum().item()
            fn += ((pred != cls) & (labels == cls)).sum().item()

    acc = correct / max(total, 1)
    # 两类各自的 F1 取平均
    f1s = []
    for cls in (0, 1):
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return acc, sum(f1s) / 2
