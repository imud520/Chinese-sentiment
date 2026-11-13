# -*- coding: utf-8 -*-
"""评测指标：准确率 Accuracy 与宏平均 F1（Macro-F1）。

宏平均 F1 = 先对"差评""好评"两类各算一个 F1，再取平均。
注意：每一类必须用各自独立的 TP/FP/FN 计数。
如果两类共用同一组计数器，算出来的是"微平均 F1"，在二分类下它恒等于准确率。
"""
import torch


@torch.no_grad()
def predict_all(model, loader, device, return_conf=False):
    """跑完整个 loader，返回 (预测标签, 真实标签)，都是 CPU 上的一维张量。

    return_conf=True 时额外返回每条预测的置信度（softmax 后最大的那个概率）。
    """
    model.eval()
    preds, golds, confs = [], [], []
    for input_ids, attention_mask, labels in loader:
        logits = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
        ).logits
        prob = torch.softmax(logits.float(), dim=-1)
        conf, pred = prob.max(dim=-1)
        preds.append(pred.cpu())
        confs.append(conf.cpu())
        golds.append(labels.cpu())
    if return_conf:
        return torch.cat(preds), torch.cat(golds), torch.cat(confs)
    return torch.cat(preds), torch.cat(golds)


def per_class_stats(pred, gold, num_classes=2):
    """每一类单独统计 precision / recall / F1。"""
    stats = []
    for cls in range(num_classes):
        tp = ((pred == cls) & (gold == cls)).sum().item()
        fp = ((pred == cls) & (gold != cls)).sum().item()
        fn = ((pred != cls) & (gold == cls)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        stats.append({"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1})
    return stats


def confusion_matrix(pred, gold, num_classes=2):
    """cm[i][j] = 真实为 i、预测为 j 的条数。"""
    cm = [[0] * num_classes for _ in range(num_classes)]
    for g, p in zip(gold.tolist(), pred.tolist()):
        cm[g][p] += 1
    return cm


def evaluate(model, loader, device):
    """返回 (accuracy, macro_f1)。loader 每个 batch 是 (input_ids, attention_mask, labels)。"""
    pred, gold = predict_all(model, loader, device)
    acc = (pred == gold).float().mean().item() if len(gold) else 0.0
    stats = per_class_stats(pred, gold)
    macro_f1 = sum(s["f1"] for s in stats) / len(stats)
    return acc, macro_f1
