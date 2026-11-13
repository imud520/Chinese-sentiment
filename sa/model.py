# -*- coding: utf-8 -*-
"""模型构建：所有模型统一走 AutoModelForSequenceClassification。

支持两种来源：
1. HuggingFace 上的预训练模型名，如 bert-base-chinese（需要联网下载一次）；
2. "tiny-random"：一个随机初始化的超小中文 BERT（2 层、隐藏维 128，约 380 万参数），
   不用联网、几十秒就能跑完——专门用来在正式训练之前验证整条流程。
"""
from transformers import AutoConfig, AutoModelForSequenceClassification, BertConfig
import transformers


def _quiet_load(fn, *args, **kwargs):
    """加载预训练权重时会打印一堆 UNEXPECTED/MISSING 提示（因为把 MLM 头换成了分类头），
    对新手来说很像报错，这里临时把日志级别调高，避免干扰视线。"""
    prev = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    try:
        return fn(*args, **kwargs)
    finally:
        transformers.logging.set_verbosity(prev)


def build_model(model_name: str, num_labels: int = 2):
    if model_name == "tiny-random":
        # 随机权重的小模型，只用于验证代码流程，不用于出结果
        cfg = BertConfig(
            vocab_size=21128,      # 与 bert-base-chinese 的词表一致
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=256,
            num_labels=num_labels,
        )
        return AutoModelForSequenceClassification.from_config(cfg)

    cfg = _quiet_load(AutoConfig.from_pretrained, model_name, num_labels=num_labels)
    return _quiet_load(AutoModelForSequenceClassification.from_pretrained, model_name, config=cfg)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_encoder(model):
    """冻结编码器，只训练最后的分类头。

    这就是对照实验的"另一组"：同样的数据、同样的训练设置，
    只改"要不要更新编码器权重"，就能看出微调到底带来了什么。
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    if hasattr(model, "bert") and hasattr(model.bert, "pooler"):
        # pooler 在某些结构里也参与分类，一并放开
        for p in model.bert.pooler.parameters():
            p.requires_grad = True
    return model
