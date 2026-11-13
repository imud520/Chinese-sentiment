# 中文情感分类（BERT 微调）

用预训练的中文 BERT 判断一句评论是**好评**还是**差评**。
整个项目核心代码约 200 行，跑通一次就能拿到能写进简历的数字。

适合作为简历项目的三条理由：

1. **任务直观**：输入一句话，输出"好评 96.3%"——面试官不用听你解释背景就懂
2. **代码短**：一个训练循环 + 一个数据集读取，没有花活
3. **数字硬**：准确率 Acc、宏平均 F1，以及"冻结编码器 vs 全参数微调"的对照表

---

## 目录结构

```
sentiment/
├── sa/
│   ├── data.py       读 CSV、分词、转张量
│   ├── model.py      建模型、冻结编码器、统计参数量
│   ├── metrics.py    Acc / F1
│   └── ckpt.py       权重保存与读取
├── train.py          训练（核心只有四步）
├── evaluate.py       测试集评测，直接打印对照表
├── predict.py        拿任意句子做演示
├── download_data.py  下载正式数据集 ChnSentiCorp（走国内镜像）
├── check_env.py      环境自检
└── data/sample/      内置小样例（84/18/18 条），不用下载就能跑通
```

---

## 快速开始（四步）

### 1. 装环境

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 2. 自检

```bash
python check_env.py
```

看到 `[OK] CUDA 可用` 和 `[OK] CUDA 内核可正常执行` 就可以往下走。

### 3. 先跑通流程（1 分钟，不下载任何模型权重）

```bash
python train.py --model tiny-random --data_dir data/sample --out_dir runs/smoke --epochs 12 --batch_size 8 --lr 1e-3
```

`tiny-random` 是随机初始化的小模型（305 万参数），只用来确认数据、训练、评测这条链路没坏。
train_loss 从 0.69 降到 0.01 以下、valid_acc 在 0.78 以上，就说明代码是对的（样例数据只有 84 条，波动很正常）。

### 4. 正式实验（两个对照，各一轮）

```bash
# 对照组：冻结编码器，只训练最后的分类头
python train.py --model bert-base-chinese --data_dir data/chnsenticorp --out_dir runs/frozen --freeze 1 --epochs 3 --lr 1e-3 --batch_size 32

# 实验组：全参数微调
python train.py --model bert-base-chinese --data_dir data/chnsenticorp --out_dir runs/full --freeze 0 --epochs 3 --lr 2e-5 --batch_size 32

# 一起出表
python evaluate.py --data_dir data/chnsenticorp --ckpt runs/frozen/best.pt runs/full/best.pt
```

正式数据还没下？先跑这一步（约 1 万条评论，几十 MB）：

```bash
python download_data.py
```

---

## 为什么设计成"冻结 vs 全参数"两组

这是本项目唯一需要你讲清楚的技术点，也是简历里最有含量的一句话。

- **冻结编码器**（linear probing）：BERT 那 1 亿参数完全不动，只学一个 2 分类的小头。训练极快、显存极小，本质是"拿预训练特征做分类"。
- **全参数微调**：所有参数都跟着数据调整。通常准确率高 2–5 个点，但训练慢、容易在小数据上过拟合。

两者的差值，就是"预训练权重到底值多少钱"的直接量化。面试官问"你为什么微调而不是直接用"，这张表就是答案。

---

## 简历写法模板

> **中文情感分类模型微调**（PyTorch / Transformers）
> - 基于 `bert-base-chinese` 在 ChnSentiCorp（约 1 万条评论）上微调二分类模型，测试集准确率 **XX.X%**、宏平均 F1 **XX.X%**
> - 设计"冻结编码器 vs 全参数微调"对照实验，量化预训练权重的贡献：**XX.X% → XX.X%**
> - 手写训练循环（AdamW + warmup 线性衰减 + 梯度裁剪），单卡 8GB 显存完成训练，单轮耗时 XX 分钟
> - 分析 XX 条错误样本，总结出否定句、反讽句是主要失效模式

（最后一条是加分项：面试官最爱问"错在哪"，能说出失败案例的人，看起来像真做过。）

---

## 常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| 下载模型很慢 / 卡住 | 默认连 huggingface.co | 设环境变量 `HF_ENDPOINT=https://hf-mirror.com`（PyCharm 配置里已内置） |
| `CUDA out of memory` | batch 太大 | `--batch_size 16` 或 `8`，bert-base 在 8GB 上 32 一般没问题 |
| 准确率一直是 50% | 学习率太小或没加载到预训练权重 | 用 `--freeze 0 --lr 2e-5`，确认日志里 `参数量: 102.27M` |
| 中文表格错位 | 终端字体问题 | 表格已按中文双宽对齐，若仍歪，把终端字体设为等宽中文字体 |
| `No module named 'sa'` | 工作目录不对 | PyCharm 的 Working directory 设为 `$PROJECT_DIR$` |
