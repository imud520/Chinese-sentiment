# -*- coding: utf-8 -*-
"""微调中文情感分类模型。

核心流程只有四步（教学版手写训练循环，每一步都能看到）：
  1. 读数据并分词
  2. 取一批数据 -> 模型前向 -> 算 loss
  3. 反向传播更新参数
  4. 每轮结束在验证集上测一次，保存最好的权重

用法示例：
  python train.py --model tiny-random   --data_dir data/sample --out_dir runs/_smoke   # 冒烟测试
  python train.py --model bert-base-chinese --data_dir data/chnsenticorp --out_dir runs/full --freeze 0
  python train.py --model bert-base-chinese --data_dir data/chnsenticorp --out_dir runs/frozen --freeze 1
"""
import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from sa.ckpt import save_ckpt
from sa.data import load_split
from sa.metrics import evaluate
from sa.model import build_model, count_params, count_trainable, freeze_encoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="bert-base-chinese",
                   help="HuggingFace 模型名，或 tiny-random（随机小模型，用于冒烟测试）")
    p.add_argument("--data_dir", default="data/sample", help="含 train.csv/valid.csv 的目录")
    p.add_argument("--out_dir", default="runs/exp", help="输出目录")
    p.add_argument("--freeze", type=int, default=0,
                   help="1 = 冻结编码器只训分类头（对照组）；0 = 全参数微调")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit_train", type=int, default=0, help="只用前 N 条训练数据（0 = 全部）")
    return p.parse_args()


def make_loader(input_ids, attention_mask, labels, batch_size, shuffle, workers):
    ds = TensorDataset(input_ids, attention_mask, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"设备: {device} | 模型: {args.model} | freeze: {args.freeze}")

    tok_name = "bert-base-chinese" if args.model == "tiny-random" else args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    print(f"加载分词器完成（{tok_name}）")

    tr_ids, tr_mask, tr_lab = load_split(Path(args.data_dir) / "train.csv", tokenizer, args.max_len)
    va_ids, va_mask, va_lab = load_split(Path(args.data_dir) / "valid.csv", tokenizer, args.max_len)
    if args.limit_train > 0:
        tr_ids, tr_mask, tr_lab = tr_ids[: args.limit_train], tr_mask[: args.limit_train], tr_lab[: args.limit_train]
    train_loader = make_loader(tr_ids, tr_mask, tr_lab, args.batch_size, True, args.num_workers)
    valid_loader = make_loader(va_ids, va_mask, va_lab, args.batch_size, False, args.num_workers)
    print(f"数据: 训练 {len(tr_lab)} 条 | 验证 {len(va_lab)} 条")

    model = build_model(args.model).to(device)
    if args.freeze:
        freeze_encoder(model)
    print(f"参数量: {count_params(model)/1e6:.2f}M | 可训练: {count_trainable(model)/1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01
    )
    total_steps = max(1, len(train_loader)) * args.epochs
    warmup_steps = max(1, int(total_steps * 0.1))

    def lr_scale(step):
        """前 10% 的步数线性升温，之后线性衰减到 0。

        手写这两行而不是用 OneCycleLR，是因为数据量很小（步数 < 20）时
        官方调度器会出 ZeroDivisionError，新手撞上很难查。
        """
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    loss_fn = torch.nn.CrossEntropyLoss()

    log_rows = []
    best_acc, best_path = -1.0, out_dir / "best.pt"
    g_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, (ids, mask, lab) in enumerate(train_loader, 1):
            ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits, lab)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            g_step += 1
            for group in optimizer.param_groups:
                group["lr"] = args.lr * lr_scale(g_step)
            running += loss.item()
            if step % 20 == 0 or step == len(train_loader):
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {running/step:.4f}")

        val_acc, val_f1 = evaluate(model, valid_loader, device)
        dt = time.time() - t0
        print(f"[epoch {epoch}] train_loss {running/len(train_loader):.4f} | "
              f"valid_acc {val_acc:.4f} valid_f1 {val_f1:.4f} | {dt:.0f}s")
        log_rows.append([epoch, f"{running/len(train_loader):.4f}", f"{val_acc:.4f}", f"{val_f1:.4f}", f"{dt:.0f}"])

        if val_acc > best_acc:
            best_acc = val_acc
            save_ckpt(best_path, model, {
                "model_name": args.model, "tokenizer_name": tok_name, "freeze": args.freeze,
                "val_acc": val_acc, "val_f1": val_f1, "epoch": epoch,
            })
            print(f"  -> 保存最优权重到 {best_path}")

    with open(out_dir / "log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "valid_acc", "valid_f1", "seconds"])
        w.writerows(log_rows)

    save_ckpt(out_dir / "last.pt", model, {
        "model_name": args.model, "tokenizer_name": tok_name, "freeze": args.freeze,
    })
    print(f"完成。best valid_acc = {best_acc:.4f}，权重在 {out_dir}")


if __name__ == "__main__":
    main()
