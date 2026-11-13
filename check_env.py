# -*- coding: utf-8 -*-
"""环境自检：一条命令告诉你哪里没配好。

在 PyCharm 里右键运行本文件，或在终端执行：python check_env.py
"""
import os
import sys
from pathlib import Path

OK, WARN, BAD = "[OK]  ", "[警告]", "[错误]"
ROOT = Path(__file__).resolve().parent


def line(tag, msg, hint=""):
    print(f"{tag} {msg}")
    if hint:
        print(f"       -> {hint}")


def check_python():
    v = sys.version_info
    print(f"\n== Python ==\n版本 {v.major}.{v.minor}.{v.micro}")
    if v < (3, 9):
        line(BAD, "版本过低，需要 3.9 以上", "装 Python 3.11 更稳")
    else:
        line(OK, "版本可用")
    print(f"解释器路径: {sys.executable}")


def check_torch():
    print("\n== PyTorch 与显卡 ==")
    try:
        import torch
    except ImportError:
        line(BAD, "没有安装 torch", "pip install torch --index-url https://download.pytorch.org/whl/cu128")
        return
    line(OK, f"torch {torch.__version__}")
    if not torch.cuda.is_available():
        line(WARN, "CUDA 不可用，只能用 CPU 训练", "确认装的是 cu128 版本，且显卡驱动正常")
        return
    line(OK, f"CUDA 可用 | 显卡 {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    line(OK, f"算力 {cap[0]}.{cap[1]} | 显存 {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    try:
        x = torch.randn(64, 64, device="cuda")
        (x @ x).sum().item()
        line(OK, "CUDA 内核可正常执行（能真正跑起来）")
    except Exception as e:
        line(BAD, f"CUDA 内核执行失败: {e}", "大概率是 torch 版本不匹配，装 cu128 版本")


def check_deps():
    print("\n== 依赖包 ==")
    for mod, pkg, need in [
        ("transformers", "transformers", True),
        ("datasets", "datasets", False),
        ("pandas", "pandas", False),
    ]:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            line(OK, f"{mod} {ver}")
        except ImportError:
            if need:
                line(BAD, f"缺少 {mod}", f"pip install {pkg}")
            else:
                line(WARN, f"缺少 {mod}（只有下载数据集时需要）", f"pip install {pkg}")


def check_data():
    print("\n== 数据 ==")
    sample = ROOT / "data" / "sample"
    for split in ["train", "valid", "test"]:
        f = sample / f"{split}.csv"
        if f.exists():
            n = sum(1 for _ in open(f, encoding="utf-8-sig")) - 1
            line(OK, f"data/sample/{split}.csv  {n} 条")
        else:
            line(BAD, f"缺少 {f}")
    real = ROOT / "data" / "chnsenticorp"
    if (real / "train.csv").exists():
        n = sum(1 for _ in open(real / "train.csv", encoding="utf-8-sig")) - 1
        line(OK, f"data/chnsenticorp/train.csv  {n} 条（正式数据集已就绪）")
    else:
        line(WARN, "正式数据集还没下载", "python download_data.py")


def check_network_hint():
    print("\n== 模型下载提示 ==")
    ep = os.environ.get("HF_ENDPOINT", "")
    if ep:
        line(OK, f"HF_ENDPOINT = {ep}")
    else:
        line(WARN, "未设置 HF_ENDPOINT，默认走 huggingface.co（国内可能很慢）",
             'PowerShell: $env:HF_ENDPOINT="https://hf-mirror.com"  或  set HF_ENDPOINT=https://hf-mirror.com')


def main():
    print("=" * 60)
    print("中文情感分类项目 - 环境自检")
    print("=" * 60)
    check_python()
    check_torch()
    check_deps()
    check_data()
    check_network_hint()
    print("\n" + "=" * 60)
    print("看到 [错误] 就先解决它；只有 [警告] 也可以继续跑通流程。")
    print("=" * 60)


if __name__ == "__main__":
    main()
