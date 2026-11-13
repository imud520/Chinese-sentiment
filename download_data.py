# -*- coding: utf-8 -*-
"""下载正式数据集 ChnSentiCorp（约 1 万条中文评论，带情感标签），存成 CSV。

默认走国内镜像 hf-mirror.com，不需要科学上网。
用法：
  python download_data.py                 # 下载到 data/chnsenticorp/
  python download_data.py --out data/xxx  # 指定目录
"""
import argparse
import os
from pathlib import Path

import pandas as pd

REPO = "lansinuote/ChnSentiCorp"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/chnsenticorp")
    args = p.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内镜像
    from datasets import load_dataset

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(REPO)
    # HF 里叫 validation，本项目统一用 valid
    mapping = [("train", "train"), ("validation", "valid"), ("test", "test")]
    for src, dst in mapping:
        df = pd.DataFrame(ds[src])
        df = df.rename(columns={"sentence": "text"})[["text", "label"]]
        df.to_csv(out / f"{dst}.csv", index=False, encoding="utf-8-sig")
        n_pos = int(df["label"].sum())
        print(f"{dst}: {len(df)} 条（好评 {n_pos} / 差评 {len(df) - n_pos}）-> {out / (dst + '.csv')}")

    print("\n下载完成。训练时用 --data_dir 指向这个目录即可。")


if __name__ == "__main__":
    main()
