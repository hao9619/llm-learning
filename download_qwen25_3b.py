#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 ModelScope 下载 Qwen2.5-VL-3B-Instruct 模型（国内镜像，AutoDL 适用）
用法: python download_qwen25_vl_3b.py
"""

from modelscope import snapshot_download
import os


def download_qwen25_vl_3b():
    """
    使用 ModelScope 下载 Qwen2.5-VL-3B-Instruct 模型
    """
    print("🚀 使用 ModelScope 下载 Qwen2.5-VL-3B-Instruct 模型...")

    model_id  = "Qwen/Qwen2.5-3B-Instruct"
    local_dir = "/home/hao/Hao9619/models/Qwen2.5-3B-Instruct"

    os.makedirs(local_dir, exist_ok=True)
    print(f"📂 目标目录: {local_dir}")

    try:
        model_path = snapshot_download(
            model_id=model_id,
            local_dir=local_dir,
            revision="master",
        )

        print("✅ 模型下载完成！")
        print(f"📍 模型位置: {model_path}")

        # 打印目录内容，方便确认文件完整性
        print("\n📋 模型文件列表:")
        for f in sorted(os.listdir(model_path)):
            fpath = os.path.join(model_path, f)
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            print(f"   {f:50s}  {size_mb:8.1f} MB")

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        raise


if __name__ == "__main__":
    download_qwen25_vl_3b()
