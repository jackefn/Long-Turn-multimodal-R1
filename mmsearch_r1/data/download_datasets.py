# 方法1: 使用 cache_dir 参数（最简单的方法）
from datasets import load_dataset

# 指定下载目录
download_dir = "/data1/xiangc/mxy/Long-Turn-multimodal-R1/datasets"

# 加载数据集并指定缓存目录
dataset = load_dataset(
    "lmms-lab/FVQA",  # 替换为你的数据集名称，例如 "openai/gsm8k"
    cache_dir=download_dir  # 指定下载目录
)

print(f"数据集已下载到: {download_dir}")
