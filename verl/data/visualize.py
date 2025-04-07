import pandas as pd

# 读取单个Parquet文件
df = pd.read_parquet("verl/data/geo3k/train.parquet")

# 显示数据
print(df.head())