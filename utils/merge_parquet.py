import pandas as pd
import os

# 1. 定义文件路径
# 根据你的截图，文件在 data/health_bench 目录下
train_path = 'data/health_bench/healthbench_train.parquet'
val_path   = 'data/health_bench/healthbench_val.parquet'
output_path = 'data/health_bench/healthbench.parquet' # 新文件的名字

def merge_files():
    # 检查文件是否存在
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("❌ 错误：找不到输入文件，请检查路径是否正确。")
        return

    print("正在读取文件...")
    
    # 2. 读取 Parquet 文件
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)

    print(f"✅ 读取成功！")
    print(f"   - Train 集数据量: {len(df_train)} 行")
    print(f"   - Val   集数据量: {len(df_val)} 行")

    # 3. 合并数据 (Concatenate)
    # ignore_index=True 会重置索引，防止两个文件的索引冲突（比如都有第0行）
    print("正在合并...")
    merged_df = pd.concat([df_train, df_val], ignore_index=True)

    # 4. 验证合并结果
    total_len = len(df_train) + len(df_val)
    if len(merged_df) == total_len:
        print(f"✅ 合并逻辑验证通过：总行数 ({len(merged_df)}) 等于两者之和。")
    else:
        print(f"⚠️ 警告：合并后的行数 ({len(merged_df)}) 与预期 ({total_len}) 不符，请检查数据。")

    # 5. 保存为新文件
    merged_df.to_parquet(output_path)
    print(f"🎉 文件已保存至: {output_path}")
    print("提示：原始的 train 和 val 文件未被修改。")

if __name__ == "__main__":
    merge_files()