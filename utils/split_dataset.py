import os
from datasets import load_dataset

# 输出：切分后的保存路径
output_dir = "data/RubricHub_v1/RuRL/RubricHub_v1/RuRL/Split"
# ===========================================

# 1. 加载本地 Parquet 数据
# data_files 可以匹配文件夹下所有的 .parquet 文件
try:
    # 注意：split="train" 是指把读取到的所有数据先统一放到 'train' 这个 key 下，方便后面处理
    dataset = load_dataset("parquet", data_files=f"data/RubricHub_v1/RuRL/RubricHub_v1/RuRL/rurbichub_v1_Medical.parquet", split="train")
    print(f"✅ 加载成功，总数据量: {len(dataset)} 条")
except Exception as e:
    print(f"❌ 加载失败，请检查路径: {e}")
    exit()

# 2. 执行 8:2 切分
# test_size=0.2 表示测试集占 20%，训练集自动占 80%
# seed=42 保证每次运行切分结果一致（可复现）
print("✂️  正在进行 8:2 切分...")
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

# 打印切分后的信息
print(f"📊 切分结果:")
print(f"   - 训练集 (Train): {len(split_dataset['train'])} 条")
print(f"   - 测试集 (Test):  {len(split_dataset['test'])} 条")

# 3. 保存切分后的数据
# 方式 A：保存为 HuggingFace Arrow 格式（加载速度最快，推荐用于训练）
# split_dataset.save_to_disk(output_dir)

# 方式 B：保存回 Parquet 格式（通用性强，方便查看）
# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

train_path = os.path.join(output_dir, "train.parquet")
test_path = os.path.join(output_dir, "test.parquet")

split_dataset['train'].to_parquet(train_path)
split_dataset['test'].to_parquet(test_path)

print("-" * 50)
print(f"✅ 数据已保存！")
print(f"   - 训练集: {train_path}")
print(f"   - 测试集: {test_path}")
print("-" * 50)
print("💡 使用提示: 以后加载时可以直接分别加载：")
print(f"train_ds = load_dataset('parquet', data_files='{train_path}')")