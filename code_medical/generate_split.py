import pandas as pd
import os

# 1. 加载主标签文件
csv_path = 'data/Data_Entry_2017.csv'
if not os.path.exists(csv_path):
    print(f"错误: 找不到 {csv_path}。请确保 CSV 文件已在 data 目录下。")
    exit()

df = pd.read_csv(csv_path)

# 2. 获取所有唯一的 Patient ID
all_patients = df['Patient ID'].unique()
print(f"总病人数: {len(all_patients)}")

# 3. 按照官方比例划分 (大约 80% 训练/验证, 20% 测试)
# 我们这里固定种子，保证每次运行结果一致
import numpy as np
np.random.seed(42)
np.random.shuffle(all_patients)

split_idx = int(len(all_patients) * 0.8)
train_val_patients = all_patients[:split_idx]
test_patients = all_patients[split_idx:]

# 4. 根据 Patient ID 筛选图片文件名
train_val_list = df[df['Patient ID'].isin(train_val_patients)]['Image Index']
test_list = df[df['Patient ID'].isin(test_patients)]['Image Index']

# 5. 保存为 txt 文件
with open('data/train_val_list.txt', 'w') as f:
    f.write('\n'.join(train_val_list.tolist()))

with open('data/test_list.txt', 'w') as f:
    f.write('\n'.join(test_list.tolist()))

print(f"成功生成划分文件！")
print(f"训练/验证集图片数: {len(train_val_list)}")
print(f"测试集图片数: {len(test_list)}")