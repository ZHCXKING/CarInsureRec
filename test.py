# 打印处理后的数据总行数
print(f"Total rows after preprocessing: {len(df)}")

# 计算包含至少一个缺失值的行数
rows_with_missing = df.isna().any(axis=1).sum()
print(f"Number of rows with at least one missing value: {rows_with_missing}")

# 计算每一列的缺失值数量（可选，用于更详细的分析）
print("Missing values per column:")
print(df.isna().sum())

# 如果你想知道具体的百分比
missing_percentage = (rows_with_missing / len(df)) * 100
print(f"Percentage of rows with missing values: {missing_percentage:.2f}%")

# 保存 DropNA 后的数据（保持你原有的逻辑）
df_drop = df.dropna(ignore_index=True)
df_drop.to_parquet('DropNaData.parquet', index=False)