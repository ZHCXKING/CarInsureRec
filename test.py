import json
from pathlib import Path
root = Path(__file__)
print(type(root))
file_path = 'data/AWM/Metadata.json'
with open(file_path, 'r') as f:
    metadata = json.load(f)
columns = metadata['tables']['table']['columns']
item_name = 'product_item'
sparse_features = []
dense_features = []
for col, info in columns.items():
    if col == item_name:
        continue
    if info['sdtype'] == 'categorical':
        sparse_features.append(col)
    elif info['sdtype'] == 'numerical':
        dense_features.append(col)
print(item_name)
print(sparse_features)
print(dense_features)