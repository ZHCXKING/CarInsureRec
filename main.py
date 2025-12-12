from src.utils import load
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from rec4torch.inputs import SparseFeat, DenseFeat, build_input_array
from rec4torch.models import DeepFM, WideDeep
from rec4torch.snippets import seed_everything
train, test = load('test', amount=5000, split_num=2500, seed=42, fillna=True)
columns = ['Age', 'DrivingExp', 'Occupation', 'NCD', 'Make', 'Car.year', 'Car.price', 'InsCov', 'Date']

embedding_dim = 4
dimension = 1
batch_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_everything(42)
epochs = 2
out_dim = train['InsCov'].nunique()

sparse_features = ['Occupation', 'NCD', 'Make']
dense_features = ['Age', 'Car.year', 'Car.price', 'DrivingExp']

mappings = {}
vocabulary_sizes = {}
for col in sparse_features:
    unique = train[col].unique()
    mapping = {v: i+1 for i, v in enumerate(unique)}
    mappings[col] = mapping
    vocabulary_sizes[col] = len(mapping) + 1
for col in sparse_features:
    mp = mappings[col]
    train[col] = train[col].map(lambda x: mp.get(x, 0))
    test[col] = test[col].map(lambda x: mp.get(x, 0))
# print(train['Occupation'].nunique(), train['Occupation'].min(), train['Occupation'].max())
# print(train['NCD'].nunique(), train['NCD'].min(), train['NCD'].max())
# print(train['Make'].nunique(), train['Make'].min(), train['Make'].max())

sparse_feature_columns = [
    SparseFeat(feat, vocabulary_sizes[feat], embedding_dim=4)
    for feat in sparse_features
]
dense_feature_columns = [
    DenseFeat(feat, dimension=dimension)
    for feat in dense_features
]

feature_columns = sparse_feature_columns + dense_feature_columns
linear_feature_columns = feature_columns
dnn_feature_columns = feature_columns

train_X, train_y = build_input_array(train, linear_feature_columns+dnn_feature_columns, target='InsCov')
train_X = torch.tensor(train_X, dtype=torch.float, device=device)
train_y = torch.tensor(train_y, dtype=torch.int64, device=device)
print(train_X.shape)
print(train_y.shape)
train_dataset = TensorDataset(train_X, train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = DeepFM(linear_feature_columns, dnn_feature_columns, out_dim=out_dim)
model.to(device)
model.compile(
    loss=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=0.01),
)
model.fit(train_loader, epochs=epochs)

test_X, test_y = build_input_array(test, linear_feature_columns+dnn_feature_columns, target='InsCov')
test_X = torch.tensor(test_X, dtype=torch.float, device=device)
predictions = model.predict(test_X)
probs = predictions.cpu().numpy()
print(probs.shape)
print(probs)