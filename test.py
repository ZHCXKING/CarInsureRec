from pathlib import Path
import json
root = Path(__file__).parents[0] / 'experiment'
print(root)
NN_MODELS = ['DCN', 'DCNv2', 'DeepFM', 'WideDeep', 'FiBiNET', 'AutoInt']
Seeds = root / 'AWM' / 'Seeds.json'
with open(Seeds, 'r') as f:
    Seeds = json.load(f)
print(Seeds['DCN'])
for model_name in NN_MODELS:
    param_file = root / 'AWM' / (model_name + "_param.json")
    with open(param_file, 'r') as f:
        params = json.load(f)
    print(params)
    for seed in Seeds[model_name]:
        print(seed)