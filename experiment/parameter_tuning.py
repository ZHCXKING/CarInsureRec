# %%
import optuna
import json
from pathlib import Path
from src.utils import load
from src.models import *
# %%
def objective(trial, model_name, train, valid, info, default_params):
    lr = trial.suggest_categorical('lr', [1e-4, 1e-3, 1e-2, 1e-1])
    dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4, 0.5])
    hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512])
    feature_dim = trial.suggest_categorical('feature_dim', [16, 32, 64, 128, 256])
    params = default_params.copy()
    params.update({
        'lr': lr,
        'dropout': dropout,
        'feature_dim': feature_dim,
        'hidden_units': [hidden_size, hidden_size // 2],
    })
    if model_name == 'AutoInt':
        attention_layers = trial.suggest_int('attention_layers', 1, 5)
        params.update({
            'attention_layers': attention_layers})
    if model_name == 'DCNv2':
        cross_layers = trial.suggest_int('cross_layers', 1, 5)
        params.update({'cross_layers': cross_layers})
    if model_name == 'DCNv2':
        model = DCNv2Recommend(info['item_name'], info['sparse_features'], info['dense_features'], seed=42, k=5, **params)
    elif model_name == 'DeepFM':
        model = DeepFMRecommend(info['item_name'], info['sparse_features'], info['dense_features'], seed=42, k=5, **params)
    elif model_name == 'WideDeep':
        model = WideDeepRecommend(info['item_name'], info['sparse_features'], info['dense_features'], seed=42, k=5, **params)
    elif model_name == 'AutoInt':
        model = AutoIntRecommend(info['item_name'], info['sparse_features'], info['dense_features'], seed=42, k=5, **params)
    elif model_name == 'FiBINET':
        model = FiBiNETRecommend(info['item_name'], info['sparse_features'], info['dense_features'], seed=42, k=5, **params)
    elif model_name == 'Hybrid':
        model = HybridRecommend(info['item_name'], info['sparse_features'], info['dense_features'], seed=42, k=5, **params)
    else:
        raise ValueError('model_name is not valid')
    model.fit(train.copy())
    val_auc = model.score_test(valid.copy(), methods=['auc'])[0]
    return val_auc
# %%
def main():
    root = Path(__file__).parents[0]
    default_params = {
        'lr': 1e-4,
        'batch_size': 1024,
        'feature_dim': 16,
        'proj_dim': 16,
        'epochs': 200,
        'lambda_nce': 1.0,
        'temperature': 0.1,
        'cross_layers': 3,
        'hidden_units': [256, 128],
        'dropout': 0.1,
        'attention_layers': 3,
        'num_heads': 2
    }
    amount = 10000
    train_ratio = 0.6
    val_ratio = 0.1
    n_trials = 20
    datasets = ['AWM', 'HIP', 'VID']
    models = ['DCNv2', 'DeepFM', 'WideDeep', 'AutoInt', 'FiBINET', 'Hybrid']
    for data_type in datasets:
        train, valid, test, info = load(data_type, amount, train_ratio, val_ratio)
        for model_name in models:
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(
                lambda trial: objective(trial, model_name, train, valid, info, default_params), n_trials=n_trials
            )
            final_params = default_params.copy()
            final_params.update(study.best_params)
            hs = final_params.pop('hidden_size')
            final_params['hidden_units'] = [hs, hs // 2]
            params_path = root / data_type / (model_name + "_params.json")
            with open(params_path, 'w') as f:
                json.dump(final_params, f, indent=4)
# %%
if __name__ == '__main__':
    main()