# %%
import optuna
import json
from pathlib import Path
from src.utils import load
from src.models import *
# --- 公共配置 ---
ROOT = Path(__file__).parents[0]
DATASETS = ['AWM', 'HIP', 'VID']
MODELS = ['DCNv2', 'DeepFM', 'WideDeep', 'FiBiNET', 'CoMICE', 'Hybrid']
DEFAULT_PARAMS = {
    'lr': 1e-3,
    'batch_size': 1024,
    'feature_dim': 32,
    'epochs': 200,
    'lambda_nce': 1.0,
    'temperature': 0.1,
    'proj_dim': 32,
    'cross_layers': 3,
    'hidden_units': [256, 128],
    'dropout': 0.1,
    'attention_layers': 3,
    'num_heads': 2,
}
# %% 阶段一：Base Model Objective
def objective_base(trial, model_name, train, valid, info, default_params):
    # 1. 搜索 Base Model 的核心参数
    #lr = trial.suggest_categorical('lr', [1e-4, 1e-3, 1e-2])
    dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.25])
    hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512])
    num_layers = trial.suggest_categorical('num_layers', [1, 2, 3])
    feature_dim = trial.suggest_categorical('feature_dim', [16, 32, 64])
    hidden_units = [hidden_size // (2 ** i) for i in range(num_layers)]
    params = default_params.copy()
    params.update({
        #'lr': lr,
        'dropout': dropout,
        'feature_dim': feature_dim,
        'hidden_units': hidden_units
    })
    if model_name in ['CoMICE', 'Hybrid']:
        params['attention_layers'] = trial.suggest_categorical('attention_layers', [1, 2, 3])
    if model_name in ['DCNv2', 'CoMICE', 'Hybrid']:
        params['cross_layers'] = trial.suggest_categorical('cross_layers', [1, 2, 3])
    if model_name in ['CoMICE']:
        params['proj_dim'] = trial.suggest_categorical('proj_dim', [8, 16, 32])
    ModelClass = globals()[f"{model_name}Recommend"]
    model = ModelClass(info['item_name'], info['sparse_features'], info['dense_features'], seed=42, k=5, **params)
    model.fit(train.copy(), valid.copy())
    return model.score_test(valid.copy(), methods=['auc'])[0]
# %% 主控制流
def run_optimization():
    amount = 10000
    train_ratio = 0.6
    val_ratio = 0.1
    n_trials_base = 10  # 阶段一试验次数
    for data_type in DATASETS:
        train, valid, test, info = load(data_type, amount, train_ratio, val_ratio)
        for model_name in MODELS:
            param_file = ROOT / data_type / (model_name + "_params.json")
            # --- Phase 1: Base Model Tuning ---
            study_base = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
            study_base.optimize(
                lambda trial: objective_base(trial, model_name, train, valid, info, DEFAULT_PARAMS),
                n_trials=n_trials_base
            )
            # 保存 Base 最优参数
            best_base_params = DEFAULT_PARAMS.copy()
            best_base_params.update(study_base.best_params)
            # 处理 hidden_units 逻辑
            hidden_size = best_base_params.pop('hidden_size')
            num_layers = best_base_params.pop('num_layers')
            best_base_params['hidden_units'] = [hidden_size // (2 ** i) for i in range(num_layers)]
            # --- 保存最终 JSON ---
            with open(param_file, 'w') as f:
                json.dump(best_base_params, f, indent=4)
if __name__ == '__main__':
    run_optimization()