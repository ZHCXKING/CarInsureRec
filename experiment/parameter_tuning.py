# %%
import optuna
import json
from pathlib import Path
from src.utils import load
from src.models import *
# --- 公共配置 ---
ROOT = Path(__file__).parents[0]
DATASETS = ['AWM', 'HIP', 'VID']
NN_MODELS = ['DCN', 'DCNv2', 'DeepFM', 'WideDeep', 'FiBiNET', 'AutoInt']
TREE_MODELS = ['RF', 'XGB', 'LGBM', 'CatB']
STATISTIC_MODELS = ['LR', 'KNN', 'NB']
NN_PARAMS = {
    'lr': 1e-3,
    'batch_size': 1024,
    'epochs': 200,
    'feature_dim': 32,
    'hidden_units': [256, 128],
    'dropout': 0.1,
    'cross_layers': 3,
    'low_rank': 64,
    'attention_layers': 3,
    'num_heads': 2,
    'proj_dim': 64,
    'lambda_nce': 1.0,
    'temperature': 0.1
}
TREE_PARAMS = {
    'learning_rate': 0.001,
    'n_estimators': 200,
    'max_depth': 5
}
STATISTIC_PARAMS = {
    'n_neighbors': 10,
    'C': 10,
    'max_iter': 1000
}
# %% 阶段一：NN Model Objective
def objective_NN(trial, model_name, train, valid, info, default_params):
    params = default_params.copy()
    params['dropout'] = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3])
    params['feature_dim'] = trial.suggest_categorical('feature_dim', [16, 32, 64])
    hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512])
    num_layers = trial.suggest_categorical('num_layers', [1, 2, 3])
    params['hidden_units'] = [hidden_size // (2 ** i) for i in range(num_layers)]
    if model_name in ['DCN', 'DCNv2']:
        params['cross_layers'] = trial.suggest_categorical('cross_layers', [1, 2, 3])
    if model_name == 'DCNv2':
        params['low_rank'] = trial.suggest_categorical('low_rank', [16, 32, 64])
    if model_name == 'AutoInt':
        params['attention_layers'] = trial.suggest_categorical('attention_layers', [1, 2, 3])
        params['num_heads'] = trial.suggest_categorical('num_heads', [1, 2, 4])
    ModelClass = globals()[f"{model_name}Recommend"]
    model = ModelClass(info['item_name'], info['sparse_features'], info['dense_features'], seed=42, k=3, **params)
    model.fit(train.copy(), valid.copy())
    return model.score_test(valid.copy(), methods=['logloss'])[0]
# %% 阶段二：CoMICE Model Objective
def objective_CoMICE(trial, model_name, train, valid, info, default_params):
    params = default_params.copy()
    params['lambda_nce'] = trial.suggest_categorical('lambda_nce', [0.1, 0.5, 1.0, 2.0])
    params['temperature'] = trial.suggest_categorical('temperature', [0.07, 0.1, 0.2])
    params['proj_dim'] = trial.suggest_categorical('proj_dim', [32, 64, 128])
    model = CoMICERecommend(info['item_name'], info['sparse_features'], info['dense_features'], seed=42, k=3, backbone=model_name, **params)
    model.fit(train.copy(), valid.copy())
    return model.score_test(valid.copy(), methods=['logloss'])[0]
# %% 阶段三：Tree Model Objective
def objective_TREE(trial, model_name, train, valid, info, default_params):
    params = default_params.copy()
    params['n_estimators'] = trial.suggest_categorical('n_estimators', [50, 100, 150, 200])
    params['max_depth'] = trial.suggest_categorical('max_depth', [3, 5, 10])
    ModelClass = globals()[f"{model_name}Recommend"]
    model = ModelClass(info['item_name'], info['sparse_features'], info['dense_features'], seed=42, k=3, **params)
    model.fit(train.copy())
    return model.score_test(valid.copy(), methods=['logloss'])[0]
# %% 阶段四：Statistic Model Objective
def objective_Statistic(trial, model_name, train, valid, info, default_params):
    params = default_params.copy()
    if model_name == 'KNN':
        params['n_neighbors'] = trial.suggest_categorical('n_neighbors', [5, 10, 20, 30, 50])
    if model_name == 'LR':
        params['C'] = trial.suggest_categorical('C', [0.001, 0.01, 0.1, 10, 50, 100])
        params['max_iter'] = trial.suggest_categorical('max_iter', [200, 500, 1000, 2000])
    ModelClass = globals()[f"{model_name}Recommend"]
    model = ModelClass(info['item_name'], info['sparse_features'], info['dense_features'], seed=42, k=3, **params)
    model.fit(train.copy())
    return model.score_test(valid.copy(), methods=['logloss'])[0]
# %% 主控制流
def run_optimization():
    amount = None
    train_ratio =0.7
    val_ratio = 0.1
    n_trials_NN = 20
    n_trials_CoMICE = 10
    n_trials_TREE = 10
    n_trials_STATISTIC = 10
    for data_type in DATASETS:
        train, valid, test, info = load(data_type, amount, train_ratio, val_ratio, is_dropna=True)
        # --- Phase 1: NN Model Tuning ---
        for model_name in NN_MODELS:
            param_file = ROOT / data_type / (model_name + "_param.json")
            study_base = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
            study_base.optimize(
                lambda trial: objective_NN(trial, model_name, train, valid, info, NN_PARAMS),
                n_trials=n_trials_NN
            )
            best_base_params = NN_PARAMS.copy()
            best_base_params.update(study_base.best_params)
            hidden_size = best_base_params.pop('hidden_size')
            num_layers = best_base_params.pop('num_layers')
            best_base_params['hidden_units'] = [hidden_size // (2 ** i) for i in range(num_layers)]
            # --- Phase 2: CoMICE Model Tuning ---
            # study_base = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
            # study_base.optimize(
            #     lambda trial: objective_CoMICE(trial, model_name, train, valid, info, best_base_params),
            #     n_trials=n_trials_CoMICE
            # )
            # best_base_params.update(study_base.best_params)
            with open(param_file, 'w') as f:
                json.dump(best_base_params, f, indent=4)
        # --- Phase 3: Tree Model Tuning ---
        for model_name in TREE_MODELS:
            param_file = ROOT / data_type / (model_name + "_param.json")
            study_base = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
            study_base.optimize(
                lambda trial: objective_TREE(trial, model_name, train, valid, info, TREE_PARAMS),
                n_trials=n_trials_TREE
            )
            best_base_params = TREE_PARAMS.copy()
            best_base_params.update(study_base.best_params)
            with open(param_file, 'w') as f:
                json.dump(best_base_params, f, indent=4)
        # --- Phase 4: STATISTIC Model Tuning ---
        for model_name in STATISTIC_MODELS:
            param_file = ROOT / data_type / (model_name + "_param.json")
            study_base = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
            study_base.optimize(
                lambda trial: objective_Statistic(trial, model_name, train, valid, info, STATISTIC_PARAMS),
                n_trials=n_trials_STATISTIC
            )
            best_base_params = STATISTIC_PARAMS.copy()
            best_base_params.update(study_base.best_params)
            with open(param_file, 'w') as f:
                json.dump(best_base_params, f, indent=4)
if __name__ == '__main__':
    run_optimization()