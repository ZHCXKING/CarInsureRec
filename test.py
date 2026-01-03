import optuna
import json
import pandas as pd
from pathlib import Path
from src.utils import load, round, add_missing_values, select_interpolation
# 假设 src.models 包含所有模型类 (NN + Tree)

ROOT = Path(".")
DATASETS = ['AWM', 'HIP', 'VID']

# 分组定义模型
NN_MODELS = ['DCN', 'DCNv2', 'DeepFM', 'WideDeep', 'FiBiNET', 'AutoInt']
TREE_MODELS = ['XGB', 'LGBM', 'CatB', 'RF']  # 对应 XGBRecommend, LGBMRecommend...

# 默认参数 (仅作参考，实际搜索时会覆盖)
DEFAULT_PARAMS_NN = {
    'lr': 1e-3, 'batch_size': 1024, 'epochs': 50,
    # ... 其他 NN 参数
}
# --------------------------------------------------------------------------
# 辅助函数：获取数据 (Complete Case Strategy)
# --------------------------------------------------------------------------
def get_data_for_tuning(data_type, use_complete_case=True):
    # (同之前的实现: AWM -> DropNA, HIP/VID -> Clean)
    # ... (代码省略，参考上一轮对话)
    pass
# --------------------------------------------------------------------------
# Objective 1: Deep Learning Models (NN)
# --------------------------------------------------------------------------
def objective_nn(trial, model_name, train, valid, info):
    # ... (同之前的 objective_base 实现)
    # 搜索 lr, dropout, layers, embedding_dim 等
    pass
# --------------------------------------------------------------------------
# Objective 2: Tree Models (XGB, LGBM, CatB, RF) [新增]
# --------------------------------------------------------------------------
def objective_tree(trial, model_name, train, valid, info):
    params = {}

    if model_name == 'XGB':
        params['n_estimators'] = trial.suggest_int('n_estimators', 100, 1000, step=100)
        params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        params['max_depth'] = trial.suggest_int('max_depth', 3, 10)
        params['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 7)
        params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
        params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.6, 1.0)
        # XGB 特定参数
        params['early_stopping_rounds'] = 50

    elif model_name == 'LGBM':
        params['n_estimators'] = trial.suggest_int('n_estimators', 100, 1000, step=100)
        params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        params['num_leaves'] = trial.suggest_int('num_leaves', 20, 150)
        params['max_depth'] = trial.suggest_int('max_depth', -1, 15)
        params['min_child_samples'] = trial.suggest_int('min_child_samples', 5, 100)
        params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
        params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.6, 1.0)

    elif model_name == 'CatB':
        params['iterations'] = trial.suggest_int('iterations', 100, 1000, step=100)
        params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        params['depth'] = trial.suggest_int('depth', 4, 10)
        params['l2_leaf_reg'] = trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True)
        params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 1.0)
        # CatBoost 也可以用 early stopping，但在 Wrapper 里需要适配

    elif model_name == 'RF':
        params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300, step=50)
        params['max_depth'] = trial.suggest_int('max_depth', 5, 30)
        params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
        params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 10)

    ModelClass = globals()[f"{model_name}Recommend"]
    model = ModelClass(info['item_name'], info['sparse_features'], info['dense_features'],
                       seed=42, k=3, **params)

    # 树模型通常没有显式的 valid_data 参数用于早停（除了 XGB/LGBM 原生接口）
    # 你的 Wrapper 中 fit 只有 train_data 参数，这里需要注意
    # 如果 Wrapper 支持传入 eval_set，则可以利用 valid
    # 否则只能 fit(train) 然后 score(valid)
    # 假设你的 fit 方法签名是 fit(train_data)
    model.fit(train)

    # 评分
    score = model.score_test(valid, methods=['auc'])[0]
    return score
# --------------------------------------------------------------------------
# 主运行逻辑
# --------------------------------------------------------------------------
def run_optimization():
    all_best_params = {}

    for data_type in DATASETS:
        all_best_params[data_type] = {}
        print(f"\n====== Optimizing {data_type} ======")

        # 1. 准备数据 (CCA Strategy)
        train_cca, valid_cca, info = get_data_for_tuning(data_type, use_complete_case=True)

        # 2. 准备数据 (Missing Strategy for CoMICE)
        train_miss, valid_miss, _ = get_data_for_tuning(data_type, use_complete_case=False)

        # === Group 1: NN Models ===
        for model_name in NN_MODELS:
            print(f"--- Tuning NN Model: {model_name} ---")
            # ... (Phase 1 & Phase 2 逻辑同前) ...
            pass  # 略去重复代码，参照上一轮回答

        # === Group 2: Tree Models ===
        for model_name in TREE_MODELS:
            print(f"--- Tuning Tree Model: {model_name} ---")
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))

            # 使用 objective_tree
            study.optimize(
                lambda t: objective_tree(t, model_name, train_cca, valid_cca, info),
                n_trials=20
            )

            all_best_params[data_type][model_name] = study.best_params
            print(f"Best {model_name} AUC: {study.best_value:.4f}")

    # 保存
    with open("all_models_best_params.json", "w") as f:
        json.dump(all_best_params, f, indent=4)