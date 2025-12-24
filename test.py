import os
import json
import torch
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
# 假设你的模型类和工具函数在这些位置，请根据实际情况调整引用
# from models import CoMICERecommend
# from src.utils import load

# 为了脚本能独立运行，这里重新引用一下之前定义的关键类和函数名
# 实际运行时请确保 CoMICERecommend 类在上下文中可用

def save_experiment_results(save_dir, model_name, best_params, model_checkpoint):
    """保存实验结果的辅助函数"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 1. 保存最佳参数为 JSON
    params_path = os.path.join(save_dir, 'best_params.json')
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=4)

    # 2. 保存模型 Checkpoint
    model_path = os.path.join(save_dir, 'best_model.pth')
    torch.save(model_checkpoint, model_path)

    print(f"[{model_name}] 结果已保存至: {save_dir}")
def objective(trial, train_data, val_data, info, model_name):
    """Optuna 的目标函数"""

    # 1. 基础固定参数
    base_params = {
        'backbone': model_name,  # 关键：指定当前使用的 Backbone
        'batch_size': 1024,
        'epochs': 30,  # 搜索阶段 Epochs 可以少一点，加快速度
        'lambda_nce': 1.0,
        'temperature': 0.1,
        'cross_layers': 3,
        'attention_layers': 3,
        'num_heads': 2,
        'proj_dim': 16,
    }

    # 2. 搜索空间定义
    search_params = {
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'feature_dim': trial.suggest_categorical('feature_dim', [16, 32, 64]),
        # 动态调整 hidden_units 结构
    }

    # 根据 hidden_size 构建 hidden_units 列表
    hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512])
    search_params['hidden_units'] = [hidden_size, hidden_size // 2]

    # 合并参数
    current_params = base_params.copy()
    current_params.update(search_params)

    # 3. 初始化模型
    # 注意：这里需要传入 info 中的元数据
    model = CoMICERecommend(
        item_name=info['item_name'],
        sparse_features=info['sparse_features'],
        dense_features=info['dense_features'],
        standard_bool=True,
        seed=42,  # 搜索阶段固定种子
        k=3,
        **current_params
    )

    # 4. 训练
    # 搜索阶段使用单纯的 fit (只用 train 数据)
    model.fit(train_data.copy())

    # 5. 验证
    # 使用 score_test 获取 AUC，optuna 需要最大化这个值
    # score_test 返回列表 [auc_score, ...]
    scores = model.score_test(val_data.copy(), methods=['auc'])
    val_auc = scores[0]

    return val_auc
def main():
    # --- 配置区域 ---
    data_types = ['AWM', 'HIP', 'VID']
    # 确保你的 CoMICERecommend 的 _build_model 支持这些名字
    models = ['DCNv2', 'DeepFM', 'WideDeep', 'AutoInt', 'FiBiNET']

    amount = 10000  # 数据采样量
    train_ratio = 0.6
    val_ratio = 0.1  # 剩下 0.3 是测试集
    n_trials = 20  # 每个模型搜索多少次
    seed = 42

    # --- 主循环 ---
    for d_type in data_types:
        print(f"\n{'=' * 20} 正在处理数据集: {d_type} {'=' * 20}")

        # 1. 加载数据 (针对每个数据集只加载一次)
        try:
            train, valid, test, info = load(
                data_type=d_type,
                amount=amount,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                seed=seed
            )
            print(f"数据加载成功. Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")
        except Exception as e:
            print(f"数据集 {d_type} 加载失败: {e}")
            continue

        for m_name in models:
            print(f"\n--- 开始搜索模型: {m_name} (Dataset: {d_type}) ---")

            # 2. 创建 Optuna Study
            study = optuna.create_study(direction='maximize')

            # 使用 lambda 或 partial 将数据传入 objective
            study.optimize(
                lambda trial: objective(trial, train, valid, info, m_name),
                n_trials=n_trials
            )

            print(f"[{m_name}] 最佳 AUC: {study.best_value:.4f}")
            print(f"[{m_name}] 最佳参数: {study.best_params}")

            # 3. 最佳参数重新训练 (Retrain)
            # 策略：使用 Train + Valid 的数据量，以最佳参数重新训练，以获得更好的泛化能力
            print(f"[{m_name}] 使用最佳参数在 (Train+Valid) 上重新训练...")

            # 准备最终参数
            final_params = {
                'backbone': m_name,
                'batch_size': 1024,
                'epochs': 30,  # 最终训练 Epoch
                'lambda_nce': 1.0,
                'temperature': 0.1,
                'cross_layers': 3,
                'attention_layers': 3,
                'num_heads': 2,
                'proj_dim': 16,
            }
            # 更新 Optuna 搜索到的参数
            final_params.update(study.best_params)
            # 处理 hidden_units 的特殊逻辑
            if 'hidden_size' in final_params:
                hs = final_params.pop('hidden_size')  # 移除中间变量
                final_params['hidden_units'] = [hs, hs // 2]

            # 初始化最终模型
            best_model = CoMICERecommend(
                item_name=info['item_name'],
                sparse_features=info['sparse_features'],
                dense_features=info['dense_features'],
                standard_bool=True,
                seed=seed,
                k=3,
                **final_params
            )

            # 合并训练集和验证集
            full_train = pd.concat([train, valid], axis=0).reset_index(drop=True)
            best_model.fit(full_train)

            # (可选) 在测试集上评估并打印
            test_scores = best_model.score_test(test.copy(), methods=['auc', 'logloss'])
            print(f"[{m_name}] 最终测试集表现 -> AUC: {test_scores[0]:.4f}, LogLoss: {test_scores[1]:.4f}")

            # 4. 保存模型和参数
            # 构建保存路径: checkpoints/数据集名/模型名/
            save_dir = os.path.join('checkpoints', d_type, m_name)

            # 构建 Checkpoint 字典
            checkpoint = {
                'init_params': {
                    'item_name': info['item_name'],
                    'sparse_features': info['sparse_features'],
                    'dense_features': info['dense_features'],
                    'standard_bool': True,
                    'seed': seed,
                    'k': 3,
                    'kwargs': final_params
                },
                'preprocessing': {
                    'mapping': best_model.mapping,
                    'vocabulary_sizes': best_model.vocabulary_sizes,
                    'scaler': best_model.scaler,
                    'unique_item': best_model.unique_item,
                    'out_dim': best_model.out_dim
                },
                'model_state_dict': best_model.model.state_dict()
            }

            # 执行保存
            save_experiment_results(save_dir, m_name, final_params, checkpoint)

            # 清理显存，防止 OOM
            del best_model
            torch.cuda.empty_cache()
if __name__ == "__main__":
    main()