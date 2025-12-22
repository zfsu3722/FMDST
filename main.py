import numpy as np
import json
from model.LSTM_Transformer import LSTM_Transformer
from model.TCN_Transformer import TCN_Transformer
from model.FMDST import FMDST
from train_evaluate import train_and_evaluate_model
from train_evaluate import fine_tune_and_evaluate_model,evaluate_data_model
from model.LSTM import ShareLSTM
from model.share_Transformer_share_fc import SharedTransformer
from Data_Preprocessing.AVG import compute_average_metrics
from Review_model.LSTM_TCN_Transformer import LSTM_TCN_Transformer




# 辅助函数：将 NumPy 类型转为 Python 原生类型，以便 JSON 序列化
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

if __name__ == "__main__":
    # """
    # 在这写入需要训练的模型名字
    models = [ShareLSTM,SharedTransformer,LSTM_Transformer, TCN_Transformer,FMDST]
    # 设置参数
    num_runs = 1  # 每个模型跑几次
    results = []

    # 外层是 run 次数，内层是模型
    for run in range(num_runs):
        print(f"\n\n===== Run {run + 1} / {num_runs} =====")

        for model in models:
            model_name = model.__name__
            print(f"\n=== Training Model: {model_name} (Run {run + 1}) ===")

            # 训练并评估模型
            result = train_and_evaluate_model(model, lr=0.002)


            # 构建结果结构
            formatted_result = {
                "model_name": model_name,
                "run_id": run + 1,  # 添加 run 编号方便后续分析
                "lr": 0.002,
                "total_metrics": {
                    "mae": result['mae'],
                    "rmse": result['rmse'],
                    "r2": result['r2']
                },
                "step_metrics": {
                    "mae_steps": result['mae_per_step'],
                    "rmse_steps": result['rmse_per_step'],
                    "r2_steps": result['r2_per_step']
                }
            }

            # 转换 NumPy 类型为 Python 原生类型
            formatted_result = convert_numpy_types(formatted_result)

            results.append(formatted_result)


            with open("result_pred60_7feature.txt", "a", encoding="utf-8") as f:
                f.write(json.dumps(formatted_result, indent=4) + "\n\n")
    #看多次训练的平均结果
    compute_average_metrics("result_pred60_7feature.txt", "result_pred60_7feature_avg.txt")



    """
    # finetune main
    models = [LSTM_Transformer, TCN_Transformer, FMDST]
    num_runs = 1
    results = []

    for run in range(num_runs):
        print(f"\n\n===== Fine-tuning Run {run + 1} / {num_runs} =====")
        for model in models:
            model_name = model.__name__
            print(f"\n=== Fine-tuning Model: {model_name} ===")

            result = fine_tune_and_evaluate_model(
                model_class=model,
                save_directory_data='clean_power_his60_future6_7features_Season',
                lr=0.002,
                num_epochs=100,
                patience=15
            )

            formatted_result = {
                "model_name": model_name,
                "run_id": run + 1,
                "lr": 0.002,
                "total_metrics": {
                    "mae": result['mae'],
                    "rmse": result['rmse'],
                    "r2": result['r2']
                },
                "step_metrics": {
                    "mae_steps": result['mae_per_step'],
                    "rmse_steps": result['rmse_per_step'],
                    "r2_steps": result['r2_per_step']
                }
            }

            formatted_result = convert_numpy_types(formatted_result)
            results.append(formatted_result)
            with open("result_pred6_7feature_finetune_season.txt", "a", encoding="utf-8") as f:
                f.write(json.dumps(formatted_result, indent=4) + "\n\n")
    """


    """
    # evaluate main
    models = [ShareLSTM, LSTM_Transformer, TCN_Transformer, FMDST]
    # models = [LSTM_TCN_Transformer]

    num_runs = 1  # 评估是确定性的，但保留 run 结构以兼容
    results = []

    for run in range(num_runs):
        print(f"\n\n===== Evaluation Run {run + 1} / {num_runs} =====")

        for model_class in models:
            model_name = model_class.__name__
            print(f"\n=== Evaluating Model: {model_name} (Run {run + 1}) ===")

            # 调用评估函数（不训练）
            result = evaluate_data_model(model_class)

            # 构建结果结构
            formatted_result = {
                "model_name": model_name,
                "run_id": run + 1,
                "total_metrics": {
                    "mae": result['mae'],
                    "rmse": result['rmse'],
                    "r2": result['r2']
                },
                "step_metrics": {
                    "mae_steps": result['mae_per_step'],
                    "rmse_steps": result['rmse_per_step'],
                    "r2_steps": result['r2_per_step']
                }
            }

            # 转换 NumPy 类型为 Python 原生类型（确保 JSON 可序列化）
            formatted_result = convert_numpy_types(formatted_result)

            results.append(formatted_result)

            # 追加写入结果文件
            with open("result_pred6_7feature_evaluate_original_data_with_missing.txt", "a", encoding="utf-8") as f:
                f.write(json.dumps(formatted_result, indent=4) + "\n\n")

        """

