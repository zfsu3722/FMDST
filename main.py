import numpy as np
import json
from model.LSTM_Transformer import LSTM_Transformer
from model.TCN_Transformer import TCN_Transformer
from model.FMDST import FMDST
from train_evaluate import train_and_evaluate_model



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
    # 在这写入需要训练的模型名字
    models = [LSTM_Transformer, TCN_Transformer,FMDST]
    # 设置参数
    num_runs = 3  # 每个模型跑几次
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


            with open("result.txt", "a", encoding="utf-8") as f:
                f.write(json.dumps(formatted_result, indent=4) + "\n\n")
    #看多次训练的平均结果
    # compute_average_metrics("result.txt", "result_avg.txt")