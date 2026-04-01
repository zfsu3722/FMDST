import numpy as np
import json

from Time_Series_Library_main.adapters.informer_adapter import InformerAdapter
from Time_Series_Library_main.adapters.patchtst_adapter import PatchTSTAdapter
from train_evaluate import train_and_evaluate_model
from Data_Preprocessing.AVG import compute_average_metrics
from model.STAEformer import STAEformerWrapper
from model.MegaCRN import MegaCRNWrapper
from model.LSTM import ShareLSTM
from model.share_Transformer_share_fc import SharedTransformer
from model.GAT_LSTM_Learning_adj import GAT_LSTM_Learning_adj_Model
from model.strep_wrapper import STRePWrapper
from model.LSTM_Transformer import LSTM_Transformer
from model.TCN_Transformer import TCN_Transformer
from model.FMDST import FMDST
from Ablation_model.LSTM_Transformer_No_multi_head import LSTM_Transformer_No_multi_head
from Ablation_model.TCN_Transformer_No_multi_head import TCN_Transformer_No_multi_head
from Ablation_model.FMDST_No_multi_head import FMDST_No_multi_head
from Ablation_model.LSTM_Transformer_No_Residual_Gate import LSTM_Transformer_No_Residual_Gate
from Ablation_model.TCN_Transformer_No_Residual_Gate import TCN_Transformer_No_Residual_Gate
from Ablation_model.FMDST_No_Residual_Gate import FMDST_No_Residual_Gate
from Ablation_model.LSTM_TCN_Transformer import LSTM_TCN_Transformer
from Ablation_model.LSTM_Transformer_No_memory import LSTM_Transformer_No_Memory





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

    # 基线和FMDST模型
    models = [MegaCRNWrapper,ShareLSTM,SharedTransformer,GAT_LSTM_Learning_adj_Model,STAEformerWrapper,STRePWrapper,InformerAdapter,PatchTSTAdapter,LSTM_Transformer,TCN_Transformer,FMDST,]
    """
    # 消融模型
    models = [LSTM_Transformer_No_multi_head,TCN_Transformer_No_multi_head,FMDST_No_multi_head,
              LSTM_Transformer_No_Residual_Gate,TCN_Transformer_No_Residual_Gate,FMDST_No_Residual_Gate,
              LSTM_Transformer_No_Memory,LSTM_TCN_Transformer]
    """
    # 设置参数
    num_runs = 6  # 每个模型跑几 次
    results = []

    # 外层是 run 次数，内层是模型
    for run in range(num_runs):
        print(f"\n\n===== Run {run + 1} / {num_runs} =====")

        for model in models:
            model_name = model.__name__
            print(f"\n=== Training Model: {model_name} (Run {run + 1}) ===")

            # 训练并评估模型
            result = train_and_evaluate_model(model,lr=0.002)


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

            # 自定义保存的文件
            with open("result_pred6_7feature_TVT7_diffblock_3_21_totalBaseline.txt", "a", encoding="utf-8") as f:
                f.write(json.dumps(formatted_result, indent=4) + "\n\n")

    #看多次训练的平均结果
    compute_average_metrics("result_pred6_7feature_TVT7_diffblock_3_21_totalBaseline.txt", "result_pred6_7feature_TVT7_diffblock_3_21_totalBaseline_avg.txt")
