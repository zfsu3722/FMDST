import json
from collections import defaultdict


def compute_average_metrics(file_path, output_file):
    """
    从指定文件中读取 JSON 格式的实验结果，计算每个模型的平均指标，并将结果保存到新文件中。

    参数:
        file_path (str): 输入文件路径，包含原始实验结果。
        output_file (str): 输出文件路径，用于保存计算后的平均结果。

    返回:
        None: 结果将直接写入输出文件。
    """
    results = []

    # 读取文件内容
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        entries = content.strip().split("\n\n")
        for entry in entries:
            try:
                result = json.loads(entry)
                results.append(result)
            except json.JSONDecodeError as e:
                print(f"JSON 解码错误: {e}，跳过该条目：{entry[:50]}...")

    # 按模型名分组
    grouped_results = defaultdict(list)
    for item in results:
        model_name = item["model_name"]
        grouped_results[model_name].append(item)

    # 计算每个模型的平均指标
    final_avg_results = []

    for model_name, runs in grouped_results.items():
        total_mae_sum = 0
        total_rmse_sum = 0
        total_r2_sum = 0

        step_mae_sum = [0] * len(runs[0]["step_metrics"]["mae_steps"])
        step_rmse_sum = [0] * len(runs[0]["step_metrics"]["rmse_steps"])
        step_r2_sum = [0] * len(runs[0]["step_metrics"]["r2_steps"])

        num_runs = len(runs)

        for run in runs:
            # 总体指标
            total_mae_sum += run["total_metrics"]["mae"]
            total_rmse_sum += run["total_metrics"]["rmse"]
            total_r2_sum += run["total_metrics"]["r2"]

            # 每一步指标
            for i, val in enumerate(run["step_metrics"]["mae_steps"]):
                step_mae_sum[i] += val
            for i, val in enumerate(run["step_metrics"]["rmse_steps"]):
                step_rmse_sum[i] += val
            for i, val in enumerate(run["step_metrics"]["r2_steps"]):
                step_r2_sum[i] += val

        avg_total_mae = round(total_mae_sum / num_runs, 4)
        avg_total_rmse = round(total_rmse_sum / num_runs, 4)
        avg_total_r2 = round(total_r2_sum / num_runs, 4)

        avg_step_mae = [round(val / num_runs, 4) for val in step_mae_sum]
        avg_step_rmse = [round(val / num_runs, 4) for val in step_rmse_sum]
        avg_step_r2 = [round(val / num_runs, 4) for val in step_r2_sum]

        final_avg_results.append({
            "model_name": model_name,
            "avg_total_metrics": {
                "mae": avg_total_mae,
                "rmse": avg_total_rmse,
                "r2": avg_total_r2
            },
            "avg_step_metrics": {
                "mae_steps": avg_step_mae,
                "rmse_steps": avg_step_rmse,
                "r2_steps": avg_step_r2
            }
        })

    # 打印或写入最终结果
    for item in final_avg_results:
        print(json.dumps(item, indent=4))

    # 写入到新的文本文件中，每条记录之间空一行，方便阅读
    with open(output_file, "w", encoding="utf-8") as f:
        for item in final_avg_results:
            json_str = json.dumps(item, indent=4, ensure_ascii=False)
            f.write(json_str + "\n\n")  # 每个模型之间空一行

    print(f"平均结果已保存至 {output_file}")