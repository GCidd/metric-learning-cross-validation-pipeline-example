import os
import pandas as pd
import numpy as np


def create_summary(results_dir: str, average: bool = True, best: str = None):
    """
    Generates a summary csv file of the results inside the results_dir directory.

    Args:
        results_dir (str):
            Directory of the results of the different models.
        average (bool):
            Whether to summarize the model's performance by averaging it from all the folds.
        best (str):
            Summarizes the model's performance by taking its best performance across all models.
    """
    for _set in ["Train", "Validation", "Fewshot", "Zeroshot"]:
        best_results = {
            "Model": [],
        }
        for model in os.listdir(results_dir):
            if os.path.isfile(os.path.join(results_dir, model)): continue
            best_results["Model"] += [model]
            full_model_results_filepath = os.path.join(
                results_dir, model, f"{_set}_results.csv")
            if not os.path.exists(full_model_results_filepath):
                continue
            df = pd.read_csv(full_model_results_filepath)
            metrics = list(df.keys())
            for metric in metrics:
                if best_results.get(metric, None) is None:
                    best_results[metric] = []
            
            if average:
                for metric in metrics:
                    performance = np.mean(df[metric].mean())
                    best_results[metric] += [performance]
            else:
                best_result = df[best].argmax()
                for metric in metrics:
                    best_results[metric] += [df[metric].values[best_result]]
        pd.DataFrame(best_results).to_csv(os.path.join(results_dir, f"{_set}_performance_summary.csv"))
