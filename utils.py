import os
from typing import Dict, List, Tuple
from pytorch_metric_learning.testers import BaseTester
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import Tensor, nn
from itertools import product
from torchmetrics.classification.stat_scores import StatScores
import pandas as pd

from datasets import PathListDataset


def get_all_embeddings(dataset: PathListDataset, model: nn.Module) -> Tuple[List, List]:
    """
    Returns dataset's embeddings generated by the model.

    Args:
        dataset (PathListDataset): Dataset to extract embeddings.
        model (nn.Module): The trunk-embedder model.

    Returns:
        Tuple[List, List]: Embeddings that were generated and labels of the classes they belong to.
    """
    tester = BaseTester(normalize_embeddings=False, dataloader_num_workers=0)
    embeddings, labels = tester.get_all_embeddings(dataset, model)
    return embeddings, labels


def plot_embeddings(embeddings: List, y: List, filepath: str = "embeddings.png") -> None:
    """
    Plots embeddings on a figure and saves it to a 'filename' file.
    The embeddings are reduced to a 2D plane using PCA.

    Args:
        embeddings (List): Embeddings to plot.
        y (List): Classes of the embeddings.
        filepath (str): File path of the figure to save. Defaults to "embeddings.png".
    """
    pcaer = PCA(n_components=2)
    points = pcaer.fit_transform(embeddings)
    plt.figure(figsize=(19.2, 10.8), dpi=100)
    classes = np.unique(y)
    for _class in classes:
        idxs = np.where(y == _class)[0]
        plt.scatter(points[idxs, 0], points[idxs, 1])
    plt.legend([str(c) for c in classes])
    plt.savefig(filepath)
    plt.close()


def evaluate_model(model: nn.Module, dataset: PathListDataset, metrics: Dict[str, StatScores]) -> Dict[str, float]:
    """
    Scores model on the dataset, calculating the provided metrics.

    Args:
        dataset (PathListDataset): Dataset to score the model on.
        model (nn.Module): Model to test on dataset.
        metrics: (Dict[str, StatScores]): A dictionary of metric name as keys and torchmetrics-metric objects to be computed as values.

    Returns:
        dict: Dictionary containing the resulting metrics.
    """
    model.eval()

    y_trues = []
    y_preds = []
    for x, y in dataset:
        y_trues.append(y.item())
        x = x.unsqueeze(0).cuda()
        pred = model(x)
        pred = torch.argmax(pred).item()
        y_preds.append(pred)
    
    results = {}
    for metric_name, metric in metrics.items():
        results[metric_name] = metric(Tensor(y_preds).type(torch.int32), Tensor(y_trues).type(torch.int32)).item()
    
    return results

def plot_cv_performance(results: dict, destination_dir: str, fig_name: str, cv: int, metrics: List[str]) -> None:
    """
    Plots the metric performance on each fold during the cross-validation process and saves them
    in a csv format.

    Args:
        results (dict): Dictionary of the results. The dict's keys are the results for the different sets (train/test/fewshot/zeroshot)
    and the values are the metric result for each fold during the cross validation.
        destination_dir (str): Destination directory where to save the plots.
        fig_name (str): Name of the file plotted.
        cv (int): Number of folds included in the results.
        metrics (List[str]): List of metrics to plot.
    """
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_subplot(111)
    for m in metrics:
        x = np.arange(cv)
        y = results[m]
        plt.plot(x, y)
        for i in range(cv):
            ax.annotate("{:.2f}".format(y[i]*100), xy=(i, y[i]))
    plt.legend(metrics)
    plt.xticks(np.arange(cv))
    plt.savefig(os.path.join(
        destination_dir, f"{fig_name}_performance.png"))
    plt.close()