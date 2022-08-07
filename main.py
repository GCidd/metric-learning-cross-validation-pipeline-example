import os
import torch
import pandas as pd
from torch import nn
from typing import Dict, List, Tuple
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchmetrics import Accuracy, F1Score, Precision, Recall, StatScores
from torchvision import transforms
from pytorch_metric_learning import losses, miners, distances, reducers, regularizers, samplers, trainers

from networks import ResnetBackboneEmbeddingNet, DensenetBackboneEmbeddingNet, MobileNetV2BackboneEmbeddingNet, VGGBackboneEmbeddingNet
from networks import MLP
from utils import plot_embeddings, get_all_embeddings, evaluate_model, plot_cv_performance
from datasets import PathListDataset
from networks import TrunkEmbedder, TrunkEmbedderClassifier

device = torch.device("cuda")
torch.multiprocessing.set_start_method('spawn')

def train(models: Dict[str, nn.Module], dataset: PathListDataset, num_epochs: int, metrics: List[StatScores], **kwargs) -> None:
    """
    Trains the models (trunk, embedder, classifier) on the dataset by initializing optimizers and loss functions for each one,
    a miner and a sampler.

    Args:
        models (Dict[str, nn.Module]): 
            A dictionary with the following keys:
                * "trunk": trunk part of the model
                * "embedder": embedder part of the model
                * "classifier": classifier part of the model
        dataset (PathListDataset):
            Dataset to train on.
        num_epochs (int):
            Number of epochs to train for.
        metrics (Dict[str, StatScores]):
            A dictionary of metric name as keys and torchmetrics-metric objects to be computed as values.
    """
    n_classes = dataset.n_classes
    batch_size_per_class = kwargs.get("batch_size_per_class", 16)
    learning_rate = kwargs.get("learning_rate", 1e-5)
    length_before_new_iter = kwargs.get("length_before_new_iter", 1_000)

    batch_size = batch_size_per_class * n_classes

    trunk_optimizer = torch.optim.Adam(
        models["trunk"].parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )
    embedder_optimizer = torch.optim.Adam(
        models["embedder"].parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )
    classifier_optimizer = torch.optim.Adam(
        models["classifier"].parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )
    optimizers = {
        "trunk_optimizer": trunk_optimizer,
        "embedder_optimizer": embedder_optimizer,
        "classifier_optimizer": classifier_optimizer
    }
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    regularizer = regularizers.ZeroMeanRegularizer()
    loss = losses.TripletMarginLoss(
        margin=0.5,
        distance=distance,
        reducer=reducer,
        embedding_regularizer=regularizer
    )
    classification_loss = torch.nn.CrossEntropyLoss()
    loss_funcs = {"metric_loss": loss,
                  "classifier_loss": classification_loss}

    miner = miners.TripletMarginMiner(
        margin=0.5,
        distance=distance,
        type_of_triplets="hard"
    )

    sampler = samplers.MPerClassSampler(
        dataset.y,
        batch_size_per_class,
        length_before_new_iter=length_before_new_iter
    )

    trainer = trainers.TrainWithClassifier(
        models=models,
        optimizers=optimizers,
        batch_size=batch_size,
        loss_funcs=loss_funcs,
        mining_funcs={"tuple_miner": miner},
        dataset=dataset,
        data_device=device,
        dataloader_num_workers=0,
        sampler=sampler,
        
    )
    trainer.train(num_epochs=num_epochs)
    dataset.empty_cache()

    trunk_embedder_classifier = TrunkEmbedderClassifier(**models)

    train_results = evaluate_model(trunk_embedder_classifier, dataset, metrics)
    
    return train_results


def validate(trunk_embedder: TrunkEmbedder, trunk_embedder_classifier: TrunkEmbedderClassifier, metrics: Dict[str, StatScores], 
         val_dataset: PathListDataset, results_dir: str) -> Dict[str, float]:
    """
    Plots the embeddings that the trunk_embedder part of the model generate and evaluates model's performance.
    The model is evaluated on both the train and val dataset using the metrics provided.

    Args:
        trunk_embedder (TrunkEmbedder): 
            Trunk and embedder parts of the model as a single model.
        trunk_embedder_classifier (TrunkEmbedderClassifier):
            Trunk, embedder and classifier parts of the model as a single model.
        metrics (Dict[str, StatScores]):
            A dictionary of metric name as keys and torchmetrics-metric objects to be computed as values.
        val_dataset (PathListDataset): 
            Validation set of the dataset.
        results_dir (str):
            Directory to save the plotted embeddings.

    Returns:
        Dict[str, float]:
            Train and val using the metrics provided.
    """
    with torch.set_grad_enabled(False):
        val_embeddings, val_labels = get_all_embeddings(
            val_dataset, trunk_embedder)
        plot_embeddings(
            val_embeddings.cpu(),
            val_labels.cpu(),
            os.path.join(results_dir, "val_embeddings.png")
        )

        val_results = evaluate_model(trunk_embedder_classifier, val_dataset, metrics)
        pd.DataFrame.from_records([val_results]).to_csv(
            os.path.join(results_dir, "val_results.csv")
        )
        
    val_dataset.empty_cache()
    return val_results


def fewshot(trunk_embedder: TrunkEmbedder, trunk_embedder_classifier: TrunkEmbedderClassifier, metrics: List[StatScores],
            fewshot_train_dataset: PathListDataset, fewshot_val_dataset: PathListDataset, results_dir: str,
            num_epochs=10, **train_kwargs) -> dict[str, StatScores]:
    """
    Performs a fewshot process with the model. The model is initially trained on the fewshot_train_dataset for ```epochs```
    and then validated on the fewshot_val_dataset.

    Args:
        trunk_embedder (TrunkEmbedder):
            Trunk and embedder parts of the model as a single model.
        trunk_embedder_classifier (TrunkEmbedderClassifier):
            Trunk, embedder and classifier parts of the model as a single model.
        metrics (Dict[str, StatScores]):
            A dictionary of metric name as keys and torchmetrics-metric objects to be computed as values.
        fewshot_train_dataset (PathListDataset): 
            Train set of the fewshot dataset. Should be much smaller than the fewshot_val_dataset.
        fewshot_val_dataset (PathListDataset): 
            Validation set of the fewshot dataset. Should be much larger than the fewshot_train_dataset.
        results_dir (str): 
            Directory to save the plotted embeddings.
        num_epochs (int):
            Number of epochs to train. Should be small.

    Returns:
        Dict: 
            Train and val using the metrics provided.
    """
    models = {
        "trunk": trunk_embedder.trunk,
        "embedder": trunk_embedder.embedder,
        "classifier": trunk_embedder_classifier.classifier
    }
    train(
        models,
        fewshot_train_dataset,
        num_epochs,
        metrics=metrics,
        **train_kwargs
    )
    fewshot_train_dataset.empty_cache()

    with torch.set_grad_enabled(False):
        fewshot_results = evaluate_model(trunk_embedder_classifier, fewshot_val_dataset, metrics)
        pd.DataFrame.from_records([fewshot_results]).to_csv(
            os.path.join(results_dir, "fewshot_results.csv")
        )
        
        fewshot_embeddings, fewshot_labels = get_all_embeddings(
            fewshot_val_dataset, trunk_embedder)
        plot_embeddings(
            fewshot_embeddings.cpu(),
            fewshot_labels.cpu(),
            os.path.join(results_dir, "fewshot_embeddings.png")
        )
    fewshot_val_dataset.empty_cache()
    return fewshot_results


def zeroshot(trunk_embedder: TrunkEmbedder, trunk_embedder_classifier: TrunkEmbedderClassifier, metrics: List[StatScores],
             zeroshot_dataset: PathListDataset, results_dir: str) -> dict:
    """
    Performs a zeroshot process with the model.

    Args:
        trunk_embedder (TrunkEmbedder): 
            Trunk and embedder parts of the model as a single model.
        trunk_embedder_classifier (TrunkEmbedderClassifier): 
            Trunk, embedder and classifier parts of the model as a single model.
        metrics (Dict[str, StatScores]):
            A dictionary of metric name as keys and torchmetrics-metric objects to be computed as values.
        zeroshot_dataset (PathListDataset): 
            Dataset on which to perform the zershot process.
        results_dir (str): 
            Directory to save the plotted embeddings.

    Returns:
        Dict: 
            Train and val using the metrics provided.
    """
    with torch.set_grad_enabled(False):
        zeroshot_results = evaluate_model(trunk_embedder_classifier, zeroshot_dataset, metrics)
        pd.DataFrame.from_records([zeroshot_results]).to_csv(
            os.path.join(results_dir, "zeroshot_results.csv")
        )
        
        zeroshot_embeddings, zeroshot_labels = get_all_embeddings(
            zeroshot_dataset, trunk_embedder)
        plot_embeddings(
            zeroshot_embeddings.cpu(),
            zeroshot_labels.cpu(),
            os.path.join(results_dir, "zeroshot_embeddings.png")
        )
    zeroshot_dataset.empty_cache()
    return zeroshot_results


def cross_validate(trunk_obj: nn.Module, trunk_kwargs: dict, embedder_layer_sizes: List[int], classifier_layer_sizes: List[int],
                   dataset_dir: str, shot_dataset_dir: str, train_transform: transforms.Compose, val_transform: transforms.Compose, 
                   metrics: Dict[str, StatScores], cv: int = 10, train_kwargs: dict = None, fewshot_kwargs: dict = None, results_dir="./results") -> None:
    """
    Performs a cv-fold cross validation on the dataset. During the cross validation, a model is created according to the model_data dictionary on the dataset in dataset_dir.
    After training the model on the training set of the dataset inside dataset_dir, it is validateded on the val set.
    Then, a fewshot and zeroshot process is performed on the dataset in shot_dataset_dir.

    Args:
        trunk (nn.Module): 
            Trunk part of the model. Uninitialized nn.Module object.
        trunk_kwargs (dict):
            Keyword arguments to use for the initialization of the trunk model.
        embedder_layer_sizes (List[int]):
            List of layer sizes for the embedder part of the model. Last size should be the same as the first size of classifier_layer_sizes.
        classifier_layer_sizes (List[int]):
            List of layer sizes for the classifier part of the model. First size should be the same as the last size of embedder_layer_sizes.
            Last size is the output of the model (number of classes).
        dataset_dir (str):
            Directory of the dataset to train the model on. It is split into train and validation sets.
        shot_dataset_dir (str):
            Directory of the dataset to perform the fewshot and zeroshot process.
        train_transform (transforms.Compose):
            Transformations to perform on the samples during training.
        val_transform (transforms.Compose):
            Transformations to perform on the samples during validating.
        metrics (Dict[str, StatScores]):
            A dictionary of metric name as keys and torchmetrics-metric objects to be computed as values.
        cv (int, optional): 
            Number of folds to perform. Defaults to 10.
        results_dir (str, optional):
            Output directory of the results. Defaults to ./results
    """
    results_per_set = {
        "Train": {},
        "Validation": {},
        "Fewshot": {},
        "Zeroshot": {},
    }
    for _set in results_per_set.keys():
        results_per_set[_set] = {
            metric_name: []
            for metric_name in metrics.keys()
        }

    if train_kwargs is None:
        train_kwargs = {}

    if fewshot_kwargs is None:
        fewshot_kwargs = {}

    X, y = PathListDataset.walk_dataset(dataset_dir)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    folds = skf.split(X, y)

    X_shot, y_shot = PathListDataset.walk_dataset(shot_dataset_dir)
    # during few shot the model only "sees" SOME of the new samples, so the val size is larger than the train val
    X_shot_train, X_shot_val, y_shot_train, y_shot_val = train_test_split(
        X_shot, y_shot, test_size=0.75, random_state=0, stratify=y_shot)
    fewshot_train_dataset = PathListDataset(
        X_shot_train, y_shot_train, transform=train_transform, device=device)
    fewshot_val_dataset = PathListDataset(
        X_shot_val, y_shot_val, transform=val_transform)

    for fold_i, (train_index, val_index) in enumerate(folds, start=1):
        print(f"Fold {fold_i}/{cv}")

        trunk = trunk_obj(**trunk_kwargs)
        output_size = trunk.last_layer_in_features
        trunk.replace_output_layer(nn.Identity())
        embedder = MLP([output_size] + embedder_layer_sizes)
        classifier = MLP([embedding_size] +
                         classifier_layer_sizes, final_relu=True)
        trunk.to(device)
        embedder.to(device)
        classifier.to(device)
        models = {
            "trunk": trunk,
            "embedder": embedder,
            "classifier": classifier
        }
        trunk_embedder = TrunkEmbedder(trunk, embedder)
        trunk_embedder_classifier = TrunkEmbedderClassifier(
            trunk, embedder, classifier)
        train_dataset = PathListDataset(
            X[train_index],
            y[train_index],
            transform=train_transform,
            device=device
        )

        validation_dataset = PathListDataset(
            X[val_index],
            y[val_index],
            transform=val_transform,
            device=device
        )

        fold_results_dir = os.path.join(results_dir, "folds", str(fold_i))
        Path(fold_results_dir).mkdir(parents=True, exist_ok=True)

        print("\tTraining model")
        train_results = train(models, train_dataset, metrics=metrics, **train_kwargs)
        pd.DataFrame.from_records([train_results]).to_csv(
            os.path.join(fold_results_dir, "train_results.csv")
        )

        train_embeddings, train_labels = get_all_embeddings(train_dataset, trunk_embedder)
        plot_embeddings(
            train_embeddings.cpu(),
            train_labels.cpu(),
            os.path.join(results_dir, "train_embeddings.png")
        )

        print("\tValidating model")
        validation_results = \
            validate(trunk_embedder, trunk_embedder_classifier, metrics,
                 validation_dataset, fold_results_dir)

        print("\tZeroshoting model")
        zeroshot_results = zeroshot(
            trunk_embedder, trunk_embedder_classifier, metrics, fewshot_val_dataset, fold_results_dir)

        print("\tFewshoting model")
        fewshot_results = fewshot(trunk_embedder, trunk_embedder_classifier, metrics,
                                  fewshot_train_dataset, fewshot_val_dataset, fold_results_dir, **fewshot_kwargs)

        for metric_name in metrics.keys():
            results_per_set["Train"][metric_name] += [train_results[metric_name]]
            results_per_set["Validation"][metric_name] += [validation_results[metric_name]]
            results_per_set["Fewshot"][metric_name] += [zeroshot_results[metric_name]]
            results_per_set["Zeroshot"][metric_name] += [fewshot_results[metric_name]]

    for set_name, results in results_per_set.items():
        plot_cv_performance(results, results_dir, fig_name=set_name, cv=cv, metrics=list(metrics.keys()))
        pd.DataFrame(results).to_csv(
            os.path.join(results_dir, f"{set_name}_results.csv"), index=False
        )
        
    
if __name__ == '__main__':
    train_dataset_dir = "D:\\preprocessed_undersampled_v2"
    shot_dataset_dir = "D:\\preprocessed_undersampled_v2"
    embedding_size = 256
    n_classes = 3
    img_size = (224, 224)

    train_kwargs = {
        "batch_size_per_class": 4,
        "num_epochs": 1,
        "learning_rate": 1e-5,
        "length_before_new_iter": 200,
    }
    fewshot_train_kwargs = {
        "batch_size_per_class": 4,
        "num_epochs": 1,
        "learning_rate": 1e-5,
        "length_before_new_iter": 100,
    }
    metrics = {
        "Accuracy": Accuracy(),
        "Precision": Precision(average="macro", num_classes=n_classes),
        "Recall": Recall(average="macro", num_classes=n_classes),
        "F1": F1Score(average="macro", num_classes=n_classes)
    }

    mean, std = PathListDataset.calculate_mean_std(
        train_dataset_dir,
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(img_size),
        ])
    )

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(img_size),
        transforms.RandomResizedCrop(
            img_size, scale=(0.7, 1.0), ratio=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(img_size),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    results_dir = "./results/VGG16"
    cross_validate(
        trunk_obj=VGGBackboneEmbeddingNet,
        trunk_kwargs={
            "model_size": 16,
            "trainable_grad_depth": 1
        },
        embedder_layer_sizes=[embedding_size],
        classifier_layer_sizes=[embedding_size, n_classes],
        dataset_dir=train_dataset_dir,
        shot_dataset_dir=shot_dataset_dir,
        train_transform=train_transform,
        val_transform=val_transform,
        metrics=metrics,
        cv=3,
        train_kwargs=train_kwargs,
        fewshot_kwargs=fewshot_train_kwargs,
        results_dir=results_dir
    )

    results_dir = "./results/ResNet18"
    cross_validate(
        trunk_obj=ResnetBackboneEmbeddingNet,
        trunk_kwargs={
            "model_size": 18,
            "trainable_grad_depth": 1
        },
        embedder_layer_sizes=[embedding_size],
        classifier_layer_sizes=[embedding_size, n_classes],
        dataset_dir=train_dataset_dir,
        shot_dataset_dir=shot_dataset_dir,
        train_transform=train_transform,
        val_transform=val_transform,
        metrics=metrics,
        cv=3,
        train_kwargs=train_kwargs,
        fewshot_kwargs=fewshot_train_kwargs,
        results_dir=results_dir
    )
    from result_processing import create_summary
    create_summary("./results", average=True)
    # create_summary("./results", best="F1")