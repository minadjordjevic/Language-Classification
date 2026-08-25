import time

import mlflow
import torch

from src.models import LanguageClassifier
from src.train import train_one_epoch, evaluate


def run_experiment(
    experiment_name,
    model_name,
    num_classes,
    train_loader,
    val_loader,
    test_loader,
    device,
    learning_rate=2e-5,
    dropout=0.1,
    epochs=2,
):
    """
    Train and evaluate one model configuration.
    """

    model = LanguageClassifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    mlflow.set_experiment("Language-Classification")

    with mlflow.start_run(run_name=experiment_name):

        mlflow.log_params({
            "model_name": model_name,
            "num_classes": num_classes,
            "learning_rate": learning_rate,
            "dropout": dropout,
            "epochs": epochs,
            "batch_size": train_loader.batch_size,
            "device": str(device),
        })

        training_start = time.time()

        for epoch in range(epochs):

            train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=None,
            device=device,
            epoch=epoch + 1,
        )

            val_metrics = evaluate(
                model=model,
                data_loader=val_loader,
                device=device,
            )

            mlflow.log_metrics({
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
            }, step=epoch)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f}"
            )

        training_time = time.time() - training_start

        test_metrics = evaluate(
            model=model,
            data_loader=test_loader,
            device=device,
        )

        mlflow.log_metrics({
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "training_time_seconds": training_time,
            "inference_time_seconds": test_metrics["time"],
        })

        print("\nTest results:")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"F1: {test_metrics['f1']:.4f}")
        print(f"Training time: {training_time:.2f}s")
        print(f"Inference time: {test_metrics['time']:.2f}s")

    return model, test_metrics
