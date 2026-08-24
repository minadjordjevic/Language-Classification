import time

import torch
from sklearn.metrics import accuracy_score, f1_score


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    scheduler,
    device
):
    model.train()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    start_time = time.time()

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = torch.nn.functional.cross_entropy(
            logits,
            labels
        )

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        predictions = torch.argmax(logits, dim=1)

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )
        all_labels.extend(
            labels.detach().cpu().numpy()
        )

    epoch_loss = total_loss / len(data_loader)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    elapsed_time = time.time() - start_time

    return {
        "loss": epoch_loss,
        "accuracy": accuracy,
        "f1": f1,
        "time": elapsed_time
    }


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    device
):
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    start_time = time.time()

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = torch.nn.functional.cross_entropy(
            logits,
            labels
        )

        total_loss += loss.item()

        predictions = torch.argmax(logits, dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )
        all_labels.extend(
            labels.cpu().numpy()
        )

    loss = total_loss / len(data_loader)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    elapsed_time = time.time() - start_time

    return {
        "loss": loss,
        "accuracy": accuracy,
        "f1": f1,
        "time": elapsed_time,
        "predictions": all_predictions,
        "labels": all_labels
    }
