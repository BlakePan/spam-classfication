import argparse
import datetime
import os

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer


def get_timestamp():
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M")
    return timestamp


def load_data(data_path):
    schema = {"label": [], "sentence": []}
    df = pd.DataFrame(schema)

    with open(data_path, "r") as f:
        for line in f.readlines():
            label, sentence = line.split("\t", 1)
            label = 0 if label == "ham" else 1
            df.loc[len(df)] = {"label": label, "sentence": sentence.lower()}

    sentences = df.sentence.values
    labels = df.label.values

    return {
        "sentences": sentences,
        "labels": labels,
    }


def get_encoding(tokenizer, sentence, max_length):
    return tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=max_length,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors="pt",
    )


def prepare_dataloader(
    pretrained_model,
    max_length,
    sentences,
    labels,
    dataloader_seed=None,
    val_ratio=0.2,
    batch_size=8,
):
    input_ids = []
    attention_masks = []

    # Encode
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    for sent in sentences:
        encoding = get_encoding(tokenizer, sent, max_length)
        input_ids.append(encoding["input_ids"])
        attention_masks.append(encoding["attention_mask"])

    # Tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Split dataset
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size=val_ratio,
        random_state=dataloader_seed,
        stratify=labels,
    )

    # Train and val sets
    train_set = TensorDataset(
        input_ids[train_idx], attention_masks[train_idx], labels[train_idx]
    )
    val_set = TensorDataset(
        input_ids[val_idx], attention_masks[val_idx], labels[val_idx]
    )

    # DataLoaders
    train_dataloader = DataLoader(
        train_set, sampler=RandomSampler(train_set), batch_size=batch_size
    )
    val_dataloader = DataLoader(
        val_set, sampler=SequentialSampler(val_set), batch_size=batch_size
    )

    return {
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
    }


def import_model(pretrained_model, num_labels, learning_rate=5e-5, eps=1e-08):
    if pretrained_model == "textattack/bert-base-uncased-yelp-polarity":
        from transformers import BertForSequenceClassification

        model = BertForSequenceClassification.from_pretrained(
            pretrained_model, num_labels=num_labels
        )
    elif pretrained_model == "distilbert-base-uncased":
        from transformers import DistilBertForSequenceClassification

        model = DistilBertForSequenceClassification.from_pretrained(
            pretrained_model, num_labels=num_labels
        )
    else:
        raise Exception("Pre-trained model not support")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=eps)

    return {
        "model": model,
        "optimizer": optimizer,
    }


def cal_metrics(preds: np.array, labels: np.array):
    acc = accuracy_score(labels, preds)
    prec, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    return {"accuracy": acc, "precision": prec, "recall": recall, "f1_score": f1}


def train_steps(model, train_dataloader, optimizer, device):
    # Set model to training mode
    model.train()

    loss = 0
    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        token_ids, attention_masks, labels = batch
        optimizer.zero_grad()

        # Forward pass
        train_output = model(token_ids, attention_mask=attention_masks, labels=labels)

        # Backward pass
        train_output.loss.backward()
        optimizer.step()

        # Update loss
        loss += train_output.loss.item()

    return loss


def val_steps(model, val_dataloader, device):
    # Set model to evaluation mode
    model.eval()

    # Metrics
    accuracy = []
    precision = []
    recall = []
    f1_score = []
    loss = 0

    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        token_ids, attention_masks, labels = batch

        with torch.no_grad():
            # Forward pass
            eval_output = model(
                token_ids, attention_mask=attention_masks, labels=labels
            )

        loss += eval_output.loss.item()

        # Cal metrics
        logits = eval_output.logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()
        labels = labels.to("cpu").numpy().flatten()
        metrics = cal_metrics(preds, labels)

        # Update metrics
        accuracy.append(metrics["accuracy"])
        if metrics["precision"] != np.nan:
            precision.append(metrics["precision"])
        if metrics["recall"] != np.nan:
            recall.append(metrics["recall"])
        if metrics["f1_score"] != np.nan:
            f1_score.append(metrics["f1_score"])

    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def main(args):
    # Load config
    with open(args.config, "r") as f:
        config_data = yaml.safe_load(f)

    # Import model
    pretrained_model = config_data.get("pretrained_model")
    num_labels = config_data.get("num_labels")
    learning_rate = float(config_data.get("learning_rate"))
    eps = float(config_data.get("eps"))
    model_res = import_model(
        pretrained_model, num_labels, learning_rate=learning_rate, eps=eps
    )
    model = model_res.get("model")
    optimizer = model_res.get("optimizer")

    # Prepare dataloader
    data_path = config_data.get("data_path")
    data_res = load_data(data_path)

    max_length = config_data.get("text_length", 32)
    dataloader_seed = config_data.get("dataloader_seed")
    val_ratio = config_data.get("val_ratio")
    batch_size = config_data.get("batch_size")
    sentences = data_res.get("sentences")
    labels = data_res.get("labels")
    dataloader_res = prepare_dataloader(
        pretrained_model,
        max_length,
        sentences,
        labels,
        dataloader_seed=dataloader_seed,
        val_ratio=val_ratio,
        batch_size=batch_size,
    )

    # Setup device
    if torch.cuda.is_available():
        cuda_index = config_data.get("cuda_index", "0")
        device = torch.device(f"cuda:{cuda_index}")
        model.cuda()
    else:
        device = torch.device("cpu")

    # Init tensorboard writer
    writer = SummaryWriter() if args.enable_tensorboard else None

    # Init folder for saving model
    save_path = f"{pretrained_model.replace('/', '_')}-{get_timestamp()}"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Training loop
    epochs = config_data.get("epochs", 4)
    train_dataloader = dataloader_res.get("train_dataloader")
    val_dataloader = dataloader_res.get("val_dataloader")
    for epoch in tqdm(range(epochs), desc="Epoch"):
        # Train and validate
        train_loss = train_steps(model, train_dataloader, optimizer, device)
        val_results = val_steps(model, val_dataloader, device)

        # Show metrics
        val_loss = val_results.get("loss")
        avg_accuracy = np.mean(val_results["accuracy"])
        avg_precision = np.mean(val_results["precision"])
        avg_recall = np.mean(val_results["recall"])
        avg_f1_score = np.mean(val_results["f1_score"])
        print()
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print("Eval metrics")
        print(f"Accuracy: {avg_accuracy}")
        print(f"Precision: {avg_precision}")
        print(f"Recall: {avg_recall}")
        print(f"F1_score: {avg_f1_score}")
        print("------")

        # Log to tensorboard
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", avg_accuracy, epoch)
            writer.add_scalar("Precision/val", avg_precision, epoch)
            writer.add_scalar("Recall/val", avg_recall, epoch)
            writer.add_scalar("F1_score/val", avg_f1_score, epoch)

        # Save model
        if epoch == 0:
            prev_val_loss = val_loss
            prev_val_acc = avg_accuracy
            prev_val_prec = avg_precision
            prev_val_recall = avg_recall
            prev_val_f1 = avg_f1_score
            trigger = False
        else:
            trigger = (
                val_loss < prev_val_loss
                or avg_accuracy > prev_val_acc
                or avg_precision > prev_val_prec
                or avg_recall > prev_val_recall
                or avg_f1_score > prev_val_f1
            )

        metrics_info = (
            "loss_{:.4f}-acc_{:.4f}-prec_{:.4f}-recall_{:.4f}-f1_{:.4f}".format(
                val_loss, avg_accuracy, avg_precision, avg_recall, avg_f1_score
            )
        )

        if trigger:
            print(f"[SAVE]: {metrics_info}")
            model.save_pretrained(f"{save_path}/{metrics_info}", from_pt=True)

        prev_val_loss = min(prev_val_loss, val_loss)
        prev_val_acc = max(prev_val_acc, avg_accuracy)
        prev_val_prec = max(prev_val_prec, avg_precision)
        prev_val_recall = max(prev_val_recall, avg_recall)
        prev_val_f1 = max(prev_val_f1, avg_f1_score)

    if writer is not None:
        writer.close()

    # Save model
    model.save_pretrained(f"{save_path}/{metrics_info}", from_pt=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add argument definitions
    parser.add_argument("-c", "--config", help="Path to the config file", required=True)
    parser.add_argument(
        "-tb", "--enable_tensorboard", help="Enable tensorboard", action="store_true"
    )

    args = parser.parse_args()
    main(args)
