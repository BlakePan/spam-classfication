import argparse
import datetime

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
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
    # TODO: share for serving
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
        labels, preds, average="micro"
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

    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        token_ids, attention_masks, labels = batch

        with torch.no_grad():
            # Forward pass
            eval_output = model(token_ids, attention_mask=attention_masks)

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

    # Training loop
    epochs = config_data.get("epochs", 4)
    train_dataloader = dataloader_res.get("train_dataloader")
    val_dataloader = dataloader_res.get("val_dataloader")
    for _ in tqdm(range(epochs), desc="Epoch"):
        train_loss = train_steps(model, train_dataloader, optimizer, device)
        val_results = val_steps(model, val_dataloader, device)

        print(f"Train Loss: {train_loss}")
        print("Eval metrics")
        print("Accuracy: {}".format(np.mean(val_results["accuracy"])))
        print("Precision: {}".format(np.mean(val_results["precision"])))
        print("Recall: {}".format(np.mean(val_results["recall"])))
        print("F1_score: {}".format(np.mean(val_results["f1_score"])))
        print("------")

    # Save model
    model_path = f"{pretrained_model.replace('/', '_')}-{get_timestamp()}.pt"
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add argument definitions
    parser.add_argument("-c", "--config", help="Path to the config file", required=True)

    args = parser.parse_args()
    main(args)
