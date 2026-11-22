import argparse
import csv
import json
import platform
import random
import time
import urllib.request
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from torch import nn
from torch.nn import functional as F
from transformers import BertForSequenceClassification, BertTokenizerFast

from moderation import ARTIFACTS, MAX_LENGTH, MODEL_ID, NAMES, WordClassifier, encode, normalize, quantize, words

DATA_URL = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
SEED = 42
BATCH_SIZE = 32


def load_data(path):
    rows = {}
    conflicts = set()
    with path.open(encoding="utf-8", newline="") as source:
        for record in csv.DictReader(source):
            message = normalize(record["tweet"])
            label = int(record["class"] != "2")
            if message in rows and rows[message] != label:
                conflicts.add(message)
            if message:
                rows[message] = label
    return [(message, label) for message, label in rows.items() if message not in conflicts]


def train_word2vec(messages, vocab):
    pairs = []
    for message in messages:
        tokens = [vocab.get(word, 1) for word in words(message)[:MAX_LENGTH]]
        for index, center in enumerate(tokens):
            for other in range(max(0, index - 2), min(len(tokens), index + 3)):
                if other != index:
                    pairs.append((center, tokens[other]))
    pairs = torch.tensor(pairs)
    centers = nn.Embedding(len(vocab) + 2, 64, padding_idx=0)
    contexts = nn.Embedding(len(vocab) + 2, 64, padding_idx=0)
    optimizer = torch.optim.SGD([*centers.parameters(), *contexts.parameters()], lr=0.3)
    frequencies = torch.bincount(pairs[:, 1], minlength=len(vocab) + 2).float().pow(0.75)
    for _ in range(5):
        for indexes in torch.randperm(len(pairs)).split(512):
            center, context = pairs[indexes].T
            negative = torch.multinomial(frequencies, len(center) * 5, replacement=True).view(-1, 5)
            vectors = centers(center)
            positive_score = (vectors * contexts(context)).sum(-1)
            negative_score = (vectors.unsqueeze(1) * contexts(negative)).sum(-1)
            loss = -(F.logsigmoid(positive_score) + F.logsigmoid(-negative_score).sum(-1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return centers.weight.detach().clone()


def fit_classifier(model, inputs, labels, optimizer, epochs, weights):
    model.train()
    for epoch in range(epochs):
        for indexes in torch.randperm(len(labels)).split(BATCH_SIZE):
            scores = model(**{key: value[indexes] for key, value in inputs.items()})
            logits = scores if isinstance(scores, torch.Tensor) else scores.logits
            loss = F.cross_entropy(logits, labels[indexes], weight=weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"{type(model).__name__} epoch {epoch + 1}/{epochs}", flush=True)
    return model.eval()


def close_match(word, target):
    row = list(range(len(target) + 1))
    for index, char in enumerate(word, 1):
        previous, row = row, [index]
        for column, expected in enumerate(target, 1):
            row.append(min(row[-1] + 1, previous[column] + 1, previous[column - 1] + (char != expected)))
    return row[-1] <= 1


def rule_predict(message, lexicon):
    return int(any(
        abs(len(word) - len(term)) <= 1 and close_match(word, term)
        for word in words(message) for term in lexicon
    ))


def measure(name, predict, test_text, test_y, model_files):
    guesses = [predict(message) for message in test_text]
    samples = test_text[:128]
    for message in samples[:10]:
        predict(message)
    durations = []
    for message in samples:
        start = time.perf_counter()
        predict(message)
        durations.append((time.perf_counter() - start) * 1000)
    return {
        "model": name,
        "macro_f1": float(f1_score(test_y, guesses, average="macro")),
        "size_mb": sum(path.stat().st_size for path in model_files) / 1_000_000,
        "latency_ms": float(np.mean(durations)),
        "latency_runs": len(durations),
        "confusion": confusion_matrix(test_y, guesses, labels=[0, 1]).tolist(),
    }, guesses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--checkpoint", default=MODEL_ID)
    args = parser.parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    ARTIFACTS.mkdir(exist_ok=True)
    csv_path = args.data or ARTIFACTS / "labeled_data.csv"
    if not csv_path.exists():
        urllib.request.urlretrieve(DATA_URL, csv_path)
    rows = load_data(csv_path)
    train, test = train_test_split(rows, train_size=4600, test_size=1000, random_state=SEED,
                                   stratify=[label for _, label in rows])
    train_text, train_y = map(list, zip(*train))
    test_text, test_y = map(list, zip(*test))
    label_tensor = torch.tensor(train_y)
    weights = (len(train_y) / (2 * torch.bincount(label_tensor))).float()
    vocab = {term: index for index, (term, _) in
             enumerate(Counter(word for message in train_text for word in words(message)).most_common(5000), 2)}
    tfidf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                          LogisticRegression(class_weight="balanced", max_iter=500, random_state=SEED))
    tfidf.fit(train_text, train_y)
    tfidf_path = ARTIFACTS / "tfidf.joblib"
    joblib.dump(tfidf, tfidf_path, compress=3)
    embeddings = train_word2vec(train_text, vocab)
    mlp = WordClassifier(embeddings)
    fit_classifier(mlp, {"tokens": encode(train_text, vocab)}, label_tensor,
                   torch.optim.SGD(mlp.parameters(), lr=0.1, momentum=0.9), 14, weights)
    mlp_path = ARTIFACTS / "mlp.pt"
    vocab_path = ARTIFACTS / "vocab.json"
    torch.save(mlp.state_dict(), mlp_path)
    vocab_path.write_text(json.dumps(vocab), encoding="utf-8")

    tokenizer = BertTokenizerFast.from_pretrained(args.checkpoint)
    tokenizer.save_pretrained(ARTIFACTS / "tokenizer")
    bert = BertForSequenceClassification.from_pretrained(args.checkpoint, num_labels=2)
    bert.config.save_pretrained(ARTIFACTS / "bert-config")
    bert_inputs = tokenizer(train_text, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
    fit_classifier(bert, bert_inputs, label_tensor, torch.optim.AdamW(bert.parameters(), lr=7e-5), 4, weights)
    bert_path = ARTIFACTS / "bert.pt"
    int8_path = ARTIFACTS / "bert-int8.pt"
    torch.save(bert.state_dict(), bert_path)
    int8 = quantize(bert)
    torch.save(int8.state_dict(), int8_path)

    lexicon = ("idiot", "stupid", "moron", "asshole", "bastard", "dumb", "fuck", "bitch", "trash", "shit")
    lexicon_path = ARTIFACTS / "lexicon.json"
    lexicon_path.write_text(json.dumps(lexicon), encoding="utf-8")
    config_files = list((ARTIFACTS / "bert-config").glob("*"))
    shared_files = list((ARTIFACTS / "tokenizer").glob("*")) + config_files

    def bert_predict(model, message):
        tokens = tokenizer(message, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        return int(model(**tokens).logits.argmax(-1).item())

    predictors = (
        ("Edit-distance rules", lambda text: rule_predict(text, lexicon), [lexicon_path]),
        (NAMES[0], lambda text: int(tfidf.predict([text])[0]), [tfidf_path]),
        (NAMES[1], lambda text: int(mlp(encode([text], vocab)).argmax(-1).item()), [mlp_path, vocab_path]),
        (NAMES[2], lambda text: bert_predict(bert, text), [bert_path] + shared_files),
        (NAMES[3], lambda text: bert_predict(int8, text), [int8_path] + shared_files),
    )
    results = []
    with torch.inference_mode():
        for name, predict, paths in predictors:
            result, guesses = measure(name, lambda message: predict(normalize(message)), test_text, test_y, paths)
            results.append(result)
            print(result, flush=True)
    failures = [{"text": message, "actual": actual, "predicted": predicted}
                for message, actual, predicted in zip(test_text, test_y, guesses)
                if actual != predicted and not rule_predict(message, lexicon) and actual == 0]
    report = {
        "results": results, "train": dict(Counter(train_y)), "test": dict(Counter(test_y)),
        "eligible_rows": len(rows), "cpu": platform.processor(), "os": platform.platform(),
        "torch_threads": torch.get_num_threads(), "seed": SEED, "data_url": DATA_URL,
        "failure": next((item for item in failures if "love" in item["text"]), failures[0] if failures else None),
    }
    (ARTIFACTS / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
