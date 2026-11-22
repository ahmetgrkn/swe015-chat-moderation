import html
import re
import unicodedata
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "models"
MAX_LENGTH = 48
MODEL_ID = "google/bert_uncased_L-2_H-128_A-2"
NAMES = ("TF-IDF", "Word2Vec + MLP", "BERT Tiny", "BERT Tiny INT8")


def normalize(message):
    message = unicodedata.normalize("NFKC", html.unescape(message)).casefold()
    message = re.sub(r"https?://\S+|www\.\S+", " ", message)
    message = re.sub(r"@\w+", " ", message)
    message = re.sub(
        r"[a-z0-9]+",
        lambda match: match[0].translate(str.maketrans("013457", "oieast"))
        if any(char.isalpha() for char in match[0]) else match[0],
        message,
    )
    message = re.sub(r"(\w)\1{2,}", r"\1\1", message)
    return " ".join(message.split())


def words(message):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", message)


def encode(messages, vocab):
    rows = []
    for message in messages:
        row = [vocab.get(word, 1) for word in words(message)[:MAX_LENGTH]]
        rows.append(row + [0] * (MAX_LENGTH - len(row)))
    return torch.tensor(rows, dtype=torch.long)


class WordClassifier(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, padding_idx=0)
        self.classifier = nn.Sequential(
            nn.Linear(embeddings.shape[1], 64), nn.ReLU(),
            nn.Dropout(0.15), nn.Linear(64, 2),
        )

    def forward(self, tokens):
        mask = tokens.ne(0).unsqueeze(-1)
        pooled = (self.embedding(tokens) * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.classifier(pooled)


def quantize(model):
    return torch.ao.quantization.quantize_dynamic(model.eval(), {nn.Linear}, dtype=torch.qint8)
