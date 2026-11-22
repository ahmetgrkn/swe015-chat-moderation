# Real-Time Chat Moderation

SWE015 use case 4. Classify an English chat message as clean or offensive. The
Gradio app masks flagged messages, reports prediction confidence, and logs CPU
latency. The default model is the dynamically quantized BERT Tiny.

## Setup

Use Python 3.11 or 3.12 on an x86-64 CPU with PyTorch dynamic INT8 support. From
this directory, create and activate a virtual environment, then install:

```sh
python -m pip install -r requirements.txt
```

Build the local model artifacts once. The script downloads the public training
CSV and the pretrained BERT checkpoint on the first run. To reuse local copies,
pass `--data path/to/labeled_data.csv --checkpoint path/to/bert-directory`.

```sh
python train.py
```

## Run

```sh
python app.py
```

Open the local address printed in the terminal. Try "Thanks for helping with
that round!" and "you are such an iiiidi0t". Select a different model and repeat
the same message to compare predictions. Keep generated `models/` private to
your local checkout or rebuild them using the training command.

## Data and evaluation

- [Davidson et al. hate speech and offensive language data](https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data). The original hate-speech and offensive-language labels both map to offensive. The neither label maps to clean. The raw data contains potentially offensive language.
- [Google BERT Tiny checkpoint](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2).
- The training script normalizes messages and removes duplicate inputs and
  label conflicts before creating a stratified split. It fits the vocabulary and
  parameters using only the training split.
- The primary comparison is TF-IDF with logistic regression, skip-gram Word2Vec
  with an MLP, fine-tuned BERT Tiny, and dynamic INT8 BERT Tiny. The edit-distance
  lexicon rule is an additional baseline.
- The CPU target is a mean latency below 50 ms per message. The report uses
  128 single-message runs following warmup. Measurements include normalization
  and inference and exclude Gradio and model loading. Sizes count serialized
  model files and their inference metadata, using decimal MB.

Training writes the measured results to `models/results.json`. This prototype
classifies isolated messages. Quoted insults, dialect, and game-specific
language can produce classification errors. Confidence is the model's score
and has not been calibrated as an error probability.
