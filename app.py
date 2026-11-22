import json
from time import perf_counter

import gradio as gr
import joblib
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from moderation import ARTIFACTS, MAX_LENGTH, NAMES, WordClassifier, encode, normalize, quantize


def load_predictors():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    tfidf = joblib.load(ARTIFACTS / "tfidf.joblib")
    vocab = json.loads((ARTIFACTS / "vocab.json").read_text(encoding="utf-8"))
    mlp_state = torch.load(ARTIFACTS / "mlp.pt", map_location="cpu", weights_only=True)
    mlp = WordClassifier(mlp_state["embedding.weight"])
    mlp.load_state_dict(mlp_state)
    mlp.eval()
    tokenizer = BertTokenizerFast.from_pretrained(ARTIFACTS / "tokenizer", local_files_only=True)
    config = BertConfig.from_pretrained(ARTIFACTS / "bert-config", local_files_only=True)

    def load_bert(filename, use_int8):
        model = BertForSequenceClassification(config).eval()
        if use_int8:
            model = quantize(model)
        state = torch.load(ARTIFACTS / filename, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model

    bert = load_bert("bert.pt", False)
    int8 = load_bert("bert-int8.pt", True)

    def predict_bert(model, message):
        tokens = tokenizer(message, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        return model(**tokens).logits.softmax(-1)[0].tolist()

    return dict(zip(NAMES, (
        lambda text: tfidf.predict_proba([text])[0],
        lambda text: mlp(encode([text], vocab)).softmax(-1)[0].tolist(),
        lambda text: predict_bert(bert, text),
        lambda text: predict_bert(int8, text),
    )))


def build_demo(predictors):
    def submit(message, model, history, log):
        message = message.strip()
        if not message:
            return "", history, log
        if len(message) > 500:
            raise gr.Error("Enter a message of at most 500 characters.")
        start = perf_counter()
        with torch.inference_mode():
            scores = predictors[model](normalize(message))
        latency = (perf_counter() - start) * 1000
        label = int(scores[1] > scores[0])
        confidence = float(scores[label])
        status = ("Clean", "Offensive")[label]
        safe_text = "[Message masked]" if label else message
        return "", history + [
            {"role": "user", "content": safe_text},
            {"role": "assistant", "content": f"{status} · Confidence {confidence:.1%}"},
        ], log + [[len(log) + 1, model, status, round(confidence * 100, 1), round(latency, 2)]]

    with gr.Blocks(title="Chat Moderation", analytics_enabled=False) as demo:
        gr.Markdown("# Chat Moderation\nTry an English chat message. Offensive messages are masked.")
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(label="Moderated chat", type="messages", height=400, render_markdown=False)
                message = gr.Textbox(label="Message", placeholder="Type a message to moderate", max_lines=3)
                send = gr.Button("Send", variant="primary")
            with gr.Column(scale=2):
                model = gr.Dropdown(NAMES, value=NAMES[-1], label="Model", elem_id="model-selector")
                log_table = gr.Dataframe(
                    headers=["#", "Model", "Label", "Confidence (%)", "Latency (ms)"],
                    datatype=["number", "str", "str", "number", "number"], interactive=False, label="Message log",
                )
                log = gr.State([])
        inputs = [message, model, chat, log]
        send.click(submit, inputs, [message, chat, log]).then(lambda rows: rows, log, log_table)
        message.submit(submit, inputs, [message, chat, log]).then(lambda rows: rows, log, log_table)
    return demo


if __name__ == "__main__":
    build_demo(load_predictors()).launch(server_name="127.0.0.1", show_error=True)
