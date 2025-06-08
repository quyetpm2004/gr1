# ner_module.py
import os
os.environ["WANDB_MODE"] = "disabled"
import torch
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from evaluate import load as load_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labels = [
    "O",
    "B-PERSON", "I-PERSON",
    "B-TASK",   "I-TASK",
    "B-PROJECT","I-PROJECT",
    "B-DEADLINE","I-DEADLINE",
    "B-PRIORITY","I-PRIORITY"
]
label2id = {lbl: idx for idx, lbl in enumerate(labels)}
id2label = {idx: lbl for lbl, idx in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)

DEFAULT_NER_MODEL_DIR = "/content/trained_module_xlmr"

def load_conll_data(file_path):
    sentences = []
    sent = []
    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                parts = line.split()
                token = " ".join(parts[:-1])
                tag = parts[-1]
                sent.append((token, tag))
            else:
                if sent:
                    sentences.append(sent)
                sent = []
        if sent:
            sentences.append(sent)
    return sentences

class NERDataset(Dataset):
    def __init__(self, raw_sentences, tokenizer, label2id, max_length=64):
        self.examples = []
        for sent in raw_sentences:
            tokens, tags = zip(*sent)
            words = list(tokens)
            encoding = tokenizer(
                words,
                is_split_into_words=True,
                return_attention_mask=True,
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            labels_aligned = []
            word_ids = encoding.word_ids()
            for idx in word_ids:
                if idx is None:
                    labels_aligned.append(-100)
                else:
                    tag = tags[idx]
                    if word_ids.count(idx) > 1:
                        if tag.startswith("B-"):
                            inside = "I-" + tag.split("B-")[1]
                            labels_aligned.append(label2id.get(inside, -100))
                        else:
                            labels_aligned.append(label2id.get(tag, -100))
                    else:
                        labels_aligned.append(label2id.get(tag, label2id.get(tag, -100)))
            # pad labels to max_length
            labels_aligned += [-100] * (max_length - len(labels_aligned))
            self.examples.append({
                "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(labels_aligned, dtype=torch.long),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            "input_ids": self.examples[idx]["input_ids"],
            "attention_mask": self.examples[idx]["attention_mask"],
            "labels": self.examples[idx]["labels"],
        }

metric = load_metric("seqeval")

def compute_metrics(p):
    logits, label_ids = p
    preds = np.argmax(logits, axis=-1)
    true_labels, pred_labels = [], []
    for b in range(label_ids.shape[0]):
        lab_seq, pred_seq = [], []
        for t in range(label_ids.shape[1]):
            if label_ids[b, t] == -100:
                continue
            lab_seq.append(id2label[label_ids[b, t]])
            pred_seq.append(id2label[preds[b, t]])
        true_labels.append(lab_seq)
        pred_labels.append(pred_seq)
    results = metric.compute(predictions=pred_labels, references=true_labels)
    return {
        "overall_precision": results.get("overall_precision", 0.0),
        "overall_recall":    results.get("overall_recall", 0.0),
        "overall_f1":        results.get("overall_f1", 0.0),
        "person_f1":   results.get("PERSON", {}).get("f1", 0.0),
        "task_f1":     results.get("TASK", {}).get("f1", 0.0),
        "deadline_f1": results.get("DEADLINE", {}).get("f1", 0.0),
        "priority_f1": results.get("PRIORITY", {}).get("f1", 0.0),
        "project_f1":  results.get("PROJECT", {}).get("f1", 0.0),
    }

def train_ner_model(conll_filepath, output_dir=DEFAULT_NER_MODEL_DIR):
    raw_sents = load_conll_data(conll_filepath)
    np.random.shuffle(raw_sents)
    split_idx = int(0.8 * len(raw_sents))
    train_sents, valid_sents = raw_sents[:split_idx], raw_sents[split_idx:]
    train_dataset = NERDataset(train_sents, tokenizer, label2id)
    valid_dataset = NERDataset(valid_sents, tokenizer, label2id)
    model = AutoModelForTokenClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    ).to(device)
    data_collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="overall_f1",
        fp16=torch.cuda.is_available()
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.train()
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[NER] Model saved at '{output_dir}'.")

def extract_entities_from_sentences(list_of_word_lists, batch_size=8):
    model = AutoModelForTokenClassification.from_pretrained(DEFAULT_NER_MODEL_DIR).to(device)
    model.eval()
    results = []
    for i in range(0, len(list_of_word_lists), batch_size):
        batch = list_of_word_lists[i:i+batch_size]
        texts = [" ".join(ws) for ws in batch]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            padding=True,
            max_length=128
        )
        # tách offset_mapping trước khi đẩy vào model
        offset_maps = enc.pop("offset_mapping").cpu()
        # đẩy inputs lên device
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        pred_ids = logits.argmax(dim=-1).cpu()
        input_ids = enc["input_ids"].cpu()
        for b_idx in range(len(batch)):
            entities = defaultdict(list)
            last_label = None
            last_start = None
            for tok_idx, pred_id in enumerate(pred_ids[b_idx]):
                start, end = offset_maps[b_idx][tok_idx].tolist()
                # bỏ qua special tokens và padding
                if start == end == 0:
                    if last_label is not None:
                        span = tokenizer.decode(
                            input_ids[b_idx][last_start:tok_idx], skip_special_tokens=True
                        ).strip()
                        entities[last_label].append(span)
                        last_label = None
                    continue
                label = id2label[pred_id.item()]
                if label.startswith("B-"):
                    if last_label is not None:
                        span = tokenizer.decode(
                            input_ids[b_idx][last_start:tok_idx], skip_special_tokens=True
                        ).strip()
                        entities[last_label].append(span)
                    last_label = label[2:]
                    last_start = tok_idx
                elif label.startswith("I-") and last_label == label[2:]:
                    continue
                else:
                    if last_label is not None:
                        span = tokenizer.decode(
                            input_ids[b_idx][last_start:tok_idx], skip_special_tokens=True
                        ).strip()
                        entities[last_label].append(span)
                        last_label = None
            # xử lý span cuối
            if last_label is not None:
                span = tokenizer.decode(
                    input_ids[b_idx][last_start:], skip_special_tokens=True
                ).strip()
                entities[last_label].append(span)
            results.append(entities)
    return results
