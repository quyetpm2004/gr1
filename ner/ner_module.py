# ner_module.py
import os
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from evaluate import load as load_metric
from vncorenlp import VnCoreNLP


# Kiểm tra GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Các nhãn cho bài toán NER
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
# Khởi tạo
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

# Cấu hình vnCoreNLP
vncorenlp = VnCoreNLP(
    r"C:\Users\DELL\workspace\vncorenlp\VnCoreNLP-1.1.1.jar",
    annotators="wseg"
)

# Đường dẫn đến folder nếu đã train model
DEFAULT_NER_MODEL_DIR = "./ner/trained_model"


# Đọc dữ liệu định dạng CoNLL (token + nhãn, blank line tách câu) 
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


# Cấu hình dataset cho NER (alignment thủ công vì tokenizer non-fast)
class NERDataset(Dataset):
    def __init__(self, raw_sentences, tokenizer, label2id, max_length=64):
        self.sentences = raw_sentences
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.examples = []
        self._build_examples()

    def _build_examples(self):
        for sent in self.sentences:
            tokens, tags = zip(*sent)
            input_ids = [self.tokenizer.convert_tokens_to_ids("<s>")]
            aligned_labels = [-100]  

            for token, tag in zip(tokens, tags):
                subwords = self.tokenizer.tokenize(token)
                sub_ids = self.tokenizer.convert_tokens_to_ids(subwords)
                input_ids.extend(sub_ids)

                lbl_id = self.label2id[tag]
                aligned_labels.append(lbl_id)
                for _ in sub_ids[1:]:
                    if tag.startswith("B-"):
                        aligned_labels.append(self.label2id["I-" + tag[2:]])
                    elif tag.startswith("I-"):
                        aligned_labels.append(lbl_id)
                    else:
                        aligned_labels.append(-100)

            input_ids.append(self.tokenizer.convert_tokens_to_ids("</s>"))
            aligned_labels.append(-100)

            attention_mask = [1] * len(input_ids)

            pad_len = self.max_length - len(input_ids)
            if pad_len > 0:
                input_ids += [self.tokenizer.pad_token_id] * pad_len
                attention_mask += [0] * pad_len
                aligned_labels += [-100] * pad_len
            else:
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                aligned_labels = aligned_labels[: self.max_length]

            self.examples.append({
                "input_ids":      torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels":         torch.tensor(aligned_labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.examples[idx]["input_ids"],
            "attention_mask": self.examples[idx]["attention_mask"],
            "labels":         self.examples[idx]["labels"],
        }


# Hàm metric tính toán (dựa vào seqeval) 
metric = load_metric("seqeval")

def compute_metrics(p):
    logits, label_ids = p
    preds = np.argmax(logits, axis=-1)

    true_labels = []
    pred_labels = []

    for b in range(label_ids.shape[0]):
        lab_seq = []
        pred_seq = []
        for t in range(label_ids.shape[1]):
            if label_ids[b, t] == -100:
                continue
            lab_seq.append(id2label[label_ids[b, t]])
            pred_seq.append(id2label[preds[b, t]])
        true_labels.append(lab_seq)
        pred_labels.append(pred_seq)

    results = metric.compute(predictions=pred_labels, references=true_labels)
    return {
        "overall_precision": results["overall_precision"],
        "overall_recall":    results["overall_recall"],
        "overall_f1":        results["overall_f1"],
        "person_f1":   results.get("PERSON", {}).get("f1", 0.0),
        "task_f1":     results.get("TASK", {}).get("f1", 0.0),
        "deadline_f1": results.get("DEADLINE", {}).get("f1", 0.0),
        "priority_f1": results.get("PRIORITY", {}).get("f1", 0.0),
        "project_f1":  results.get("PROJECT", {}).get("f1", 0.0),
    }


# Hàm huấn luyện model
def train_ner_model(conll_filepath, output_dir=DEFAULT_NER_MODEL_DIR):
    raw_sents = load_conll_data(conll_filepath)
    np.random.shuffle(raw_sents)
    split_idx = int(0.8 * len(raw_sents))
    train_sents = raw_sents[:split_idx]
    valid_sents = raw_sents[split_idx:]

    train_dataset = NERDataset(train_sents, tokenizer, label2id)
    valid_dataset = NERDataset(valid_sents, tokenizer, label2id)

    # Tạo model từ PhoBERT
    model = AutoModelForTokenClassification.from_pretrained(
        "vinai/phobert-base",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    ).to(device)

    # Data collator để pad tự động và align nhãn = -100 cho pad
    data_collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=3e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
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
    print(f"[NER] Đã huấn luyện xong và lưu model vào '{output_dir}'.")


# Tách nhiều câu 1 lần batch_size=8 8 câu 1 lần
def extract_entities_from_sentences(list_of_word_lists, batch_size=8):
    model = AutoModelForTokenClassification.from_pretrained(DEFAULT_NER_MODEL_DIR).to(device)
    model.eval()
    results = []

    for i in range(0, len(list_of_word_lists), batch_size):
        batch = list_of_word_lists[i : i + batch_size]
        joined = [" ".join(ws) for ws in batch]
        enc = tokenizer(
            joined,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits  

        preds = logits.argmax(dim=-1).cpu().numpy()      
        input_ids = enc["input_ids"].cpu().numpy()       

        for b_idx in range(len(batch)):
            seq_input_ids = input_ids[b_idx]
            seq_pred_ids = preds[b_idx]
            tokens = tokenizer.convert_ids_to_tokens(seq_input_ids)

            entities = defaultdict(list)
            current_label = None
            current_tokens = []

            for tok, pid in zip(tokens, seq_pred_ids):
                if tok in tokenizer.all_special_tokens:
                    if current_label and current_tokens:
                        entities[current_label].append(" ".join(current_tokens))
                        current_tokens, current_label = [], None
                    continue

                word_piece = tok.replace("▁", "")
                lbl_str = id2label[pid]

                if lbl_str.startswith("B-"):
                    if current_label and current_tokens:
                        entities[current_label].append(" ".join(current_tokens))
                    current_label = lbl_str.split("B-")[1]
                    current_tokens = [word_piece]

                elif lbl_str.startswith("I-") and current_label == lbl_str.split("I-")[1]:
                    current_tokens.append(word_piece)

                else:
                    if current_label and current_tokens:
                        entities[current_label].append(" ".join(current_tokens))
                    current_label, current_tokens = None, []

            if current_label and current_tokens:
                entities[current_label].append(" ".join(current_tokens))

            results.append(entities)

    return results
