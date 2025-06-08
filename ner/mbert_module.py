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

# Kiểm tra GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Các nhãn cho bài toán Task Management NER (BIO tagging)
labels = [
    "O",
    "B-TASK",   "I-TASK",
    "B-PERSON", "I-PERSON",
    "B-PROJECT","I-PROJECT",
    "B-DEADLINE","I-DEADLINE",
    "B-PRIORITY","I-PRIORITY"
]
label2id = {lbl: idx for idx, lbl in enumerate(labels)}
id2label = {idx: lbl for lbl, idx in label2id.items()}

# Khởi tạo tokenizer 
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased", use_fast=True)

# Đường dẫn đến folder lưu model nếu đã train
DEFAULT_NER_MODEL_DIR = "/content/trained_module_mbert"


def load_conll_data(file_path):
    """
    Đọc dữ liệu định dạng CoNLL (mỗi dòng "token label", blank line tách câu).
    Trả về: List[List[(token, label)]], mỗi inner list là một câu.
    """
    sentences = []
    sent = []
    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                parts = line.split()
                token = parts[0]
                tag = parts[1] if len(parts) > 1 else "O"
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
        self.sentences = raw_sentences
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.examples = []
        self._build_examples()

    def _build_examples(self):
        for sent in self.sentences:
            tokens, tags = zip(*sent)
            cls_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
            sep_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
            pad_id = self.tokenizer.pad_token_id

            input_ids = [cls_id]
            aligned_labels = [-100]  # Không tính loss trên CLS

            # Tokenize từng token gốc
            for token, tag in zip(tokens, tags):
                subwords = self.tokenizer.tokenize(token)
                sub_ids = self.tokenizer.convert_tokens_to_ids(subwords)
                input_ids.extend(sub_ids)

                # Gán nhãn cho subword đầu
                lbl_id = self.label2id.get(tag, 0)
                aligned_labels.append(lbl_id)
                # Các subword tiếp theo trong cùng token
                for _ in sub_ids[1:]:
                    if tag.startswith("B-"):
                        inner_label = "I-" + tag.split("B-")[1]
                        aligned_labels.append(self.label2id.get(inner_label, 0))
                    elif tag.startswith("I-"):
                        aligned_labels.append(lbl_id)
                    else:
                        aligned_labels.append(-100)

            # Thêm [SEP]
            input_ids.append(sep_id)
            aligned_labels.append(-100)

            attention_mask = [1] * len(input_ids)

            # Padding hoặc cắt bớt
            pad_len = self.max_length - len(input_ids)
            if pad_len > 0:
                input_ids += [pad_id] * pad_len
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
        "task_f1":     results.get("TASK", {}).get("f1", 0.0),
        "person_f1":   results.get("PERSON", {}).get("f1", 0.0),
        "project_f1":  results.get("PROJECT", {}).get("f1", 0.0),
        "deadline_f1": results.get("DEADLINE", {}).get("f1", 0.0),
        "priority_f1": results.get("PRIORITY", {}).get("f1", 0.0),
    }


def train_ner_model(conll_filepath, output_dir=DEFAULT_NER_MODEL_DIR):
    raw_sents = load_conll_data(conll_filepath)
    np.random.shuffle(raw_sents)
    split_idx = int(0.8 * len(raw_sents))
    train_sents = raw_sents[:split_idx]
    valid_sents = raw_sents[split_idx:]

    train_dataset = NERDataset(train_sents, tokenizer, label2id, max_length=64)
    valid_dataset = NERDataset(valid_sents, tokenizer, label2id, max_length=64)

    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-multilingual-cased",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    ).to(device)

    data_collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="overall_f1",
        fp16=torch.cuda.is_available(),
        report_to=None
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


def extract_entities_from_sentences(list_of_word_lists, batch_size=8):
    model = AutoModelForTokenClassification.from_pretrained(
        DEFAULT_NER_MODEL_DIR,
        id2label=id2label,
        label2id=label2id
    ).to(device)
    model.eval()
    results = []

    for i in range(0, len(list_of_word_lists), batch_size):
        batch = list_of_word_lists[i : i + batch_size]
        enc = tokenizer(
            batch,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits
        preds = logits.argmax(dim=-1).cpu().numpy()
        word_ids_batch = [enc.word_ids(batch_index=bi) for bi in range(len(batch))]

        for bi, words in enumerate(batch):
            entities = defaultdict(list)
            cur_lbl = None
            cur_words = []
            prev_word_idx = None

            wid_seq = word_ids_batch[bi]
            pred_seq = preds[bi]

            for idx, word_idx in enumerate(wid_seq):
                # special token hoặc lặp word_idx → flush + skip
                if word_idx is None or word_idx == prev_word_idx:
                    if word_idx is None and cur_lbl:
                        entities[cur_lbl].append(" ".join(cur_words))
                        cur_lbl, cur_words = None, []
                    prev_word_idx = word_idx
                    continue

                prev_word_idx = word_idx
                label_str = id2label[pred_seq[idx]]
                word = words[word_idx]
                typ = label_str.split("-",1)[1] if "-" in label_str else None

                if label_str.startswith("B-"):
                    if cur_lbl:
                        entities[cur_lbl].append(" ".join(cur_words))
                    cur_lbl, cur_words = typ, [word]

                elif label_str.startswith("I-"):
                    if cur_lbl == typ:
                        cur_words.append(word)
                    else:
                        # I- đứng đầu → coi như B-
                        if cur_lbl:
                            entities[cur_lbl].append(" ".join(cur_words))
                        cur_lbl, cur_words = typ, [word]

                else:  # "O"
                    if cur_lbl:
                        entities[cur_lbl].append(" ".join(cur_words))
                        cur_lbl, cur_words = None, []

            # flush cuối câu
            if cur_lbl:
                entities[cur_lbl].append(" ".join(cur_words))

            results.append(entities)

    return results


