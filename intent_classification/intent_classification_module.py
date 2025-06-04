import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import re
from vncorenlp import VnCoreNLP
import os

class TaskExtractionModule:
    def __init__(self, vncorenlp_path, model_name="vinai/phobert-base", model_path="best_model.pt"):
        # Khởi tạo VnCoreNLP 
        self.vncorenlp = VnCoreNLP(vncorenlp_path, annotators="wseg")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        segmented = self.vncorenlp.tokenize(text)
        return " ".join(w for sent in segmented for w in sent)

    # Hàm dự đoán task
    def predict_task(self, sentence):
        sent = self.preprocess_text(sentence)
        encoding = self.tokenizer(
            sent,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**encoding)
            probs = torch.softmax(output.logits, dim=1)
            predicted = torch.argmax(probs, dim=1).item()
            return predicted, probs[0][predicted].item()

    # Lấy ra các câu chứa task
    def extract_tasks(self, meeting_file):
        with open(meeting_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Tiền xử lý: tách câu theo dấu chấm hoặc xuống dòng
        sentences = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r'[.]', line)
            for p in parts:
                p = p.strip()
                if p:
                    sentences.append(p)

        # Thử extract tên dự án (Sử dụng rule-based đơn giản)
        project_pattern = r"BIÊN BẢN CUỘC HỌP DỰ ÁN\s+(.+?)\n|Dự án\s*:\s*(.+?)\n"
        text_all = "".join(lines)
        match = re.search(project_pattern, text_all, re.IGNORECASE)
        if match:
            project_name = next(group for group in match.groups() if group is not None).strip()
        else:
            project_name = None
    

        # Lặp qua từng câu, predict nếu label=1 thì thêm vào tasks
        tasks = []
        for sent in sentences:
            label, conf = self.predict_task(sent)
            if label == 1:
                tasks.append(sent)

        return project_name, tasks

    # Hàm train model
    def train_model(self, train_csv, epochs=3):
        # Cấu trúc dataset gồm 2 cột text và label/ label = 1 là có task, 0 là no task
        df = pd.read_csv(train_csv)
        if "text" not in df.columns or "label" not in df.columns:
            raise KeyError("File CSV phải có cột 'text' và cột 'label'")
        df["text"] = df["text"].apply(self.preprocess_text)
        df_pre = df.copy()

        # Định nghĩa Dataset
        class TaskDataset(Dataset):
            def __init__(self, df_data, tokenizer):
                self.texts = df_data["text"].tolist()
                self.labels = df_data["label"].tolist()
                self.tokenizer = tokenizer

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                encoding = self.tokenizer(
                    self.texts[idx],
                    truncation=True, padding="max_length", max_length=64,
                    return_tensors="pt"
                )
                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": torch.tensor(self.labels[idx], dtype=torch.long)
                }

        # Chia tập train/val
        full_dataset = TaskDataset(df_pre, tokenizer=self.tokenizer)
        total_len = len(full_dataset)
        train_size = int(0.8 * total_len)
        val_size = total_len - train_size
        train_set, val_set = random_split(full_dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=16)

        # Khởi tạo optimizer + scheduler
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        num_training_steps = len(train_loader) * epochs
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        # Hàm đánh giá chất lượng train
        def evaluate(model, val_loader):
            model.eval()
            preds, trues = [], []
            total_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Evaluating"):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = model(**batch)
                    total_loss += outputs.loss.item()
                    logits = outputs.logits
                    preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    trues.extend(batch["labels"].cpu().numpy())
            acc = accuracy_score(trues, preds)
            f1 = f1_score(trues, preds, zero_division=0)
            return total_loss / len(val_loader), acc, f1

        # Loop huấn luyện epoch = 3
        best_val_acc = 0.0
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

            # Đánh giá trên validation
            val_loss, val_acc, val_f1 = evaluate(self.model, val_loader)
            print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

            # Lưu model tốt nhất
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.model_path)
                print(f"  → Saved best model (Val Acc: {val_acc:.4f})")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
        else:
            raise FileNotFoundError(f"Model file {self.model_path} not found.")

    def close(self):
        self.vncorenlp.close()


# Chạy thử không liên quan đến pipeline chính
def main():
    module = TaskExtractionModule(
        vncorenlp_path=r"C:\Users\DELL\workspace\vncorenlp\VnCoreNLP-1.1.1.jar",
        model_path="best_model.pt"
    )

    # Nếu chưa có checkpoint, train; ngược lại chỉ load
    if not os.path.exists(module.model_path):
        print("Model chưa tồn tại, bắt đầu train...")
        module.train_model("./dataset/train.csv", epochs=3)
    else:
        print("Tìm thấy checkpoint, bỏ qua train.")

    module.load_model()

    # Ví dụ trích xuất task từ file văn bản
    project, tasks = module.extract_tasks("vanbanloi.txt")
    print("Project:", project)
    print("Tasks:", tasks)
    print("Tasks:", len(tasks)) # Xem số lượng câu chứa task có đúng không

    module.close()


if __name__ == "__main__":
    main()
