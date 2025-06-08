# Đề tài: Tự động trích xuất công việc từ biên bản cuộc họp dự án phần mềm

## Giới thiệu

Chương trình này bao gồm hai module chính:

1. **Intent Classification (TaskExtractionModule)**
   Xác định và trích xuất các câu chứa công việc (task) từ biên bản cuộc họp.

2. **NER (Named Entity Recognition)**
   Trích xuất các thực thể (PERSON, TASK, PROJECT, DEADLINE, PRIORITY) từ những câu đã được xác định là có công việc.

Sau khi cài đặt, có thể:

- Huấn luyện hoặc tải sẵn model Intent Classification để phân loại câu “có task”.
- Huấn luyện hoặc tải sẵn model NER (PhoBERT) để gán nhãn token và trích thực thể.
- Chạy file `main.py` để đọc file biên bản, tách câu task, rồi trích entity, cuối cùng in kết quả ra console.

---

## Môi trường và phụ thuộc

1. **Python**: 3.8 trở lên
2. **Java 8+** (để chạy VnCoreNLP)
3. **Thư viện Python**:

   - `torch` (PyTorch)
   - `transformers` (Hugging Face Transformers)
   - `evaluate` (để tính metric seqeval)
   - `vncorenlp` (VnCoreNLP Python wrapper)
   - `numpy`

   Có thể cài đặt nhanh bằng `pip`:

   ```bash
   pip install torch transformers evaluate vncorenlp numpy
   ```

4. **VnCoreNLP**:

   - Tải [VnCoreNLP-1.1.1.jar](https://github.com/vncorenlp/VnCoreNLP/releases).

     - Tải hai file model của module word‐segmenter:

       `vi-vocab`:
       https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab

       `wordsegmenter.rdr`:
       https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr

     - Tạo thư mục C:\> mkdir vncorenlp\models\wordsegmenter
     - Di chuyển
       `C:\vncorenlp> move \path\to\VnCoreNLP-1.1.1.jar` .
       `C:\vncorenlp\models\wordsegmenter> move \path\to\vi-vocab` .
       `C:\vncorenlp\models\wordsegmenter> move \path\to\wordsegmenter.rdr` .

   - Giải nén vào một thư mục bất kỳ, ví dụ `C:vncorenlp\VnCoreNLP-1.1.1.jar`.

   - Nếu sử dụng trên gg cobal cài đặt nhanh:

     # Tạo thư mục chứa

     !mkdir -p vncorenlp/models/wordsegmenter

     # Tải engine và model

     !wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
     !wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
     !wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr

     # Di chuyển vào đúng chỗ

     !mv VnCoreNLP-1.1.1.jar vncorenlp/
     !mv vi-vocab vncorenlp/models/wordsegmenter/
     !mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/

---

## Cấu trúc thư mục

```
root/
├── intent_classification/
│   └── intent_classification_module.py      # Module Intent Classification
│
├── ner/
│   └── phobert_module.py                        # Module PhoBERT NER
│   └── mbert_module                             # Module mBERT NER
│   └── xlmr_module                              # Module XLM-R NER
│
├── main.py                                      # File chính để chạy toàn bộ pipeline
│
│
├── meeting-docs/                            #Tập các văn bản thử nghiệm
│   └── meeting_001.txt
│   └── .....
│── dataset/
│   └── dataset1.txt                         #dataset1 của ner để train
│   └── dataset2.txt                         #dataset2 của ner để train
│   └── train.csv                            #dataset của intent classification để train
│
└── README.md                                # File hướng dẫn này
```

---

## Bước 1: Chuẩn bị dữ liệu

1. **Dữ liệu training NER (CoNLL-like)**

   - File `dataset1.txt` chứa các dòng `[token] [label]` mỗi dòng.
   - Folder trained_model lưu các checkpoint đã được huấn luyện được
   - Nếu chưa có, có thể tự xây dựng theo hướng dẫn trong `ner_module.py`.
   - Dòng trống để tách câu. Ví dụ:

     ```
     Trung B-PERSON
     phải O
     viết B-TASK
     báo_cáo I-TASK
     ...

     Anh O
     Hiếu B-PERSON
     triển_khai B-TASK
     ...

     ```

2. **Dữ liệu training Intent Classification**
   - Tập dữ liệu câu đã gắn nhãn “có task”/“không task” (nhị phân).
   - File model checkpoint (như `intentclassification/best_model.pt`) đã được huấn luyện trước.
   - Nếu chưa có, có thể tự xây dựng theo hướng dẫn trong `extract_has_task_module.py`.

---

## Bước 2: Cấu hình VnCoreNLP

Trong cả hai module NER và Intent Classification, cần chỉ định đúng đường dẫn đến file JAR của VnCoreNLP. Ví dụ trong code:

```python
vncorenlp = VnCoreNLP(
    r"C:\vncorenlp\VnCoreNLP-1.1.1.jar",
    annotators="wseg"
)
```

- Hãy sửa `r"C:\vncorenlp\VnCoreNLP-1.1.1.jar"` cho phù hợp với máy.

---

## Bước 3: Huấn luyện và lưu model NER

Nếu chưa có thư mục `./trained_model_phobert/`, chương trình `main.py` sẽ tự động gọi hàm `train_ner_model()`. Nếu muốn huấn luyện thủ công, có thể:

```bash
python -c "from phobert_module import train_ner_model; train_ner_model('dataset/dataset1.txt', './trained_model_phobert')"
```

- Kết quả sẽ được lưu dưới `./trained_model_phobert/` bao gồm:

  ```
  trained_model/
  ├── config.json
  ├── pytorch_model.bin
  ├── tokenizer_config.json
  ├── merges.txt
  ├── vocab.txt
  └── ...
  ```

- Với các module của model khác cũng tương tự

---

## Bước 4: Chuẩn bị model Intent Classification

1. Đảm bảo file checkpoint của Intent Classification (ví dụ `intentclassification/best_model.pt`) đã tồn tại.
2. Trong `main.py`, đường dẫn này được truyền vào khi khởi tạo:

   ```python
   task_module = TaskExtractionModule(
       vncorenlp_path=r"C:\vncorenlp\VnCoreNLP-1.1.1.jar",
       model_path="intentclassification/best_model.pt"
   )
   task_module.load_model()
   ```

- Nếu bạn muốn tự train Intent Classification, có thể chạy trực tiếp !python intentclassification.py

---

## Bước 5: Chạy toàn bộ pipeline

Sau khi đã có:

- **NER model** trong `./trained_model_phobert/`
- **Intent model** (checkpoint) tại `intentclassification/best_model.pt`
- **VnCoreNLP** đã cài đặt đúng đường dẫn
- **File biên bản** (ví dụ `meeting-docs/meeting_001.txt`)

Chạy:

```bash
python main.py
```

Ví dụ đầu ra:

```
Kết quả 1:
  Sentence:   "Nguyễn Văn A sẽ hoàn thiện báo cáo trước ngày 05/05/2025, ưu tiên cao"
  PERSON:     Nguyễn Văn A
  TASK:       hoàn thiện báo cáo
  PROJECT:    ABC
  DEADLINE:   05/05/2025
  PRIORITY:   cao
  STATUS:     Chưa làm
--------------------------------------------------
Kết quả 2:
  Sentence:   "Trần Thị B triển khai module X trước ngày 07/05/2025, ưu tiên trung bình"
  PERSON:     Trần Thị B
  TASK:       triển khai module X
  PROJECT:    ABC
  DEADLINE:   07/05/2025
  PRIORITY:   trung bình
  STATUS:     Chưa làm
--------------------------------------------------
```

---
