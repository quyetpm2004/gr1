# main_extraction.py
import os
import torch
from vncorenlp import VnCoreNLP

# IMPORT từ module intent classification (đã có sẵn)
from intent_classification_module import TaskExtractionModule

# IMPORT từ module NER mới vừa tạo
from phobert_module import (
    train_ner_model,
    extract_entities_from_sentences,
    DEFAULT_NER_MODEL_DIR
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vncorenlp_dir = "/content/vncorenlp/VnCoreNLP-1.1.1.jar"

vncorenlp = VnCoreNLP(
    vncorenlp_dir,
    annotators="wseg"
)

# Khởi tạo TaskExtractionModule (Intent Classification)
task_module = TaskExtractionModule(
    vncorenlp_path=vncorenlp_dir,
    model_path="/content/best_model.pt"    
)

task_module.load_model()



# Nếu chưa có model NER thì train
if not os.path.exists(DEFAULT_NER_MODEL_DIR):
    print("[MAIN] Chưa tìm thấy NER model đã huấn luyện, bắt đầu huấn luyện NER...")
    train_conll_file = "./dataset/dataset1.txt"
    train_ner_model(train_conll_file, DEFAULT_NER_MODEL_DIR)
else:
    print(f"[MAIN] Đang load NER model từ '{DEFAULT_NER_MODEL_DIR}'...")

# Dự đoán status
status_keywords = {
    "Đang làm": ["tiếp tục", "đang thực hiện", "đang tiến hành", "đang", "lại"],
    "Đã làm":   ["hoàn thành", "đã xong", "kết thúc"]
}

# Loại bỏ tách từ do subword và vncorenlp gây ra
def clean_token(token: str) -> str:
    no_subword = token.replace("@@ ", "")
    no_underscore = no_subword.replace("_", " ")
    return no_underscore

def extract_entities_from_meeting_minutes(minutes_text, batch_size=8):
    # Lấy project_name và danh sách câu chứa task từ module IC
    project_name, task_sentences = task_module.extract_tasks(minutes_text)

    if not task_sentences:
        return []

    # Tokenize word-level mỗi câu bằng VnCoreNLP
    all_word_lists = [vncorenlp.tokenize(sent)[0] for sent in task_sentences]

    # Gọi NER
    ner_results = extract_entities_from_sentences(all_word_lists, batch_size=batch_size)

    # Ghép kết quả NER vào output chung
    sentence_results = []
    for sent_idx, sent in enumerate(task_sentences):
        entities = ner_results[sent_idx]
        def merge_list(lst):
            return " ".join(lst) if lst else ""

        persons    = merge_list(entities.get("PERSON", []))
        tasks      = merge_list(entities.get("TASK", []))
        projects   = merge_list(entities.get("PROJECT", [])) or (project_name or "N/A")
        deadlines  = merge_list(entities.get("DEADLINE", []))
        priorities = merge_list(entities.get("PRIORITY", []))

        # Xác định trạng thái: nếu trong câu có từ khóa “hoàn thành”, “đã xong”… => "Đã làm", 
        # Nếu có “đang”/“tiếp tục”… => "Đang làm", ngược lại "Chưa làm"
        text_low = sent.lower()
        status = "Chưa làm"
        for st, kws in status_keywords.items():
            if any(kw in text_low for kw in kws):
                status = st
                break

        sentence_results.append({
            "sentence":    clean_token(sent),
            "persons":     clean_token(persons),
            "task":        clean_token(tasks),
            "project":     clean_token(projects),
            "deadline":    clean_token(deadlines),
            "priority":    clean_token(priorities),
            "status":      clean_token(status)
        })

    return sentence_results


# Hàm main
if __name__ == "__main__":
    minutes_file = "/content/meeting_001.txt"
    if not os.path.exists(minutes_file):
        print(f"[MAIN] File biên bản '{minutes_file}' không tồn tại.")
        exit(1)

    results = extract_entities_from_meeting_minutes(minutes_file)
    if not results:
        print("[MAIN] Không tìm thấy câu task nào trong biên bản.")
    else:
        for idx, info in enumerate(results, 1):
            print(f"- Câu {idx}, person: \"{info['persons']}\", task: \"{info['task']}\", deadline: \"{info['deadline']}\", project: \"{info['project']}\", priority: \"{info['priority']}\", status: \"{info['status']}\"")

    # Close vncorenlp
    task_module.close()
    vncorenlp.close()

# So sánh chất lượng công cụ
# Chạy tất cả các file biên bản trong thư mục và lưu kết quả vào JSON và Excel để so sánh với Gemini 2.0
# if __name__ == "__main__":

#     MEETING_FOLDER = "/content/meetings/"
#     JSON_LOG_FILE = "all_results.json"
#     EXCEL_LOG_FILE = "my_all_results_output.xlsx"

#     # Lấy danh sách tất cả file txt trong thư mục
#     all_files = [
#         os.path.join(MEETING_FOLDER, f)
#         for f in os.listdir(MEETING_FOLDER)
#         if f.lower().endswith(".txt")
#     ]
#     all_files.sort()  

#     if os.path.exists(JSON_LOG_FILE):
#         with open(JSON_LOG_FILE, "r", encoding="utf-8") as f:
#             all_data = json.load(f)
#     else:
#         all_data = []

#     # Loop qua từng file
#     for idx, meeting_file in enumerate(all_files, 1):

#         with open(meeting_file, "r", encoding="utf-8") as f:
#             minutes_text = f.read()

#         results = extract_entities_from_meeting_minutes(minutes_text)
#         if not results:
#             continue

#         # Gắn tên source_file vào kết quả
#         for r in results:
#             r["source_file"] = os.path.basename(meeting_file)

#         all_data.extend(results)

#     # Lưu JSON tổng
#     with open(JSON_LOG_FILE, "w", encoding="utf-8") as f:
#         json.dump(all_data, f, ensure_ascii=False, indent=4)

#     # Xuất Excel tổng
#     df_all = pd.DataFrame(all_data)
#     df_all.to_excel(EXCEL_LOG_FILE, index=False)

#     # Đóng
#     task_module.close()
#     vncorenlp.close()