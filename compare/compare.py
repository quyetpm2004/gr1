import pandas as pd

# Load
df_my = pd.read_excel("all_results.xlsx")
df_gemini = pd.read_excel("output.xlsx")

# Chuẩn hóa tên cột nếu cần
df_my = df_my.rename(columns={"persons": "person", "tasks": "task", "deadlines": "deadline"})
df_gemini = df_gemini.rename(columns={"persons": "person", "tasks": "task", "deadlines": "deadline"})

# Đảm bảo cột đủ
needed_cols = ["person", "task", "deadline", "project", "priority", "status", "source_file", "sentence"]
for col in needed_cols:
    if col not in df_my.columns:
        df_my[col] = ""
    if col not in df_gemini.columns:
        df_gemini[col] = ""

# Tạo chỉ số thứ tự trong từng file biên bản
df_my["row"] = df_my.groupby("source_file").cumcount()
df_gemini["row"] = df_gemini.groupby("source_file").cumcount()

# Merge theo source_file + row
df_merge = pd.merge(
    df_my,
    df_gemini,
    on=["source_file", "row"],
    suffixes=('_my', '_gemini')
)

print("[INFO] ✅ Merged shape:", df_merge.shape)

# So sánh từng field
for col in ["person", "task", "deadline", "project", "priority", "status"]:
    df_merge[f"{col}_match"] = df_merge[f"{col}_my"] == df_merge[f"{col}_gemini"]

# Exact match toàn record
df_merge["record_match"] = df_merge[
    [f"{col}_match" for col in ["person", "task", "deadline", "project", "priority", "status"]]
].all(axis=1)

# Tính tỷ lệ match
for col in ["person", "task", "deadline", "project", "priority", "status"]:
    rate = df_merge[f"{col}_match"].mean() * 100
    print(f"{col} match rate: {rate:.2f}%")


# Xuất Excel
df_merge.to_excel("comparison_results.xlsx", index=False)
