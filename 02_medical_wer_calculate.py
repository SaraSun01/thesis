import pandas as pd

# === 文件路径（你可以改为你的实际文件路径）===
file_path = "01_ALL_fuzzy_match.csv"

# === 加载数据 ===
df = pd.read_csv(file_path)

# === 分类统计 ===
S = (df["match_type"] == "substitution").sum()
D = (df["match_type"] == "missing").sum()
I = 0  # 当前数据未提供“插入错误”信息
N = len(df)

# === 计算 mWER ===
mwer = (S + D + I) / N if N > 0 else 0

# === 输出结果 ===
print("=== ALL MWER Evaluation ===")
print(f"Total terms        : {N}")
print(f"Substitutions (S)  : {S}")
print(f"Deletions (D)      : {D}")
print(f"Insertions (I)     : {I}")
print(f"Medical WER (mWER) : {mwer:.2%}")
