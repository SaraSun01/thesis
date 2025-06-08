import pandas as pd

# === load the file===
file_path = "01_ALL_fuzzy_match.csv"
df = pd.read_csv(file_path)

# === 分类统计 ===
S = (df["match_type"] == "substitution").sum()
D = (df["match_type"] == "missing").sum()
I = (df["match_type"] == "incertion").sum()
N = len(df)

# === 计算 mWER ===
mwer = (S + D + I) / N if N > 0 else 0

# === 输出结果 ===
print("=== MWER Evaluation ===")
print(f"Total terms        : {N}")
print(f"Substitutions (S)  : {S}")
print(f"Deletions (D)      : {D}")
print(f"Insertions (I)     : {I}")
print(f"Medical WER (mWER) : {mwer:.2%}")
