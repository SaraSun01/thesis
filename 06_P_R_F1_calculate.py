import pandas as pd

# === path of input file ===
input_file = "01_ALL_fuzzy_match.csv"

# === load CSV file===
df = pd.read_csv(input_file)

# === count TP / FP / FN ===
tp = (df["match_type"] == "correct").sum()
fp = (df["match_type"] == "substitution").sum()
fn = (df["match_type"] == "missing").sum()

# === claculate Precision / Recall / F1 ===
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# === print results ===
print("=== ALL Terms P_R_F1 Results ===")
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1 Score:  {f1:.2%}")
