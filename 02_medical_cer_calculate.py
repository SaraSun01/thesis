import pandas as pd
import editdistance

# Step 1: 读取 CSV 文件
# 替换为你的实际文件路径
file_path = '01_ALL_fuzzy_match.csv'
df = pd.read_csv(file_path)

# Step 2: 选取匹配列，统一转为小写（也可用 upper）
gold_terms = df['gold_term'].astype(str).str.lower().tolist()
hypo_terms = df['matched_hyp_phrase'].astype(str).str.lower().tolist()

# Step 3: 拼接成一个整体字符串（中间用空格隔开）
gold_sequence = ' '.join(gold_terms)
hypo_sequence = ' '.join(hypo_terms)

# Step 4: 计算编辑距离（Levenshtein Distance）
distance = editdistance.eval(gold_sequence, hypo_sequence)

# Step 5: 计算总字符数
total_chars = len(gold_sequence)

# Step 6: 计算 M-CER
m_cer = distance / total_chars

# Step 7: 输出结果
print(f"Total Gold Characters: {total_chars}")
print(f"Edit Distance: {distance}")
print(f"ALL Medical CER (M-CER): {m_cer:.4f} ({m_cer*100:.2f}%)")
