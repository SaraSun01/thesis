import pandas as pd
import editdistance

# Step 1: load the file
file_path = '01_ALL_fuzzy_match.csv'
df = pd.read_csv(file_path)

# Step 2: Select matching columns and convert them to lowercase 
gold_terms = df['gold_term'].astype(str).str.lower().tolist()
hypo_terms = df['matched_hyp_phrase'].astype(str).str.lower().tolist()

# Step 3: Concatenate into a whole string (separated by spaces)
gold_sequence = ' '.join(gold_terms)
hypo_sequence = ' '.join(hypo_terms)

# Step 4: Calculate edit distance（Levenshtein Distance）
distance = editdistance.eval(gold_sequence, hypo_sequence)

# Step 5: Calculate the total number of characters
total_chars = len(gold_sequence)

# Step 6: Calculation of M-CER
m_cer = distance / total_chars

# Step 7: Output
print(f"Total Gold Characters: {total_chars}")
print(f"Edit Distance: {distance}")
print(f"ALL Medical CER (M-CER): {m_cer:.4f} ({m_cer*100:.2f}%)")
