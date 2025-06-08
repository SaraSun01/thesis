import pandas as pd
import editdistance
import re

def normalize_text(text):
    """Remove punctuation, unify lowercase, and standardize spaces"""
    text = text.lower()  
    text = re.sub(r'[^\w\s]', '', text)  
    text = re.sub(r'\s+', ' ', text)  
    return text.strip()

# Step 1: Read the alignment file
file_path = '03_all_k2_aligned_transcripts.csv'  
df = pd.read_csv(file_path)

# Step 2: Remove null values
df = df.dropna(subset=['gold', 'hyp'])

# Step 3: Preprocess all gold and hyp strings
gold_clean = df['gold'].astype(str).apply(normalize_text)
hyp_clean = df['hyp'].astype(str).apply(normalize_text)

# Step 4: Concatenate into a whole string of text
gold_text = ' '.join(gold_clean)
hyp_text = ' '.join(hyp_clean)

# Step 5: Calculate CER
distance = editdistance.eval(gold_text, hyp_text)
total_chars = len(gold_text)
cer = distance / total_chars

# Step 6: Output
print(f"Total Gold Characters: {total_chars}")
print(f"Edit Distance: {distance}")
print(f"Overall CER：{cer:.4f} = {cer*100:.2f}%")
