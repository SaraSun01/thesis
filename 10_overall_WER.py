import pandas as pd
from jiwer import compute_measures, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip
import re

# Read the alignment transcript
df = pd.read_csv("03_all_k2_aligned_transcripts.csv")

# Remove empty content
df = df[df['gold'].notna() & df['hyp'].notna() & df['gold'].str.strip().astype(bool) & df['hyp'].str.strip().astype(bool)]

# Custom rule functions
def remove_nonverbal(text):
    return re.sub(r'\[.*?\]|\<.*?\>', '', text)

def normalize_spellings(text):
    return re.sub(r'covid[\s\-]?19', 'covid19', text)

def merge_hyphenated_words(text):
    return re.sub(r'(\w+)-(\w+)', r'\1\2', text)

# Get gold and hyp text lists
gold_texts = df['gold'].tolist()
hyp_texts = df['hyp'].tolist()

# Definite transformation
transformation = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip(),
    lambda x: remove_nonverbal(x),
    lambda x: normalize_spellings(x),
    lambda x: merge_hyphenated_words(x)
])

# Clean the text
gold_texts_clean = [transformation(t) for t in gold_texts]
hyp_texts_clean = [transformation(t) for t in hyp_texts]

# Splice into large text
gold_text = " ".join(gold_texts_clean)
hyp_text = " ".join(hyp_texts_clean)
truth_words = len(gold_text.split())

# Calculate WER
measures = compute_measures(gold_text, hyp_text)
overall_wer = measures['wer']
substitutions = measures['substitutions']
insertions = measures['insertions']
deletions = measures['deletions']
hits = measures['hits']

# Output
print(f"=== Evaluation Results ===")
print(f"Total Reference Words: {truth_words}")
print(f"Overall WER: {overall_wer:.2%}")
print(f"Substitutions: {substitutions}")
print(f"Insertions: {insertions}")
print(f"Deletions: {deletions}")
print(f"Hits (correct words): {hits}")
print(f"===========================")
