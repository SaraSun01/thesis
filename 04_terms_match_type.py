import pandas as pd
from rapidfuzz import process, fuzz

# This script is about generate a result about the type of fuzzy match between the term in hypothesis trancript and the term in gold transcript, 
# both for medical terms and named entities

# === Parameter settings ===
CORRECT_THRESHOLD = 100
SUBSTITUTION_THRESHOLD = 75
MAX_NGRAM = 3  # Maximum n-gram window

# === Utility function: Generate n-gram token fragments ===
def generate_ngrams(tokens, max_n=3):
    ngrams = []
    for n in range(1, max_n + 1):
        ngrams += [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return ngrams

# === Loading Aligned Text（includes gold + hyp） ===
aligned_df = pd.read_csv("03_kaldi_aligned_transcripts.csv")

# === Loading medical terms in gold transcript ===
drug_df = pd.read_csv("01_gold_filtered_clinical_finding.csv")
drug_df = drug_df[drug_df['type'] == 'clinical finding'] #take clinical finding as an example
drug_df = drug_df[['transcript_id', 'candidate']]

# === Constructing the gold medical term dictionary: {uuid: [term1, term2, ...]} ===
gold_terms_dict = drug_df.groupby("transcript_id")["candidate"].apply(list).to_dict()

# === Matching result list ===
results = []

for _, row in aligned_df.iterrows():
    uid = row['id']
    hyp_raw = str(row['hyp']).lower()
    hyp_tokens = hyp_raw.split()
    hyp_ngrams = generate_ngrams(hyp_tokens, MAX_NGRAM)
    
    gold_terms = gold_terms_dict.get(uid, [])
    for gold_term in gold_terms:
        term_lower = gold_term.lower()
        best_match, score, _ = process.extractOne(term_lower, hyp_ngrams, scorer=fuzz.ratio)
        
        if score >= CORRECT_THRESHOLD:
            match_type = "correct"
        elif score >= SUBSTITUTION_THRESHOLD:
            match_type = "substitution"
        else:
            match_type = "missing"
            best_match = ""

        results.append({
            "id": uid,
            "gold_term": gold_term,
            "matched_hyp_phrase": best_match,
            "match_score": round(score, 2),
            "match_type": match_type
        })

# === Saving as csv ===
result_df = pd.DataFrame(results)
result_df.to_csv("clinical_finding_term_fuzzy_match.csv", index=False)
print("Finished! Saved as: clinical_finding_term_fuzzy_match.csv")


