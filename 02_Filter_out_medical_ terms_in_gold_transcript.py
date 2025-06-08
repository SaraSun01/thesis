import json
import pandas as pd
from rapidfuzz import process, fuzz
import spacy

# === input file path ===
transcript_path = "all_gold_transcriptions_20250520.json" # input the path of your gold transciption
clinical_finding_path = "02_snomed_drug_terms_all.csv" # input the file name of medical terms that filtered by SNOMED
output_path = "fuzzy_matched_output_drug.csv"  # output file name

# === load the file ===
with open(transcript_path, "r", encoding="utf-8") as f:
    transcript_dict = json.load(f)

clinical_finding_df = pd.read_csv(clinical_finding_path)

# === load spaCy model ===
nlp = spacy.load("nl_core_news_sm")
stopwords = nlp.Defaults.stop_words

# ===Tokenization + combination of 1-gram and 2-gram (including cleaning) ===
def generate_candidates(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if token.is_alpha]
    # Cleaning: Remove stopwords and words with length less than or equal to 3
    tokens = [t for t in tokens if t.lower() not in stopwords and len(t) > 3]
    one_grams = tokens
    two_grams = [' '.join([tokens[i], tokens[i+1]]) for i in range(len(tokens)-1)]
    return list(set(one_grams + two_grams))

# === Perform fuzzy matching (keep only the top 3 matches) ===
def fuzzy_match_terms(candidates, term_df, score_threshold=90):
    term_list = term_df["term"].tolist()
    match_pool = []
    for candidate in candidates:
        match, score, idx = process.extractOne(
            candidate, term_list, scorer=fuzz.ratio
        )
        if score >= score_threshold:
            matched_row = term_df.iloc[idx]
            match_pool.append({
                "candidate": candidate,
                "matched_term": match,
                "score": score,
                "type": matched_row["type"],
                "conceptId": matched_row["conceptId"]
            })
    # Returns the top 3 matches with the highest scores
    return sorted(match_pool, key=lambda x: x["score"], reverse=True)[:3]

# === Main loop: utterance-by-utterance matching ===
all_matches = []

for uid, text in transcript_dict.items():
    candidates = generate_candidates(text)
    clinical_finding_matches = fuzzy_match_terms(candidates, clinical_finding_df)
    for m in clinical_finding_matches:
        m["transcript_id"] = uid
    all_matches.extend(clinical_finding_matches)

# === save output ===
matched_df = pd.DataFrame(all_matches)
matched_df.to_csv(output_path, index=False)
print(f"Finished!Saved as {output_path}")
