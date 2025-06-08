import pandas as pd
from rapidfuzz import process, fuzz

# === 参数设定 ===
CORRECT_THRESHOLD = 100
SUBSTITUTION_THRESHOLD = 75
MAX_NGRAM = 3  # 最大n-gram窗口

# === 工具函数：生成 n-gram token 片段 ===
def generate_ngrams(tokens, max_n=3):
    ngrams = []
    for n in range(1, max_n + 1):
        ngrams += [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return ngrams

# === 加载对齐文本（含 gold + hyp） ===
aligned_df = pd.read_csv("03_kaldi_aligned_transcripts.csv")

# === 加载 drug gold 匹配（filtered drug matches） ===
drug_df = pd.read_csv("01_gold_filtered_clinical_finding.csv")
drug_df = drug_df[drug_df['type'] == 'clinical finding']
drug_df = drug_df[['transcript_id', 'candidate']]

# === 构建 gold drug term 字典: {uuid: [term1, term2, ...]} ===
gold_terms_dict = drug_df.groupby("transcript_id")["candidate"].apply(list).to_dict()

# === 匹配结果列表 ===
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

# === 保存输出 ===
result_df = pd.DataFrame(results)
result_df.to_csv("clinical_finding_term_fuzzy_match.csv", index=False)
print("✅ 术语 fuzzy 匹配分析完成，输出文件: clinical_finding_term_fuzzy_match.csv")


