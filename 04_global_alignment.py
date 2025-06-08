import pandas as pd
import json

# 1. load gold transcript 
with open('03_all_gold_transcriptions_20250520.json', 'r', encoding='utf-8') as f:
    gold_dict = json.load(f)

gold_df = pd.DataFrame(list(gold_dict.items()), columns=['id', 'gold'])

# 2. load hypothesis file
hyp_df = pd.read_csv('03_gm_kaldi_030.tsv', sep='\t', header=None, names=["id", "hyp"])
hyp_df = hyp_df[['id', 'hyp']]

#  check and print duplicate ids 
duplicates = hyp_df[hyp_df.duplicated(subset='id', keep=False)]
print(f"[!] There are {len(duplicates)} dulicate reocords in hyp_df，the number of dupilacte ids is {duplicates['id'].nunique()} ：")
print(duplicates.sort_values('id'))

# check if there are duplicate ids in gold_df 
dup_gold = gold_df[gold_df.duplicated(subset='id', keep=False)]
print(f"There are {len(dup_gold)} duplicate records in gold_df，the number of dupilacte ids is {dup_gold['id'].nunique()} 个。")

# check which ids in hypo file but not in gold
extra_in_hyp = set(hyp_df['id']) - set(gold_df['id'])
print(f"hyp 中有 {len(extra_in_hyp)} 个 id 不在 gold 中：", extra_in_hyp)

# Find the ids that cannot be matched in gold
unmatched_ids = set(gold_df['id']) - set(hyp_df['id'])
unmatched_df = gold_df[gold_df['id'].isin(unmatched_ids)]

# output as csv
unmatched_df.to_csv('unmatched_gold_only.csv', index=False, encoding='utf-8')
print(f"There are {len(unmatched_df)} utterances in gold that failed to match hypo，saved as 'unmatched_gold_only.csv'")


# 3. Merge two DataFrames (aligned by id)
aligned_df = pd.merge(gold_df, hyp_df, on='id')

# 4. save as csv 
aligned_df.to_csv('kaldi_aligned_transcripts.csv', index=False, encoding='utf-8')

print(f"Finished！There are {len(aligned_df)} utterances in total，saved as 'kaldi_aligned_transcripts.csv'.")
