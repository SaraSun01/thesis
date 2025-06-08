import pandas as pd
import json

# 1. 加载 gold transcript JSON
with open('03_all_gold_transcriptions_20250520.json', 'r', encoding='utf-8') as f:
    gold_dict = json.load(f)

gold_df = pd.DataFrame(list(gold_dict.items()), columns=['id', 'gold'])

# 2. 加载 hypothesis CSV
hyp_df = pd.read_csv('03_gm_kaldi_030.tsv', sep='\t', header=None, names=["id", "hyp"])
hyp_df = hyp_df[['id', 'hyp']]

# ✅ 检查并打印重复 id 及其内容
duplicates = hyp_df[hyp_df.duplicated(subset='id', keep=False)]
print(f"[!] 共发现 {len(duplicates)} 条重复记录，重复 id 数量为 {duplicates['id'].nunique()} 个：")
print(duplicates.sort_values('id'))

# 检查 gold_df 中是否有重复的 id
dup_gold = gold_df[gold_df.duplicated(subset='id', keep=False)]
print(f"gold_df 中有 {len(dup_gold)} 条重复记录，重复 id 数量为 {dup_gold['id'].nunique()} 个。")

# 查看 hypo 中有哪些 id 不在 gold 中
extra_in_hyp = set(hyp_df['id']) - set(gold_df['id'])
print(f"hyp 中有 {len(extra_in_hyp)} 个 id 不在 gold 中：", extra_in_hyp)

# 找出 gold 中未能 match 到的 id
unmatched_ids = set(gold_df['id']) - set(hyp_df['id'])
unmatched_df = gold_df[gold_df['id'].isin(unmatched_ids)]

# 输出到 CSV 文件
unmatched_df.to_csv('unmatched_gold_only.csv', index=False, encoding='utf-8')
print(f"[i] gold 中有 {len(unmatched_df)} 条样本未能匹配到 hyp，已保存到 'unmatched_gold_only.csv'")


# 3. 合并两个 DataFrame（通过 UUID 对齐）
aligned_df = pd.merge(gold_df, hyp_df, on='id')

# 4. 保存对齐结果
aligned_df.to_csv('k2_aligned_transcripts.csv', index=False, encoding='utf-8')

print(f"对齐完成！共 {len(aligned_df)} 条样本，已保存到 'k2_aligned_transcripts.csv'.")
