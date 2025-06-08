import pandas as pd
from jiwer import compute_measures, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip
import re

# 读取对齐后的 transcript 文件
df = pd.read_csv("03_all_k2_aligned_transcripts.csv")

# 去除空内容
df = df[df['gold'].notna() & df['hyp'].notna() & df['gold'].str.strip().astype(bool) & df['hyp'].str.strip().astype(bool)]

# 自定义规则函数
def remove_nonverbal(text):
    return re.sub(r'\[.*?\]|\<.*?\>', '', text)

def normalize_spellings(text):
    return re.sub(r'covid[\s\-]?19', 'covid19', text)

def merge_hyphenated_words(text):
    return re.sub(r'(\w+)-(\w+)', r'\1\2', text)

# 获取 gold 和 hyp 文本列表
gold_texts = df['gold'].tolist()
hyp_texts = df['hyp'].tolist()

# 定义 transformation
transformation = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip(),
    lambda x: remove_nonverbal(x),
    lambda x: normalize_spellings(x),
    lambda x: merge_hyphenated_words(x)
])

# 清理文本
gold_texts_clean = [transformation(t) for t in gold_texts]
hyp_texts_clean = [transformation(t) for t in hyp_texts]

# 拼接成大文本
gold_text = " ".join(gold_texts_clean)
hyp_text = " ".join(hyp_texts_clean)
truth_words = len(gold_text.split())

# 计算 WER
measures = compute_measures(gold_text, hyp_text)
overall_wer = measures['wer']
substitutions = measures['substitutions']
insertions = measures['insertions']
deletions = measures['deletions']
hits = measures['hits']

# 打印结果
print(f"=== Evaluation Results ===")
print(f"Total Reference Words: {truth_words}")
print(f"Overall WER: {overall_wer:.2%}")
print(f"Substitutions: {substitutions}")
print(f"Insertions: {insertions}")
print(f"Deletions: {deletions}")
print(f"Hits (correct words): {hits}")
print(f"===========================")
