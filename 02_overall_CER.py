import pandas as pd
import editdistance
import re

def normalize_text(text):
    """去标点、统一小写、规范空格"""
    text = text.lower()  # 不区分大小写
    text = re.sub(r'[^\w\s]', '', text)  # 移除标点符号
    text = re.sub(r'\s+', ' ', text)  # 合并多个空格为1个
    return text.strip()

# Step 1: 读取文件
file_path = '03_all_k2_aligned_transcripts.csv'  # 修改为你的实际文件名
df = pd.read_csv(file_path)

# Step 2: 移除空值
df = df.dropna(subset=['gold', 'hyp'])

# Step 3: 预处理所有 gold 和 hyp 字符串
gold_clean = df['gold'].astype(str).apply(normalize_text)
hyp_clean = df['hyp'].astype(str).apply(normalize_text)

# Step 4: 拼接为整串文本
gold_text = ' '.join(gold_clean)
hyp_text = ' '.join(hyp_clean)

# Step 5: 计算 CER
distance = editdistance.eval(gold_text, hyp_text)
total_chars = len(gold_text)
cer = distance / total_chars

# Step 6: 输出结果
print(f"Total Gold Characters: {total_chars}")
print(f"Edit Distance: {distance}")
print(f"Overall CER：{cer:.4f} = {cer*100:.2f}%")
