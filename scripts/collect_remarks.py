import pandas as pd
import unicodedata
from pathlib import Path

df = pd.read_csv('./data/listings.csv', encoding='utf-8')
remarks = df['remarks'].tolist()

def normalize(s: str) -> str:
    s = s.strip().lower()
    s = unicodedata.normalize('NFKD', s)
    return s

normalized_remarks = [normalize(r) for r in remarks]

with open('./data/normalized_remarks.txt', 'w', encoding='utf-8') as f:
    for r in normalized_remarks:
        f.write(r + '\n')
