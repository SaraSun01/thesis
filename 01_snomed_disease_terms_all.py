import pandas as pd

# ------------------- 配置项 -------------------
RELATIONSHIP_FILE = "sct2_Relationship_Snapshot_NL1000146_20250331.txt"
DESCRIPTION_FILE = "sct2_Description_Snapshot-nl_NL1000146_20250331.txt"  
OUTPUT_FILE = "snomed_disease_terms_all.csv"

DISEASE_PARENTS = {"40733004","27624003","49601007","118940003","50043002","928000","74732009"}      # Disease (disorder)
LANGUAGE_CODE = "nl"                # choose language
# ---------------------------------------------

# ------------------ 函数部分 -------------------
def get_all_descendants(parent_ids, rel_df):
    isa_df = rel_df[
        (rel_df["typeId"] == "116680003") & 
        (rel_df["active"] == "1")
    ]
    descendants = set()
    queue = list(parent_ids)
    while queue:
        current = queue.pop()
        children = isa_df[isa_df["destinationId"] == current]["sourceId"].tolist()
        for child in children:
            if child not in descendants:
                descendants.add(child)
                queue.append(child)
    return descendants
# ----------------------------------------------

# 1. read files
print("reading relationship and description files...")
rel_df = pd.read_csv(RELATIONSHIP_FILE, sep="\t", dtype=str)
desc_df = pd.read_csv(DESCRIPTION_FILE, sep="\t", dtype=str)

# 2. get conceptId of descendants
print("searching disease terms...")
disease_ids = get_all_descendants(DISEASE_PARENTS, rel_df)

# 3. screen terms
desc_df = desc_df[
    (desc_df["active"] == "1") & 
    (desc_df["languageCode"] == LANGUAGE_CODE)
]

# 4. extract disease terms
disease_terms = desc_df[desc_df["conceptId"].isin(disease_ids)].copy()
disease_terms["type"] = "disease"


# 5. merge and remove duplicates
terms_df = pd.concat([disease_terms])
terms_df = terms_df.drop_duplicates(subset=["term", "conceptId", "type"])
terms_df = terms_df[["term", "type", "conceptId", "active"]]

# 6. output
terms_df.to_csv(OUTPUT_FILE, index=False)
print(f"Finished! A total of {len(terms_df)} terms were extracted, saved as {OUTPUT_FILE}")
