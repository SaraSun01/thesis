import spacy
import json
import csv
import re
from pathlib import Path
from collections import defaultdict

# 1.Load language model
nlp = spacy.load("nl_core_news_lg")  

# 2. Add custom entity rules
ruler = nlp.add_pipe("entity_ruler", before="ner")

patterns = [
    {"label": "ABBR", "pattern": [{"TEXT": {"REGEX": "^[A-Z]{2,}$"}}]},  # All capital letters may be abbreviations
    # Add known specific abbreviations
    {"label": "ABBR", "pattern": "SO"},
    {"label": "ABBR", "pattern": "BIA"},
    {"label": "ABBR", "pattern": "TSH"},
    {"label": "ABBR", "pattern": "VS"},
    {"label": "ABBR", "pattern": "HDL"},
    {"label": "ABBR", "pattern": "NBT"},
    {"label": "ABBR", "pattern": "PLVT"},
    # Added known location
    {"label": "LOC", "pattern": [{"LOWER": "den"}, {"LOWER": "bosch"}]},
    {"label": "LOC", "pattern": "Berkenlaan"},
]
ruler.add_patterns(patterns)

# 3. Adding extra Functions
def clean_text(text):
    text = re.sub(r"<unk>", "", text)
    text = re.sub(r"<dubbele_punt>", "", text)
    text = re.sub(r"<slash>", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def post_process_entities(entities, text)
    corrected = []
    for ent_text, ent_label in entities:
        if ent_text.isupper() and len(ent_text) >= 2:
            corrected.append((ent_text, "ABBR"))
        elif any(loc_word in ent_text.lower() for loc_word in ["laan", "straat", "plein", "weg"]):
            corrected.append((ent_text, "LOC"))
        elif any(org_word in ent_text.lower() for org_word in ["bedrijf", "stichting", "vereniging"]):
            corrected.append((ent_text, "ORG"))
        else:
            corrected.append((ent_text, ent_label))
    
    return corrected

def extract_quantity_units(doc):
    units = []
    common_units = ["dag", "dagen", "week", "weken", "maand", "maanden",
                    "tabletten", "tablet", "pillen", "capsules", "keer", "mg", "ml"]

    for i in range(len(doc) - 1):
        token1, token2 = doc[i], doc[i + 1]
        if token1.pos_ == "NUM" and token2.text.lower() in common_units:
            phrase = f"{token1.text} {token2.text}"
            units.append((phrase, "CARDINAL_UNIT"))
    return units

def load_entity_dictionary():
    known_entities = {
        "Duren": "PER",
        "Albert heijn": "BRAND",
        "Claudia": "PER",
        "medipoint": "BRAND",
        "Dongen": "PER",
        "den Bosch": "LOC",
        # add more here
    }
    return known_entities

def apply_dictionary_lookup(entities, text, known_entities):
    result = list(entities) 
    # Check if each entity in the dictionary is in the gold text
    for entity_text, entity_type in known_entities.items():
        pattern = r'\b' + re.escape(entity_text.lower()) + r'\b'
        if re.search(pattern, text.lower()):
            found = False
            for i, (ent_text, ent_type) in enumerate(result):
                if ent_text.lower() == entity_text.lower():
                    result[i] = (ent_text, entity_type)
                    found = True
                    break 
            if not found:
                result.append((entity_text, entity_type))
    
    return result

# 4. Main processing flow
def process_transcripts(transcripts_data):
    results = []
    known_entities = load_entity_dictionary()
    
    for uid, text in transcripts_data.items():
        cleaned_text = clean_text(text)
        doc = nlp(cleaned_text)
        quantity_units = extract_quantity_units(doc)


        # Basic entity extraction
        spacy_entities = [(ent.text, ent.label_) for ent in doc.ents]
        # Application post-processing and dictionary lookup
        processed_entities = post_process_entities(spacy_entities, cleaned_text)
        final_entities = apply_dictionary_lookup(processed_entities, cleaned_text, known_entities)
        # Extract all (possibly multi-word) entity names
        proper_nouns = set(ent_text for ent_text, _ in final_entities)

        for proper_noun in proper_nouns:
            entity_label = "UNKNOWN"
            for entity_text, entity_type in final_entities:
                if entity_text.lower() == proper_noun.lower():
                    entity_label = entity_type
                    break
            
            results.append({
                "id": uid,
                "proper_noun": proper_noun,
                "label": entity_label
            })
        for phrase, label in quantity_units:
            results.append({
                "id": uid,
                "proper_noun": phrase,
                "label": label
            })

    return results

# 5. Main execution process
if __name__ == "__main__":
    # Reading Transcript Files
    with open("all_gold_transcriptions_20250520.json", "r", encoding="utf-8") as f:
        transcripts = json.load(f)
    
    # Processing Transcription
    results = process_transcripts(transcripts)
# output file
    with open("proper_noun_classification.csv", "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["id", "proper_noun", "label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("Finished! Saved as 'proper_noun_classification.csv'")
