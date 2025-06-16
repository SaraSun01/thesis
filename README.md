# A Comparative Evaluation of Closed- and Open-Vocabulary ASR Systems for the Recognition of Dutch Healthcare Terms

This repository contains the code used in the thesis project *A Comparative Evaluation of Closed- and Open-Vocabulary ASR Systems for the Recognition of Dutch Healthcare Terms*.  
The project aims to evaluate and compare two ASR systems—one closed-vocabulary and one open-vocabulary—by analyzing their performance in recognizing domain-specific Dutch healthcare terms.

## Dependencies

Please install the following Python packages before running the scripts:

- `pandas`
- `rapidfuzz`
- `spacy`
- `editdistance`
- `jiwer`

Install them using pip:

```bash
pip install pandas rapidfuzz spacy editdistance jiwer
```

## File Overview and Execution Order
Scripts should be executed in the following order (01–10):

### 01_extract_SNOMED_medical_terms.py
Parses the SNOMED CT relationship and description files to extract healthcare terms from specific categories (e.g., diseases, drugs). Outputs term name and concept ID to a CSV.

### 02_Filter_out_medical_terms_in_gold_transcript.py
Uses the term list from script 01 to identify and extract occurrences of these terms from gold-standard transcripts (in JSON format). Outputs a CSV of identified terms by category.

### 03_extract_named_entities.py
Uses spaCy to extract named entities (e.g., person names, locations) from gold transcripts.

### 04_global_alignment.py
Performs global alignment between the gold transcript and ASR hypothesis for fuzzy string matching. Should be run separately for each ASR system.

### 05_terms_match_type.py
Generates detailed fuzzy match results between hypothesis terms and gold terms, for both medical terms and named entities. Requires outputs from scripts 02/03 and 04.

### 06_P_R_F1_calculate.py
Calculates precision, recall, and F1-score for each category of recognized terms.

### 07_medical_cer_calculate.py
Computes character-level CER (M-CER) for each type of term.

### 08_medical_wer_calculate.py
Computes word-level WER (M-WER) for each type of term.

### 09_overall_CER.py
Computes overall character error rate (CER) for full transcripts.

### 10_overall_WER.py
Computes overall word error rate (WER) for full transcripts. 
---

## Contact
For questions about the code or thesis, please contact: s.sun.19@student.rug.nl
