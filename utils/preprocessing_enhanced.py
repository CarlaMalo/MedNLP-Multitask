# preprocessing_enhanced.py
"""
Enhanced text preprocessing for medical documents.
Includes automatic custom stopword generation and loading.
"""

import spacy
import re
import pandas as pd
import os
import numpy as np
from collections import Counter
from typing import Set, List

# Load spaCy model once for efficiency (disable NER/parser to save time)
nlp = spacy.load("en_core_web_md", disable=["ner", "parser"])

# Clinically meaningful abbreviations that are short but semantically rich
CLINICAL_ABBREV = {"bp", "hr", "ct", "mri", "dm", "mi", "copd", "htn"}

def get_short_text_idx(df, n_chars):
    short_idx = [i for i, t in enumerate(df['text']) if len(t) < n_chars]
    short_df = df.loc[short_idx, ['label', 'text']]
    print(short_df.to_string(index=True))
    return short_idx


def extract_capital_phrases(text: str, min_length: int = 6) -> List[str]:
    """
    Extract all-capital phrases from text (medical template headers).
    
    Parameters:
    -----------
    text : str
        Input medical text
    min_length : int
        Minimum phrase length to consider
        
    Returns:
    --------
    List[str]
        List of extracted capital phrases
        
    Examples:
    ---------
    "PREOPERATIVE DIAGNOSIS: pneumonia" -> ['PREOPERATIVE DIAGNOSIS']
    """
    pattern = r'\b[A-Z][A-Z\s/\-]{' + str(min_length - 1) + r',}\b'
    matches = re.findall(pattern, text)
    
    cleaned = []
    for match in matches:
        match = match.strip()
        match = re.sub(r'[^\w\s/\-]$', '', match)
        if len(match) >= min_length:
            cleaned.append(match)
    
    return cleaned


def get_specific_stopwords(
    df: pd.DataFrame,
    text_col: str = 'text',
    min_stopword_len: int = 6,
    min_frequency: int = 30
) -> Set[str]:
    """
    Generate custom stopwords from a corpus of medical documents.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing medical text
    text_col : str
        Name of column containing text
    min_stopword_len : int
        Minimum length of stopword phrases
    min_frequency : int
        Minimum occurrence frequency to be considered stopword
        
    Returns:
    --------
    Set[str]
        Set of custom stopwords
    """
    print(f"Extracting capital phrases from {len(df)} documents...")
    
    all_phrases = []
    for text in df[text_col]:
        if isinstance(text, str):
            phrases = extract_capital_phrases(text, min_length=min_stopword_len)
            all_phrases.extend(phrases)
    
    phrase_counts = Counter(all_phrases)
    
    stopwords = set([
        phrase for phrase, count in phrase_counts.items()
        if count >= min_frequency
    ])
    
    print(f"Found {len(stopwords)} custom stopwords (frequency >= {min_frequency})")
    
    top_phrases = phrase_counts.most_common(10)
    print("\nTop 10 most common phrases:")
    for phrase, count in top_phrases:
        print(f"  {phrase}: {count}")
    
    return stopwords


def load_stopwords(filepath: str) -> Set[str]:
    """
    Load stopwords from an explicit file path.
    """
    if not os.path.exists(filepath):
        print(f"[preprocessing] {filepath} not found - using empty stopword list")
        return set()

    df = pd.read_csv(filepath, header=None)
    stopwords = set(
        w.lower().strip()
        for w in df[0].tolist()
        if isinstance(w, str)
    )
    print(f"[preprocessing] Loaded {len(stopwords)} stopwords from {filepath}")
    return stopwords




# This must match previous behavior exactly
CLINICAL_ABBREV = {"bp", "hr", "ct", "mri", "dm", "mi", "copd", "htn"}


def preprocess_text(
    text: str,
    enable=False,
    lowercase=False,
    lemmatize=False,
    remove_stopwords=False,
    remove_numbers=False,
    keep_short_tokens=False,
    remove_medical_boilerplate=False,
    custom_stopwords=None,
):
    """
    EXACT same logic as preprocess_text_custom_stopwords(),
    but allowing explicit custom_stopwords passing.
    """
    if not isinstance(text, str):
        return ""
    if not enable:
        return text

    # Normalize
    if lowercase:
        text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    if remove_numbers:
        text = re.sub(r"\b\d+\b", " ", text)

    # Ensure stopwords always exist
    if custom_stopwords is None:
        custom_stopwords = set()

    tokens = []
    doc = nlp(text)

    for token in doc:
        # lemma or original
        t = token.lemma_ if lemmatize else token.text

        # filtering sequence — EXACT original ordering
        if remove_stopwords and token.is_stop:
            continue

        if remove_medical_boilerplate and t in custom_stopwords:
            continue

        if len(t) <= 1:
            continue

        # ORIGINAL behavior: append FIRST
        tokens.append(t)

        # AFTER append — same logic as before
        # only apply this rule when keep_short_tokens=False
        if not keep_short_tokens:
            if len(t) <= 3 and t not in CLINICAL_ABBREV:
                continue

    return " ".join(tokens).strip()



def get_specific_stopwords(df, min_stopword_len=5):
    caps = []
    word = ''
    def _find_cap(s):
        for ele in str(s):
            if ord(ele) < 65 or ord(ele) > 90:
                return 0
        return 1
    texts = df['text']
    for text in texts: 
        for char in text:
            if _find_cap(char):
                word += char
            elif len(word) > 0:
                if char == ':': # stop if the word has : at the end
                    caps.append(word)
                    word = ''
                elif char == ' ' and word[-1] != ' ': # continue if there is a space detected but not 2 in a row
                    word += char
                elif len(word) >= min_stopword_len: # final char is neither caps nor :, maybe >1 space in a row
                    word = word[:-1] if word[-1] == ' ' else word
                    caps.append(word)
                    word = ''
                else: # 2 spaces in a row and length < min_stopword_len
                    word = ''
    return caps