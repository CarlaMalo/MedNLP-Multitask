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
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# Clinically meaningful abbreviations that are short but semantically rich
CLINICAL_ABBREV = {"bp", "hr", "ct", "mri", "dm", "mi", "copd", "htn"}


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


def save_stopwords(stopwords: Set[str], filepath: str = 'custom_stopwords.csv'):
    """
    Save stopwords to CSV file.
    
    Parameters:
    -----------
    stopwords : Set[str]
        Set of stopwords to save
    filepath : str
        Output file path
    """
    df = pd.DataFrame(sorted(stopwords))
    df.to_csv(filepath, sep=',', header=False, index=False)
    print(f"Saved {len(stopwords)} stopwords to {filepath}")


def load_stopwords(filepath: str = 'custom_stopwords.csv') -> Set[str]:
    """
    Load stopwords from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to stopwords file
        
    Returns:
    --------
    Set[str]
        Set of stopwords
    """
    custom_stopword_path = os.path.join(os.path.dirname(__file__), filepath)
    if os.path.exists(custom_stopword_path):
        try:
            custom_stopwords_df = pd.read_csv(custom_stopword_path, header=None)
            stopwords = set(w.lower().strip() for w in custom_stopwords_df[0].tolist() if isinstance(w, str))
            print(f"[preprocessing] Loaded {len(stopwords)} custom stopwords from {filepath}")
            return stopwords
        except Exception as e:
            print(f"[preprocessing] Could not load {filepath}: {e}")
            return set()
    else:
        print(f"[preprocessing] {filepath} not found - using empty stopword list")
        return set()




import re
import spacy

# Load spaCy model only once
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# This must match previous behavior exactly
CLINICAL_ABBREV = {"bp", "hr", "ct", "mri", "dm", "mi", "copd", "htn"}


def preprocess_text(
    text: str,
    enable=True,
    lowercase=True,
    lemmatize=True,
    remove_stopwords=True,
    remove_numbers=False,
    keep_short_tokens=True,
    remove_medical_boilerplate=True,
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


