# preprocessing.py
import spacy
import re

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# Template boilerplate common across specialties


# Clinically meaningful abbreviations
CLINICAL_ABBREV = {"bp","hr","ct","mri","dm","mi","copd","htn","hx","dx","sx"}

# # Negation terms important to semantics
# NEGATION = {"no", "not", "denies", "deny", "without"}


def preprocess_text(
    text: str,
    enable: bool = True,
    lowercase: bool = True,
    lemmatize: bool = True,
    remove_stopwords: bool = True,
    remove_numbers: bool = True,
    keep_short_tokens: bool = True,
    remove_medical_boilerplate: bool = True,
    **kwargs
):
    MEDICAL_STOPWORDS = {
        "history", "present", "illness", "hpi", "subjective", "objective",
        "assessment", "procedure", "consult", "chief", "complaint",
        "reason", "evaluation", "management", "findings", "impression",
        "performed","patient", "biopsy", "diagnosis"
        
        # generic clinical subject
        "patient", "patients", "pt", "pts", "male", "female",
        "man", "woman", "boy", "girl", "person", "people",
        "family", "mother", "father", "mom", "dad", "old"

        # visit / clinical workflow
        "visit", "followup", "consult", "consultation",
        "admission", "discharge", "clinic", "hospital",
        "floor", "bed", "nurse", "resident", "rounds",
        "presenting", "presented", "referral", "referred",

        # subjective template
        "subjective", "objective", "assessment",
        "plan", "impression", "review", "concern",
        "complaint", "chief", "chief complaint",

        # history template
        "history", "hx", "hpi", "illness", "recent",
        "symptom", "symptoms", "chronic", "acute",
        "course", "status", "stable", "unstable",
        "baseline", "ongoing", "follow",

        # exam boilerplate
        "exam", "examination", "examined", "findings",
        "clear", "normal", "abnormal", "unremarkable",
        "negative", "positive", "equal", "intact",
        "noted", "notes", "appears",

        # generic measurement words
        "rate", "level", "range", "mm", "cm",
        "degree", "mild", "moderate", "severe",
        "high", "low", "elevated", "slightly",

        # non-discriminative anatomy
        "left", "right", "bilateral", "upper", "lower",
        "distal", "proximal", "anterior", "posterior",
        "lateral", "medial", "superior", "inferior",

        # common verbs (clinical)
        "show", "shows", "shown", "reveal", "reveals",
        "demonstrate", "demonstrates", "indicate", "appear"
        "appears", "present", "presented", "suggest",

        # procedure boilerplate verbs
        "performed", "performed on", "performed with",
        "removed", "placed", "inserted", "discussed",
        "discuss", "explain", "explained", "reviewed",
        "assessed", "evaluated", "noted", "recommended",

        # surgical boilerplate
        "procedure", "procedures", "operative", "operation",
        "surgery", "surgical", "incision", "closure",
        "bleeding", "complication", "complications",
        "estimated", "blood loss", "minimal",
        "anesthesia", "preoperative", "postoperative",

        # body location generic terms
        "area", "region", "side", "site", "portion",
        "segment", "structure", "surface",

        # clinical negation patterns
        "no", "not", "denies", "deny", "without", "none",
        "unable", "able", "refused", "refuse",

        # vital signs template
        "bp", "hr", "rr", "temp", "pr", "pulse",
        "pressure", "breathing", "respiration",
        "oxygen", "saturation", 

        # medication / dosage noise
        "mg", "mcg", "ml", "g", "l", "tablet",
        "daily", "nightly", "morning", "evening",
        "dose", "dosed", "dosage", "po", "iv",

        # imaging boilerplate
        "image", "images", "imaging", "scan", "scans",
        "technique", "protocol", "view", "views",

        # administrative identifiers
        "mrn", "dob", "id", "chart", "code", "signature",
        "dated", "dictated", "transcribed",

        # common filler
        "also", "however", "therefore", "overall",
        "currently", "initially", "finally",
        "approximately", "significantly", "likely",

        # time words
        "day", "days", "night", "year", "years",
        "week", "weeks", "month", "months",
        "today", "yesterday", "tomorrow",
        "daily", "hour", "hours", "after"

        # admission/discharge noise
        "admit", "admitted", "discharge", "transferred",
        "transfer", "follow", "followed", "ordered",
        "order", "ordering",

        # generic clinical outcome
        "improved", "worsened", "stable", "resolved",
        "resolved", "persistent", "intermittent",
        "ongoing", "recurrent",

        # consent / risk template
        "risk", "benefit", "risks", "benefits",
        "probable", "possible", "discussed",
        "consent", "consented",

        # surgical instruments noise
        "needle", "suture", "scalpel", "gauge",
        "trocars", "scope", "blade", "drain",

        # hospital workflow nouns
        "floor", "room", "unit", "ward",
        "staff", "team", "provider",
        "care", "treatment", "therapy",

        # generic radiology vocabulary
        "axial", "sagittal", "coronal",
        "image", "window", "slice", "series",
        "sequence", "contrast", "artifact",
    }
    if not isinstance(text, str):
        return ""
    
    if not enable:
        return text

    if lowercase:
        text = text.lower()

    # Remove symbols but keep whitespace
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove stand-alone numbers
    if remove_numbers:
        text = re.sub(r"\b\d+\b", " ", text)

    doc = nlp(text)
    tokens = []

    for token in doc:
        
        # # keep negations
        # if token.text in NEGATION:
        #     tokens.append(token.text)
        #     continue

        t = token.lemma_ if lemmatize else token.text

        if remove_stopwords and token.is_stop:
            continue

        if remove_medical_boilerplate and t in MEDICAL_STOPWORDS:
            continue

        if len(t) <= 1:
            continue

        # length-based filtering but preserve clinical abbreviations
        if not keep_short_tokens and len(t) <= 3 and t not in CLINICAL_ABBREV:
            continue

        tokens.append(t)

    return " ".join(tokens).strip()
