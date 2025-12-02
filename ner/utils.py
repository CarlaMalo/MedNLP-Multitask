# ================================================
# File for Task 2 (to evaluate NER model predictions)
# ===============================================

def find_span(text, phrase):
    """
    Locate (start, end) in the sentence.
    If multiple choices, take the first one.
    Retrun None if not found。
    Input:
        text: str
        phrase: str
    Return:
        (start,end) or None
    """
    start = text.lower().find(phrase.lower())
    if start == -1:
        return None
    return (start, start + len(phrase))

def extract_spans(nlp, data):
    """
    Extract gold and predicted spans from data.
    Input:
        nlp: spacy NLP pipeline with NER component
        data: list of dict, each dict has keys "text" and "annotation"
              e.g., {"text": "...", "annotation": [["ankle","BODY_PART"], ...]}
    Return:
        gold_spans: list of list of (start,end,label)
        pred_spans: list of list of (start,end,label)
        labels: list of all labels in the dataset
    """
    gold_spans = []
    pred_spans = []
    labels = []   # For all ground-truth annotion in the test

    for item in data:
        text = item["text"]
        ann_list = item["annotation"]  # [["ankle","BODY_PART"], ...]
        
        doc = nlp(text)

        # -------- gold spans --------
        gold = []
        for phrase, label in ann_list:
            span = find_span(text, phrase)
            if span is not None:
                gold.append((span[0], span[1], label))
                labels.append(label)
            # else:
            #     print("Warning: phrase not found:", phrase)

        gold_spans.append(gold)

        # -------- predicted spans --------
        pred = []
        for ent in doc.ents:
            pred.append((ent.start_char, ent.end_char, ent.label_))

        pred_spans.append(pred)
    return gold_spans, pred_spans, labels

def convert_to_labels(gold_spans, pred_spans):
    """
    Prepare true/pred labels for evaluation.
    Input:
        gold_spans: list of list of (start,end,label)
        pred_spans: list of list of (start,end,label)
    Return:
        true_labels: list of str
        pred_labels: list of str
    """
    
    true_labels = []
    pred_labels = []

    for g_spans, p_spans in zip(gold_spans, pred_spans):

        #  (start,end,label) precise matching
        g_set = set(g_spans)
        p_set = set(p_spans)

        # True positives
        for span in g_set:
            if span in p_set:
                true_labels.append(span[2])
                pred_labels.append(span[2])
            else:
                true_labels.append(span[2])
                pred_labels.append("NONE")  # missed

        # False positives
        for span in p_set:
            if span not in g_set:
                true_labels.append("NONE")     # if no gold annotation
                pred_labels.append(span[2])    # wrong
                
    return true_labels, pred_labels