# ================================================
# File for Task 2 (to evaluate NER model predictions)
# ===============================================
def __find_contained_rows(starts_, ends_):
    contained = []

    for i, (s1, e1) in enumerate(zip(starts_, ends_)):
        for j, (s2, e2) in enumerate(zip(starts_, ends_)):
            if i == j:
                continue

            # span i is inside span j
            if s2 <= s1 and e1 <= e2:
                contained.append((i,j))
                break

    return contained

def remove_overlapping_spans(entities_list, is_annotations_list=False):
    if is_annotations_list:
        annotations_list = entities_list
        entities_list = [entities_dict["entities"] for _, entities_dict in annotations_list]
    for e_i, entities in enumerate(entities_list):
        # check for overlapping indices
        starts_ = [x[0] for x in entities]
        ends_ = [x[1] for x in entities]
        to_delete = set()
        overlaps = __find_contained_rows(starts_, ends_)
        if len(overlaps):
            for i,j in overlaps:
                span_len_1 = ends_[i] - starts_[i]
                span_len_2 = ends_[j] - starts_[j]
                # Keep only the entity with the longest span
                if span_len_1 >= span_len_2:
                    to_delete.add(j)
                else:
                    to_delete.add(i)
        # Delete from end so indices stay valid
        for idx in sorted(to_delete, reverse=True):
            entities.pop(idx)
        entities_list[e_i] = entities
    if is_annotations_list:
        for i, entities in enumerate(entities_list):
            annotations_list[i][1]["entities"] = entities_list[i]
        entities_list = annotations_list.copy()
    return entities_list

def extract_spans(nlp, sentences):
    """
    Extract gold and predicted spans from data.
    Input:
        nlp: spacy NLP pipeline with NER component
        data: list of sentences
    Return:
        pred_spans: list of [start,end,label]
    """
    pred_spans = []

    for sentence in sentences:
        doc = nlp(sentence)

        # -------- predicted spans --------
        pred = []
        for ent in doc.ents:
            pred.append([ent.start_char, ent.end_char, ent.label_])
        pred_spans.append(pred)

    return pred_spans

def get_label_lists(gold_spans, pred_spans):
    """
    Prepare true/pred labels for evaluation. Expects that annotations from  identified all 
    Input:
        gold_spans: list of [start,end,label]
        pred_spans: list of [start,end,label]
    Return:
        true_labels: list of str
        pred_labels: list of str
        sentence_ids: list of int
    """
    
    true_labels = []
    pred_labels = []
    sentence_ids = []

    # Iterate through each sentence's identified entities
    for sent_idx, (g_spans, p_spans) in enumerate(zip(gold_spans, pred_spans)):

        # True positives - overlap detected between the start and end indices in gold and pred, with same label assigned
        p_included = set()
        # Iterate through all gold entities identified
        for g_i, (g_start,g_end,g_label) in enumerate(g_spans):
            match_found = False # initialise match flag for the entity

            # Check if any predicted labels have overlapping spans as the current gold span
            for p_i, (p_start, p_end, p_label) in enumerate(p_spans): # Get list of starts and ends and labels in pred
                # overlap exists in span
                if not (p_end < g_start or p_start > g_end): 
                    true_labels.append(g_label)
                    pred_labels.append(p_label)
                    sentence_ids.append(sent_idx)
                    match_found = True
                    p_included.add(p_i)
                    break
            # overlap does not exist - NER failed to identify word/phrase as an entity
            if not match_found:
                true_labels.append(g_label)
                pred_labels.append("NONE")
                sentence_ids.append(sent_idx)


        # Check if pred labels exist where they don't in gold (exclude cases where both exist - already handled above)
        for p_i, (p_start, p_end, p_label) in enumerate(p_spans): 
            if p_i in p_included:
                continue
            match_found=False
            for g_i, (g_start,g_end,g_label) in enumerate(g_spans):
                # Check for where gold span overlaps with pred span
                if not (g_end < p_start or g_start > p_end): 
                    print(f'Error: label with overlapping spans not detected in previous loop.\n\
                          Gold: {g_spans[g_i]}, Pred: {p_spans[p_i]}')
                    match_found = True
            if not match_found:
                true_labels.append("NONE")     # if no gold annotation
                pred_labels.append(p_label)    # Extra annotation from pred set
                sentence_ids.append(sent_idx)
                
    return true_labels, pred_labels, sentence_ids