# ================================================
# File for Task 2 (to evaluate NER model predictions)
# ===============================================
def __find_contained_rows(starts_, ends_):
    assert (len(starts_) == len(ends_))
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
        entities_list = []
        for _, entities_dict in annotations_list:
            entities_list.append(entities_dict["entities"])
    for e_i, entities in enumerate(entities_list):
        if len(entities) == 0:
            continue
        # check for overlapping indices
        starts_ = []
        ends_ = []
        for x in entities:
            starts_.append(x[0])
            ends_.append(x[1])
        to_delete = set()
        overlaps = __find_contained_rows(starts_, ends_)
        if len(overlaps)>0:
            for i, j in overlaps:
                assert i < len(starts_)
                assert j < len(starts_)
                span_len_1 = ends_[i] - starts_[i]
                span_len_2 = ends_[j] - starts_[j]
                # Keep only the entity with the longest span
                if span_len_1 >= span_len_2:
                    to_delete.add(j)
                else:
                    to_delete.add(i)
        # Delete from end so indices stay valid
        if len(to_delete) > 0:
            for idx in sorted(to_delete, reverse=True):
                entities.pop(idx)
        entities_list[e_i] = entities
    if is_annotations_list:
        for i, entities in enumerate(entities_list):
            annotations_list[i][1]["entities"] = entities_list[i]
        entities_list = annotations_list.copy()
    return __remove_overlaps(entities_list, is_annotations_list)

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
                if p_i in p_included:
                    continue
                # overlap exists in span
                if p_end >= g_start and p_start <= g_end: 
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
            if p_i not in p_included:
                true_labels.append("NONE")     # if no gold annotation
                pred_labels.append(p_label)    # Extra annotation from pred set
                sentence_ids.append(sent_idx)
    #print("True labels\n", true_labels)
    #print("pred\n", pred_labels)
    #print("sent ids\n", sentence_ids)
    return true_labels, pred_labels, sentence_ids

def __remove_overlaps(entities, is_annotations_list):
    cleaned_spans = []
    annotations_list = None
    if is_annotations_list:
        annotations_list = entities
        entities = []
        for _, entities_dict in annotations_list:
            entities.append(entities_dict["entities"])
    if not entities:
        return entities if not is_annotations_list else annotations_list
        
    for sentence_entities in entities:
        if not sentence_entities:
            cleaned_spans.append([])
            continue

        # sort by start, then length descending
        sentence_entities.sort(key=lambda x: (x[0], -(x[1]-x[0])))

        cleaned_sentence = []
        for s, e, label in sentence_entities:
            overlap = False
            for cs, ce, _ in cleaned_sentence:
                if not (e <= cs or s >= ce):  # any overlap
                    overlap = True
                    break
            if not overlap:
                cleaned_sentence.append([s, e, label])

        cleaned_spans.append(cleaned_sentence)
    if is_annotations_list:
        if len(cleaned_spans):
            for i, entity in enumerate(cleaned_spans):
                annotations_list[i][1]["entities"] = cleaned_spans[i]
        return annotations_list
    return cleaned_spans
    # print("In:")
    # print(entities)
    # # entities: [[start, end, label], ...]
    # annotations_list = None
    # if is_annotations_list:
    #     annotations_list = entities
    #     entities = []
    #     for _, entities_dict in annotations_list:
    #         entities.append(entities_dict["entities"])
    # if not entities:
    #     return []
    # for i, doc_ents in enumerate(entities):
    #     doc_ents.sort(key=lambda x: (x[0], -(x[1]-x[0])) if len(x) else [])  # start asc, length desc
    #     entities[i] = doc_ents
    # cleaned = []
    # for row in entities:
    #     if len(row) > 0:
    #         s, e, label = row[0]
    #         overlap = False
    #         for cleaned_row in cleaned:
    #             if len(cleaned_row) >0:
    #                 cs, ce, _ = cleaned_row
    #                 if not (e <= cs or s >= ce):  # any overlap
    #                     overlap = True
    #                     break
    #         if not overlap:
    #             cleaned.append([s, e, label])
    #     else:
    #         cleaned.append([])
    # if is_annotations_list:
    #     if len(cleaned):
    #         for i, entity in enumerate(cleaned):
    #             annotations_list[i][1]["entities"] = [cleaned[i]]
    #         cleaned = annotations_list.copy()
    #     else:
    #         cleaned = annotations_list.copy()
    # print("out:")
    # print(cleaned)
    # return cleaned