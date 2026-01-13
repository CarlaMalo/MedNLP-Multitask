from openai import OpenAI
import os
import re
import ast
import json

def chatgpt_annotate_text(labels, text, num_examples_per_label):
    client = OpenAI(
        api_key = os.environ.get("OPENAI_API_KEY")
    )
    assistant_msg = f"""
                    For each entity name, extract 30 unique keywords or short phrases from the text that are most strongly associated with that entity.
                    Association should be determined by semantic similarity between the entity name and the candidate word/phrase, not by co-occurrence.
                    Keyword Compression Rule:
                        For every candidate word/phrase, reduce it to the most semantically meaningful single word if possible. Exclude adjectives.
                        Use a multi-word phrase only when:
                        (1) the head word alone would be too vague or misleading, or 
                        (2) the phrase contains essential qualifiers.
                        Do not include redundant variants. Choose the shortest form that preserves the specific meaning.
                    Rank all candidate phrases primarily by semantic similarity (highest first).
                    If two candidates have very similar semantic similarity scores, break the tie using frequency in the text (higher frequency first). If these candidates have the same root word, use only the root word.

                    A candidate phrase should be excluded if it has both low semantic similarity and low frequency and there exist other candidates that have higher semantic similarity and higher frequency.
                    However, a candidate with high semantic similarity should not be excluded only because its frequency is low.

                    After applying these rules, return the top {num_examples_per_label} ranked phrases for the entity.

                    Guidelines:

                    Only use words or phrases that appear in the text.

                    Ignore stopwords.
                    
                    Phrases should be 2–4 words and must appear contiguously in the text.

                    Each entity must have exactly {num_examples_per_label} items.

                    Return a Python dictionary as a string where:

                    the keys are the entity names,

                    the values are lists of {num_examples_per_label} lists [index, keyword_or_phrase] where index starts from 0.
                """
    messages = [{'role': 'system', 
                'content': """You are an assistant that performs entity-focused keyword extraction.
                    Given: 
                        a list of entity names 
                        a block of text 
                    your task is to extract keywords associated with each entity."""},
                    
                {'role': 'assistant', 
                'content': str(assistant_msg)},
                {"role": "user", 
                 "content": [{"type": "text", 'text': ','.join(labels)}, {'type':'text', 'text': ' '.join(text)}]}]
    completion = client.chat.completions.create(
            model="gpt-4o", messages=messages
        )
    reply = completion.choices[0].message.content
    match = re.search(r"```(?:python)?\s*(.*?)```", reply, flags=re.DOTALL)
    if match:
        inner = match.group(1).strip()
    else:
        # fallback: no fences, assume raw content is the dict
        inner = reply.strip()
    result = ast.literal_eval(inner)
    return result


def annotate_sentences_and_save(all_sentences, annotations_dict, labels, annotations_filepath):
    annotations_list = []
    
    for sentence in all_sentences:
        if not isinstance(sentence, str):
            sentence = str(sentence)
        entities = []
        for label, keywords in annotations_dict.items():
            for keyword in keywords:
                start_idxs = [match.start() for match in re.finditer(keyword, sentence)]
                if len(start_idxs):
                    for idx in start_idxs:
                        entities.append([idx, idx+len(keyword), label] )
        annotations_list.append([sentence, {'entities':entities}])
    output = {'classes': labels, 'annotations':annotations_list}
    with open(annotations_filepath, 'w') as fp:
        json.dump(output, fp)
    return
