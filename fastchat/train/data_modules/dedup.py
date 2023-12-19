import chromadb
from chromadb.config import Settings
import random
from fastchat.modules.embedder_adapter import Embedder, get_embedder
from fastchat.conversation import (
    SeparatorStyle,
)
from fastchat.model.model_adapter import get_conversation_template
import copy

def dedup_dataset(dataset, prompt_template='chat-orca', target_text_len=100, n_results=100, distance_threshold = 0.6):
    conv = get_conversation_template(prompt_template)
    system_message = conv.system_message
    sep_style = conv.sep_style
    sep = conv.sep
    prompt_user, prompt_bot = conv.roles

    len_sep_style = 0
    if sep_style == SeparatorStyle.ADD_COLON_TWO:
        len_sep_style = 1

    len_front = len(system_message) + len(sep) + len(prompt_user) + len_sep_style + 1
    len_rear = len(sep) + len(prompt_bot) + len_sep_style
    def filter_question(data):
        return { 
            **data,
            'prompt': data['prompt'][len_front:-len_rear][:target_text_len]
        }

    question_dataset = dataset.map(filter_question)
    
    chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
    embedder = get_embedder("ddobokki/klue-roberta-base-nli-sts-ko-en")
    collection = chroma_client.create_collection(name="context", embedding_function=embedder.embed, metadata={"hnsw:space": "cosine"})
    ids = []
    # add
    texts = question_dataset['prompt']
    last_id = -1
    new_ids = [f"id{i+last_id+1}" for i in range(len(texts))]
    ids += new_ids
    collection.add(documents=texts, ids=new_ids)

    query_ids = copy.deepcopy(new_ids)
    selected_ids = []
    duplicated_ids = []

    while query_ids:
        current_id = random.choice(query_ids)
        selected_ids.append(current_id)
        search_strings = [texts[int(current_id[2:])]]
        if collection.count() == 0:
            print("Warning: collection is empty. Forced break")
            break
        result = collection.query(query_texts=search_strings, n_results=min(n_results, len(query_ids)), include=['distances']) #'documents'

        search_ids = result['ids'][0]
        distances = result['distances'][0]
        remove_ids = []
        for idx in range(len(search_ids)):
            sid = search_ids[idx]
            dist = distances[idx]
            if dist < distance_threshold:
                remove_ids.append(sid)

        for rid in remove_ids:
            if rid in query_ids:
                query_ids.remove(rid)
        duplicated_ids += remove_ids
        collection.delete(ids=remove_ids)

        print(f"Total:{len(new_ids)} Selected:{len(selected_ids)} current_dup:{len(remove_ids)} vector_store:{collection.count()} remained:{len(query_ids)} total_dup:{len(duplicated_ids)}", '\t\t\t\t\t', end='\r')
    
    print('finished dedup data:', f"Total:{len(new_ids)} Selected:{len(selected_ids)} current_dup:{len(remove_ids)} vector_store:{collection.count()} remained:{len(query_ids)} total_dup:{len(duplicated_ids)}")

    selected_ids = [int(sid[2:]) for sid in set(selected_ids)]

    dataset = dataset.select(selected_ids)
    
    return dataset