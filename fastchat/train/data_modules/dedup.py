import chromadb
from chromadb.config import Settings
import random
from fastchat.modules.embedder_adapter import Embedder, get_embedder
from fastchat.conversation import (
    SeparatorStyle,
)
from fastchat.model.model_adapter import get_conversation_template
import copy
import requests

def dedup_by_similarity(dataset, prompt_template='chat-orca', target_text_len=100, n_results=100, distance_threshold = 0.6):
    
    if prompt_template == 'chat-orca':
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
                # **data,
                'prompt': data['prompt'][len_front:-len_rear][:target_text_len]
            }

    if prompt_template == 'vicuna':
        def filter_question(data):
            return {
                'prompt': data['conversations'][0]['value'][:target_text_len]
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

    weird_ids = []
    error_ids = []
    while query_ids:
        current_id = random.choice(query_ids)
        if current_id in selected_ids:
            print("Warning: this is weird..")
            weird_ids.append(current_id)
            continue
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

        if remove_ids:
            duplicated_ids += remove_ids
            collection.delete(ids=remove_ids)
        else:
            print("Warning: this is error..")
            error_ids.append(current_id)

        print(f"Total:{len(new_ids)} Selected:{len(selected_ids)} current_dup:{len(remove_ids)} vector_store:{collection.count()} remained:{len(query_ids)} total_dup:{len(duplicated_ids)}", '\t\t\t\t\t', end='\r')

    print('finished dedup data:', f"Total:{len(new_ids)} Selected:{len(selected_ids)} current_dup:{len(remove_ids)} vector_store:{collection.count()} remained:{len(query_ids)} total_dup:{len(duplicated_ids)}")

    selected_ids = [int(sid[2:]) for sid in set(selected_ids)]

    dataset = dataset.select(selected_ids)

    return dataset

def dedup_non_pair(dataset, data_format='sft'):
    def validate_non_pair(data):
        dedup_flag = False
        if data_format == 'sft':
            conversations = data['conversations']

            if len(conversations) == 0: # empty
                return False

            if conversations[0]["from"] != 'human': # skip first if it's not human
                conversations = conversations[1:]

            for idx, conv in enumerate(conversations): # check right pairs
                role = conv['from']
                if idx % 2 == 0 and role != 'human':
                    dedup_flag = True
                    break
                elif idx % 2 == 1 and role != 'gpt':
                    dedup_flag = True
                    break
        elif data_format == 'dpo':
            if data['input'] and data['chosen'] and data['rejected']:
                return True
            else:
                return False
                
        # if dedup_flag:
        #     print(conv)
        
        return not dedup_flag
    
    start = len(dataset)
    dataset = dataset.filter(validate_non_pair)
    print(f"{start - len(dataset)}/{start} deduped")
    return dataset

def dedup_repetition(dataset, data_format='sft'):
    def validate_repetition(data):
        dedup_flag = False
        if data_format == 'sft':
            convs = data['conversations']
        elif data_format == 'dpo':
            convs = [data['input'], data['chosen'], data['rejected']]
        else:
            print("data_format should be [sft, dpo]")
            return
        
        for conv in convs:
            if data_format == 'sft':
                _from = conv['from']
                _value = conv['value']
            else:
                _value = conv
        
            words = _value.split(" ")
            if len(words) < 3: continue

            # unigram
            word1, word2 = words[0], words[1]
            for word3 in words[2:]:
                # print(word1, word2)
                if word1 == '' or word1 == ' ':
                    continue
                if word1 == word2 and word2 == word3:
                    dedup_flag = True
                    break
                word1, word2 = word2, word3
            if dedup_flag: 
                break
            
            continue_dup_flag = False
            if len(_value) > 50:
                for _vid in range(len(_value)-50):
                    letter = _value[_vid]
                    continue_dup_flag = True
                    for letter2 in _value[_vid:_vid+50]:
                        if letter == letter2:
                            continue
                        else:
                            continue_dup_flag = False
                            break
                    if continue_dup_flag:
                        break
                        
            if continue_dup_flag:
                dedup_flag = True
                break
            
            # bigram
            word1, word2, word3 = words[0], words[1], words[2]
            for word4 in words[3:]:
                if word1 == '' or word1 == ' ':
                    continue
                if word1 == word3 and word2 == word4:
                    dedup_flag = True
                    break
                word1, word2, word3 = word2, word3, word4
                
        # if dedup_flag:
        #     print(conv)
        
        return not dedup_flag
    
    start = len(dataset)
    dataset = dataset.filter(validate_repetition)
    print(f"{start - len(dataset)}/{start} deduped")
    return dataset

def dedup_math(dataset, data_format='sft'):
    def validate_math(data):
        dedup_flag = False
        if data_format == 'sft':
            convs = data['conversations']
        elif data_format == 'dpo':
            convs = [data['input'], data['chosen'], data['rejected']]
        else:
            print("data_format should be [sft, dpo]")
            return
        
        for conv in convs:
            if data_format == 'sft':
                _from = conv['from']
                _value = conv['value']
            else:
                _value = conv
            words = _value.split("\\")
            if len(words) > 3:
                dedup_flag = True
                break

        # if dedup_flag:
        #     print(conv)
        
        return not dedup_flag
    
    start = len(dataset)
    dataset = dataset.filter(validate_math)
    print(f"{start - len(dataset)}/{start} deduped")
    return dataset

def dedup_too_much_token(dataset, data_format='sft', max_token=3800, 
                        api_server_url = "http://localhost:21122",
                        model_name = "M-DIE-M-10.7B_gpt4_ep3-GPTQ"):
    
    def validate_too_much_token(data):
        dedup_flag = False
        if data_format == 'sft':
            convs = data['conversations']
        elif data_format == 'dpo':
            convs = [data['input'], data['chosen'], data['rejected']]
        else:
            print("data_format should be [sft, dpo]")
            return
        
        # do simple test first
        num_text = 0
        for conv in convs:
            if data_format == 'sft':
                _from = conv['from']
                _value = conv['value']
            else:
                _value = conv
            
            num_text += len(_value)
            
        if num_text < max_token // 2:
            return not dedup_flag
        
        num_token = 0
        for conv in convs:
            if data_format == 'sft':
                _from = conv['from']
                _value = conv['value']
            else:
                _value = conv
            
            input_json = {
                "model_name": model_name,
                "prompt": _value,
            }

            ret = requests.post(api_server_url + "/count_token", json=input_json)

            output_json = ret.json()
            num_token += output_json['count']
            
            if num_token > max_token:
                dedup_flag = True
                break

        # if dedup_flag:
        #     print(conv)
        
        return not dedup_flag
    
    start = len(dataset)
    dataset = dataset.filter(validate_too_much_token)
    print(f"{start - len(dataset)}/{start} deduped")
    return dataset

# for sharegpt_ko
def dedup_short(dataset, data_format='sft', threshold_len=10):
    def validate_short(data):
        dedup_flag = False
        if data_format == 'sft':
            convs = data['conversations']
        elif data_format == 'dpo':
            convs = [data['input'], data['chosen'], data['rejected']]
        else:
            print("data_format should be [sft, dpo]")
            return
        
        for conv in convs:
            if data_format == 'sft':
                _from = conv['from']
                _value = conv['value']
            else:
                _value = conv
            
            if len(_value) < threshold_len:
                dedup_flag = True
                break

        # if dedup_flag:
        #     print(conv)
        
        return not dedup_flag
    
    start = len(dataset)
    dataset = dataset.filter(validate_short)
    print(f"{start - len(dataset)}/{start} deduped")
    return dataset