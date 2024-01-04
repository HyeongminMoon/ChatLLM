from typing import Dict, Optional
from datasets import Dataset, load_dataset

from fastchat.model.model_adapter import get_conversation_template

def extract_anthropic_prompt(prompt_and_response, search_term="\n\nAssistant:"):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]

def load_dpo_data_module(dataset_path):
    dataset = load_dataset(dataset_path)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)    
    

def load_dpo_dataset(dataset_path, split='train'):
    if dataset_path.endswith("json") or dataset_path.endswith("json.kr"):
        dataset = load_dataset("json", data_files=dataset_path, split=split)
    elif dataset_path.endswith("parquet"):
        dataset = load_dataset("parquet", data_files=dataset_path, split=split)
    else:
        dataset = load_dataset(dataset_path, split=split)
        
    return dataset

class ados_DPODataset:
    def __init__(
        self, 
        dataset_path="/data/llm_datasets/custom/ados/dpo/ados_dpo_v2.json",
        eval_dataset_path = "", #/data/llm_datasets/Ultrafeedback_binarized.ko.hankang/test_prefs.json.kr
        data_format='chat-orca',
        # search_term='\n\n### Assistant:',
        num_train=None,
        num_eval=None,
    ):
        self.dataset_path = dataset_path
        self.eval_dataset_path = dataset_path
        if eval_dataset_path:
            self.eval_dataset_path = eval_dataset_path
        self.data_format = data_format
        # self.search_term = search_term
        self.num_train = num_train
        self.num_eval = num_eval
    
    def get_prompt_and_response(self, data):
        conv = get_conversation_template(self.data_format)
        if data['system']:
            conv.system_message = conv.tasks['system_instruct'].format(system=data['system'])
        conv.append_message(conv.roles[0], data['input'])
        conv.append_message(conv.roles[1], '')
        prompt = conv.get_prompt()
        conv.update_last_message(data['chosen'])
        chosen = conv.get_prompt()[len(prompt):]
        conv.update_last_message(data['rejected'])
        rejected = conv.get_prompt()[len(prompt):]
        
        return prompt, chosen, rejected
    
    def make_dpo_data_module(self):
        def split_prompt_and_responses(data) -> Dict[str, str]:
            prompt, chosen, rejected = self.get_prompt_and_response(data)
            # prompt = extract_anthropic_prompt(prompt_and_response, self.search_term)
            # promopt_rejected = extract_anthropic_prompt(prompt_and_response_rejected, self.search_term)
            return {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
                             
        train_dataset = load_dpo_dataset(self.dataset_path)
        eval_dataset = load_dpo_dataset(self.eval_dataset_path)

        original_columns = list(train_dataset.features.keys())
        original_columns_eval = list(eval_dataset.features.keys())

        if self.num_train is not None:
            train_dataset = train_dataset.select(range(min(len(train_dataset), self.num_train)))
        if self.num_eval is not None:
            eval_dataset = eval_dataset.select(range(min(len(train_dataset), self.num_eval)))

        train_dataset = train_dataset.map(split_prompt_and_responses, remove_columns=original_columns)
        eval_dataset = eval_dataset.map(split_prompt_and_responses, remove_columns=original_columns_eval)

        return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

class hankang_DPODataset:
    def __init__(
        self, 
        dataset_path="/data/llm_datasets/Ultrafeedback_binarized.ko.hankang/",
        data_format='chat-orca',
        search_term='\n\n### Assistant:',
        num_train=None,
        num_eval=None,
    ):
        self.dataset_path = dataset_path
        self.data_format = data_format
        self.search_term = search_term
        self.num_train = num_train
        self.num_eval = num_eval
    
    def get_prompt_and_response(self, data):
        conv = get_conversation_template(self.data_format)

        for idx, _conv in enumerate(data):
            role = _conv['role']
            content = _conv['content_kr']
            if idx % 2 == 0 and role == 'user':
                conv.append_message(conv.roles[0], content)
            elif idx % 2 == 1 and role == 'assistant':
                conv.append_message(conv.roles[1], content)
            else:
                print("Warning: data type invaild")

        if len(conv.messages) == 0:
            print("Warning: data is empty")
        if len(conv.messages) % 2 != 0:
            print("Warning: data has weird pair")

        return conv.get_prompt()
    
    def make_dpo_data_module(self):
        def validate_prompt_and_responses(data) -> bool:
            try:
                prompt_and_response = self.get_prompt_and_response(data['chosen'])
                prompt_and_response_rejected = self.get_prompt_and_response(data['rejected'])
                prompt = extract_anthropic_prompt(prompt_and_response, self.search_term)
                promopt_rejected = extract_anthropic_prompt(prompt_and_response_rejected, self.search_term)
            except AssertionError:
                return False

            return True

        def split_prompt_and_responses(data) -> Dict[str, str]:
            prompt_and_response = self.get_prompt_and_response(data['chosen'])
            prompt_and_response_rejected = self.get_prompt_and_response(data['rejected'])
            prompt = extract_anthropic_prompt(prompt_and_response, self.search_term)
            promopt_rejected = extract_anthropic_prompt(prompt_and_response_rejected, self.search_term)
            return {
                "prompt": prompt,
                "chosen": prompt_and_response[len(prompt) :],
                "rejected": prompt_and_response_rejected[len(promopt_rejected) :],
            }
                             
                             
        dataset = load_dataset(self.dataset_path)

        train_dataset = dataset['train']
        eval_dataset = dataset['test']

        original_columns = list(train_dataset.features.keys())

        if self.num_train is not None:
            train_dataset = train_dataset.select(range(min(len(train_dataset), self.num_train)))
        if self.num_eval is not None:
            eval_dataset = eval_dataset.select(range(min(len(train_dataset), self.num_eval)))

        train_dataset = train_dataset.filter(validate_prompt_and_responses)
        train_dataset = train_dataset.map(split_prompt_and_responses, remove_columns=original_columns)

        eval_dataset = eval_dataset.filter(validate_prompt_and_responses)
        eval_dataset = eval_dataset.map(split_prompt_and_responses, remove_columns=original_columns)

        return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)