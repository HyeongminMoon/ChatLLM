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