from transformers import AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

from typing import List, Optional, Union
from datasets import load_dataset, Dataset


class BaseDatasetAdapter:
    def match(self, data_path: str, format: Optional[str] = None) -> bool:
        True

    def load_data(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        format: Optional[str] = None,
        n_samples: Optional[int] = None,
    ) -> Dataset:
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=data_path)
        else:
            dataset = load_dataset(data_path)

        if n_samples:
            dataset = dataset["train"].train_test_split(
                test_size=min(n_samples, dataset["train"].num_rows), shuffle=True
            )["test"]

        return dataset


dataset_adapters: List[BaseDatasetAdapter] = []


def register_dataset_adapter(cls):
    dataset_adapters.append(cls())


def get_dataset_adapter(
    data_path: str, format: Optional[str] = None
) -> BaseDatasetAdapter:
    """Get a dataset adapter for a format."""
    for adapter in dataset_adapters:
        if adapter.match(data_path, format):
            return adapter
    raise ValueError(f"No valid dataset adapter for {format}")


class VicunaDatasetAdapter(BaseDatasetAdapter):
    name = "vicuna_v1.1"
    system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    roles = {"human": "USER", "gpt": "ASSISTANT"}
    sep = " "
    sep2 = "</s>"

    def match(self, data_path: str, format: Optional[str] = None):
        if format:
            return "vicuna" in format.lower()
        return "vicuna" in data_path.lower()

    def load_data(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        # format: Optional[str] = None, # should be removed later
        n_samples: int = None,
        remove_columns: Optional[List] = None,
        max_length: Optional[int] = None,
        padding: Union[str, bool] = "max_length",
    ):
        # load dataset
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=data_path)
        else:
            dataset = load_dataset(data_path)

        if n_samples:
            dataset = dataset["train"].train_test_split(
                test_size=min(n_samples, dataset["train"].num_rows), shuffle=True
            )["test"]

        def tokenize(examples):
            seps = [self.sep, self.sep2]

            sources = examples["conversations"]
            conversations = []
            for i, source in enumerate(sources):
                if self.roles[source[0]["from"]] != "USER":
                    # Skip the first one if it is not from human
                    source = source[1:]

                prompt = self.system + seps[0]
                for j, conversation in enumerate(source):
                    role = self.roles[conversation["from"]]
                    if "value" in conversation and conversation["value"]:
                        prompt += role + ": " + conversation["value"] + seps[j % 2]
                    else:
                        prompt += role + ":"

                conversations.append(prompt)

            model_max_length = (
                max_length if max_length is not None else tokenizer.model_max_length
            )

            input_ids = tokenizer(
                conversations,
                return_tensors="pt",
                padding=padding,
                max_length=model_max_length,
                truncation=True,
            ).input_ids
            targets = input_ids.clone()

            mask_sep = self.sep + self.roles["gpt"] + ": "
            for conversation, target in zip(conversations, targets):
                total_len = int(target.ne(tokenizer.pad_token_id).sum())

                rounds = conversation.split(self.sep2)
                cur_len = 1
                target[:cur_len] = IGNORE_TOKEN_ID
                for i, rou in enumerate(rounds):
                    if rou == "":
                        break

                    parts = rou.split(mask_sep)
                    if len(parts) != 2:
                        break
                    parts[0] += mask_sep
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                    target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

                    cur_len += round_len
                target[cur_len:] = IGNORE_TOKEN_ID

                if False:
                    z = target.clone()
                    z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                    rank0_print(tokenizer.decode(z))

                if cur_len < model_max_length:
                    if cur_len != total_len:
                        target[:] = IGNORE_TOKEN_ID

            return dict(
                input_ids=input_ids,
                labels=targets,
                attention_mask=input_ids.ne(tokenizer.pad_token_id),
            )

        dataset = dataset.map(tokenize, batched=True, remove_columns=remove_columns)

        return dataset


class KoAlpacaDatasetAdapter(BaseDatasetAdapter):
    name = "alpaca"

    def match(self, data_path: str, format: Optional[str] = None):
        if format:
            return "alpaca" in format.lower()
        return "alpaca" in data_path.lower()

    def load_data(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        # format: Optional[str] = None, # should be removed later
        n_samples: int = None,
        remove_columns: Optional[List] = None,
        max_length: Optional[int] = None,
        padding: Union[str, bool] = "max_length",
    ):
        # load dataset
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=data_path)
        else:
            dataset = load_dataset(data_path)

        if n_samples:
            dataset = dataset["train"].train_test_split(
                test_size=min(n_samples, dataset["train"].num_rows), shuffle=True
            )["test"]

        def tokenize(examples):
            instructions = examples.get("instruction")
            inputs = examples.get("input")
            if not inputs:
                inputs = [None] * len(instructions)
            outputs = examples.get("output")

            prompts = []
            texts = []
            input_ids = []
            attention_mask = []
            for istr, inp, opt in zip(instructions, inputs, outputs):
                if inp:
                    prompt = f"### Instruction(istr):\n{istr}\n\n### Input(입력):\n{inp}\n\n### Response(응답):"
                    text = prompt + opt
                else:
                    prompt = f"### Instruction(명령어):\n{istr}\n\n### Response(응답):"
                    text = prompt + opt
                # if len(tokenizer(prompt)["input_ids"]) >= tokenizer.model_max_length:
                #     continue

                model_max_length = (
                    max_length if max_length is not None else tokenizer.model_max_length
                )

                tokenized_data = tokenizer(
                    text, padding=padding, max_length=model_max_length
                )

                input_ids.append(tokenized_data["input_ids"][:model_max_length])
                attention_mask.append(
                    tokenized_data["attention_mask"][:model_max_length]
                )
                prompts.append(prompt)
                texts.append(text)

            return dict(
                input_ids=input_ids, attention_mask=attention_mask, prompt=prompts
            )

        dataset = dataset.map(tokenize, batched=False, remove_columns=remove_columns)

        return dataset

class OrcaDatasetAdapter(BaseDatasetAdapter):
    name = "orca"

    def match(self, data_path: str, format: Optional[str] = None):
        if format:
            return "orca" in format.lower()
        return "orca" in data_path.lower()

    def load_data(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        # format: Optional[str] = None, # should be removed later
        n_samples: int = None,
        remove_columns: Optional[List] = None,
        max_length: Optional[int] = None,
        padding: Union[str, bool] = "max_length",
    ):
        # load dataset
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=data_path)
        else:
            dataset = load_dataset(data_path)

        if n_samples:
            dataset = dataset["train"].train_test_split(
                test_size=min(n_samples, dataset["train"].num_rows), shuffle=True
            )["test"]

        def tokenize(examples):
            instructions = examples.get("instruction")
            inputs = examples.get("input")
            if not inputs:
                inputs = [None] * len(instructions)
            outputs = examples.get("output")

            prompts = []
            texts = []
            input_ids = []
            attention_mask = []
            for istr, inp, opt in zip(instructions, inputs, outputs):
                if inp:
                    prompt = f"### System:\n{istr}\n\n### User:\n{inp}\n\n### Assistant:"
                    text = prompt + opt
                else:
                    prompt = f"### User:\n{istr}\n\n### Assistant:"
                    text = prompt + opt
                # if len(tokenizer(prompt)["input_ids"]) >= tokenizer.model_max_length:
                #     continue

                model_max_length = (
                    max_length if max_length is not None else tokenizer.model_max_length
                )

                tokenized_data = tokenizer(
                    text, padding=padding, max_length=model_max_length
                )

                input_ids.append(tokenized_data["input_ids"][:model_max_length])
                attention_mask.append(
                    tokenized_data["attention_mask"][:model_max_length]
                )
                prompts.append(prompt)
                texts.append(text)

            return dict(
                input_ids=input_ids, attention_mask=attention_mask, prompt=prompts
            )

        dataset = dataset.map(tokenize, batched=True, remove_columns=remove_columns)

        return dataset
    
register_dataset_adapter(VicunaDatasetAdapter)
register_dataset_adapter(KoAlpacaDatasetAdapter)
register_dataset_adapter(OrcaDatasetAdapter)
register_dataset_adapter(BaseDatasetAdapter)
