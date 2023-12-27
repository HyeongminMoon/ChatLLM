# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence, List, Union

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from transformers import deepspeed as ds
import deepspeed
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: Union[List[str], str] = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False
    data_format: str = "vicuna"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    max_grad_norm: float = field(default=1.0)
    group_by_length: bool = field(default=False)


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
#     """Collects the state dict and dump to disk."""
#     state_dict = trainer.model.state_dict()
#     if trainer.args.should_save:
#         cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
#         del state_dict
#         trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
#     model = trainer.model
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    model = trainer.model
    state_dict = trainer.model.state_dict()
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
    if trainer.args.should_save:
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    data_format="vicuna_v1.1",
    # additional_system_message=None,
    custom_system_message=None,
) -> Dict:
    conv = get_conversation_template(data_format)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    if custom_system_message is not None:
        conv.system_message = custom_system_message

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        # padding="max_length", #memo todo False=GPTNeox max_length=Llama
        padding=False,
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        tokenizer: transformers.PreTrainedTokenizer,
        data_format="vicuna",
    ):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, self.data_format)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
        self.data_format = data_format

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        tokenizer: transformers.PreTrainedTokenizer,
        data_format="chat-orca",
    ):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.data_format = data_format
        self.conv = get_conversation_template(data_format)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        data_format = self.data_format
        # additional_system_message = None
        custom_system_message = None
        task_name = None
        if "task" in self.raw_data[i]:
            task_name = self.raw_data[i].get("task")
        elif "task_name" in self.raw_data[i]:
            task_name = self.raw_data[i].get("task_name")
        if task_name is not None:
            if task_name == "instruct":
                custom_system_message = self.conv.tasks[task_name].format(instruction=self.raw_data[i]['instruction'])
            elif task_name == "system_instruct":
                custom_system_message = self.conv.tasks[task_name].format(system=self.raw_data[i]['system'])
            else:
                custom_system_message = self.conv.tasks[task_name]
                
            #     additional_system_message = (
            #         f" Your reply should be based on the context below.\n\nContext:{self.raw_data[i]['instruction']}"
            #         if data_format == "chat-orca"
            #         else f" 당신의 답변은 아래 맥락에 기반하여 대답해야 합니다.\n\n맥락:{self.raw_data[i]['instruction']}"
            #     )
            # if task_name == "enkotranslation":
            #     data_format = (
            #         "enkotranslation-orca"
            #         if data_format == "chat-orca"
            #         else "enkotranslation-ko-orca"
            #     )
            # if task_name == "koentranslation":
            #     data_format = (
            #         "koentranslation-orca"
            #         if data_format == "chat-orca"
            #         else "koentranslation-ko-orca"
            #     )
            # if task_name == "summarization":
            #     data_format = (
            #         "summarization-orca"
            #         if data_format == "chat-orca"
            #         else "summarization-ko-orca"
            #     )

        ret = preprocess(
            [self.raw_data[i]["conversations"]],
            self.tokenizer,
            data_format,
            custom_system_message=custom_system_message,
        )
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret

def load_sft_dataset(dataset_path, split='train'):
    dataset = load_dataset("json", data_files=dataset_path, split=split)

    if "id" not in dataset.features:
        dataset = dataset.map(lambda x: {"id": "null"})

    if hasattr(dataset.features["id"], "dtype") and dataset.features["id"].dtype == "int64":
        dataset = dataset.map(lambda x: {"id": "id" + str(x["id"])})
    
    return dataset
    

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    #TODO data_path가 List일 경우, 붙이고 섞기
    if isinstance(data_args.data_path, list):
        train_json = concatenate_datasets([load_sft_dataset(path, 'train') for path in data_args.data_path])
    else:
        train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(
        train_json, tokenizer=tokenizer, data_format=data_args.data_format
    )

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(
            eval_json, tokenizer=tokenizer, data_format=data_args.data_format
        )
    else:#TODO: split eval
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    print("ds.is_deepspeed_zero3_enabled():", ds.is_deepspeed_zero3_enabled())

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        max_memory={i: "81920MiB" for i in range(torch.cuda.device_count())},
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        # use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Start trainner
    # deepspeed.initialize(model)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    model.config.use_cache = True
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
