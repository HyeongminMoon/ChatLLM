# Usage: deepspeed train_lora.py --deepspeed <$PATH_TO_DEEPSPEED_CONFIG>

# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
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
import logging
import pathlib
from typing import Dict, Optional, List
import os
import shutil

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers
from transformers import Trainer, BitsAndBytesConfig, deepspeed
import torch
from datasets import Dataset, load_dataset

from fastchat.train.train import (
    DataArguments,
    ModelArguments,
    make_supervised_data_module,
)

from fastchat.train.llama2_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

from trl import DPOTrainer
from fastchat.train.data_modules.dpo_dataset import (
    hankang_DPODataset,
    ados_DPODataset,
    load_dpo_data_module,
)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")  # QLoRA 데이터 클 때 깨지는 이슈 -> 변경해 볼 것
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    flash_attn: bool = False
    max_grad_norm: float = field(default=1.0)
    remove_unused_columns: bool = False,


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    load_in_8bit: bool = False
    lora_modules_to_save: List[str] = field(
        default_factory=lambda: ["embed_tokens", "lm_head"]
    )

@dataclass
class DPOArguments:
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    max_length: Optional[int] = field(default=512, metadata={"help": "max length of each sample"})
    max_prompt_length: Optional[int] = field(default=128, metadata={"help": "max length of each sample's prompt"})
    max_target_length: Optional[int] = field(
        default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    
def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

def train():
    # 0. prepare arguments and utilities
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments, DPOArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
        dpo_args,
    ) = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)
    if lora_args.q_lora:
        shutil.copy(
            "train_dpo_qlora.sh", os.path.join(training_args.output_dir, "train_dpo_qlora.sh")
        )
    else:
        shutil.copy(
            "train_dpo_lora.sh", os.path.join(training_args.output_dir, "train_dpo_lora.sh")
        )

    # 4. load dataset
    dpo_dataset = ados_DPODataset()
    dpo_datamodule = dpo_dataset.make_dpo_data_module()
    # train_dataset = dpo_datamodule['train_dataset'].shuffle(42)
    # eval_dataset = dpo_datamodule['eval_dataset'].shuffle(42)
        
    # 1. load model
    if training_args.flash_attn:
        replace_llama_attn_with_flash_attn()

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = (
        {"": int(os.environ.get("LOCAL_RANK") or 0)}
        if ddp
        and not deepspeed.is_deepspeed_zero3_enabled()
        and not lora_args.load_in_8bit
        else None
    )
    if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
        if lora_args.q_lora and training_args.local_rank == 0:
            logging.warning(
                "FSDP and ZeRO3 are both currently incompatible with QLoRA."
            )

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    if training_args.local_rank == 0:
        print("world_size:", world_size)
        print(
            "deepspeed.is_deepspeed_zero3_enabled():",
            deepspeed.is_deepspeed_zero3_enabled(),
        )
        print("device_map:", device_map)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        load_in_8bit=lora_args.load_in_8bit,
        torch_dtype=compute_dtype,
        max_memory={i: "81920MiB" for i in range(torch.cuda.device_count())},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        if lora_args.q_lora
        else None,
    )
    # 2. attach lora(qlora)
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        modules_to_save=lora_args.lora_modules_to_save,  # 23230913 added (v0.22)
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
    )

    if lora_args.q_lora or lora_args.load_in_8bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )
        if not ddp and torch.cuda.device_count() > 1:
            # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
            model.is_parallelizable = True
            model.model_parallel = True

    if lora_args.load_in_8bit:
        for param in model.parameters():
            # Check if parameter dtype is  Float (float32) due to prepare_model_for_kbit_training
            if param.dtype != compute_dtype:
                param.data = param.data.to(compute_dtype)

    model = get_peft_model(model, lora_config)
    if training_args.flash_attn:
        for name, module in model.named_modules():
            if "norm" in name:
                module = module.to(compute_dtype)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    module = module.to(compute_dtype)
    if training_args.deepspeed is not None and training_args.local_rank == 0:
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    model.config.use_cache = False

    # 3. load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        # use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token  # todo: 파라미터화
    # token extending fix
    # if len(tokenizer) != 32000:
    #     model.resize_token_embeddings(len(tokenizer))


    
    # 5. initialize the DPO trainer
    # trainer = Trainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=training_args,
    #     **data_module,
    # )
    dpo_trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        beta=dpo_args.beta,
        # train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        max_length=dpo_args.max_length,
        max_prompt_length=dpo_args.max_prompt_length,
        max_target_length=dpo_args.max_target_length,
        **dpo_datamodule,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        dpo_trainer.train(resume_from_checkpoint=True)
    else:
        dpo_trainer.train()
    dpo_trainer.save_state()

    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        # use deepspeed engine internal function to gather state dict
        # state_dict_zero3 contains whole parameters of base and lora adapters
        # we will not extract lora parameters since peft save_pretrained will do that
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/peft_model.py#L125
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/utils/save_and_load.py#L19
        state_dict_zero3 = dpo_trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
        if training_args.local_rank == 0:
            state_dict = state_dict_zero3
    else:
        # in other mode we use original code from fastchat team, to make sure our change is minimum
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), lora_args.lora_bias
        )

    if training_args.local_rank == 0:
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)


if __name__ == "__main__":
    train()
