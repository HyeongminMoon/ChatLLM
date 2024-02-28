from fastchat.model.model_adapter import (
    BaseModelAdapter,
    get_conv_template,
    register_model_adapter,
)
from fastchat.conversation import Conversation


class AULMAdapter(BaseModelAdapter):
    """The model adapter for AULM"""

    def match(self, model_path: str):
        return "aulm" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        conv_template = get_conv_template("aulm")
        conv_template.config = {
            "max_new_tokens": 4096,
        }
        return conv_template


### llama-2 orca 스타일 학습용
class ChatOrcaAdapter(BaseModelAdapter):
    """The model adapter for chat-orca"""

    def match(self, model_path: str):
        return ("SOLAR-10.7B-Instruct-v1.0" in model_path.lower() 
                or "chat-orca" in model_path.lower() 
                or "die" in model_path.lower()
               )

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        conv_template = get_conv_template("chat-orca")
        conv_template.config = {
            "max_new_tokens": 4096,
        }
        return conv_template

### ados ollm
class AdosOLLMAdapter(BaseModelAdapter):
    """The model adapter for ados-ollm"""

    def match(self, model_path: str):
        return ("ados-ollm" in model_path.lower()
                or "OLLM" in model_path
                or model_path in ["DIE_10.7b_sft_v4_dpo_v2_ep3", "MingAI-70B-chat-orca_v0.42_2_dpo-GPTQ", "M-DIE-M-10.7B_gpt4_dpo_ep2"]
               )

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        conv_template = get_conv_template("ados-ollm")
        conv_template.config = {
            "max_new_tokens": 4096,
        }
        return conv_template
    

class EnkotranslationOrcaAdapter(BaseModelAdapter):
    """The model adapter for enkotranslation-orca"""

    def match(self, model_path: str):
        return "enkotranslation-orca" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        conv_template = get_conv_template("enkotranslation-orca")
        conv_template.config = {
            "max_new_tokens": 4096,
        }
        return conv_template


class KoentranslationOrcaAdapter(BaseModelAdapter):
    """The model adapter for koentranslation-orca"""

    def match(self, model_path: str):
        return "koentranslation-orca" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        conv_template = get_conv_template("koentranslation-orca")
        conv_template.config = {
            "max_new_tokens": 4096,
        }
        return conv_template


class SummarizationOrcaAdapter(BaseModelAdapter):
    """The model adapter for summarization-orca"""

    def match(self, model_path: str):
        return "summarization-orca" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        conv_template = get_conv_template("summarization-orca")
        conv_template.config = {
            "max_new_tokens": 4096,
        }
        return conv_template


### polyglot-ko orca 스타일 학습용
class EnkotranslationKoOrcaAdapter(BaseModelAdapter):
    """The model adapter for enkotranslation-ko-orca"""

    def match(self, model_path: str):
        return "enkotranslation-ko-orca" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        conv_template = get_conv_template("enkotranslation-ko-orca")
        conv_template.config = {
            "max_new_tokens": 4096,
        }
        return conv_template


class KoentranslationKoOrcaAdapter(BaseModelAdapter):
    """The model adapter for koentranslation-ko-orca"""

    def match(self, model_path: str):
        return "koentranslation-ko-orca" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        conv_template = get_conv_template("koentranslation-ko-orca")
        conv_template.config = {
            "max_new_tokens": 4096,
        }
        return conv_template


class SummarizationKoOrcaAdapter(BaseModelAdapter):
    """The model adapter for summarization-ko-orca"""

    def match(self, model_path: str):
        return "summarization-ko-orca" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        conv_template = get_conv_template("summarization-ko-orca")
        conv_template.config = {
            "max_new_tokens": 4096,
        }
        return conv_template


class ChatKoOrcaAdapter(BaseModelAdapter):
    """The model adapter for chat-ko-orca"""

    def match(self, model_path: str):
        return "chat-ko-orca" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        conv_template = get_conv_template("chat-ko-orca")
        conv_template.config = {
            "max_new_tokens": 4096,
        }
        return conv_template


class AdosRefineAdapter(BaseModelAdapter):
    """The model adapter for AdosRefine"""

    def match(self, model_path: str):
        return "adosrefine" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        # model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            # device_map="auto",
            # torch_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            device_map="auto",
            # **from_pretrained_kwargs,
            torch_dtype=torch.float16,
        )

        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        conv_template = get_conv_template("adosrefine")
        conv_template.config = {
            "max_new_tokens": 4096,
        }
        return conv_template


class FreeWilly2Adapter(BaseModelAdapter):
    """The model adapter for freewilly2"""

    def match(self, model_path: str):
        return "freewilly2" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        conv_template = get_conv_template("freewilly2")
        conv_template.config = {
            "max_new_tokens": 4096,
        }
        return conv_template


class Platypus2Adapter(BaseModelAdapter):
    """The model adapter for Platypus2"""

    def match(self, model_path: str):
        return "platypus2" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        conv_template = get_conv_template("platypus2")
        conv_template.config = {
            "max_new_tokens": 4096,
        }
        return conv_template


class KoPolyglotAdapter(BaseModelAdapter):
    "Model adapater for polyglot-ko"

    def match(self, model_path: str):
        prefixes = ["polyglot-ko", "kopolyglot", "kmodelp", "kullm"]
        for prefix in prefixes:
            if prefix in model_path.lower():
                return True
        return False

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, revision=revision
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            # max_seq_len=4096,
            **from_pretrained_kwargs,
        )
        # model.config['max_position_embeddings'] = 4096
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        conv_template = get_conv_template("Koalpaca")
        if "kullm" in model_path:
            conv_template.config = {  # custom config for kullm
                "top_p": 0.1,
                "repetition_penalty": 1.18,
                "temperature": 1.0,
            }
        return conv_template

class DolphinMixAdapter(BaseModelAdapter):
    """Model adapter for dolphin-2.5-mixtral-8x7b-GPTQ"""

    def match(self, model_path: str):
        return "dolphin" in model_path.lower() and "mixtral" in model_path.lower()
    
    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=self.use_fast_tokenizer, revision=revision
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        ).eval()
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("dolphin-2.2.1-mistral-7b")

class MixtralAdapter(BaseModelAdapter):
    """The model adapter for Mixtral AI models"""

    def match(self, model_path: str):
        return "mixtral" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("mistral")

class llama2dpoAdapter(BaseModelAdapter):
    """The model adapter for llama2-13b-dpo"""
    def match(self, model_path: str):
        return "llama2-13b-dpo" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        conv_template = get_conv_template("starchat")
        return conv_template

class GugugoAdapter(BaseModelAdapter):
    """The model adapter for Gugugo-koen-7B-V1.1"""
    def match(self, model_path: str):
        return "gugugo" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        conv_template = get_conv_template("gugugo_enko")
        return conv_template

class SynatraAdapter(BaseModelAdapter):
    """The model adapter for Synatra-7B-v0.3-Translation"""
    def match(self, model_path: str):
        return "synatra" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        conv_template = get_conv_template("synatra_enko")
        return conv_template

class PIEChatAdapter(BaseModelAdapter):
    """The model adapter for PIE
    To run this model, you need to ensure additional flash attention installation:
    ``` bash
    git clone https://github.com/Dao-AILab/flash-attention
    cd flash-attention && pip install .
    pip install csrc/layer_norm
    pip install csrc/rotary
    ```

    Since from 2.0, the following change happened
    - `flash_attn_unpadded_func` -> `flash_attn_varlen_func`
    - `flash_attn_unpadded_qkvpacked_func` -> `flash_attn_varlen_qkvpacked_func`
    - `flash_attn_unpadded_kvpacked_func` -> `flash_attn_varlen_kvpacked_func`
    You may need to revise the code in: https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/modeling_qwen.py#L69
    to from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_unpadded_func
    """

    def match(self, model_path: str):
        return "pie" in model_path.lower()

    def float_set(self, config, option):
        config.bf16 = False
        config.fp16 = False
        config.fp32 = False

        if option == "bf16":
            config.bf16 = True
        elif option == "fp16":
            config.fp16 = True
        elif option == "fp32":
            config.fp32 = True
        else:
            print("Invalid option. Please choose one from 'bf16', 'fp16' and 'fp32'.")

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        from transformers.generation import GenerationConfig

        revision = from_pretrained_kwargs.get("revision", "main")
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        # NOTE: if you use the old version of model file, please remove the comments below
        # config.use_flash_attn = False
        self.float_set(config, "fp16")
        generation_config = GenerationConfig.from_pretrained(
            model_path, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **from_pretrained_kwargs,
        ).eval()
        if hasattr(model.config, "use_dynamic_ntk") and model.config.use_dynamic_ntk:
            model.config.max_sequence_length = 16384
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, revision=revision
        )
        tokenizer.eos_token_id = config.eos_token_id
        tokenizer.bos_token_id = config.bos_token_id
        tokenizer.pad_token_id = generation_config.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("qwen")

class CodeLlamaAdapter(BaseModelAdapter):
    """The model adapter for CodeLlama-70B-Instruct-GPTQ"""
    def match(self, model_path: str):
        return "codellama-70b" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        conv_template = get_conv_template("codellama_70b")
        return conv_template
    
class KiquAdapter(BaseModelAdapter):
    """kiqu-70b-awq"""
    def match(self, model_path: str):
        return "kiqu" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        conv_template = get_conv_template("mistral")
        return conv_template
    
# ados models
register_model_adapter(AdosRefineAdapter)
register_model_adapter(AdosOLLMAdapter)

# qwen
register_model_adapter(PIEChatAdapter)

# orca
register_model_adapter(ChatOrcaAdapter)
register_model_adapter(SummarizationOrcaAdapter)
register_model_adapter(KoentranslationOrcaAdapter)
register_model_adapter(EnkotranslationOrcaAdapter)

# ko-orca
register_model_adapter(ChatKoOrcaAdapter)
register_model_adapter(SummarizationKoOrcaAdapter)
register_model_adapter(KoentranslationKoOrcaAdapter)
register_model_adapter(EnkotranslationKoOrcaAdapter)

# korean models
register_model_adapter(AULMAdapter)
register_model_adapter(KoPolyglotAdapter)
register_model_adapter(KiquAdapter)

# others
register_model_adapter(FreeWilly2Adapter)
register_model_adapter(Platypus2Adapter)
register_model_adapter(DolphinMixAdapter)
register_model_adapter(MixtralAdapter)
register_model_adapter(llama2dpoAdapter)
register_model_adapter(CodeLlamaAdapter)

# translation
register_model_adapter(GugugoAdapter)
register_model_adapter(SynatraAdapter)
