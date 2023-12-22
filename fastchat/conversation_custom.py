import dataclasses
from enum import auto, IntEnum
from typing import List, Any, Dict
from datetime import datetime

from fastchat.conversation import (
    conv_templates,
    register_conv_template,
    Conversation,
    SeparatorStyle,
)

### llama-2 orca 스타일 학습용
# chat-orca template
register_conv_template(
    Conversation(
        name="chat-orca",
        system_message="### System:\nThis is a system prompt, please behave and help the user.",
        roles=("### User", "### Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="</s>",
    )
)

# enkotranslation-orca template
register_conv_template(
    Conversation(
        name="enkotranslation-orca",
        system_message="### System:\nThis is a system prompt, Convert english sentences to korean sentences. Regenerate the data by referring to the original content, not just translating it.",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="</s>",
    )
)

# koentranslation-orca template
register_conv_template(
    Conversation(
        name="koentranslation-orca",
        system_message="### System:\nThis is a system prompt, Convert korean sentences to english sentences. Regenerate the data by referring to the original content, not just translating it.",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="</s>",
    )
)

# summarization-orca template
register_conv_template(
    Conversation(
        name="summarization-orca",
        system_message="### System:\nThis is a system prompt, Summarize below sentences.",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="</s>",
    )
)


### polyglot-ko orca 스타일 학습용
# 아직 없음

# 다음은 유저와 어시스턴트의 대화이다. 한국어 대화 형식으로 변환하라. 단순 번역이 아닌, 원문 내용을 참조하여 데이터를 재생성하라.
# freewilly2-adosrefine template
register_conv_template(
    Conversation(
        name="adosrefine",
        system_message="### System:\nThis is a conversation between a user and an assistant. Convert it to Korean dialog format. Regenerate the data by referring to the original content, not just translating it.\n\n### Instruction: Q: Hi\nA: Hi, How can I help you today?\n\n### Response: Q: 안녕\nA: 안녕하세요, 오늘은 어떻게 도와드릴까요?",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n\n",
    )
)

## korean models
# AdosBot template
register_conv_template(
    Conversation(
        name="AdosBot",
        system_message="You are an artificial intelligence assistant named Adosbot developed by Mohomin. And this is a chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

# AdosTarotBot template
register_conv_template(
    Conversation(
        name="AdosTarotBot",
        system_message="You are an artificial intelligence assistant named AdosTarotBot developed by Mohomin. And this is a chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. ",
        roles=("USER", "ASSISTANT"),
        messages=(
            (
                "USER",
                "I want you to act as a tarot counselor. I will speak to you in English and you will reply to me in English to explain the result of tarot. At first, you need to ask to me what I want to be consult about. And you will pick three tarot cards at random. I want you to explain the results of tarot cards in relation to the consultation topic I mentioned. I want you to keep your reply neat, limiting the reply to 100 words.",
            ),
            (
                "ASSISTANT",
                "Sure, I can act as a tarot counselor for you. To begin, please let me know what you would like to consult about. Once you have provided me with the topic, I will randomly select three tarot cards and provide you with a brief explanation of their meaning in relation to your consultation. Please note that the tarot is a tool for divination and its interpretation is subjective, so the meaning may vary depending on your own perspective and experiences.",
            ),
        ),
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

# Koalpaca template
register_conv_template(
    Conversation(
        name="Koalpaca",
        system_message="",
        roles=("### 질문", "### 답변"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n\n",
        stop_str="###",
    )
)

# Platypus2 template
register_conv_template(
    Conversation(
        name="platypus2",
        system_message="",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="</s>",
    )
)

# AULM template
register_conv_template(
    Conversation(
        name="aulm",
        system_message="당신은 한국어 챗봇 아우름입니다. 사용자에게 도움이 되고 유익한 내용을 제공해야합니다. 답변은 길고 자세해야 하며, 반드시 친절한 설명을 포함해야합니다.",
        roles=("### 사용자", "### 챗봇"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="<|endoftext|>",
        stop_str="<|endoftext|>",
    )
)

### polyglot-ko orca 스타일 학습용
# chat-orca template
register_conv_template(
    Conversation(
        name="chat-ko-orca",
        system_message="### 명령어:\n이것은 명령 프롬프트입니다, 당신은 사용자에게 도움이 되고 유익한 내용을 제공하는 어시스턴트입니다. 답변은 길고 자세해야 하며, 반드시 친절한 설명을 포함해야합니다.",
        roles=("### 사용자", "### 어시스턴트"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="<|endoftext|>",
        stop_str="<|endoftext|>",
    )
)

# enkotranslation-orca template
register_conv_template(
    Conversation(
        name="enkotranslation-ko-orca",
        system_message="### 명령어:\n이것은 명령 프롬프트입니다, 영어 문장을 한국어 문장으로 변환하세요. 단순히 번역하는 것이 아닌, 본래의 뜻에 기반하여 데이터를 재생성하세요.",
        roles=("### 질문", "### 답변"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="<|endoftext|>",
        stop_str="<|endoftext|>",
    )
)

# koentranslation-orca template
register_conv_template(
    Conversation(
        name="koentranslation-ko-orca",
        system_message="### 명령어:\n이것은 명령 프롬프트입니다, 한국어 문장을 영어 문장으로 변환하세요. 단순히 번역하는 것이 아닌, 본래의 뜻에 기반하여 데이터를 재생성하세요.",
        roles=("### 질문", "### 답변"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="<|endoftext|>",
        stop_str="<|endoftext|>",
    )
)

# summarization-orca template
register_conv_template(
    Conversation(
        name="summarization-ko-orca",
        system_message="### 명령어:\n이것은 명령 프롬프트입니다, 아래의 내용을 요약하세요.",
        roles=("### 질문", "### 답변"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="<|endoftext|>",
        stop_str="<|endoftext|>",
    )
)

# gugugo template
register_conv_template(
    Conversation(
        name="gugugo_enko",
        system_message="",
        roles=("### 영어:", "### 한국어:"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="\n",
        stop_str=["</끝>", "###"],
    )
)

# Synatra template
register_conv_template(
    Conversation(
        name="synatra_enko",
        system_message="<|im_start|>system\n주어진 문장을 한국어로 번역해라.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="<|im_end|>\n",
        stop_str="<|im_end|>",
    )
)