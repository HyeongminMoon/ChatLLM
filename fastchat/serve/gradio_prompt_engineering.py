"""
Prompt Engineering tab.
"""

import json
import time
import glob
import os
import gradio as gr
import numpy as np

import requests

from fastchat.constants import (
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    INACTIVE_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
)
from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.gradio_block_arena_named import flash_buttons
from fastchat.serve.gradio_web_server import (
    State,
    bot_response,
    get_conv_log_filename,
    no_change_btn,
    enable_btn,
    disable_btn,
    learn_more_md,
    get_model_description_md,
    ip_expiration_dict,
    model_worker_stream_iter,
    controller_url,
)
from fastchat.utils import (
    build_logger,
    violates_moderation,
)

logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")

controller_url = None
enable_moderation = False

template_dir = 'data/prompt_templates'
prompt_templates = {}
template_keys = sorted(glob.glob(os.path.join(template_dir, '*.json')))
for key in template_keys:
    with open(key, 'r') as f:
        prompt_templates[os.path.basename(key)[:-5]] = json.load(f)

def load_prompt_templates(request: gr.Request):
    global prompt_templates
    prompt_templates = {}
    template_keys = sorted(glob.glob(os.path.join(template_dir, '*.json')))
    for key in template_keys:
        with open(key, 'r') as f:
            prompt_templates[os.path.basename(key)[:-5]] = json.load(f)
            
    return gr.Dropdown.update(
        choices=list(prompt_templates.keys()),
    )
    
def set_global_vars_pte(controller_url_, enable_moderation_):
    global controller_url, enable_moderation
    controller_url = controller_url_
    enable_moderation = enable_moderation_


def load_demo_prompt_engineering(models, url_params):
    states = (None,) * num_sides

    model_left = models[0] if len(models) > 0 else ""
    if len(models) > 1:
        weights = ([8] * 4 + [4] * 8 + [1] * 32)[: len(models) - 1]
        weights = weights / np.sum(weights)
        model_right = np.random.choice(models[1:], p=weights)
    else:
        model_right = model_left

    selector_updates = (
        gr.Dropdown.update(choices=models, value=model_left, visible=True),
        gr.Dropdown.update(choices=models, value=model_right, visible=True),
    )

    return (
        states
        + selector_updates
        + (gr.Chatbot.update(visible=True),) * num_sides
        + (
            gr.Textbox.update(visible=True),
            gr.Box.update(visible=True),
            gr.Row.update(visible=True),
            gr.Row.update(visible=True),
            gr.Accordion.update(visible=True),
        )
    )


def clear_history(request: gr.Request):
    logger.info(f"clear_history (named). ip: {request.client.host}")
    return [None] * num_sides + [None] * num_sides + [""] + [disable_btn] * 6

def clear_textbox(request: gr.Request):
    logger.info(f"clear_textbox (pte). ip: {request.client.host}")
    return ""

def model_change(model_selector, request: gr.Request):
    logger.info(f"model_change (pte). ip: {request.client.host}")
    state = State(model_selector)
    return state

def template_change(template_selector, request: gr.Request):
    logger.info(f"template_change (pte). ip: {request.client.host}")
    
    pt = prompt_templates[template_selector]
    sep = pt["separator"]
    conversations = pt["few_shot_learning"]
    
    prompt = pt["system"] + sep
    
    for i in range(len(conversations)):
        conv = conversations[i]
        prompt += pt["user_prefix"] + f": {conv[0]}"
        prompt += sep + pt["bot_prefix"] + f": {conv[1]}" + pt["stop_token"]
    
    prompt += pt["user_prefix"] + ": %PUT_YOUR_INPUT_HERE%"
    prompt += sep + pt["bot_prefix"] + ":"
    
    
    return prompt, pt["system"], pt["user_prefix"], "%PUT_YOUR_INPUT_HERE%", pt["bot_prefix"], pt["separator"].replace('\n', '\\n').replace('\t', '\\t'), pt["stop_token"], pt["few_shot_learning"]

def prompt_change(system_prompt, user_prompt, user_input, bot_prompt, sep_prompt, stop_token, few_shot_textbox, request: gr.Request):
    logger.info(f"prompt_change (pte). ip: {request.client.host}")
    
    sep = sep_prompt.replace('\\n', '\n').replace('\\t', '\t')
    
    conversations = []
    try:
        conversations = json.loads(few_shot_textbox.replace('\'', '\"'))
    except:
        return "Error: Few shot learning conversations format is wrong. Please fix it and apply again. It should be composed of double lists."
    
    prompt = system_prompt + sep
    
    for i in range(len(conversations)):
        conv = conversations[i]
        prompt += user_prompt + f": {conv[0]}"
        prompt += sep + bot_prompt + f": {conv[1]}" + stop_token
    
    prompt += user_prompt + f": {user_input}"
    prompt += sep + bot_prompt + ":"
    
    return prompt

def enable_save_option(request: gr.Request):
    logger.info(f"enable_save_option (pte). ip: {request.client.host}")
    return (
        gr.Button.update(visible=False),
        gr.Textbox.update(visible=True),
        gr.Button.update(visible=True),
        gr.Button.update(visible=True),
    )

def disable_save_option(request: gr.Request):
    logger.info(f"disable_save_option (pte). ip: {request.client.host}")
    return (
        gr.Button.update(visible=True),
        gr.Textbox.update(visible=False),
        gr.Button.update(visible=False),
        gr.Button.update(visible=False),
    )

def stop_response(state, request: gr.Request): 
    logger.info(f"stop (pte). ip: {request.client.host}")
    _ = requests.post(state.worker_addr + "/worker_stop_stream", timeout=5)

def save_prompt_template(prompt_save_name, system_prompt, user_prompt, bot_prompt, sep_prompt, stop_token, few_shot_textbox, request: gr.Request):
    logger.info(f"save_prompt_template (pte). ip: {request.client.host}")
    if not prompt_save_name:
        return f"Save Failed. (Invaild name)"
    dst_path = os.path.join(template_dir, prompt_save_name + '.json')
    if os.path.exists(dst_path):
        return f"Save Failed. (Template '{prompt_save_name}' already exists)"
    
    few_shot_prompt = []
    
    try:
        few_shot_prompt = json.loads(few_shot_textbox.replace('\'', '\"'))
    except:
        return "Error: Few shot learning conversations format is wrong. Please fix it and save again. It should be composed of double lists."
    
    temp = {
        "system": system_prompt,
        "user_prefix": user_prompt,
        "bot_prefix": bot_prompt,
        "separator": sep_prompt.replace('\\n', '\n').replace('\\t', '\t'),
        "stop_token": stop_token,
        "few_shot_learning": few_shot_prompt,
    }
    
    with open(dst_path, "w") as f:
        json.dump(temp, f)
    
    return f"Successfully saved template as {prompt_save_name}."


def bot_text_generate(
    state,
    model_selector,
    text_box,
    temperature,
    top_p,
    repetition_penalty,
    max_new_tokens,
    snippet_checkbox,
    user_prompt, bot_prompt, sep_prompt, stop_token,
    request: gr.Request,
):
    logger.info(f"bot_text_generate (pte). ip: {request.client.host}")

    if state is None:
        state = State(model_selector)
    
    conv = state.conv
    model_name = state.model_name
    
    # Query worker address
    ret = requests.post(
        controller_url + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    state.worker_addr = worker_addr
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        conv.update_last_message(SERVER_ERROR_MSG)
        yield (
            state,
            text_box,
        )
        return
    
    # Construct Prompt
    prompt = text_box

    stream_iter = model_worker_stream_iter(
        conv,
        model_name,
        worker_addr,
        prompt,
        temperature,
        repetition_penalty,
        top_p,
        max_new_tokens,
    )
    
    # Update text box
    for data in stream_iter:
        if data["error_code"] == 0:
            output = data["text"].strip()
            # if "vicuna" in model_name:
            #     output = post_process_code(output)
            yield (state, text_box + output)
            
    if snippet_checkbox:
        sep = sep_prompt.replace('\\n', '\n').replace('\\t', '\t')
        yield (state, text_box + output + stop_token + user_prompt +": %PUT_YOUR_INPUT_HERE%" + sep + bot_prompt + ":")


def build_prompt_engineering(models):
    notice_markdown = """
    test
"""
    state = gr.State()
    model_selector = None
    templates = prompt_templates.keys()
    
    with gr.Row():
        model_selector = gr.Dropdown(
            choices=models,
            value=models[0] if len(models) > 0 else "",
            interactive=True,
            show_label=True,
            label="Choose a model",
            container=True,
        )
        
        template_selector = gr.Dropdown(
            choices=templates,
            value="",
            interactive=True,
            label="Choose a prompt template",
            container=True,
            show_label=True,
        )

    with gr.Row():
        textbox = gr.Textbox(
            show_label=False,
            placeholder="Enter text and press SHIFT+ENTER",
            visible=True,
            container=False,
            lines=20,
        )

    with gr.Row():
        send_btn = gr.Button(value="Generate Text", visible=True)
        
    with gr.Row():
        stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=True)
        clear_btn = gr.Button(value="🗑️ Clear", interactive=True)

    with gr.Accordion("Advanced Options", open=True, visible=True) as advanced_row:
        input_explanation = gr.Markdown("%PUT_YOUR_INPUT_HERE%은 사용자 input값으로 대체하여 사용해야 합니다.")
        user_input = gr.Textbox(
            label="user input",
            placeholder="user input",
            interactive=True,
            container=False,
        )
        
        snippet_checkbox = gr.Checkbox(
            value=False,
            info="생성 완료 후 자동으로 다음 대화 템플릿 적용하기",
            interactive=True,
            container=False,
            label="사용",
        )
        
        prompt_explanation = gr.Markdown("**Prompt template 구성요소**")
        system_prompt = gr.Textbox(
            label="system",
            placeholder="System command",
            info="Prompt의 맨 앞에 위치하며, 인공지능의 역할을 정의합니다.",
            interactive=True,
        )
        
        user_prompt = gr.Textbox(
            label="user prefix",
            placeholder="user prefix",
            interactive=True,
        )
        
        bot_prompt = gr.Textbox(
            label="bot prefix",
            placeholder="bot prefix",
            interactive=True,
        )
        
        sep_prompt = gr.Textbox(
            label="separator",
            placeholder="separator",
            info="각 요소 사이의 분리자입니다. \\n=Enter \\t=Tab",
            interactive=True,
        )
        
        stop_token = gr.Textbox(
            label="(Optional)stop token",
            placeholder="stop token",
            info="Multi turn으로 학습된 모델에서 사용합니다. vicuna:</s> llama-2:<|endoftext|>",
            interactive=True,
        )
        
        few_shot_explanation = gr.Markdown("**Few shot learning 구성요소**\n\nFew shot learning이란 LLM이 몇가지의 예시만 보고도 그와 유사한 대답을 할 수 있게 하는 방법입니다.")
        few_shot_textbox = gr.Textbox(
            label="conversations",
            info="이중 리스트의 형태로 작성해야 합니다. ex) [[\"반가워!\", \"반갑습니다. 무엇을 도와드릴까요?\"], [\"UFO가 뭐야?\", \"UFO는 Unidentified Flying Object라는 뜻입니다.\"]]",
            placeholder="Conversations",
            interactive=True,
            lines=3,
        )
            

        prompt_apply_btn = gr.Button(value="Apply", interactive=True, variant='primary')
        with gr.Row():
            prompt_save_btn = gr.Button(value="Save as", interactive=True, scale=50)
            prompt_save_name = gr.Textbox(placeholder="Enter template name", interactive=True, container=False, visible=False, scale=50)
            prompt_save_confirm_btn = gr.Button(value="Confirm", interactive=True, visible=False, variant='stop', scale=20)
            prompt_save_cancel_btn = gr.Button(value="Cancel", interactive=True, visible=False, scale=20)
        logbox = gr.Markdown()
        
    with gr.Accordion("Parameters", open=False, visible=True) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
            info="토큰 선택시 무작위성을 조절합니다. 낮을수록 사실적인 대답을, 높을수록 창조적인 대답을 합니다.",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.8,
            step=0.1,
            interactive=True,
            label="Top P",
            info="샘플링 결정성을 조절합니다. 낮을수록 사실적인 대답을, 높을수록 창조적인 대답을 합니다.",
        )
        repetition_penalty = gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Repetition Penalty",
            info="반복 패널티를 조절합니다. 높을수록 단어가 반복되는 현상이 줄어듭니다.",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=4096,
            value=2048,
            step=64,
            interactive=True,
            label="Max output tokens",
            info="생성하는 최대 토큰 개수를 조절합니다.",
        )

    stop_btn.click(stop_response, state, None)
    clear_btn.click(clear_textbox, None, textbox)

    model_selector.change(model_change, model_selector, state).then(
        load_prompt_templates, None, template_selector,
    )
    template_selector.select(
        template_change, 
        template_selector, 
        [textbox, system_prompt, user_prompt, user_input, bot_prompt, sep_prompt, stop_token, few_shot_textbox],
    ).then(
        load_prompt_templates, None, template_selector,
    )
    prompt_apply_btn.click(
        prompt_change,
        [system_prompt, user_prompt, user_input, bot_prompt, sep_prompt, stop_token, few_shot_textbox],
        textbox,
    )
    
    
    prompt_save_btn.click(
        enable_save_option,
        None,
        [prompt_save_btn, prompt_save_name, prompt_save_confirm_btn, prompt_save_cancel_btn],
    )
    prompt_save_confirm_btn.click(
        save_prompt_template, 
        [prompt_save_name, system_prompt, user_prompt, bot_prompt, sep_prompt, stop_token, few_shot_textbox],
        logbox,
    ).then(
        disable_save_option, None, [prompt_save_btn, prompt_save_name, prompt_save_confirm_btn, prompt_save_cancel_btn],
    ).then(
        load_prompt_templates, None, template_selector,
    )
    prompt_save_cancel_btn.click(disable_save_option, None, [prompt_save_btn, prompt_save_name, prompt_save_confirm_btn, prompt_save_cancel_btn])
    
    
    textbox.submit(bot_text_generate,
                   [state, model_selector, textbox, temperature, top_p, repetition_penalty, max_output_tokens, snippet_checkbox, user_prompt, bot_prompt, sep_prompt, stop_token,],
                   [state, textbox],
    )
    send_btn.click(bot_text_generate,
                   [state, model_selector, textbox, temperature, top_p, repetition_penalty, max_output_tokens, snippet_checkbox, user_prompt, bot_prompt, sep_prompt, stop_token,],
                   [state, textbox],
    )
        

    return (
        state,
        model_selector,
        textbox,
        send_btn,
        clear_btn,
        parameter_row,
    )
