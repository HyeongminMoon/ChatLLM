"""
Chatbot Arena (side-by-side) tab.
Users chat with two chosen models.
"""

import json
import time

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
)
from fastchat.utils import (
    build_logger,
    violates_moderation,
)


logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")

cur_num_sides = 2
num_sides = 4
enable_moderation = False


def set_global_vars_named(enable_moderation_):
    global enable_moderation
    enable_moderation = enable_moderation_


def load_demo_side_by_side_named(models, url_params):
    states = (None,) * num_sides

    model_left = models[0] if len(models) > 0 else ""
    if len(models) > 1:
        weights = ([8] * 4 + [4] * 8 + [1] * 32)[: len(models) - 1]
        weights = weights / np.sum(weights)
        model_right = np.random.choice(models[1:], p=weights)
    else:
        model_right = model_left

    selector_updates = (
        gr.Dropdown.update(choices=models, value=models[0] if len(models) > 0 else "", visible=True),
        gr.Dropdown.update(choices=models, value=models[1] if len(models) > 1 else "", visible=True),
        gr.Dropdown.update(choices=models, value=models[2] if len(models) > 2 else "", visible=True),
        gr.Dropdown.update(choices=models, value=models[3] if len(models) > 3 else "", visible=True),
    )

    return (
        states
        + selector_updates
        + (gr.Chatbot.update(visible=True),) * num_sides
        + (
            gr.Textbox.update(visible=True),
            gr.Box.update(visible=True),
            gr.Row.update(visible=False),
            gr.Row.update(visible=True),
            gr.Accordion.update(visible=True),
        )
    )


def vote_last_response(states, vote_type, model_selectors, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "models": [x for x in model_selectors],
            "states": [x.dict() for x in states],
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def leftvote_last_response(
    state0, state1, state2, state3,
    model_selector0, model_selector1, model_selector2, model_selector3,
    request: gr.Request
):
    logger.info(f"leftvote (named). ip: {request.client.host}")
    vote_last_response(
        [state0, state1, state2, state3], "leftvote", [model_selector0, model_selector1, model_selector2, model_selector3], request
    )
    return ("",) + (disable_btn,) * 4


def rightvote_last_response(
    state0, state1, state2, state3,
    model_selector0, model_selector1, model_selector2, model_selector3,
    request: gr.Request
):
    logger.info(f"rightvote (named). ip: {request.client.host}")
    vote_last_response(
        [state0, state1, state2, state3], "rightvote", [model_selector0, model_selector1, model_selector2, model_selector3], request
    )
    return ("",) + (disable_btn,) * 4


def tievote_last_response(
    state0, state1, state2, state3,
    model_selector0, model_selector1, model_selector2, model_selector3,
    request: gr.Request
):
    logger.info(f"tievote (named). ip: {request.client.host}")
    vote_last_response(
       [state0, state1, state2, state3], "tievote", [model_selector0, model_selector1, model_selector2, model_selector3], request
    )
    return ("",) + (disable_btn,) * 4


def bothbad_vote_last_response(
    state0, state1, state2, state3,
    model_selector0, model_selector1, model_selector2, model_selector3,
    request: gr.Request
):
    logger.info(f"bothbad_vote (named). ip: {request.client.host}")
    vote_last_response(
        [state0, state1, state2, state3], "bothbad_vote", [model_selector0, model_selector1, model_selector2, model_selector3], request
    )
    return ("",) + (disable_btn,) * 4


def regenerate(
    state0, state1, state2, state3,
    request: gr.Request
):
    logger.info(f"regenerate (named). ip: {request.client.host}")
    states = [state0, state1, state2, state3]
    for i in range(num_sides):
        states[i].conv.update_last_message(None)
    return states + [x.to_gradio_chatbot() for x in states] + [""] + [disable_btn] * 4 + [enable_btn] + [disable_btn] * 2


def clear_history(request: gr.Request):
    logger.info(f"clear_history (named). ip: {request.client.host}")
    return [None] * num_sides + [None] * num_sides + [""] + [disable_btn] * 7


def share_click(
    state0, state1, state2, state3,
    model_selector0, model_selector1, model_selector2, model_selector3,
    request: gr.Request
):
    logger.info(f"share (named). ip: {request.client.host}")
    if state0 is not None and state1 is not None:
        vote_last_response(
            [state0, state1], "share", [model_selector0, model_selector1], request
        )

def append_model(
    state0, state1, state2, state3,
    model_selector0, model_selector1, model_selector2, model_selector3,
    chatbot0, chatbot1, chatbot2, chatbot3,
    request: gr.Request
):
    global cur_num_sides
    logger.info(f"append_model Model{cur_num_sides} (named). ip: {request.client.host}")
    # states[num_sides] = gr.State()
    # model_selectors[num_sides] = None
    # chatbots[num_sides] = None
    # num_sides += 1
    return

def stop_response(state0, state1, state2, state3,
                  model_selector0, model_selector1, model_selector2, model_selector3,
                  request: gr.Request): 
    logger.info(f"stop (named). ip: {request.client.host}")
    
    states = [state0, state1, state2, state3]
    model_selectors = [model_selector0, model_selector1, model_selector2, model_selector3]
    
    for state in states:
        try:
            _ = requests.post(state.worker_addr + "/worker_stop_stream", json={"session_id": state.conv_id}, timeout=5)
        except:
            pass
    return ("",) + (disable_btn,)

def add_text(
    state0, state1, state2, state3,
    model_selector0, model_selector1, model_selector2, model_selector3,
    text, request: gr.Request
):
    ip = request.client.host
    logger.info(f"add_text (named). ip: {ip}. len: {len(text)}")
    states = [state0, state1,  state2, state3]
    model_selectors = [model_selector0, model_selector1, model_selector2, model_selector3]

    # Init states if necessary
    for i in range(num_sides):
        if states[i] is None:
            states[i] = State(model_selectors[i])

    if len(text) <= 0:
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [""]
            + [
                no_change_btn,
            ]
            * 7
        )

    if ip_expiration_dict[ip] < time.time():
        logger.info(f"inactive (named). ip: {request.client.host}. text: {text}")
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [INACTIVE_MSG]
            + [
                no_change_btn,
            ]
            * 7
        )

    if enable_moderation:
        flagged = violates_moderation(text)
        if flagged:
            logger.info(
                f"violate moderation (named). ip: {request.client.host}. text: {text}"
            )
            for i in range(num_sides):
                states[i].skip_next = True
            return (
                states
                + [x.to_gradio_chatbot() for x in states]
                + [MODERATION_MSG]
                + [
                    no_change_btn,
                ]
                * 7
            )

    conv = states[0].conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {request.client.host}. text: {text}")
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [CONVERSATION_LIMIT_MSG]
            + [
                no_change_btn,
            ]
            * 7
        )

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    for i in range(num_sides):
        if model_selectors[i]:
            states[i].conv.append_message(states[i].conv.roles[0], text)
            states[i].conv.append_message(states[i].conv.roles[1], None)
            states[i].skip_next = False

    return (
        states
        + [x.to_gradio_chatbot() for x in states]
        + [""]
        + [
            disable_btn,
        ]
        * 7
    )


def bot_response_multi(
    state0, state1, state2, state3,
    temperature,
    top_p,
    repetition_penalty,
    max_new_tokens,
    request: gr.Request,
):
    logger.info(f"bot_response_multi (named). ip: {request.client.host}")

    if state0.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (
            state0,
            state1,
            state2,
            state3,
            state0.to_gradio_chatbot(),
            state1.to_gradio_chatbot(),
            state2.to_gradio_chatbot(),
            state3.to_gradio_chatbot(),
        ) + (no_change_btn,) * 7
        return

    states = [state0, state1, state2, state3]
    gen = []
    for i in range(num_sides):
        gen.append(
            bot_response(
                states[i],
                temperature,
                top_p,
                repetition_penalty,
                max_new_tokens,
                request,
            )
        )

    chatbots = [None] * num_sides
    while True:
        stop = True
        for i in range(num_sides):
            try:
                ret = next(gen[i])
                states[i], chatbots[i] = ret[0], ret[1]
                stop = False
            except StopIteration:
                pass
        yield states + chatbots + [disable_btn] * 4 + [enable_btn] + [disable_btn] * 2
        if stop:
            break


def flash_buttons():
    btn_updates = [
        [enable_btn] * 4 + [disable_btn] * 1 + [enable_btn] * 2,
        [enable_btn] * 7,
    ]
    for i in range(5):
        yield btn_updates[i % 2]
        time.sleep(0.2)


def build_side_by_side_ui_named(models):
    notice_markdown = """
    - 최대 4개의 모델을 동시에 비교하세요.
"""
# # ⚔️  Chatbot Arena ⚔️ 
# ### Rules
# - Chat with two models side-by-side and vote for which one is better!
# - You pick the models you want to chat with.
# - You can do multiple rounds of conversations before voting.
# - Click "Clear history" to start a new round.
# - | [Blog](https://lmsys.org/blog/2023-05-03-arena/) | [GitHub](https://github.com/lm-sys/FastChat) | [Paper](https://arxiv.org/abs/2306.05685) | [Twitter](https://twitter.com/lmsysorg) | [Discord](https://discord.gg/HSWAKCrnFx) |

# ### Terms of use
# By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. **The service collects user dialogue data and reserves the right to distribute it under a Creative Commons Attribution (CC-BY) license.** The demo works better on desktop devices with a wide screen.

# ### Choose two models to chat with (view [leaderboard](?leaderboard))

    states = [gr.State() for _ in range(num_sides)]
    model_selectors = [None] * num_sides
    chatbots = [None] * num_sides

    # model_description_md = get_model_description_md(models)
    notice = gr.Markdown(
        notice_markdown, elem_id="notice_markdown"
    )

    with gr.Box(elem_id="share-region-named"):
        with gr.Row():
            for i in range(num_sides):
                # if i < 2:
                with gr.Column():
                    model_selectors[i] = gr.Dropdown(
                        choices=models,
                        value=models[i] if len(models) > i else "",
                        interactive=True,
                        show_label=False,
                        container=False,
                        scale=200,
                    )
                # else:
                #     with gr.Column(scale=20):
                #         model_selectors[i] = gr.Dropdown(
                #             choices=models,
                #             value= "",
                #             interactive=True,
                #             show_label=False,
                #             container=False,
                #             visible=False,
                #         )
            # with gr.Column(scale=1, max_width=20):
            # append_model_btn = gr.Button(value="+", elem_id=f"plus", visible=True, scale=1, size='sm')

        with gr.Row():
            for i in range(num_sides):
                # label = "Model A" if i == 0 else "Model B"
                label = f"Model {i}"
                # if i < 2:
                with gr.Column(scale=20):
                    chatbots[i] = gr.Chatbot(
                        label=label, elem_id=f"chatbot_multi", visible=False, height=550, 
                    )
                # else:
                #     with gr.Column(scale=20):
                #         chatbots[i] = gr.Chatbot(
                #             label=label, elem_id=f"chatbot", visible=False, height=550, 
                #         )
            # with gr.Column(scale=1, min_width=20, height=550):
            #     gr.Button(value="+", elem_id=f"plus", visible=False, height=550)

        with gr.Box() as button_row:
            with gr.Row():
                leftvote_btn = gr.Button(value="👈  A is better", interactive=False, visible=False)
                rightvote_btn = gr.Button(value="👉  B is better", interactive=False, visible=False)
                tie_btn = gr.Button(value="🤝  Tie", interactive=False, visible=False)
                bothbad_btn = gr.Button(value="👎  Both are bad", interactive=False, visible=False)

    with gr.Row():
        with gr.Column(scale=20):
            textbox = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press ENTER",
                visible=False,
                container=False,
            )
        with gr.Column(scale=1, min_width=50):
            send_btn = gr.Button(value="Send", visible=False)

    with gr.Row() as button_row2:
        stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        clear_btn = gr.Button(value="🧹  Clear history", interactive=False)
        # share_btn = gr.Button(value="📷  Share", visible=False)

    with gr.Accordion("Parameters", open=False, visible=True) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.8,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        repetition_penalty = gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Repetition Penalty",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=10240,
            value=2048,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    gr.Markdown(learn_more_md)

    # Register listeners
    btn_list = [
        leftvote_btn,
        rightvote_btn,
        tie_btn,
        bothbad_btn,
        stop_btn,
        regenerate_btn,
        clear_btn,
    ]
    leftvote_btn.click(
        leftvote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    rightvote_btn.click(
        rightvote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    tie_btn.click(
        tievote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    bothbad_btn.click(
        bothbad_vote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    regenerate_btn.click(
        regenerate, states, states + chatbots + [textbox] + btn_list
    ).then(
        bot_response_multi,
        states + [temperature, top_p, repetition_penalty, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )
    clear_btn.click(clear_history, None, states + chatbots + [textbox] + btn_list)

    stop_btn.click(
        stop_response,
        states + model_selectors,
        [textbox, stop_btn]
    )
    
    # append_model_btn.click(append_model, states + model_selectors + chatbots, states + model_selectors + chatbots)
    
    share_js = """
function (a, b, c, d) {
    const captureElement = document.querySelector('#share-region-named');
    html2canvas(captureElement)
        .then(canvas => {
            canvas.style.display = 'none'
            document.body.appendChild(canvas)
            return canvas
        })
        .then(canvas => {
            const image = canvas.toDataURL('image/png')
            const a = document.createElement('a')
            a.setAttribute('download', 'chatbot-arena.png')
            a.setAttribute('href', image)
            a.click()
            canvas.remove()
        });
    return [a, b, c, d];
}
"""
    # share_btn.click(share_click, states + model_selectors, [], _js=share_js)

    for i in range(num_sides):
        model_selectors[i].change(
            clear_history, None, states + chatbots + [textbox] + btn_list
        )

    textbox.submit(
        add_text,
        states + model_selectors + [textbox],
        states + chatbots + [textbox] + btn_list,
    ).then(
        bot_response_multi,
        states + [temperature, top_p, repetition_penalty, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )
    send_btn.click(
        add_text,
        states + model_selectors + [textbox],
        states + chatbots + [textbox] + btn_list,
    ).then(
        bot_response_multi,
        states + [temperature, top_p, repetition_penalty, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )

    return (
        states,
        model_selectors,
        chatbots,
        textbox,
        send_btn,
        button_row,
        button_row2,
        parameter_row,
    )
